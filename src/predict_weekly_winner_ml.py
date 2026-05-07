"""ML 版周频选股: 线性回归 baseline + LightGBM, 用预测值驱动选股 / TP.

流程:
    1. 构造 ML 数据集 = features (11 列) + labels (y_mean/max/min)
    2. 单次切分 train/test (2025-01 之前为训练, 之后为测试)
    3. 训练两个模型, 各自预测 y_mean / y_max / y_min
    4. 评估指标: IC (rank correlation), R^2
    5. 用预测值做选股 + 动态 TP, 跑回测对比 SPY 和 v1 启发式

策略规则 (沿用 v1, 仅 TP 改成预测值驱动):
    每周五收盘:
        对所有 TECH_50 股票预测 (y_mean_hat, y_max_hat, y_min_hat)
        按 y_mean_hat 排序, 选 top-K
    下周一开盘等权买入. 出场优先级:
        1. 当日 Low <= 买入价 × (1 - SL_FIXED)             (固定 -3% 止损)
        2. 当日 High >= 买入价 × (1 + TP_FACTOR × y_max_hat) (动态止盈, 0.7 倍预测高)
        3. 周五 Close                                       (时间止损)

用法:
    python -m src.predict_weekly_winner_ml
    python -m src.predict_weekly_winner_ml --k 5 --train-end 2025-01-01
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

import lightgbm as lgb

from src.build_panel import load_panel
from src.features import build_features
from src.labels import build_labels
from src.predict_weekly_winner import (
    TradeResult, get_rebalance_dates, perf_stats, simulate_trade, fmt,
)
from src.universe import BENCHMARK, TECH_50

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_COLS = [
    "mom_1w", "mom_4w", "mom_12w", "rev_1w",
    "vol_4w", "vol_12w", "rsi_14",
    "dist_from_high", "vol_ratio_1_4",
    "spy_mom_4w", "vix_z",
]
TARGET_COLS = ["y_mean", "y_max", "y_min"]

# 回测参数
SL_FIXED = 0.03         # 固定止损 -3%
TP_FACTOR = 0.7         # 止盈触发 = 0.7 × 预测最大涨幅
TP_MIN = 0.015          # TP 不低于 +1.5%, 防止预测太悲观时 TP 被压到 0
TP_MAX = 0.10           # TP 不高于 +10%, 防止预测过于乐观


# -------------------------- 数据集准备 ---------------------------------------

def build_ml_dataset(panel: pd.DataFrame) -> pd.DataFrame:
    """构造 (features + labels) 长面板, 仅 TECH_50, dropna 后返回."""
    feats = build_features(panel)
    labels = build_labels(panel)
    df = feats.merge(labels, on=["ts_code", "trade_date"], how="inner")
    df = df[df["ts_code"].isin(TECH_50)].copy()
    df = df.dropna(subset=FEATURE_COLS + TARGET_COLS).reset_index(drop=True)
    return df


# -------------------------- 模型: 线性回归 + LightGBM -----------------------

def train_predict_ridge(
    train: pd.DataFrame, test: pd.DataFrame,
) -> tuple[dict[str, Ridge], pd.DataFrame]:
    """每个 target 训一个 Ridge, 返回 (模型字典, test 加上预测列的 DataFrame)."""
    models: dict[str, Ridge] = {}
    out = test[["ts_code", "trade_date"]].copy()
    X_train = train[FEATURE_COLS].values
    X_test = test[FEATURE_COLS].values
    for tgt in TARGET_COLS:
        m = Ridge(alpha=1.0, random_state=42)
        m.fit(X_train, train[tgt].values)
        out[f"{tgt}_hat"] = m.predict(X_test)
        models[tgt] = m
    return models, out


def train_predict_lgbm(
    train: pd.DataFrame, test: pd.DataFrame,
) -> tuple[dict[str, lgb.LGBMRegressor], pd.DataFrame]:
    """每个 target 训一个 LightGBM."""
    models: dict[str, lgb.LGBMRegressor] = {}
    out = test[["ts_code", "trade_date"]].copy()
    X_train = train[FEATURE_COLS].values
    X_test = test[FEATURE_COLS].values
    for tgt in TARGET_COLS:
        m = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1,
        )
        m.fit(X_train, train[tgt].values)
        out[f"{tgt}_hat"] = m.predict(X_test)
        models[tgt] = m
    return models, out


# -------------------------- 评估: IC / R^2 ----------------------------------

def evaluate_predictions(test: pd.DataFrame, preds: pd.DataFrame, label: str) -> None:
    """打印 R^2 (整体准确度) + IC (横截面排序相关性) + IC t-stat."""
    print(f"\n--- {label} 预测质量 ---")
    merged = test.merge(preds, on=["ts_code", "trade_date"])
    for tgt in TARGET_COLS:
        y_true = merged[tgt].values
        y_hat = merged[f"{tgt}_hat"].values
        r2 = r2_score(y_true, y_hat)
        # IC: 每个交易日做一次 spearman, 取均值; t-stat = mean / (std/sqrt(n))
        ic_per_day = []
        for d, g in merged.groupby("trade_date"):
            if len(g) >= 5:
                rho, _ = spearmanr(g[tgt].values, g[f"{tgt}_hat"].values)
                if np.isfinite(rho):
                    ic_per_day.append(rho)
        ic_arr = np.array(ic_per_day) if ic_per_day else np.array([np.nan])
        ic_mean = ic_arr.mean()
        ic_std = ic_arr.std() if len(ic_arr) > 1 else np.nan
        t_stat = ic_mean / (ic_std / np.sqrt(len(ic_arr))) if ic_std > 0 else np.nan
        print(f"  {tgt:<8}  R²={r2:>7.4f}   IC均值={ic_mean:>+.4f}  "
              f"IC_t={t_stat:>+5.2f}  样本日={len(ic_arr)}")


# -------------------------- 回测: 用预测做选股 + 动态 TP --------------------

def run_backtest_ml(
    panel: pd.DataFrame, preds: pd.DataFrame, k: int,
    sl: float = SL_FIXED, tp_factor: float = TP_FACTOR,
) -> tuple[list[TradeResult], pd.DataFrame]:
    """走预测驱动的回测.

    在 preds 覆盖的所有交易日中, 找出 rebalance Friday, 选 top-K, 模拟下周.
    """
    rebals_all = get_rebalance_dates(panel)
    pred_dates = set(pd.to_datetime(preds["trade_date"]).unique())
    # 只在有预测的 Friday 上交易
    rebals = [d for d in rebals_all if d in pred_dates]

    panel_by_ticker = {tc: g.sort_values("trade_date").reset_index(drop=True)
                       for tc, g in panel.groupby("ts_code")}
    preds_by_date = {d: g for d, g in preds.groupby("trade_date")}

    trades: list[TradeResult] = []
    weekly_rows: list[dict] = []

    for i in range(len(rebals) - 1):
        rebal = rebals[i]
        next_rebal = rebals[i + 1]

        pred_today = preds_by_date.get(rebal)
        if pred_today is None or len(pred_today) < k:
            continue

        # 选 y_mean_hat top-K
        top = pred_today.nlargest(k, "y_mean_hat")
        top_k_codes = top["ts_code"].tolist()
        top_y_max_hat = dict(zip(top["ts_code"], top["y_max_hat"]))

        week_trades: list[TradeResult] = []
        for ts_code in top_k_codes:
            full = panel_by_ticker.get(ts_code)
            if full is None:
                continue
            days = full[(full["trade_date"] > rebal) & (full["trade_date"] <= next_rebal)]
            if days.empty:
                continue
            buy_price = float(days.iloc[0]["open"])

            # 动态 TP, 截断
            ymax_hat = float(top_y_max_hat[ts_code])
            tp = float(np.clip(tp_factor * ymax_hat, TP_MIN, TP_MAX))

            tr = simulate_trade(days.reset_index(drop=True),
                                buy_price=buy_price, tp=tp, sl=sl,
                                rebal=rebal, ts_code=ts_code)
            week_trades.append(tr)
            trades.append(tr)

        if not week_trades:
            continue

        port_ret = float(np.mean([t.ret for t in week_trades]))

        # SPY 基准
        spy = panel_by_ticker[BENCHMARK]
        spy_days = spy[(spy["trade_date"] > rebal) & (spy["trade_date"] <= next_rebal)]
        if len(spy_days) >= 1:
            spy_buy = float(spy_days.iloc[0]["open"])
            spy_sell = float(spy_days.iloc[-1]["close"])
            bench_ret = (spy_sell - spy_buy) / spy_buy
        else:
            bench_ret = np.nan

        weekly_rows.append({
            "rebalance": rebal,
            "port_ret": port_ret,
            "bench_ret": bench_ret,
            "n_trades": len(week_trades),
            "n_tp": sum(1 for t in week_trades if t.exit_reason == "TP"),
            "n_sl": sum(1 for t in week_trades if t.exit_reason == "SL"),
            "n_time": sum(1 for t in week_trades if t.exit_reason == "TIME"),
        })

    weekly = pd.DataFrame(weekly_rows)
    if not weekly.empty:
        weekly["port_equity"] = (1.0 + weekly["port_ret"]).cumprod()
        weekly["bench_equity"] = (1.0 + weekly["bench_ret"]).cumprod()
    return trades, weekly


# -------------------------- 主流程 -------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--train-end", type=str, default="2025-01-01",
                   help="训练集截止日 (含), 默认 2025-01-01; 之后做 OOS 测试")
    return p.parse_args()


def report_backtest(label: str, trades: list[TradeResult], weekly: pd.DataFrame) -> None:
    """打印一次回测的统计."""
    print(f"\n--- {label} 回测 ---")
    if weekly.empty:
        print("  无周收益")
        return
    rows = [
        perf_stats(weekly["port_ret"], f"{label} 策略"),
        perf_stats(weekly["bench_ret"], f"{label} SPY"),
    ]
    for r in rows:
        print("  " + fmt(r))
    n_total = len(trades)
    n_tp = sum(1 for t in trades if t.exit_reason == "TP")
    n_sl = sum(1 for t in trades if t.exit_reason == "SL")
    n_time = sum(1 for t in trades if t.exit_reason == "TIME")
    print(f"  出场分布: TP {n_tp}/{n_total} ({n_tp/n_total:.1%}), "
          f"SL {n_sl}/{n_total} ({n_sl/n_total:.1%}), "
          f"TIME {n_time}/{n_total} ({n_time/n_total:.1%})")
    print(f"  期末净值: 策略 {weekly['port_equity'].iloc[-1]:.3f}, "
          f"基准 {weekly['bench_equity'].iloc[-1]:.3f}")


def main() -> None:
    args = parse_args()
    train_end = pd.Timestamp(args.train_end)

    print("加载面板 / 构造特征 + 标签...")
    panel = load_panel()
    df = build_ml_dataset(panel)
    print(f"ML 数据集: {len(df):,} 行 (TECH_50 only, dropna 后)")
    print(f"日期范围  : {df['trade_date'].min().date()} ~ {df['trade_date'].max().date()}")
    print(f"特征: {len(FEATURE_COLS)} 个, 目标: {len(TARGET_COLS)} 个")

    train = df[df["trade_date"] < train_end].copy()
    test = df[df["trade_date"] >= train_end].copy()
    print(f"\n训练集: {len(train):,} 行 ({train['trade_date'].min().date()} ~ "
          f"{train['trade_date'].max().date()})")
    print(f"测试集: {len(test):,} 行 ({test['trade_date'].min().date()} ~ "
          f"{test['trade_date'].max().date()})")

    print("\n训练 Ridge ...")
    _, ridge_preds = train_predict_ridge(train, test)
    evaluate_predictions(test, ridge_preds, "Ridge")

    print("\n训练 LightGBM ...")
    _, lgbm_preds = train_predict_lgbm(train, test)
    evaluate_predictions(test, lgbm_preds, "LightGBM")

    # 把 LightGBM 的特征重要度也打出来
    print("\nLightGBM 特征重要度 (gain, 对 y_mean):")
    train_X = train[FEATURE_COLS].values
    m = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1, importance_type="gain",
    )
    m.fit(train_X, train["y_mean"].values)
    imp = pd.Series(m.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    for name, val in imp.items():
        print(f"  {name:<18} {val:>10.0f}")

    # 用预测做回测
    print("\n========== 回测 (test 期间) ==========")
    ridge_trades, ridge_weekly = run_backtest_ml(panel, ridge_preds, k=args.k)
    report_backtest("Ridge", ridge_trades, ridge_weekly)

    lgbm_trades, lgbm_weekly = run_backtest_ml(panel, lgbm_preds, k=args.k)
    report_backtest("LightGBM", lgbm_trades, lgbm_weekly)


if __name__ == "__main__":
    main()
