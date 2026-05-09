"""v1 周频选股策略: 人工因子打分 + Take-Profit / Stop-Loss / Time-stop 回测.

策略规则 (v1 定稿):
  信号:  每周最后一个交易日 (典型周五) 收盘, 用 5 个因子的 z-score 加权打分
  入场:  下周第一个交易日 (典型周一) 按开盘价等权买入 top-K 只
  出场:  按以下优先级 (同日 SL 和 TP 都触发时, 保守假设 SL 先发生):
           1. 当日 Low  <= 买入价 × (1 - SL)  →  以 SL 价出场
           2. 当日 High >= 买入价 × (1 + TP)  →  以 TP 价出场
           3. 下周最后一个交易日 Close       →  时间止损出场
  基准:  同期持有 SPY 的等长收益

特征权重 (v1 拍脑袋, 后续可调):
    + 1.0 * mom_4w_z          # 月动量, 主信号
    + 0.5 * mom_1w_z          # 周动量
    - 0.5 * vol_4w_z          # 高波动率惩罚
    + 0.3 * dist_from_high_z  # 接近 52 周高 = 强势
    + 0.2 * vol_ratio_1_4_z   # 近期成交活跃

用法:
    python -m src.predict_weekly_winner
    python -m src.predict_weekly_winner --k 5 --tp 0.04 --sl 0.03
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.build_panel import load_panel
from src.features import build_features, cross_sectional_zscore
from src.universe import BENCHMARK, TECH_50

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_COLS = ["mom_1w", "mom_4w", "vol_4w", "dist_from_high", "vol_ratio_1_4"]
SCORE_WEIGHTS = {
    "mom_4w_z":           1.0,
    "mom_1w_z":           0.5,
    "vol_4w_z":          -0.5,
    "dist_from_high_z":   0.3,
    "vol_ratio_1_4_z":    0.2,
}


# -------------------------- 数据准备 ----------------------------------------

def get_rebalance_dates(panel: pd.DataFrame) -> list[pd.Timestamp]:
    """每个交易周的最后一个交易日 (一般周五, 假期时是周四).

    用 ISO 年-周 分组, 取每组最大 trade_date.
    """
    dates = panel["trade_date"].drop_duplicates().sort_values()
    iso = dates.dt.isocalendar()
    df = pd.DataFrame({"date": dates.values, "year": iso.year.values, "week": iso.week.values})
    last = df.groupby(["year", "week"], as_index=False)["date"].max()
    return sorted(last["date"].tolist())


def compute_scores(features_today: pd.DataFrame) -> pd.Series:
    """加权 z-score, 返回 ts_code -> score 的 Series."""
    score = sum(features_today[col] * w for col, w in SCORE_WEIGHTS.items())
    return pd.Series(score.values, index=features_today["ts_code"].values)


# -------------------------- 单笔交易模拟 -------------------------------------

@dataclass
class TradeResult:
    ts_code: str
    rebalance_date: pd.Timestamp
    buy_date: pd.Timestamp
    buy_price: float
    exit_date: pd.Timestamp
    exit_price: float
    exit_reason: str  # 'TP' / 'SL' / 'TIME'
    hold_days: int
    ret: float


def simulate_trade(
    days: pd.DataFrame, buy_price: float, tp: float, sl: float,
    rebal: pd.Timestamp, ts_code: str,
    sl_use_close: bool = False,
) -> TradeResult:
    """给定下周持仓期内的日 OHLC, 模拟一笔交易.

    days 必须按 trade_date 升序, 第一行就是买入日.

    sl_use_close=False (默认, 旧行为):
        盘中触发 SL — 当日 low <= sl_price 即砍仓, 出场价 = sl_price.
        同日 H/L 都触发时, 假设 SL 先发生 (保守).

    sl_use_close=True (新选项):
        收盘触发 SL — 只看 close, 当日 close <= sl_price 才砍, 出场价 = close.
        TP 仍在盘中检查 (high >= tp_price), 因此同日先 TP 后 SL.
        优点: 过滤掉日内"插针"假摔; 缺点: 真崩盘当日多亏 1-2%.
    """
    sl_price = buy_price * (1.0 - sl)
    tp_price = buy_price * (1.0 + tp)
    buy_date = days.iloc[0]["trade_date"]

    for i, row in enumerate(days.itertuples(index=False), 1):
        if sl_use_close:
            # TP 优先 (盘中可触发), SL 等到收盘后才确定
            if row.high >= tp_price:
                return TradeResult(
                    ts_code=ts_code, rebalance_date=rebal,
                    buy_date=buy_date, buy_price=buy_price,
                    exit_date=row.trade_date, exit_price=tp_price,
                    exit_reason="TP", hold_days=i,
                    ret=(tp_price - buy_price) / buy_price,
                )
            if row.close <= sl_price:
                return TradeResult(
                    ts_code=ts_code, rebalance_date=rebal,
                    buy_date=buy_date, buy_price=buy_price,
                    exit_date=row.trade_date, exit_price=float(row.close),
                    exit_reason="SL", hold_days=i,
                    ret=(row.close - buy_price) / buy_price,
                )
        else:
            # 老逻辑: SL 优先 (保守), 都用盘中价
            if row.low <= sl_price:
                return TradeResult(
                    ts_code=ts_code, rebalance_date=rebal,
                    buy_date=buy_date, buy_price=buy_price,
                    exit_date=row.trade_date, exit_price=sl_price,
                    exit_reason="SL", hold_days=i,
                    ret=(sl_price - buy_price) / buy_price,
                )
            if row.high >= tp_price:
                return TradeResult(
                    ts_code=ts_code, rebalance_date=rebal,
                    buy_date=buy_date, buy_price=buy_price,
                    exit_date=row.trade_date, exit_price=tp_price,
                    exit_reason="TP", hold_days=i,
                    ret=(tp_price - buy_price) / buy_price,
                )

    # 时间止损: 最后一日收盘出
    last = days.iloc[-1]
    return TradeResult(
        ts_code=ts_code, rebalance_date=rebal,
        buy_date=buy_date, buy_price=buy_price,
        exit_date=last["trade_date"], exit_price=float(last["close"]),
        exit_reason="TIME", hold_days=len(days),
        ret=(last["close"] - buy_price) / buy_price,
    )


# -------------------------- 主回测循环 ---------------------------------------

def run_backtest(
    panel: pd.DataFrame, features: pd.DataFrame,
    k: int, tp: float, sl: float,
) -> tuple[list[TradeResult], pd.DataFrame]:
    """walk-forward 周频回测.

    返回:
      trades:       全部单笔交易记录
      weekly:       每周一行的 DataFrame (含组合周收益、SPY 基准、累计净值)
    """
    rebals = get_rebalance_dates(panel)
    panel_by_ticker = {tc: g.sort_values("trade_date").reset_index(drop=True)
                       for tc, g in panel.groupby("ts_code")}

    trades: list[TradeResult] = []
    weekly_rows: list[dict] = []

    for i in range(len(rebals) - 1):
        rebal = rebals[i]
        next_rebal = rebals[i + 1]

        # 选 top-K 标的 (只在 TECH_50 里挑, 不挑 ETF)
        feats_today = features[
            (features["trade_date"] == rebal) & (features["ts_code"].isin(TECH_50))
        ].dropna(subset=FEATURE_COLS)
        if len(feats_today) < k:
            continue

        scores = compute_scores(feats_today)
        top_k = scores.nlargest(k).index.tolist()

        # 模拟每只的下周持仓
        week_trades: list[TradeResult] = []
        for ts_code in top_k:
            full = panel_by_ticker.get(ts_code)
            if full is None:
                continue
            days = full[(full["trade_date"] > rebal) & (full["trade_date"] <= next_rebal)]
            if days.empty:
                continue
            buy_price = float(days.iloc[0]["open"])
            tr = simulate_trade(days.reset_index(drop=True),
                                buy_price=buy_price, tp=tp, sl=sl,
                                rebal=rebal, ts_code=ts_code)
            week_trades.append(tr)
            trades.append(tr)

        if not week_trades:
            continue

        port_ret = float(np.mean([t.ret for t in week_trades]))

        # SPY 基准: 同期 Mon open -> Fri close
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
            "next_rebalance": next_rebal,
            "top_k": ",".join(top_k),
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


# -------------------------- 统计指标 -----------------------------------------

WEEKS_PER_YEAR = 52


def perf_stats(returns: pd.Series, label: str) -> dict:
    r = returns.dropna()
    if r.empty or r.std() == 0:
        return {"label": label, "ann_ret": 0.0, "ann_vol": 0.0,
                "sharpe": 0.0, "mdd": 0.0, "win_rate": 0.0, "weeks": len(r)}
    equity = (1.0 + r).cumprod()
    ann_ret = equity.iloc[-1] ** (WEEKS_PER_YEAR / len(r)) - 1
    ann_vol = r.std() * np.sqrt(WEEKS_PER_YEAR)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    mdd = (equity / equity.cummax() - 1).min()
    win = (r > 0).mean()
    return {"label": label, "ann_ret": ann_ret, "ann_vol": ann_vol,
            "sharpe": sharpe, "mdd": mdd, "win_rate": win, "weeks": len(r)}


def fmt(s: dict) -> str:
    return (f"{s['label']:<18} "
            f"年化={s['ann_ret']:>7.2%}  "
            f"波动={s['ann_vol']:>6.2%}  "
            f"Sharpe={s['sharpe']:>5.2f}  "
            f"最大回撤={s['mdd']:>7.2%}  "
            f"胜率={s['win_rate']:>5.1%}  "
            f"周数={s['weeks']:>3d}")


# -------------------------- CLI ---------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--k", type=int, default=5, help="每周持仓数 (top-K), 默认 5")
    p.add_argument("--tp", type=float, default=0.04, help="止盈阈值, 默认 0.04 (即 +4%)")
    p.add_argument("--sl", type=float, default=0.03, help="止损阈值, 默认 0.03 (即 -3%)")
    p.add_argument("--save-trades", type=Path, default=None,
                   help="把每笔交易写到 CSV, 路径相对项目根")
    p.add_argument("--save-weekly", type=Path, default=None,
                   help="把每周收益写到 CSV")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("加载面板 + 计算特征...")
    panel = load_panel()
    feats = build_features(panel)
    feats = cross_sectional_zscore(feats, FEATURE_COLS)

    print(f"\n=== v1 人工打分策略  K={args.k}  TP={args.tp:.0%}  SL={args.sl:.0%} ===\n")
    trades, weekly = run_backtest(panel, feats, k=args.k, tp=args.tp, sl=args.sl)

    if weekly.empty:
        print("没有产出任何周收益, 检查数据/特征.")
        return

    # 整体表现
    rows = [
        perf_stats(weekly["port_ret"], "策略 (top-K 等权)"),
        perf_stats(weekly["bench_ret"], "基准 (SPY B&H)"),
    ]
    for r in rows:
        print(fmt(r))

    # 出场原因分布
    n_total = len(trades)
    n_tp = sum(1 for t in trades if t.exit_reason == "TP")
    n_sl = sum(1 for t in trades if t.exit_reason == "SL")
    n_time = sum(1 for t in trades if t.exit_reason == "TIME")
    print(f"\n出场原因分布 (总单数 {n_total}):")
    print(f"  止盈 TP   : {n_tp:>4d} ({n_tp/n_total:>5.1%})")
    print(f"  止损 SL   : {n_sl:>4d} ({n_sl/n_total:>5.1%})")
    print(f"  时间止损  : {n_time:>4d} ({n_time/n_total:>5.1%})")
    avg_hold = np.mean([t.hold_days for t in trades])
    print(f"  平均持仓天数: {avg_hold:.2f} 天")

    print(f"\n回测期: {weekly['rebalance'].min().date()} ~ {weekly['rebalance'].max().date()}")
    print(f"最终净值: 策略 {weekly['port_equity'].iloc[-1]:.3f}, "
          f"基准 {weekly['bench_equity'].iloc[-1]:.3f}")

    # 落盘 (可选)
    if args.save_trades:
        path = PROJECT_ROOT / args.save_trades
        pd.DataFrame([t.__dict__ for t in trades]).to_csv(path, index=False)
        print(f"\n交易明细已写: {path}")
    if args.save_weekly:
        path = PROJECT_ROOT / args.save_weekly
        weekly.to_csv(path, index=False)
        print(f"周度收益已写: {path}")


if __name__ == "__main__":
    main()
