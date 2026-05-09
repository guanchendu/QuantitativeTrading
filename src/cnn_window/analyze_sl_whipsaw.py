"""SL whipsaw 分析: 找出被 -3% 砍仓但后来又涨回去的"冤枉"交易.

口径:
    对每一笔被 SL 砍掉的交易, 假设我们没设 SL, 看看到周五收盘会变什么样.
    然后按 (周五收盘价 - SL 出场价) 排序, 列出最"可惜"的 N 笔.

衡量"反弹"的 3 个口径:
    1. recover_friday : (周五收盘 - 买入价) / 买入价      ← 实际"如果不止损"的最终结果
    2. max_after_sl   : (SL 触发日之后最高价 - 买入价)/买入价  ← 期间能不能反扑到正区间
    3. saved_amount   : recover_friday - sl_ret           ← 这一笔被 SL 砍掉了多少潜在收益

用法:
    python -m src.cnn_window.analyze_sl_whipsaw                       # 默认 K=5, SL=3%
    python -m src.cnn_window.analyze_sl_whipsaw --top 30 --k 5
    python -m src.cnn_window.analyze_sl_whipsaw --save data/sl_whipsaw.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.build_panel import load_panel
from src.predict_weekly_winner import get_rebalance_dates
from src.predict_weekly_winner_ml import SL_FIXED, TP_FACTOR, TP_MAX, TP_MIN

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREDS = PROJECT_ROOT / "data" / "preds_cnn.csv"


def analyze_whipsaws(
    panel: pd.DataFrame,
    preds: pd.DataFrame,
    k: int = 5,
    sl: float = SL_FIXED,
    tp_factor: float = TP_FACTOR,
) -> pd.DataFrame:
    """复刻一遍 backtest 的选股 / 模拟逻辑, 但同时记录"如果不 SL 会怎样"."""
    rebals_all = get_rebalance_dates(panel)
    pred_dates = set(pd.to_datetime(preds["trade_date"]).unique())
    rebals = [d for d in rebals_all if d in pred_dates]

    panel_by_ticker = {tc: g.sort_values("trade_date").reset_index(drop=True)
                       for tc, g in panel.groupby("ts_code")}
    preds_by_date = {d: g for d, g in preds.groupby("trade_date")}

    sl_trades: list[dict] = []

    for i in range(len(rebals) - 1):
        rebal = rebals[i]
        next_rebal = rebals[i + 1]

        pred_today = preds_by_date.get(rebal)
        if pred_today is None or len(pred_today) < k:
            continue
        top = pred_today.nlargest(k, "y_mean_hat")

        for _, row in top.iterrows():
            ts_code = row["ts_code"]
            ymax_hat = float(row["y_max_hat"])
            tp = float(np.clip(tp_factor * ymax_hat, TP_MIN, TP_MAX))

            full = panel_by_ticker.get(ts_code)
            if full is None:
                continue
            days = full[
                (full["trade_date"] > rebal) & (full["trade_date"] <= next_rebal)
            ].reset_index(drop=True)
            if days.empty:
                continue

            buy_price = float(days.iloc[0]["open"])
            sl_price = buy_price * (1 - sl)
            tp_price = buy_price * (1 + tp)

            # 找出场日: 同 simulate_trade 的优先级 (SL 先于 TP)
            exit_idx = None
            exit_reason = None
            for idx in range(len(days)):
                d = days.iloc[idx]
                if d["low"] <= sl_price:
                    exit_idx, exit_reason = idx, "SL"
                    break
                if d["high"] >= tp_price:
                    exit_idx, exit_reason = idx, "TP"
                    break
            if exit_reason != "SL":
                continue   # 我们只关心 SL 单

            sl_date = days.iloc[exit_idx]["trade_date"]
            sl_ret = -sl

            # 假如不 SL: 持到周五收盘
            friday_close = float(days.iloc[-1]["close"])
            recover_friday = (friday_close - buy_price) / buy_price

            # SL 后到周五的最高价 (能反扑多高)
            after_sl = days.iloc[exit_idx:]
            max_high_after = float(after_sl["high"].max())
            max_recovery = (max_high_after - buy_price) / buy_price

            sl_trades.append({
                "ts_code":         ts_code,
                "rebal":           rebal.date(),
                "buy_date":        days.iloc[0]["trade_date"].date(),
                "sl_date":         sl_date.date(),
                "friday":          days.iloc[-1]["trade_date"].date(),
                "buy_price":       round(buy_price, 2),
                "sl_price":        round(sl_price, 2),
                "friday_close":    round(friday_close, 2),
                "max_high_after_sl": round(max_high_after, 2),
                "sl_ret":          sl_ret,
                "recover_friday":  recover_friday,
                "max_recovery":    max_recovery,
                "saved_if_no_sl":  recover_friday - sl_ret,   # 不 SL 能多赚多少 (可正可负)
            })

    return pd.DataFrame(sl_trades)


def report(df: pd.DataFrame, top_n: int = 20) -> None:
    if df.empty:
        print("没有 SL 触发的交易, 这不太合理 — 检查输入.")
        return

    n = len(df)
    n_recover = (df["recover_friday"] > -SL_FIXED).sum()       # 周五时不至于亏 3%
    n_breakeven = (df["recover_friday"] >= 0).sum()             # 周五时回到正
    n_max_above = (df["max_recovery"] > 0).sum()                # 期间曾创正区间

    avg_sl = df["sl_ret"].mean()
    avg_recover = df["recover_friday"].mean()
    total_saved = df["saved_if_no_sl"].sum()

    print(f"\n=== SL Whipsaw 分析 (共 {n} 笔 SL 交易) ===\n")
    print(f"  实际 SL 平均收益       : {avg_sl:>+7.2%}  (定义就是 -3%)")
    print(f"  若不 SL, 持到周五平均   : {avg_recover:>+7.2%}")
    print(f"  其中:")
    print(f"    周五收盘没 SL 也能少亏 (>-3%): {n_recover} 笔 ({n_recover/n:.1%})")
    print(f"    周五收盘回到正  (>= 0)      : {n_breakeven} 笔 ({n_breakeven/n:.1%})")
    print(f"    SL 后曾创正区间 (期间最高>0): {n_max_above} 笔 ({n_max_above/n:.1%})")

    print(f"\n  累计被砍掉的潜在收益: {total_saved:+.2%}  "
          f"(若不 SL, 单笔总和能多 {total_saved:+.2%})")
    print(f"  按 K=5 等权折算到组合周收益: {total_saved/5:+.2%}")

    # 最"可惜"的 — 周五涨最多的
    print(f"\n--- Top {top_n} '最冤枉的' SL 交易 (按周五收盘涨幅排序) ---\n")
    cols = ["ts_code", "buy_date", "sl_date", "friday",
            "buy_price", "sl_price", "friday_close",
            "sl_ret", "recover_friday", "saved_if_no_sl"]
    show = df.sort_values("recover_friday", ascending=False).head(top_n)[cols].copy()
    # 百分比格式
    for c in ["sl_ret", "recover_friday", "saved_if_no_sl"]:
        show[c] = show[c].map(lambda x: f"{x:+.2%}")
    print(show.to_string(index=False))

    # 最"伤"的 — 反过来, SL 真的救了我们的 (周五跌得更狠)
    saved_well = df[df["recover_friday"] < df["sl_ret"]]
    if not saved_well.empty:
        print(f"\n  ── 反观: SL 救命的次数 ──")
        print(f"  {len(saved_well)} 笔 ({len(saved_well)/n:.1%}) "
              f"周五收盘比 SL 价更低, 即 SL 帮你避免了更大亏损")
        print(f"  这些交易若不 SL, 平均会亏到: "
              f"{saved_well['recover_friday'].mean():+.2%}")

    # 频次最高的"被冤枉股票"
    recovered = df[df["recover_friday"] > -SL_FIXED]
    if not recovered.empty:
        counts = recovered["ts_code"].value_counts().head(10)
        print(f"\n  ── 最常被冤枉的股票 (Top 10) ──")
        for code, c in counts.items():
            avg_rec = recovered[recovered["ts_code"] == code]["recover_friday"].mean()
            print(f"    {code:<6}  {c} 次  平均若不 SL 周五收益 {avg_rec:+.2%}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--preds", type=Path, default=DEFAULT_PREDS)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--sl", type=float, default=SL_FIXED)
    p.add_argument("--tp-factor", type=float, default=TP_FACTOR)
    p.add_argument("--top", type=int, default=20, help="列出多少笔最冤枉的, 默认 20")
    p.add_argument("--save", type=Path, default=None,
                   help="把所有 SL 交易明细 CSV 保存, 路径相对项目根")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"加载预测: {args.preds}")
    preds = pd.read_csv(args.preds, parse_dates=["trade_date"])
    print(f"  {len(preds):,} 行")

    print("加载面板...")
    panel = load_panel()
    print(f"  {len(panel):,} 行")

    df = analyze_whipsaws(panel, preds, k=args.k, sl=args.sl, tp_factor=args.tp_factor)
    report(df, top_n=args.top)

    if args.save:
        path = PROJECT_ROOT / args.save
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"\n明细已写: {path}  ({len(df)} 行)")


if __name__ == "__main__":
    main()
