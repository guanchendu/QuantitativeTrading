"""跑 CNN 预测产出的 data/preds_cnn.csv 的周频选股回测.

直接复用 src.predict_weekly_winner_ml.run_backtest_ml — 同一套规则:
    每周五收盘按 y_mean_hat 选 top-K
    周一开盘等权买入
    出场: SL 固定 -3% / 动态 TP = 0.7 × y_max_hat (clip 到 1.5%~10%) / 周五时间止损

回测产出:
    - 终端: 策略 / SPY 的年化 / 波动 / Sharpe / 最大回撤 / 胜率 + 出场分布
    - 可选: data/cnn_trades.csv (每笔交易) / data/cnn_weekly.csv (每周收益)

用法:
    python -m src.cnn_window.backtest                            # 默认 K=5
    python -m src.cnn_window.backtest --k 3 --tp-factor 0.5
    python -m src.cnn_window.backtest --preds data/preds_cnn.csv \\
        --save-trades data/cnn_trades.csv --save-weekly data/cnn_weekly.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.build_panel import load_panel
from src.predict_weekly_winner_ml import (
    SL_FIXED, TP_FACTOR,
    report_backtest, run_backtest_ml,
)


def print_pnl_breakdown(trades: list, k: int = 5) -> None:
    """按出场原因 (TP / SL / TIME) 拆 P&L, 看每类各赚 / 亏多少.

    展示口径:
        - 平均单笔: 这一类交易的平均收益率
        - 最好 / 最差: 单笔区间
        - 盈利 / 亏损 / 持平: 笔数分布
        - 单笔总和: 简单求和 (∑ret), 不复利, 用来感受相对量级
        - 周组合贡献 ≈ 单笔总和 / K, 仍是简化口径,
          真实累计收益看 weekly['port_equity'].iloc[-1]
    """
    if not trades:
        return

    rets_by_reason = {"TP": [], "SL": [], "TIME": []}
    for t in trades:
        if t.exit_reason in rets_by_reason:
            rets_by_reason[t.exit_reason].append(t.ret)

    n_total = len(trades)
    print(f"\n--- 出场原因 P&L 明细 (共 {n_total} 笔, K={k}) ---")

    cn_label = {"TP": "止盈", "SL": "止损", "TIME": "时间到期"}
    for reason in ("TP", "SL", "TIME"):
        rets = np.array(rets_by_reason[reason], dtype=float)
        if len(rets) == 0:
            continue
        n = len(rets)
        wins = int((rets > 0).sum())
        losses = int((rets < 0).sum())
        flat = int((rets == 0).sum())
        s = rets.sum()
        print(f"\n【{cn_label[reason]} {reason}】 {n} 笔 ({n / n_total:.1%})")
        print(f"  平均单笔   : {rets.mean():>+7.2%}")
        print(f"  最好 / 最差: {rets.max():>+7.2%}  /  {rets.min():>+7.2%}")
        print(f"  盈 / 亏 / 平: {wins:>3d}  /  {losses:>3d}  /  {flat:>3d}")
        print(f"  单笔收益总和: {s:>+8.2%}")
        print(f"  周组合贡献 ≈: {s / k:>+8.2%}    (单笔总和 / K, 简化估算)")

    # 极端样本: 给你看具体哪几笔最赚 / 最亏
    sorted_trades = sorted(trades, key=lambda t: t.ret)
    print("\n  ── 单笔最差 5 笔 ──")
    for t in sorted_trades[:5]:
        print(f"    {t.ts_code:<6} 买于 {t.buy_date.date()}  "
              f"出场 {t.exit_date.date()} ({t.exit_reason})  "
              f"持 {t.hold_days} 天  收益 {t.ret:>+7.2%}")
    print("  ── 单笔最好 5 笔 ──")
    for t in sorted_trades[-5:][::-1]:
        print(f"    {t.ts_code:<6} 买于 {t.buy_date.date()}  "
              f"出场 {t.exit_date.date()} ({t.exit_reason})  "
              f"持 {t.hold_days} 天  收益 {t.ret:>+7.2%}")


def print_capital_utilization(trades: list, k: int = 5,
                              days_per_week: int = 5) -> None:
    """资金利用率诊断: 触发 TP / SL 后到下周一之前, 钱是闲置的.

    口径:
        实际仓位日数 = sum(hold_days)            每只股票各持几天加总
        满仓位日数   = 周数 × K × 一周交易日数    若每只都持满
        利用率      = 实际 / 满仓
        闲置率      = 1 - 利用率                  这部分钱没产生收益, 拖年化
    """
    if not trades:
        return

    hold_days = np.array([t.hold_days for t in trades], dtype=float)
    n = len(trades)
    n_weeks = n / k                                 # 每周 K 笔, 反推周数
    actual = hold_days.sum()
    ideal = n_weeks * k * days_per_week
    util = actual / ideal if ideal > 0 else 0.0

    print(f"\n--- 资金利用率诊断 ---")
    print(f"  交易笔数         : {n}")
    print(f"  平均持仓天数     : {hold_days.mean():.2f} 天   "
          f"(满仓应为 {days_per_week} 天)")
    print(f"  持仓天数  最短/最长: {int(hold_days.min())} / {int(hold_days.max())} 天")
    print(f"  实际仓位日数     : {int(actual):>5d}")
    print(f"  满仓仓位日数     : {int(ideal):>5d}")
    print(f"  仓位日利用率     : {util:>6.1%}   ← 资金有效工作时间比例")
    print(f"  闲置资金占比     : {1-util:>6.1%}   ← TP/SL 出场后到下周一前 0 收益")

    # 按出场原因分别看
    print(f"\n  分类持仓天数:")
    for reason in ("TP", "SL", "TIME"):
        rs = np.array([t.hold_days for t in trades if t.exit_reason == reason])
        if len(rs):
            print(f"    {reason:<5} 平均 {rs.mean():.2f} 天   "
                  f"最短 {int(rs.min())}  最长 {int(rs.max())}   "
                  f"贡献仓位日数 {int(rs.sum()):>4d}")

    # 推算: 如果出场后能立即换仓, 每年理论上能多赚多少 (粗估)
    if util < 0.95:
        print(f"\n  💡 改进空间: 若 TP / SL 后立即换入下一名股票, "
              f"利用率可拉到 ~95%+")
        print(f"     按当前年化推算, 这部分闲置资金多赚的潜力 ≈ "
              f"{(1 - util) * 0.10:.1%} ~ {(1 - util) * 0.15:.1%} 年化"
              f" (假设新仓位也能赚 10-15%)")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREDS = PROJECT_ROOT / "data" / "preds_cnn.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--preds", type=Path, default=DEFAULT_PREDS,
                   help="train.py 产出的预测 CSV, 默认 data/preds_cnn.csv")
    p.add_argument("--k", type=int, default=5, help="每周持仓数, 默认 5")
    p.add_argument("--sl", type=float, default=SL_FIXED,
                   help=f"固定止损, 默认 {SL_FIXED}")
    p.add_argument("--tp-factor", type=float, default=TP_FACTOR,
                   help=f"动态 TP = tp_factor × y_max_hat, 默认 {TP_FACTOR}")
    p.add_argument("--sl-mode", choices=["intraday", "close"], default="close",
                   help="SL 触发口径: intraday=盘中 low 触发(老行为), "
                        "close=收盘价触发(默认, 过滤日内插针)")
    p.add_argument("--save-trades", type=Path, default=None,
                   help="保存每笔交易明细 CSV, 路径相对项目根")
    p.add_argument("--save-weekly", type=Path, default=None,
                   help="保存每周组合收益 CSV")
    p.add_argument("--label", default="CNN", help="终端打印用的策略名, 默认 CNN")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.preds.exists():
        raise FileNotFoundError(
            f"找不到预测 CSV: {args.preds}\n"
            f"先跑: python -m src.cnn_window.train"
        )

    print(f"加载预测: {args.preds}")
    preds = pd.read_csv(args.preds, parse_dates=["trade_date"])
    need_cols = {"ts_code", "trade_date", "y_mean_hat", "y_max_hat", "y_min_hat"}
    missing = need_cols - set(preds.columns)
    if missing:
        raise ValueError(f"预测 CSV 缺少列: {missing}")
    print(f"  {len(preds):,} 行, "
          f"{preds['trade_date'].min().date()} ~ {preds['trade_date'].max().date()}, "
          f"{preds['ts_code'].nunique()} 只股票")

    print("\n加载面板...")
    panel = load_panel()
    print(f"  {len(panel):,} 行, {panel['ts_code'].nunique()} 只标的")

    sl_use_close = (args.sl_mode == "close")
    print(f"\n=== 回测  K={args.k}  SL={args.sl:.0%}  "
          f"TP_factor={args.tp_factor}  SL_mode={args.sl_mode} ===")
    trades, weekly = run_backtest_ml(
        panel, preds, k=args.k, sl=args.sl, tp_factor=args.tp_factor,
        sl_use_close=sl_use_close,
    )
    report_backtest(args.label, trades, weekly)
    print_pnl_breakdown(trades, k=args.k)
    print_capital_utilization(trades, k=args.k)

    if args.save_trades:
        path = PROJECT_ROOT / args.save_trades
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([t.__dict__ for t in trades]).to_csv(path, index=False)
        print(f"\n交易明细已写: {path}")
    if args.save_weekly:
        path = PROJECT_ROOT / args.save_weekly
        path.parent.mkdir(parents=True, exist_ok=True)
        weekly.to_csv(path, index=False)
        print(f"周度收益已写: {path}")


if __name__ == "__main__":
    main()
