"""标签构造: 给每个 (ts_code, t) 计算"未来 5 个交易日"的 3 个回归 target.

时间口径:
    t = 当前交易日 (典型周五收盘)
    用 close[t] 做参考 → 计算 t+1 ~ t+5 这 5 个交易日的:
        y_mean_ret = mean(close[t+1..t+5]) / close[t] - 1
        y_max_ret  = max(high[t+1..t+5])  / close[t] - 1
        y_min_ret  = min(low[t+1..t+5])   / close[t] - 1

为什么用 close[t] 而不是 open[t+1] 做参考?
    - 与特征的"as-of t"时间点对齐, 模型学的是"给定 t 时的状态, 下周相对 t 收盘的 range"
    - open[t+1] 实盘交易价 vs close[t] 通常只差一个 gap, 误差可控
    - 训练标签和回测执行口径分离, 是常见做法

不向 panel 加列, 单独返回 long format DataFrame, 保持模块独立.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LOOKAHEAD_DAYS = 5  # "下周" = 下 5 个交易日


def _per_ticker_labels(g: pd.DataFrame) -> pd.DataFrame:
    """单只标的 (已按 trade_date 升序) 的标签."""
    g = g.sort_values("trade_date").reset_index(drop=True)
    close = g["close"].to_numpy()
    high = g["high"].to_numpy()
    low = g["low"].to_numpy()
    n = len(g)

    y_mean = np.full(n, np.nan)
    y_max = np.full(n, np.nan)
    y_min = np.full(n, np.nan)

    for i in range(n - LOOKAHEAD_DAYS):
        ref = close[i]
        if ref <= 0 or not np.isfinite(ref):
            continue
        next_close = close[i + 1: i + 1 + LOOKAHEAD_DAYS]
        next_high = high[i + 1: i + 1 + LOOKAHEAD_DAYS]
        next_low = low[i + 1: i + 1 + LOOKAHEAD_DAYS]
        y_mean[i] = next_close.mean() / ref - 1.0
        y_max[i] = next_high.max() / ref - 1.0
        y_min[i] = next_low.min() / ref - 1.0

    out = g[["ts_code", "trade_date"]].copy()
    out["y_mean"] = y_mean
    out["y_max"] = y_max
    out["y_min"] = y_min
    return out


def build_labels(panel: pd.DataFrame) -> pd.DataFrame:
    """对每只标的计算下周 3 个 target, 拼起来返回 long format."""
    panel = panel.sort_values(["ts_code", "trade_date"])
    parts = [_per_ticker_labels(g) for _, g in panel.groupby("ts_code", sort=False)]
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    """sanity check."""
    from src.build_panel import load_panel

    panel = load_panel()
    labels = build_labels(panel)
    print(f"标签行数: {len(labels):,}")
    print(f"NaN 统计 (尾部 5 行因 lookahead 不够预期会有):")
    print(labels.isna().sum())
    print(f"\nAAPL 最早能看到标签的 5 行:")
    aapl = labels[labels["ts_code"] == "AAPL"].dropna()
    print(aapl.head())
    print(f"\nAAPL 最晚能看到标签的 5 行:")
    print(aapl.tail())


if __name__ == "__main__":
    main()
