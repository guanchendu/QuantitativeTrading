"""特征工程: 把面板转成可用于打分 / 建模的特征表.

重要纪律:
    所有特征在时刻 t 必须只用 t 时刻 (含) 之前的数据计算, 否则 look-ahead bias.
    pandas rolling 默认就是这个行为 (window 截止到当前 index), 我们利用之.

个股特征 (每行 = (ts_code, trade_date)):
    mom_1w           : 最近 5 日累计收益 = close[t]/close[t-5] - 1
    mom_4w           : 最近 20 日累计收益
    mom_12w          : 最近 60 日累计收益 (长期动量)
    rev_1w           : -mom_1w (短期反转, 学术上有 alpha)
    vol_4w           : 最近 20 日日收益标准差
    vol_12w          : 最近 60 日日收益标准差
    rsi_14           : 14 日 RSI
    dist_from_high   : close[t] / max(close[t-252..t]) - 1, [-1, 0]
    vol_ratio_1_4    : 5日均量 / 20日均量

宏观特征 (按 trade_date 广播给所有股票, 同一日同一值):
    spy_mom_4w       : SPY 最近 20 日收益 (大盘趋势)
    vix_z            : VIXY 收盘价的 252 日 z-score (恐慌情绪)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.where(loss > 0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _per_ticker_features(g: pd.DataFrame) -> pd.DataFrame:
    """对单只标的 (已按 trade_date 升序) 计算所有个股特征."""
    out = g[["ts_code", "trade_date"]].copy()
    close = g["close"]
    vol = g["vol"]
    daily_ret = close.pct_change()

    out["mom_1w"] = close.pct_change(periods=5)
    out["mom_4w"] = close.pct_change(periods=20)
    out["mom_12w"] = close.pct_change(periods=60)
    out["rev_1w"] = -out["mom_1w"]
    out["vol_4w"] = daily_ret.rolling(20).std()
    out["vol_12w"] = daily_ret.rolling(60).std()
    out["rsi_14"] = _rsi(close, 14)
    rolling_max_252 = close.rolling(252, min_periods=252).max()
    out["dist_from_high"] = close / rolling_max_252 - 1.0
    vol_1w = vol.rolling(5).mean()
    vol_4w = vol.rolling(20).mean()
    out["vol_ratio_1_4"] = vol_1w / vol_4w
    return out


def _macro_features(panel: pd.DataFrame) -> pd.DataFrame:
    """从 SPY 和 VIXY 计算宏观特征, 返回 (trade_date, spy_mom_4w, vix_z)."""
    spy = (panel[panel["ts_code"] == "SPY"]
           .sort_values("trade_date")
           .set_index("trade_date"))
    vixy = (panel[panel["ts_code"] == "VIXY"]
            .sort_values("trade_date")
            .set_index("trade_date"))

    spy_mom_4w = spy["close"].pct_change(periods=20)
    vixy_close = vixy["close"]
    vix_z = (vixy_close - vixy_close.rolling(252).mean()) / vixy_close.rolling(252).std()

    macro = pd.DataFrame({"spy_mom_4w": spy_mom_4w, "vix_z": vix_z}).reset_index()
    return macro


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """对面板每只标的计算个股特征, 再 join 上宏观特征."""
    panel = panel.sort_values(["ts_code", "trade_date"])
    parts = [_per_ticker_features(g) for _, g in panel.groupby("ts_code", sort=False)]
    feats = pd.concat(parts, ignore_index=True)

    macro = _macro_features(panel)
    feats = feats.merge(macro, on="trade_date", how="left")
    return feats


def cross_sectional_zscore(
    df: pd.DataFrame,
    feature_cols: list[str],
    group_col: str = "trade_date",
) -> pd.DataFrame:
    """对每个 trade_date 的横截面做 z-score 标准化.

    这一步是"打分"的前置: 让不同特征在同一日期内可比, 否则 mom_4w (~0.05)
    和 vol_ratio_1_4 (~1.2) 量纲完全不同, 直接加权毫无意义.
    """
    out = df.copy()
    for col in feature_cols:
        grp = out.groupby(group_col)[col]
        mean = grp.transform("mean")
        std = grp.transform("std")
        # std == 0 时全部置 0, 避免除零 NaN
        z = (out[col] - mean) / std.where(std > 0, np.nan)
        out[f"{col}_z"] = z.fillna(0.0)
    return out


def main() -> None:
    """sanity check: 打印特征构造后的样例."""
    from src.build_panel import load_panel

    panel = load_panel()
    feats = build_features(panel)
    feats = cross_sectional_zscore(
        feats,
        ["mom_1w", "mom_4w", "vol_4w", "dist_from_high", "vol_ratio_1_4"],
    )
    print(f"特征行数: {len(feats):,}")
    print(f"列: {list(feats.columns)}")
    print(f"\nNaN 统计 (前 252 行因 52w 窗口不够预期会有):")
    print(feats.isna().sum())
    print(f"\n最近一个交易日 5 只样本:")
    latest = feats[feats["trade_date"] == feats["trade_date"].max()]
    print(latest.head())


if __name__ == "__main__":
    main()
