"""从 data/ 下的所有 CSV 拼成一个长面板 (long format) DataFrame.

输出 schema:
    ts_code        str          标的代码
    trade_date     datetime64   交易日
    open/high/low/close/vol  float / int
    kind           str          'stock' (TECH_50) 或 'etf' (INDEX_ETFS)

按 (ts_code, trade_date) 升序排列, 方便后续计算滚动特征.

用法 (模块):
    from src.build_panel import load_panel
    panel = load_panel()        # 默认从 data/ 加载

用法 (CLI, 仅 sanity check):
    python src/build_panel.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.universe import INDEX_ETFS, TECH_50

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def _load_one(ticker: str, kind: str, data_dir: Path) -> pd.DataFrame:
    matches = sorted(data_dir.glob(f"{ticker}_*.csv"))
    if not matches:
        raise FileNotFoundError(f"{data_dir} 下没有 {ticker} 的 CSV (是否还没拉?)")
    df = pd.read_csv(matches[-1])
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df["kind"] = kind
    return df[["ts_code", "trade_date", "open", "high", "low", "close", "vol", "kind"]]


def load_panel(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """读取所有 56 只标的, 拼成长面板."""
    frames: list[pd.DataFrame] = []
    for t in TECH_50:
        frames.append(_load_one(t, "stock", data_dir))
    for t in INDEX_ETFS:
        frames.append(_load_one(t, "etf", data_dir))

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    return panel


def main() -> None:
    panel = load_panel()
    print(f"面板总行数: {len(panel):,}")
    print(f"标的数    : {panel['ts_code'].nunique()}")
    print(f"日期范围  : {panel['trade_date'].min().date()} ~ {panel['trade_date'].max().date()}")
    print(f"每只行数  :")
    counts = panel.groupby("ts_code").size()
    print(f"  min={counts.min()}, max={counts.max()}, median={int(counts.median())}")
    print(f"\nhead:\n{panel.head(3)}\n")
    print(f"tail:\n{panel.tail(3)}")


if __name__ == "__main__":
    main()
