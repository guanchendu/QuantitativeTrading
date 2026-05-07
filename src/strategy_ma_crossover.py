"""双均线交叉策略 + 简易回测.

规则:
  - 短期均线 (SMA_short) 上穿长期均线 (SMA_long): 次日开盘满仓买入
  - 短期均线下穿长期均线: 次日开盘清仓
  - 不加杠杆, 不做空, 单票各自独立回测, 最后再做等权组合

用法:
    python src/strategy_ma_crossover.py
    python src/strategy_ma_crossover.py --short 10 --long 30
    python src/strategy_ma_crossover.py --tickers GOOG NVDA
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
# DEFAULT_TICKERS = ["CLF", "META", "NVDA", "TSLA"]
DEFAULT_TICKERS = [
    "AAPL",  "MSFT",  "NVDA",  "GOOGL", "AMZN",
    "META",  "TSLA",  "AVGO",  "ORCL",  "CRM",
    "ADBE",  "AMD",   "NFLX",  "CSCO",  "ACN",
    "INTC",  "QCOM",  "TXN",   "IBM",   "INTU",
    "NOW",   "AMAT",  "PANW",  "MU",    "LRCX",
    "ADI",   "KLAC",  "ANET",  "CDNS",  "SNPS",
    "MRVL",  "CRWD",  "FTNT",  "WDAY",  "PLTR",
    "ADSK",  "NXPI",  "MCHP",  "SNOW",  "DDOG",
    "TEAM",  "MDB",   "NET",   "ZS",    "DELL",
    "UBER",  "SHOP",  "SPOT",  "ABNB",  "GOOG",
]
TRADING_DAYS = 252


def load_ticker(ticker: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """读取 {ticker}_YYYYMMDD.csv, 按日期升序返回."""
    matches = sorted(data_dir.glob(f"{ticker}_*.csv"))
    if not matches:
        raise FileNotFoundError(f"{data_dir} 下没有找到 {ticker} 的 CSV")
    df = pd.read_csv(matches[-1])
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df = df.sort_values("trade_date").reset_index(drop=True)
    return df[["trade_date", "open", "high", "low", "close", "vol"]]


def generate_signals(df: pd.DataFrame, short: int, long: int) -> pd.DataFrame:
    """生成持仓信号: 1 = 持有, 0 = 空仓. 信号 T-1 触发, T 日开盘成交."""
    out = df.copy()
    out["sma_short"] = out["close"].rolling(short).mean()
    out["sma_long"] = out["close"].rolling(long).mean()
    out["raw_signal"] = (out["sma_short"] > out["sma_long"]).astype(int)
    # 当日信号根据当日收盘均线生成, 次日开盘建仓 -> shift(1)
    out["position"] = out["raw_signal"].shift(1).fillna(0).astype(int)
    return out


def backtest(df: pd.DataFrame) -> pd.DataFrame:
    """以次日开盘价成交, 按日开盘->收盘 + 隔夜跳空近似日收益."""
    out = df.copy()
    # 当日收益: 持仓状态下吃今日 close/前一收盘 的涨跌
    out["ret"] = out["close"].pct_change().fillna(0.0)
    out["strategy_ret"] = out["position"] * out["ret"]
    out["equity"] = (1.0 + out["strategy_ret"]).cumprod()
    out["bh_equity"] = (1.0 + out["ret"]).cumprod()
    return out


def perf_stats(returns: pd.Series, label: str) -> dict:
    """年化收益 / 年化波动 / 夏普 / 最大回撤."""
    returns = returns.dropna()
    if returns.empty or returns.std() == 0:
        return {"label": label, "ann_ret": 0.0, "ann_vol": 0.0, "sharpe": 0.0, "mdd": 0.0}
    equity = (1.0 + returns).cumprod()
    ann_ret = equity.iloc[-1] ** (TRADING_DAYS / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    drawdown = equity / equity.cummax() - 1
    return {
        "label": label,
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "mdd": drawdown.min(),
    }


def format_row(stats: dict) -> str:
    return (
        f"{stats['label']:<22} "
        f"年化={stats['ann_ret']:>7.2%}  "
        f"波动={stats['ann_vol']:>6.2%}  "
        f"Sharpe={stats['sharpe']:>5.2f}  "
        f"最大回撤={stats['mdd']:>7.2%}"
    )


def run(tickers: list[str], short: int, long: int) -> None:
    if short >= long:
        raise ValueError(f"short ({short}) 必须小于 long ({long})")

    all_strategy_rets: dict[str, pd.Series] = {}
    all_bh_rets: dict[str, pd.Series] = {}
    rows: list[dict] = []

    print(f"\n=== 双均线策略  short={short}  long={long} ===\n")
    for ticker in tickers:
        df = load_ticker(ticker)
        sig = generate_signals(df, short, long)
        bt = backtest(sig)
        bt = bt.set_index("trade_date")

        all_strategy_rets[ticker] = bt["strategy_ret"]
        all_bh_rets[ticker] = bt["ret"]

        rows.append(perf_stats(bt["strategy_ret"], f"{ticker} 策略"))
        rows.append(perf_stats(bt["ret"], f"{ticker} Buy&Hold"))

    # 等权组合
    strat_df = pd.DataFrame(all_strategy_rets).dropna(how="all").fillna(0)
    bh_df = pd.DataFrame(all_bh_rets).dropna(how="all").fillna(0)
    rows.append(perf_stats(strat_df.mean(axis=1), "等权组合 策略"))
    rows.append(perf_stats(bh_df.mean(axis=1), "等权组合 Buy&Hold"))

    for r in rows:
        print(format_row(r))
    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="双均线交叉策略回测")
    p.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    p.add_argument("--short", type=int, default=5, help="短期均线窗口, 默认 20")
    p.add_argument("--long", type=int, default=10, help="长期均线窗口, 默认 60")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.tickers, args.short, args.long)
