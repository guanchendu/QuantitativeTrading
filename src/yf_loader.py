"""yfinance 数据加载器, 输出与 Tushare us_daily 兼容的 CSV.

为什么换源:
    项目早期使用的第三方 Tushare 入口 (124.220.22.110:8020) 对部分股票
    只缓存最近一周, 且完全不收 ETF. yfinance 免费、覆盖完整 (含 SPY/QQQ/GLD
    等所有 ETF), 并且对所有 50 只目标科技股都有完整 3 年日线.

CSV 字段 (与 Tushare us_daily 对齐, 让 strategy_ma_crossover.py 可直接消费):
    ts_code, trade_date, close, open, high, low,
    pre_close, pct_change, vol, amount, vwap

注意:
    - 价格使用 yfinance auto_adjust=True (前复权), 适合做相对收益回测.
    - amount 近似为 close * vol (yfinance 不提供精确成交额).
    - vwap 近似为典型价 (high+high+close)/3 不可得, 这里用 (h+l+c)/3.
    - 行序与原 Tushare CSV 保持一致: 最新日期在最前.

用法:
    python src/yf_loader.py AAPL
    python src/yf_loader.py AAPL MSFT NVDA --years 5
"""
from __future__ import annotations

import argparse
import time
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data"
DATE_FORMAT = "%Y%m%d"


def years_window(years: int = 3, today: date | None = None) -> tuple[date, date]:
    """返回 (start, end) = (今日往前推 years 年, 今日)."""
    today = today or date.today()
    try:
        start = today.replace(year=today.year - years)
    except ValueError:
        # 闰日特殊处理
        start = today.replace(year=today.year - years, day=28)
    return start, today


def _to_tushare_format(ticker: str, raw: "pd.DataFrame") -> "pd.DataFrame":
    """把 yfinance 单只标的的 DataFrame 转成 Tushare us_daily 格式."""
    import pandas as pd

    if raw is None or raw.empty:
        return pd.DataFrame()
    # 单 ticker 偶尔也是 MultiIndex columns, 统一拍平
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.copy()
        raw.columns = raw.columns.get_level_values(0)

    df = raw.reset_index().rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "vol",
    })
    df["ts_code"] = ticker
    df["trade_date"] = df["Date"].dt.strftime(DATE_FORMAT)
    df = df.sort_values("trade_date").reset_index(drop=True)  # 先按时间升序好算
    df["pre_close"] = df["close"].shift(1)
    df["pct_change"] = (df["close"] - df["pre_close"]) / df["pre_close"] * 100.0
    df["amount"] = df["close"] * df["vol"]
    df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3.0

    # 与原有 NVDA_*.csv 一致: 最新日期在前
    df = df.sort_values("trade_date", ascending=False).reset_index(drop=True)
    return df[[
        "ts_code", "trade_date", "close", "open", "high", "low",
        "pre_close", "pct_change", "vol", "amount", "vwap",
    ]]


def load_yf_daily(ticker: str, start: date, end: date) -> "pd.DataFrame":
    """从 yfinance 拉单只标的, 返回 Tushare us_daily 格式的 DataFrame.

    end 是闭区间 (含当日); yfinance 的 end 是开区间, 所以内部会 +1 天.
    """
    import yfinance as yf

    ticker = ticker.strip().upper()
    raw = yf.download(
        ticker,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    return _to_tushare_format(ticker, raw)


def load_yf_daily_batch(
    tickers: list[str], start: date, end: date,
) -> dict[str, "pd.DataFrame"]:
    """一次批量拉多只, 走 yfinance 的 group_by='ticker' 模式.

    比循环 load_yf_daily 友好得多: 一个 HTTP 调用搞定全部, 不容易触发限速.
    返回 {ticker: 单标的 DataFrame}, 失败 / 空数据的 ticker 不在结果里.
    """
    import pandas as pd
    import yfinance as yf

    tickers = [t.strip().upper() for t in tickers]
    raw = yf.download(
        tickers=" ".join(tickers),
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="ticker",
    )
    if raw is None or raw.empty:
        return {}

    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        # 单 ticker 时 raw 没有外层 ticker 列; 多 ticker 时 raw[t] 取子表
        if len(tickers) == 1:
            sub = raw
        else:
            try:
                sub = raw[t]
            except KeyError:
                continue
        if sub is None or sub.empty or sub.dropna(how="all").empty:
            continue
        sub = sub.dropna(how="all")
        df = _to_tushare_format(t, sub)
        if not df.empty:
            out[t] = df
    return out


def save_csv(ticker: str, df: "pd.DataFrame", output_dir: Path, today: date) -> Path:
    """写入 {ticker}_{today}.csv 并删除同股票的旧日期文件."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{ticker}_{today.strftime(DATE_FORMAT)}.csv"
    for old in output_dir.glob(f"{ticker}_*.csv"):
        if old.resolve() != out.resolve():
            old.unlink()
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="使用 yfinance 拉美股日线 (Tushare us_daily 兼容格式)",
    )
    p.add_argument("tickers", nargs="+", help="股票代码, 例如 AAPL SPY GLD")
    p.add_argument("--years", type=int, default=5, help="拉取多少年的数据, 默认 3")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--sleep", type=float, default=0.1,
                   help="每次调用间隔秒数, 防限速")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    start, end = years_window(args.years)
    today = date.today()
    print(f"区间 {start.strftime(DATE_FORMAT)} -> {end.strftime(DATE_FORMAT)}\n")

    ok, fail = 0, []
    for t in args.tickers:
        t = t.strip().upper()
        try:
            df = load_yf_daily(t, start, end)
        except Exception as exc:
            fail.append((t, str(exc)))
            print(f"❌ {t:<6} 异常: {exc}")
            continue
        if df.empty:
            fail.append((t, "空数据"))
            print(f"❌ {t:<6} 空数据")
            continue
        path = save_csv(t, df, args.output_dir, today)
        ok += 1
        print(f"✅ {t:<6} {len(df):>4} 行 -> {path.name}")
        time.sleep(args.sleep)

    print(f"\n汇总: 成功 {ok}, 失败 {len(fail)}")
    for t, err in fail:
        print(f"  {t}: {err}")


if __name__ == "__main__":
    main()
