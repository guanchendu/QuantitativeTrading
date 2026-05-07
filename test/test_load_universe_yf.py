"""冒烟测试: 用 yfinance 拉美股科技股 50 只 + 6 个指数 ETF.

与 test_load_universe.py 的差异:
    - 数据源换成 yfinance (test_load_universe.py 用的是第三方 Tushare 入口,
      它对部分股票只有最近一周历史, 且完全不收 ETF)
    - 输出 CSV 字段、文件命名都与 Tushare 版完全兼容,
      strategy_ma_crossover.py 不需要改动.

用法:
    python test/test_load_universe_yf.py                # 全量 56 只
    python test/test_load_universe_yf.py --limit 3     # 只跑前 3 只
    python test/test_load_universe_yf.py --no-save     # 只检查能否拉到, 不写盘
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.yf_loader import load_yf_daily_batch, save_csv, years_window

# 美股市值前 50 大科技 / 互联网公司 (静态列表, 大致按 2026 年市值排序)
TECH_50 = [
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

INDEX_ETFS = [
    "SPY",   # 标普 500
    "QQQ",   # 纳斯达克 100
    "DIA",   # 道琼斯 30
    "IWM",   # 罗素 2000
    "XLK",   # 科技板块
    "VIXY",  # VIX 短期期货 (恐慌指数代理)
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="只跑前 N 只 (用于快速冒烟)")
    parser.add_argument("--years", type=int, default=3, help="拉取多少年, 默认 3")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    start, end = years_window(args.years)
    today = date.today()
    universe = TECH_50 + INDEX_ETFS
    if args.limit:
        universe = universe[: args.limit]

    print(f"数据源: yfinance (批量模式, 一次 HTTP 调用)")
    print(f"区间   : {start} -> {end}")
    print(f"标的数 : {len(universe)} (科技股 {len(TECH_50)} + 指数 ETF {len(INDEX_ETFS)})\n")

    print("正在批量下载, 请稍候...")
    result = load_yf_daily_batch(universe, start, end)
    print(f"批量返回 {len(result)} 只\n")

    ok: list[tuple[str, int]] = []
    fail: list[tuple[str, str]] = []

    for i, code in enumerate(universe, 1):
        df = result.get(code)
        if df is None or df.empty:
            fail.append((code, "空数据"))
            print(f"[{i:>2}/{len(universe)}] {code:<6} ❌ 空数据")
            continue
        ok.append((code, len(df)))
        tail = ""
        if not args.no_save:
            path = save_csv(code, df, PROJECT_ROOT / "data", today)
            tail = f" -> {path.name}"
        print(f"[{i:>2}/{len(universe)}] {code:<6} ✅ {len(df):>4} 行{tail}")

    print(f"\n汇总: 成功 {len(ok)} / 失败 {len(fail)}")
    if fail:
        print("失败明细:")
        for code, err in fail:
            print(f"  {code:<6} {err}")


if __name__ == "__main__":
    main()
