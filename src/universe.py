"""股票池常量, 多个模块共享."""
from __future__ import annotations

# 美股市值前 50 大科技 / 互联网公司 (策略候选池)
TECH_50: list[str] = [
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

# 指数 ETF (做宏观特征 + 基准对比)
INDEX_ETFS: list[str] = [
    "SPY",   # 标普 500
    "QQQ",   # 纳斯达克 100
    "DIA",   # 道琼斯 30
    "IWM",   # 罗素 2000
    "XLK",   # 科技板块
    "VIXY",  # VIX 短期期货代理
]

ALL_TICKERS: list[str] = TECH_50 + INDEX_ETFS

# 基准 (策略对照基线)
BENCHMARK: str = "SPY"
