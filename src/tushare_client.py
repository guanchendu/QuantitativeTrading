"""共享的 Tushare Pro 客户端初始化.

后续所有访问 Tushare 数据的脚本都应该通过 get_pro() 拿到 pro 实例,
统一注入 token 和自定义 HTTP 入口.

用法:
    from src.tushare_client import get_pro
    pro = get_pro()
    df = pro.index_basic(limit=5)
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def get_pro():
    """返回配置好 token + 自定义 http_url 的 Tushare pro 客户端.

    注意: pro._DataApi__http_url 必须在 ts.pro_api(...) 之后设置,
    否则会回退到默认入口, 导致 "Token 不对" 之类的报错.
    """
    import tushare as ts
    from DONOTGIT_API import request_API, request_HTTP_URL

    pro = ts.pro_api(request_API())
    pro._DataApi__http_url = request_HTTP_URL()
    return pro


if __name__ == "__main__":
    pro = get_pro()
    df = pro.index_basic(limit=5)
    print(df)
