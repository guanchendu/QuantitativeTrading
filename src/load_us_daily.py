from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data"
DATE_FORMAT = "%Y%m%d"
CODE_ALIASES = {
    "APPL": "AAPL",
}


class TushareDownloadError(RuntimeError):
    """Raised when Tushare rejects or fails a market data request."""


def three_year_window(today: date | None = None) -> tuple[str, str]:
    """Return the Tushare date range from three years ago through today."""
    current_date = today or date.today()
    try:
        start_date = current_date.replace(year=current_date.year - 3)
    except ValueError:
        # Handles February 29 on leap years.
        start_date = current_date.replace(year=current_date.year - 3, day=28)

    return start_date.strftime(DATE_FORMAT), current_date.strftime(DATE_FORMAT)


def normalize_us_stock_code(stock_code: str) -> str:
    """Normalize user input such as 'goog' into the code Tushare expects."""
    code = stock_code.strip().upper()
    if not code:
        raise ValueError("stock_code cannot be empty.")
    return CODE_ALIASES.get(code, code)


def dated_output_path(stock_code: str, output_dir: Path, today: date | None = None) -> Path:
    """Build a per-stock CSV path using the operating system date."""
    code = normalize_us_stock_code(stock_code)
    current_date = today or date.today()
    return output_dir / f"{code}_{current_date.strftime(DATE_FORMAT)}.csv"


def remove_old_stock_csvs(stock_code: str, output_dir: Path, keep_path: Path) -> list[Path]:
    """Remove previous CSV files for the same stock code from the output directory."""
    code = normalize_us_stock_code(stock_code)
    removed_paths = []

    if not output_dir.exists():
        return removed_paths

    for csv_path in output_dir.glob(f"{code}_*.csv"):
        if csv_path.resolve() == keep_path.resolve():
            continue
        csv_path.unlink()
        removed_paths.append(csv_path)

    return removed_paths


def get_tushare_client():
    """Initialize and return a Tushare pro client using the local API helper."""
    try:
        import tushare as ts
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'tushare'. Install it in the current Python environment "
            "before downloading market data."
        ) from exc

    sys.path.insert(0, str(PROJECT_ROOT))
    from DONOTGIT_API import request_API

    token = request_API()
    ts.set_token(token)
    return ts.pro_api()


def load_us_daily_data(
    stock_code: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    sleep_seconds: float = 0.3,
) -> "pd.DataFrame":
    """Download one US stock's daily data and replace that stock's dated CSV."""
    code = normalize_us_stock_code(stock_code)
    start_date, end_date = three_year_window()
    output_path = dated_output_path(code, output_dir)
    pro = get_tushare_client()

    print(f"正在下载 {code}: {start_date} -> {end_date}")
    try:
        data = pro.us_daily(
            ts_code=code,
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as exc:
        message = str(exc)
        if "频率超限" in message or "freq" in message.lower():
            raise TushareDownloadError(
                f"Tushare 今日 us_daily 调用额度已用完，未删除任何旧的 {code} CSV。"
            ) from exc
        raise TushareDownloadError(f"Tushare 下载 {code} 失败: {message}") from exc

    time.sleep(sleep_seconds)

    if data.empty:
        raise RuntimeError(f"{code} 在 {start_date} 到 {end_date} 之间没有返回数据。")

    output_dir.mkdir(parents=True, exist_ok=True)
    removed_paths = remove_old_stock_csvs(code, output_dir, keep_path=output_path)
    for removed_path in removed_paths:
        print(f"已删除旧文件: {removed_path}")

    data.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"已写入新文件: {output_path}")
    print(data.head())
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download US daily stock data from Tushare into a per-stock dated CSV."
    )
    parser.add_argument(
        "stock_code",
        nargs="?",
        help="US stock ticker, for example AAPL, GOOG, GOOGL, MSFT.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"CSV output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stock_code = args.stock_code or input("请输入公司代码，例如 AAPL 或 GOOG: ")
    try:
        load_us_daily_data(stock_code=stock_code, output_dir=args.output_dir)
    except (ModuleNotFoundError, ValueError, RuntimeError, TushareDownloadError) as exc:
        raise SystemExit(f"错误: {exc}") from exc


if __name__ == "__main__":
    main()
