"""把面板数据切成 (lookback × C) 的窗口, 喂给 1D-CNN.

设计要点:
    1. 共享权重 — 50 只科技股全部进同一个数据集, 一个 CNN 训练
    2. 4 个无量纲通道 — 直接喂 OHLCV 会被价格水平 / 成交量量级污染
    3. 窗口内归一化 — 每个 60 天窗口独立做 z-score, 跨股票 / 跨时间可比
    4. 标签 vol 归一化 — y / sigma_4w, 让高波动股不主导 MSE 损失;
       预测时再乘回 vol_4w, 输出绝对收益, 兼容现有回测

详见: src/cnn_window/README.md
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.universe import TECH_50

LOOKBACK = 60
LOOKAHEAD = 5            # 与 src/labels.py 保持一致
VOL_WINDOW = 20          # 标签归一化用的波动率窗口 (= mom_4w / vol_4w 同口径)
N_CHANNELS = 4

# 通道说明 (顺序和模型 in_channels 必须对齐):
#   0. log_return   = log(close[t] / close[t-1])         价格变动 (无量纲)
#   1. hl_pct       = (high - low) / close               日内波幅 (无量纲)
#   2. co_pct       = (close - open) / open              日内跳空+涨跌 (无量纲)
#   3. log_vol      = log(volume)                        成交量 (后续在窗口内 z-score)


def _per_ticker_channels(g: pd.DataFrame) -> np.ndarray:
    """单只标的按 trade_date 升序生成 [N, 4] 原始通道矩阵 (尚未做窗口归一化).

    注意: 全部用 float32 — MPS 不支持 float64, 提前对齐避免后续 .float() 转换.
    """
    g = g.sort_values("trade_date").reset_index(drop=True)
    close = g["close"].to_numpy(dtype=np.float32)
    open_ = g["open"].to_numpy(dtype=np.float32)
    high = g["high"].to_numpy(dtype=np.float32)
    low = g["low"].to_numpy(dtype=np.float32)
    vol = g["vol"].to_numpy(dtype=np.float32)

    # log_return: 第 0 天没有前一日, 置 0
    log_ret = np.zeros_like(close)
    safe_close_prev = np.clip(close[:-1], 1e-8, None)
    log_ret[1:] = np.log(close[1:] / safe_close_prev)

    hl_pct = (high - low) / np.clip(close, 1e-8, None)
    co_pct = (close - open_) / np.clip(open_, 1e-8, None)
    log_vol = np.log(np.clip(vol, 1.0, None))  # 成交量为 0 时压到 log(1)=0

    return np.stack([log_ret, hl_pct, co_pct, log_vol], axis=1).astype(np.float32)


def _normalize_window(window: np.ndarray) -> np.ndarray:
    """窗口内 z-score, 各通道独立. 全部用窗口"自己"的统计量, 严格无未来信息.

    log_return 等通道本身已经无量纲, 但做 z-score 仍有用:
        - 让窗口内"近期相对历史"的相对幅度被显式编码
        - BatchNorm 之前先把分布拉到 ~ N(0, 1), 训练更稳
    """
    mean = window.mean(axis=0, keepdims=True)
    std = window.std(axis=0, keepdims=True) + 1e-6
    return (window - mean) / std


def _per_ticker_labels(g: pd.DataFrame) -> np.ndarray:
    """[N, 3] = (y_mean, y_max, y_min). 复制 src/labels.py 逻辑, 但避开循环依赖."""
    g = g.sort_values("trade_date").reset_index(drop=True)
    close = g["close"].to_numpy(dtype=np.float32)
    high = g["high"].to_numpy(dtype=np.float32)
    low = g["low"].to_numpy(dtype=np.float32)
    n = len(g)
    out = np.full((n, 3), np.nan, dtype=np.float32)
    for i in range(n - LOOKAHEAD):
        ref = close[i]
        if ref <= 0 or not np.isfinite(ref):
            continue
        out[i, 0] = close[i + 1: i + 1 + LOOKAHEAD].mean() / ref - 1.0
        out[i, 1] = high[i + 1: i + 1 + LOOKAHEAD].max() / ref - 1.0
        out[i, 2] = low[i + 1: i + 1 + LOOKAHEAD].min() / ref - 1.0
    return out


def _vol_4w(g: pd.DataFrame) -> np.ndarray:
    """20 日日收益标准差, 滚动算到 t (含 t). t 之前不足 20 天的为 NaN."""
    close = g["close"].to_numpy(dtype=np.float32)
    n = len(close)
    ret = np.zeros(n, dtype=np.float32)
    ret[1:] = close[1:] / np.clip(close[:-1], 1e-8, None) - 1.0
    out = np.full(n, np.nan, dtype=np.float32)
    for i in range(VOL_WINDOW, n):
        out[i] = ret[i - VOL_WINDOW + 1: i + 1].std()
    return out


class WindowDataset(Dataset):
    """长面板 → 滑动窗口数据集.

    每个样本:
        x       : torch.float32, shape [C=4, T=60]      (Conv1d 期望 [B, C, T])
        y       : torch.float32, shape [3]              vol-归一化后的 (mean, max, min)

    侧表 (不进入 __getitem__, 用于切分 / 回测对齐):
        self.dates  : np.datetime64[N]
        self.codes  : str[N]
        self.vol4w  : float32[N]                        预测时用来"反归一化"

    样本筛选 (会丢弃以下情况):
        - 252 日 / 60 日窗口数据不足 → channels 含 NaN
        - 未来 5 日不存在 → labels 含 NaN
        - vol_4w 不足 20 天或几乎为 0
    """

    def __init__(
        self,
        panel: pd.DataFrame,
        tickers: list[str] | None = None,
        lookback: int = LOOKBACK,
        normalize_window: bool = True,
        vol_normalize_label: bool = True,
    ):
        if tickers is None:
            tickers = TECH_50
        panel = panel[panel["ts_code"].isin(tickers)].copy()

        self.lookback = lookback
        self.normalize_window = normalize_window
        self.vol_normalize_label = vol_normalize_label

        X_list: list[np.ndarray] = []
        y_list: list[np.ndarray] = []
        date_list: list[np.datetime64] = []
        code_list: list[str] = []
        vol_list: list[float] = []

        for ts_code, g in panel.groupby("ts_code", sort=False):
            g = g.sort_values("trade_date").reset_index(drop=True)
            channels = _per_ticker_channels(g)         # [N, 4]
            labels = _per_ticker_labels(g)             # [N, 3]
            vol4w = _vol_4w(g)                         # [N]
            dates = g["trade_date"].to_numpy()         # datetime64[ns]

            n = len(g)
            for t in range(lookback - 1, n - LOOKAHEAD):
                if not np.all(np.isfinite(labels[t])):
                    continue
                v = float(vol4w[t])
                if not np.isfinite(v) or v < 1e-5:
                    continue

                w = channels[t - lookback + 1: t + 1]   # [60, 4]
                if not np.all(np.isfinite(w)):
                    continue
                if normalize_window:
                    w = _normalize_window(w)

                y = labels[t].copy()
                if vol_normalize_label:
                    y = y / v                           # 单位化到 ~ N(0, 1) 量级

                X_list.append(w.astype(np.float32))
                y_list.append(y.astype(np.float32))
                date_list.append(dates[t])
                code_list.append(ts_code)
                vol_list.append(v)

        if X_list:
            self.X = np.stack(X_list)                   # [M, 60, 4]
            self.y = np.stack(y_list)                   # [M, 3]
            self.dates = np.array(date_list, dtype="datetime64[ns]")
            self.codes = np.array(code_list)
            self.vol4w = np.array(vol_list, dtype=np.float32)
        else:
            self.X = np.empty((0, lookback, N_CHANNELS), dtype=np.float32)
            self.y = np.empty((0, 3), dtype=np.float32)
            self.dates = np.array([], dtype="datetime64[ns]")
            self.codes = np.array([], dtype=object)
            self.vol4w = np.array([], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        # [60, 4] → [4, 60]: Conv1d 期望 (batch, channel, time)
        x = torch.from_numpy(self.X[idx]).transpose(0, 1).contiguous()
        y = torch.from_numpy(self.y[idx])
        return x, y

    def date_index(self, lo: pd.Timestamp | None = None,
                   hi: pd.Timestamp | None = None) -> np.ndarray:
        """返回 dates 落在 [lo, hi) 区间的样本索引. None 表示开区间."""
        mask = np.ones(len(self.dates), dtype=bool)
        if lo is not None:
            mask &= self.dates >= np.datetime64(lo, "ns")
        if hi is not None:
            mask &= self.dates < np.datetime64(hi, "ns")
        return np.where(mask)[0]


def main() -> None:
    """sanity check: 打印数据集形状 + 一个样例."""
    from src.build_panel import load_panel

    panel = load_panel()
    ds = WindowDataset(panel)
    print(f"窗口数: {len(ds):,}")
    print(f"X shape (单样本): {ds.X[0].shape if len(ds) else 'empty'}")
    print(f"y shape (单样本): {ds.y[0].shape if len(ds) else 'empty'}")
    print(f"日期范围: {ds.dates.min()} ~ {ds.dates.max()}")
    print(f"标的数  : {len(set(ds.codes))}")
    print(f"vol4w    p1 / p50 / p99: "
          f"{np.percentile(ds.vol4w, 1):.4f} / "
          f"{np.percentile(ds.vol4w, 50):.4f} / "
          f"{np.percentile(ds.vol4w, 99):.4f}")
    if len(ds):
        x0, y0 = ds[0]
        print(f"\n样本 0: x.dtype={x0.dtype}, x.shape={tuple(x0.shape)}, "
              f"y={y0.tolist()}, code={ds.codes[0]}, date={ds.dates[0]}")


if __name__ == "__main__":
    main()
