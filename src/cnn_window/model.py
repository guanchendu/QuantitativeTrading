"""1D-CNN: 把 [B, 4, 60] 的窗口 → [B, 3] (y_mean, y_max, y_min) 的预测.

形状变化 (默认 hidden=32, 输入 [B, 4, 60]):
    Conv1d(4 → 32,  k=5, pad=2)  : [B, 32,  60]
    BN + ReLU + MaxPool(2)        : [B, 32,  30]
    Conv1d(32 → 64, k=3, pad=1)  : [B, 64,  30]
    BN + ReLU + MaxPool(2)        : [B, 64,  15]
    Conv1d(64 → 64, k=3, pad=1)  : [B, 64,  15]
    BN + ReLU + AdaptiveAvgPool1d : [B, 64,   1]
    Flatten + Dropout + Linear    : [B, 3]

参数量 ≈ 5*4*32 + 3*32*64 + 3*64*64 + 64*3 ≈ 21k. 故意保持小, 防过拟合.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# 与 window_dataset 对齐 — 改通道数请同步两边
DEFAULT_IN_CHANNELS = 4
DEFAULT_OUT_DIM = 3            # y_mean, y_max, y_min


class WindowCNN(nn.Module):
    """轻量 1D-CNN, 共享权重处理所有股票.

    BatchNorm1d 用于稳定训练; 在 MPS 后端上原生支持, 不会回退到 CPU.
    """

    def __init__(
        self,
        in_channels: int = DEFAULT_IN_CHANNELS,
        hidden: int = 32,
        out_dim: int = DEFAULT_OUT_DIM,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            # block 1 : 60 -> 30
            nn.Conv1d(in_channels, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            # block 2 : 30 -> 15
            nn.Conv1d(hidden, hidden * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            # block 3 : 15 -> 1 (global pooling)
            nn.Conv1d(hidden * 2, hidden * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),

            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C=4, T=60] → [B, 3]
        return self.net(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    """Sanity check: 跑一次 forward, 验证形状 / dtype / device."""
    model = WindowCNN()
    print(f"参数量: {count_params(model):,}")

    # 模拟 batch=8 的输入
    x = torch.randn(8, DEFAULT_IN_CHANNELS, 60, dtype=torch.float32)
    y = model(x)
    print(f"in:  {tuple(x.shape)}  dtype={x.dtype}")
    print(f"out: {tuple(y.shape)}  dtype={y.dtype}")
    assert y.shape == (8, DEFAULT_OUT_DIM)
    assert y.dtype == torch.float32

    # 简单测一下 MPS (如果有)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        m = WindowCNN().to("mps")
        x_mps = x.to("mps")
        y_mps = m(x_mps)
        print(f"mps out: {tuple(y_mps.shape)}  device={y_mps.device}")


if __name__ == "__main__":
    main()
