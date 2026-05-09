# CNN Window Predictor

> 用 1D-CNN 把过去 60 个交易日的 OHLCV 模式压成"未来 5 日相对收益"预测。
> **所有股票共享一套权重** (pooled training); 输出三列, 直接接现有回测。

---

## 目录

- [文件结构](#文件结构)
- [数据流](#数据流)
- [设计细节](#设计细节)
  - [1. 输入: 4 个无量纲通道](#1-输入-4-个无量纲通道)
  - [2. 窗口内归一化](#2-窗口内归一化)
  - [3. 标签 vol 归一化](#3-标签-vol-归一化)
  - [4. 共享权重的合理性](#4-共享权重的合理性)
  - [5. 模型结构](#5-模型结构)
  - [6. Walk-Forward + Embargo](#6-walk-forward--embargo)
- [macOS / MPS 注意事项](#macos--mps-注意事项)
- [运行](#运行)
- [对接现有回测](#对接现有回测)
- [性能 / 复杂度](#性能--复杂度)
- [已知限制 / 后续可做](#已知限制--后续可做)

---

## 文件结构

```
src/cnn_window/
├── __init__.py
├── window_dataset.py    # 把面板切成 (lookback × C) 窗口的 PyTorch Dataset
├── model.py             # 1D-CNN 模型定义
├── train.py             # walk-forward 训练 + 输出预测 CSV
├── backtest.py          # 读预测 CSV → 跑周频选股回测
└── README.md            # 你正在看的文件
```

不依赖现有的 `src/features.py` (那是给 LightGBM 用的标量特征);
但**复用** `src/build_panel.py` 加载面板, **复用** `src/predict_weekly_winner_ml.py` 的
`run_backtest_ml()` 跑回测 — 同一套选股 / TP / SL / time-stop 规则, 模型可比。

---

## 数据流

```
data/*.csv
    │
    ▼
src.build_panel.load_panel()              ← 长面板 [ts_code, trade_date, OHLCV]
    │
    ▼
window_dataset.WindowDataset              ← 滑窗 + 通道构造 + 归一化 + 标签
    │  X: [N, 4, 60]   y: [N, 3]
    │  meta: dates, codes, vol4w
    ▼
DataLoader (batch=256, shuffle=True)
    │
    ▼
model.WindowCNN  (共享权重)               ← Conv1d ×3 + GlobalAvgPool + FC
    │  forward: [B, 4, 60] → [B, 3]
    ▼
train.walk_forward()                      ← expanding 窗口 + embargo
    │  每 20 天重训一次, 全程 OOS
    ▼
data/preds_cnn.csv                        ← (ts_code, trade_date, y_mean_hat, y_max_hat, y_min_hat)
    │
    ▼
src.predict_weekly_winner_ml.run_backtest_ml()  ← 选 top-K + 动态 TP / SL
```

---

## 设计细节

### 1. 输入: 4 个无量纲通道

直接喂 OHLCV 有两个**致命问题**:
- 价格水平漂移: NVDA 从 \$20 涨到 \$300, 模型在训练分布和测试分布上看到的"输入"完全不一样。
- 跨股票不可比: AAPL \$200 和 PLTR \$20 同样涨 1%, 绝对数值差 10 倍。

所以**先把每个时间步映射到 4 个无量纲量**:

| 通道 | 公式 | 含义 |
|------|------|------|
| `log_return` | `log(close[t] / close[t-1])` | 当日对数收益, 第 0 天置 0 |
| `hl_pct`     | `(high - low) / close`      | 日内波幅 |
| `co_pct`     | `(close - open) / open`      | 日内跳空+涨跌 |
| `log_vol`    | `log(volume)`                | 成交量, 后续在窗口内 z-score |

代码: [`window_dataset.py::_per_ticker_channels`](window_dataset.py)

### 2. 窗口内归一化

每个 60 天窗口**独立**做 z-score, 各通道分别归一化:

```python
window_z = (window - window.mean(0)) / (window.std(0) + 1e-6)
```

- 严格无未来信息: 只用窗口"自己"的统计量。
- `log_return` 等通道虽然已经无量纲, 仍做 z-score — 让 BatchNorm 之前的输入更接近 N(0, 1), 训练更稳。
- `log_vol` 不做窗口归一化的话, **所有股票最后被成交量水平主导** (NVDA ~10⁸ vs PLTR ~10⁶ 差两个数量级)。

代码: [`window_dataset.py::_normalize_window`](window_dataset.py)

### 3. 标签 vol 归一化

原始标签 `y_mean = future_5d.mean() / close[t] - 1` 有量纲问题:

```
高波动股 (PLTR, CRWD): 平均 |y| ≈ 0.05
低波动股 (CSCO, ACN ): 平均 |y| ≈ 0.015
                                   ↑ 同样的 MSE 损失, 高波动股贡献的梯度大 3 倍
                                     → 模型被高波动股"绑架"
```

**解法**: 把每个标签除以 `vol_4w[t]` (过去 20 日日收益标准差):

```
y_normalized = y_raw / vol_4w[t]    # 类似"周收益的 t 分数 / Sharpe"
```

训练时模型见到的 `y` 在 ±2 量级, 各股票均衡。

但回测需要**绝对收益**口径 (TP=4% 这种阈值需要绝对量纲),所以**预测时再乘回去**:

```python
y_hat_abs = y_hat_norm * vol_4w[t]
```

代码: [`window_dataset.py::_vol_4w`](window_dataset.py), 反归一化在 [`train.py::walk_forward`](train.py)。

可以用 `--no-vol-norm` 关掉, 做 ablation 对比。

### 4. 共享权重的合理性

50 只股票 → 一个 CNN, 而不是 50 个独立 CNN。理由:

| 选择 | 训练样本 | 评价 |
|------|---------|------|
| 50 只各训一个 | 690 / 模型 | 严重欠拟合 — CNN 容量需要至少几千样本 |
| **共享一个** ⭐ | 50 × 690 ≈ 34500 | 50× 数据增强 |

**底层假设**: 技术形态(突破、回踩、放量)是通用的, 不是某一只股票的怪癖。
窗口归一化 + 标签 vol 归一化让"通用"这个假设成立。

学术上叫 *stock-agnostic model* / *universal model*, 是 Numerai、WorldQuant 这些机构的标配。

### 5. 模型结构

定义在 [`model.py::WindowCNN`](model.py)。形状变化:

```
输入                                  [B,  4, 60]
  Conv1d(4 → 32,   kernel=5, pad=2)  [B, 32, 60]
  BatchNorm1d + ReLU
  MaxPool1d(2)                        [B, 32, 30]
  Conv1d(32 → 64,  kernel=3, pad=1)  [B, 64, 30]
  BatchNorm1d + ReLU
  MaxPool1d(2)                        [B, 64, 15]
  Conv1d(64 → 64,  kernel=3, pad=1)  [B, 64, 15]
  BatchNorm1d + ReLU
  AdaptiveAvgPool1d(1)                [B, 64,  1]
  Flatten + Dropout(0.2)              [B, 64]
  Linear(64 → 3)                      [B,  3]
```

参数量约 **21k**, 故意保持小 — 金融数据信噪比低, 容量大反而过拟合。

**为什么 GlobalAvgPool 而不是 Flatten 后接 Linear?**
- AvgPool 对窗口长度不敏感, 改 lookback (60 → 80) 不用改模型结构。
- Flatten 会让最后一层有 64 × 15 = 960 维输入 → 参数量翻 15 倍, 过拟合风险大。

**为什么 BatchNorm?**
- 窗口归一化已经把分布拉到 ~ N(0, 1), 但 Conv1d 之后激活值会漂移, BN 维持稳定。
- 注意训练 / eval 切换: 推理时用累积的 running mean/var, 必须 `model.eval()`。
- MPS 后端原生支持 BatchNorm1d, 不会回退到 CPU。

### 6. Walk-Forward + Embargo

时间序列 ML 的两条铁律:

**① 不能 shuffle 后随机切分 train/test**
窗口高度重叠 (相邻样本共享 59/60 的输入), 随机 shuffle 后 test 几乎包含在 train 里。

**② 必须有 embargo (隔离期)**
我们的标签是未来 5 日的均值/极值。如果 train 截止到 `周五 t`, test 从下周一 `t+3` 开始,
**train 集最后一行的标签包含了 test 集前几天的数据** → 数据泄露。

```
[████████ Train ████████ | embargo 5d | Test ████████]
                                              ↑
                              至少留 LOOKAHEAD_DAYS=5 天空白
```

实现:

```python
cutoff = refit_date - pd.Timedelta(days=embargo_days)   # 5 天
train_idx = np.where(dates_all < cutoff)[0]             # 严格小于
```

**Walk-Forward 节奏**: 每 20 个交易日 (~1 个月) 重训一次, 用 expanding 窗口 (从最早数据到 cutoff)。
不用 rolling 窗口 — 我们数据本来就少, 扔掉早期数据太奢侈。

代码: [`train.py::walk_forward`](train.py)

---

## macOS / MPS 注意事项

PyTorch MPS 后端在 macOS 上**有一些坑**, 这个项目里都处理好了, 但还是值得知道:

### 1. **MPS 不支持 float64**
所有 numpy 数组在 [`window_dataset.py`](window_dataset.py) 里都强制 `dtype=np.float32`,
`torch.from_numpy` 会保持 float32, 模型默认 float32, 全程一致。

如果你看到 `RuntimeError: Cannot convert a Float64 tensor to MPS`, 说明某处偷偷用了 float64,
检查 `to(device)` 之前的 dtype。

### 2. **DataLoader 设置**
```python
DataLoader(..., num_workers=0, pin_memory=False)
```
- `num_workers > 0` 在 MPS 上**不会加速**, 反而因为 process spawn 开销变慢。
- `pin_memory` 只对 CUDA 有意义, MPS 用了会报 warning。

### 3. **MPS 显存清理**
```python
if device.type == "mps":
    torch.mps.empty_cache()
```
MPS 不会自动释放上一折的张量, 长 walk-forward 会逐渐吃满内存。
[`train.py::empty_cache`](train.py) 在每折结束后调用。

### 4. **某些算子会静默回退到 CPU**
本项目用的算子 (`Conv1d`, `BatchNorm1d`, `MaxPool1d`, `AdaptiveAvgPool1d`, `Linear`,
`Dropout`, `MSELoss`, `AdamW`) 都是 MPS 原生支持的, 不会回退。
如果你扩展模型加了新算子 (比如 `LayerNorm` 在某些 PyTorch 版本上有问题), 设置环境变量:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1   # 允许 CPU 回退, 不报错
```

### 5. **随机种子**
```python
torch.manual_seed(42)
```
`torch.manual_seed` 已经覆盖 CPU + MPS + CUDA。
不需要 `torch.mps.manual_seed_all` (那个 API 不存在)。

### 6. **强制 CPU 调试**
如果模型行为异常, 用 `--cpu` 强制走 CPU。MPS 上的浮点累积偶尔和 CPU 略有差异 (1e-5 量级),
非数值问题用 CPU 跑能排除"是不是 MPS 算错了"的可能。

---

## 运行

### 0. 安装依赖

```bash
pip install torch scipy        # 项目已有 numpy / pandas / lightgbm
```

### 1. 跑 dataset sanity check

```bash
python -m src.cnn_window.window_dataset
```

期望输出:
```
窗口数: 30,000+
X shape (单样本): (60, 4)
y shape (单样本): (3,)
日期范围: 2023-XX-XX ~ 2026-05-07
标的数  : 50
vol4w    p1 / p50 / p99: 0.0095 / 0.0220 / 0.0490
```

### 2. 跑模型 forward

```bash
python -m src.cnn_window.model
```

会打印参数量 + 在 MPS 上跑一次 forward 验证。

### 3. 跑训练 (产出预测 CSV)

```bash
# 默认 (MPS, 20 epochs, 256 batch)
python -m src.cnn_window.train

# 自定义
python -m src.cnn_window.train \
    --test-start 2025-01-01 \
    --epochs 25 \
    --batch-size 512 \
    --retrain-every 20 \
    --ic                   # 训完打印 OOS IC

# 调试: 走 CPU + 关掉归一化做 ablation
python -m src.cnn_window.train --cpu --no-vol-norm --no-window-norm
```

输出: `data/preds_cnn.csv`

### 4. 跑回测

```bash
# 默认 K=5, SL=-3%, TP_factor=0.7
python -m src.cnn_window.backtest

# 调参
python -m src.cnn_window.backtest --k 3 --tp-factor 0.5

# 落盘交易明细 + 周收益 (后续画图 / 详细分析用)
python -m src.cnn_window.backtest \
    --save-trades data/cnn_trades.csv \
    --save-weekly data/cnn_weekly.csv
```

终端会打印年化 / 波动 / Sharpe / 最大回撤 / 胜率 + 出场分布(TP / SL / TIME),
和 `predict_weekly_winner_ml.py` 完全一样的格式 — 可以直接对比 v1 启发式 vs Ridge vs LightGBM vs CNN。

---

## 对接现有回测

如果想在 notebook 里更精细地分析, 也可以直接用底层 API:

```python
import pandas as pd
from src.build_panel import load_panel
from src.predict_weekly_winner_ml import run_backtest_ml, report_backtest

panel = load_panel()
preds = pd.read_csv("data/preds_cnn.csv", parse_dates=["trade_date"])

trades, weekly = run_backtest_ml(panel, preds, k=5)
report_backtest("CNN", trades, weekly)
# trades  : list[TradeResult] — 每笔交易, 含 exit_reason / hold_days / ret
# weekly  : DataFrame         — 每周组合收益 + SPY 基准
```

---

## 性能 / 复杂度

在 M1 / M2 Pro Mac 上, 默认参数 (50 只 × 3 年数据, ~34500 窗口):

| 设备 | 单折训练 (20 epoch, 27000 train 样本) | 一次完整 walk-forward (~17 折) |
|------|---------------------------------------|--------------------------------|
| MPS  | ~30-60 秒                            | ~10-15 分钟                    |
| CPU  | ~3-5 分钟                            | ~50-80 分钟                    |

如果嫌慢:
- `--epochs 10` (减半训练时间, IC 通常只降 10% 左右)
- `--retrain-every 60` (季度重训, 折数减少 3 倍)
- 加 warm-start: 修改 [`train.py::walk_forward`](train.py), 复用上一折模型权重

---

## 已知限制 / 后续可做

| 限制 | 影响 | 后续可做 |
|------|------|---------|
| 每折独立训练, 无 warm-start | 训练时间偏长 | 用上一折模型 init, 只训 5 epoch |
| 单一模型, 没集成 | 单次结果有 ~10% 噪声 | 5 个不同 seed 训 5 个, 预测取均值 |
| 没有验证集做 early stopping | epoch 可能过 / 不足 | 留训练集最后 10% 作 val, 看 val loss 决定停止 |
| 通道只用 OHLCV 衍生 | 缺宏观信息 | 加 SPY / VIXY 通道 (注意广播对齐) |
| 标签是 5 日均值/极值 | 是路径依赖, 模型被迫学 | 改预测 t+5 单点收益, 模型更纯净 |
| 没有 stock embedding | 没用上股票 ID 信息 | 加 8 维 learnable embedding 拼到 FC 前 |
| 训练期 < 1 年时模型不稳 | 早期 OOS 折噪声大 | 把 `--test-start` 推后到 2024 年中以后 |

