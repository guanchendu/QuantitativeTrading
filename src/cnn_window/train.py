"""walk-forward 训练 + 出预测 CSV.

输出格式 (data/preds_cnn.csv):
    ts_code, trade_date, y_mean_hat, y_max_hat, y_min_hat   ← 绝对收益口径

直接接现有回测:
    from src.predict_weekly_winner_ml import run_backtest_ml, report_backtest
    preds = pd.read_csv("data/preds_cnn.csv", parse_dates=["trade_date"])
    trades, weekly = run_backtest_ml(panel, preds, k=5)

用法:
    python -m src.cnn_window.train                           # 默认参数
    python -m src.cnn_window.train --test-start 2025-01-01 --epochs 25
    python -m src.cnn_window.train --cpu                     # 调试时强制 CPU
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from src.build_panel import load_panel
from src.cnn_window.model import WindowCNN, count_params
from src.cnn_window.window_dataset import WindowDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "preds_cnn.csv"

EMBARGO_DAYS = 5         # = LOOKAHEAD_DAYS, 防 5 日重叠 label 泄露
RETRAIN_FREQ_DAYS = 20   # 每 ~1 个月重训一次


# ----------------------------- 设备 / 随机种子 -----------------------------

def pick_device(force_cpu: bool = False) -> torch.device:
    """macOS: 优先 MPS; 没有 MPS 但有 CUDA 就用 CUDA; 否则 CPU."""
    if force_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    """复现性. MPS 没有独立的 manual_seed_all, torch.manual_seed 已覆盖."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def empty_cache(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


# ----------------------------- 训练 / 推理循环 -----------------------------

def train_one_fold(
    dataset: WindowDataset,
    train_idx: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float = 1e-4,
    verbose: bool = False,
) -> WindowCNN:
    """从零训一个 CNN. 返回训完的模型 (eval 模式).

    每折独立训练, 不做 warm-start — 实现简单、对比公平.
    若想加速, 可改成"前一折模型 → 这一折继续训"(warm-start), 通常能省 1/3 时间.
    """
    model = WindowCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()

    train_subset = Subset(dataset, train_idx.tolist())
    # macOS / MPS 注意:
    #   - num_workers=0  : MPS 多 worker 不会加速, 反而增加 spawn 开销
    #   - pin_memory=False: pin_memory 仅 CUDA 有意义
    loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False,
    )

    model.train()
    last_loss = float("nan")
    for ep in range(epochs):
        total = 0.0
        n = 0
        for x, y in loader:
            # MPS 不支持 float64 — 提前已用 float32, 这里直接 .to(device) 即可
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
        last_loss = total / max(n, 1)
        if verbose and (ep % max(1, epochs // 5) == 0 or ep == epochs - 1):
            print(f"    epoch {ep+1:>3d}/{epochs}  loss={last_loss:.5f}")

    return model


@torch.no_grad()
def predict_indices(
    model: WindowCNN,
    dataset: WindowDataset,
    idx: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """对给定索引出预测, 返回 [N, 3] (vol-归一化口径)."""
    model.eval()
    out = np.empty((len(idx), 3), dtype=np.float32)
    subset = Subset(dataset, idx.tolist())
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    p = 0
    for x, _ in loader:
        x = x.to(device)
        pred = model(x).cpu().numpy()
        out[p: p + len(pred)] = pred
        p += len(pred)
    return out


# ----------------------------- walk-forward 主流程 -------------------------

def walk_forward(
    dataset: WindowDataset,
    test_start: pd.Timestamp,
    embargo_days: int = EMBARGO_DAYS,
    retrain_freq_days: int = RETRAIN_FREQ_DAYS,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: torch.device | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """每 retrain_freq_days 天重训一次, 用 expanding 窗口 + embargo.

    输出 DataFrame 的 y_*_hat 已经 **乘回 vol_4w**, 是绝对收益口径,
    可以直接喂 src.predict_weekly_winner_ml.run_backtest_ml().
    """
    device = device or pick_device()
    print(f"训练设备: {device}")
    print(f"模型参数量: {count_params(WindowCNN()):,}")

    dates_all = pd.to_datetime(dataset.dates)
    test_mask = dates_all >= test_start
    test_dates_unique = sorted(set(dates_all[test_mask]))
    if not test_dates_unique:
        raise RuntimeError(f"{test_start.date()} 之后没有任何样本日期")

    # 重训点: 序列开头 + 每 retrain_freq_days 天后续点
    refit_dates: list[pd.Timestamp] = [test_dates_unique[0]]
    for d in test_dates_unique:
        if (d - refit_dates[-1]).days >= retrain_freq_days:
            refit_dates.append(d)

    print(f"测试期: {test_dates_unique[0].date()} ~ {test_dates_unique[-1].date()}  "
          f"(共 {len(test_dates_unique)} 个交易日, {len(refit_dates)} 次重训)\n")

    preds_rows: list[pd.DataFrame] = []

    for refit_idx, refit_date in enumerate(refit_dates):
        # 训练集 = 严格早于 (refit_date - embargo) 的样本
        cutoff = refit_date - pd.Timedelta(days=embargo_days)
        train_idx = np.where(dates_all < cutoff)[0]
        if len(train_idx) < 200:
            print(f"[{refit_date.date()}] 训练样本仅 {len(train_idx)}, 跳过")
            continue

        # 这次重训覆盖的预测日: refit_date 起, 到下一个 refit_date 之前
        next_refit = refit_dates[refit_idx + 1] if refit_idx + 1 < len(refit_dates) else None
        if next_refit is None:
            seg_mask = dates_all >= refit_date
        else:
            seg_mask = (dates_all >= refit_date) & (dates_all < next_refit)
        seg_idx = np.where(seg_mask)[0]
        if len(seg_idx) == 0:
            continue

        t0 = time.time()
        model = train_one_fold(
            dataset, train_idx, device,
            epochs=epochs, batch_size=batch_size, lr=lr, verbose=verbose,
        )
        train_sec = time.time() - t0
        y_hat_norm = predict_indices(model, dataset, seg_idx, device, batch_size)

        # vol-反归一化: 训练用的是 y / sigma, 这里乘回 sigma 得到绝对收益预测
        vol = dataset.vol4w[seg_idx][:, None]              # [N, 1]
        y_hat_abs = y_hat_norm * vol if dataset.vol_normalize_label else y_hat_norm

        preds_rows.append(pd.DataFrame({
            "ts_code":   dataset.codes[seg_idx],
            "trade_date": dates_all[seg_idx].values,
            "y_mean_hat": y_hat_abs[:, 0],
            "y_max_hat":  y_hat_abs[:, 1],
            "y_min_hat":  y_hat_abs[:, 2],
        }))

        print(f"[{refit_date.date()}] train={len(train_idx):>5}  "
              f"predict={len(seg_idx):>4}  耗时={train_sec:5.1f}s")

        del model
        empty_cache(device)

    if not preds_rows:
        return pd.DataFrame(columns=[
            "ts_code", "trade_date", "y_mean_hat", "y_max_hat", "y_min_hat",
        ])
    return pd.concat(preds_rows, ignore_index=True)


# ----------------------------- 评估 (可选) ---------------------------------

def quick_ic(preds: pd.DataFrame, dataset: WindowDataset) -> None:
    """打印 OOS 的横截面 spearman IC 均值 — 快速判断有没有 alpha."""
    from scipy.stats import spearmanr

    # 把真实 y (绝对收益口径) 拼回去
    df = preds.copy()
    df = df.merge(
        pd.DataFrame({
            "ts_code":  dataset.codes,
            "trade_date": pd.to_datetime(dataset.dates),
            "y_mean":   dataset.y[:, 0] * (dataset.vol4w if dataset.vol_normalize_label else 1.0),
            "y_max":    dataset.y[:, 1] * (dataset.vol4w if dataset.vol_normalize_label else 1.0),
            "y_min":    dataset.y[:, 2] * (dataset.vol4w if dataset.vol_normalize_label else 1.0),
        }),
        on=["ts_code", "trade_date"], how="inner",
    )

    print("\n--- OOS IC (spearman, 按交易日横截面) ---")
    for tgt in ["y_mean", "y_max", "y_min"]:
        ics = []
        for d, g in df.groupby("trade_date"):
            if len(g) >= 5:
                rho, _ = spearmanr(g[tgt].values, g[f"{tgt}_hat"].values)
                if np.isfinite(rho):
                    ics.append(rho)
        ics = np.array(ics) if ics else np.array([np.nan])
        ic_mean = ics.mean()
        ic_std = ics.std() if len(ics) > 1 else np.nan
        t_stat = ic_mean / (ic_std / np.sqrt(len(ics))) if ic_std > 0 else np.nan
        print(f"  {tgt:<8}  IC均值={ic_mean:>+.4f}  IC_t={t_stat:>+5.2f}  "
              f"样本日={len(ics)}")


# ----------------------------- CLI ----------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--test-start", default="2025-01-01", help="OOS 起点 (含)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--retrain-every", type=int, default=RETRAIN_FREQ_DAYS,
                   help="每多少天重训一次模型, 默认 20")
    p.add_argument("--embargo", type=int, default=EMBARGO_DAYS,
                   help="train / test 间的隔离天数, 默认 5 (= LOOKAHEAD)")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--cpu", action="store_true", help="强制 CPU, 调试用")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-vol-norm", action="store_true",
                   help="关闭标签 vol 归一化 (ablation 用)")
    p.add_argument("--no-window-norm", action="store_true",
                   help="关闭窗口内归一化 (ablation 用)")
    p.add_argument("--ic", action="store_true", help="跑完打印 OOS IC")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    print("加载面板...")
    panel = load_panel()
    print(f"面板: {len(panel):,} 行, {panel['ts_code'].nunique()} 只标的")

    print("\n构造窗口数据集 ...")
    ds = WindowDataset(
        panel,
        normalize_window=not args.no_window_norm,
        vol_normalize_label=not args.no_vol_norm,
    )
    n_codes = len(set(ds.codes))
    print(f"数据集: {len(ds):,} 个窗口  ({n_codes} 只 × ~{len(ds) // max(n_codes, 1)}/股)")
    print(f"日期    : {pd.Timestamp(ds.dates.min()).date()} ~ "
          f"{pd.Timestamp(ds.dates.max()).date()}")
    print(f"窗口归一化={not args.no_window_norm}, "
          f"标签 vol 归一化={not args.no_vol_norm}\n")

    device = pick_device(args.cpu)
    preds = walk_forward(
        ds,
        test_start=pd.Timestamp(args.test_start),
        embargo_days=args.embargo,
        retrain_freq_days=args.retrain_every,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        verbose=args.verbose,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(args.output, index=False)
    print(f"\n预测已写: {args.output}  ({len(preds):,} 行)")

    if args.ic:
        quick_ic(preds, ds)


if __name__ == "__main__":
    main()
