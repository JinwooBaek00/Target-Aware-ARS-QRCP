# Recommended filename: E9_EndtoEnd_Training_Speedup.py
# E9.py (ICML-SA-ready: End-to-End Training Speedup | CIFAR-100 | SORG vs Baselines)
# -*- coding: utf-8 -*-

import os

# ============================================================
# [Critical] Stabilize CPU thread usage BEFORE numpy/sklearn import
# ============================================================
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import json
import time
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ============================================================
# [Rule #1] Force caches/data under /nfs/hpc/share/baekji
# ============================================================
BASE_NFS = "/nfs/hpc/share/baekji"
TORCH_HOME = os.path.join(BASE_NFS, "torch_home")
TORCH_DATASETS = os.path.join(BASE_NFS, "torch_datasets")
FEATURE_CACHE = os.path.join(BASE_NFS, "E1_feature_cache")  # Reuse E1 cache (CIFAR100 + ResNet18 features)

os.makedirs(TORCH_HOME, exist_ok=True)
os.makedirs(TORCH_DATASETS, exist_ok=True)
os.makedirs(FEATURE_CACHE, exist_ok=True)

os.environ["TORCH_HOME"] = TORCH_HOME
os.environ["HF_HOME"] = os.path.join(BASE_NFS, "hf_cache")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Clean logs
import warnings
warnings.filterwarnings("ignore")

# Align torch CPU threads with OMP
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))

# ============================================================
# Core SORG algorithm
# ============================================================
try:
    from SORG import SORG
except Exception as e:
    raise ImportError(f"SORG import failed: {e}")


# ============================================================
# (Optional) Baseline selectors from Baselines.py
#   - If Baselines is not available, we fall back to local implementations
# ============================================================
def _fallback_select_random_group(y: np.ndarray, k: int, seed: int) -> np.ndarray:
    """
    Fallback implementation for random baseline (global uniform sampling).
    If Baselines.select_random_group is available, that will be used instead.
    """
    n = len(y)
    k = int(min(k, n))
    rng = np.random.RandomState(seed)
    return rng.choice(n, size=k, replace=False).astype(int)


def _fallback_select_herding_group(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """
    Fallback implementation for Group Herding baseline (class-wise mean matching).
    If Baselines.select_herding_group is available, that will be used instead.

    X : (N, d) standardized features
    y : (N,) class labels
    k : total budget
    """
    classes = np.unique(y)
    n_classes = len(classes)
    k_per_class = k // n_classes
    remainder = k % n_classes

    selected_indices: List[int] = []

    for i, c in enumerate(classes):
        current_k = k_per_class + (1 if i < remainder else 0)
        if current_k <= 0:
            continue

        idx = np.where(y == c)[0]
        X_c = X[idx]
        if len(idx) == 0:
            continue
        if current_k > len(idx):
            current_k = len(idx)

        # Standard Herding per class
        mu = np.mean(X_c, axis=0)
        current_sum = np.zeros_like(mu)
        mask = np.ones(len(idx), dtype=bool)

        for t in range(current_k):
            target = (t + 1) * mu - current_sum
            scores = X_c @ target
            scores[~mask] = -np.inf
            best = int(np.argmax(scores))
            if not mask[best]:
                break
            selected_indices.append(idx[best])
            mask[best] = False
            current_sum += X_c[best]

    return np.array(selected_indices, dtype=int)


try:
    # Expected API in Baselines.py (you can adapt Baselines to match this)
    from Baselines import select_random_group, select_herding_group  # type: ignore
except Exception:
    # If Baselines.py is not ready, fall back to the local implementations
    select_random_group = _fallback_select_random_group
    select_herding_group = _fallback_select_herding_group


# ============================================================
# Config
# ============================================================
@dataclass
class E9Config:
    dataset: str = "cifar100"
    encoder: str = "resnet18"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Coreset budget for subset runs (e.g., 20%)
    budget_frac: float = 0.20

    # Training hyperparameters (end-to-end)
    num_epochs: int = 200
    batch_size: int = 128
    num_workers: int = 4
    base_lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4

    # SORG hyperparameters (borrowed from E8 "good region")
    sorg_alpha: float = 1.0
    sorg_p_norm: float = 4.0

    # Randomness
    seed: int = 0

    # Output prefix
    out_prefix: str = "E9_End2End"

    # For reproducibility / plotting
    save_plots: bool = True


CFG = E9Config()


# ============================================================
# Helpers
# ============================================================
def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ============================================================
# Feature cache (reuse E1/E8 style)
# ============================================================
def cifar_feat_paths(dataset: str, encoder: str) -> Dict[str, str]:
    tag = f"{dataset}_{encoder}"
    return {
        "train_X": os.path.join(FEATURE_CACHE, f"{tag}_train_X.npy"),
        "train_y": os.path.join(FEATURE_CACHE, f"{tag}_train_y_fine.npy"),
        "test_X": os.path.join(FEATURE_CACHE, f"{tag}_test_X.npy"),
        "test_y": os.path.join(FEATURE_CACHE, f"{tag}_test_y_fine.npy"),
        "meta": os.path.join(FEATURE_CACHE, f"{tag}_meta.json"),
    }


def get_resnet18_encoder(device: str):
    """
    Pretrained ResNet18 encoder for feature extraction (same as E1/E8).
    Used only for selection, NOT for end-to-end training.
    """
    try:
        w = torchvision.models.ResNet18_Weights.DEFAULT
        m = torchvision.models.resnet18(weights=w)
        transform = w.transforms()
    except Exception:
        m = torchvision.models.resnet18(pretrained=True)
        transform = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
    feat = nn.Sequential(*list(m.children())[:-1]).to(device)
    feat.eval()
    return feat, transform


@torch.no_grad()
def extract_all(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        z = model(x).flatten(1)
        feats.append(z.cpu().numpy().astype(np.float32))
        labels.append(y.numpy().astype(np.int64))
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


def get_feature_data(cfg: E9Config):
    """
    Load CIFAR-100 feature cache (E1/E8) or build it once if missing.
    """
    paths = cifar_feat_paths(cfg.dataset, cfg.encoder)
    needed = ("train_X", "train_y", "test_X", "test_y", "meta")
    if all(os.path.exists(paths[k]) for k in needed):
        print(">>> Loading cached CIFAR100 features (E1/E8 cache)...")
        Xt = np.load(paths["train_X"])
        Yt = np.load(paths["train_y"])
        Xte = np.load(paths["test_X"])
        Yte = np.load(paths["test_y"])
        return Xt, Yt, Xte, Yte

    print(">>> Feature cache missing. Extracting CIFAR100 features once (for E9)...")
    model, transform = get_resnet18_encoder(cfg.device)

    train_ds = torchvision.datasets.CIFAR100(
        TORCH_DATASETS, train=True, download=True, transform=transform
    )
    test_ds = torchvision.datasets.CIFAR100(
        TORCH_DATASETS, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=512,
        num_workers=cfg.num_workers,
        shuffle=False,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=512,
        num_workers=cfg.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    Xt, Yt = extract_all(model, train_loader, cfg.device)
    Xte, Yte = extract_all(model, test_loader, cfg.device)

    # Save minimal meta
    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump(
            {
                "feat_dim": int(Xt.shape[1]),
                "dataset": cfg.dataset,
                "encoder": cfg.encoder,
            },
            f,
            indent=2,
        )

    np.save(paths["train_X"], Xt)
    np.save(paths["train_y"], Yt)
    np.save(paths["test_X"], Xte)
    np.save(paths["test_y"], Yte)

    return Xt, Yt, Xte, Yte


# ============================================================
# SORG group selection for coreset
# ============================================================
def select_sorg_group(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    alpha: float,
    p_norm: float,
) -> np.ndarray:
    """
    Group-wise SORG with explicit guidance r = class-mean.

    This mirrors E8's design:
      - Per-class SORG
      - Guidance vector r = mean feature of that class
      - k is split across classes as evenly as possible
    """
    classes = np.unique(y)
    n_classes = len(classes)
    k_per_class = k // n_classes
    remainder = k % n_classes

    selected_indices: List[np.ndarray] = []

    for i, c in enumerate(classes):
        current_k = k_per_class + (1 if i < remainder else 0)
        if current_k <= 0:
            continue

        idx_global = np.where(y == c)[0]
        X_c = X[idx_global]
        if X_c.shape[0] == 0:
            continue
        if current_k > X_c.shape[0]:
            current_k = X_c.shape[0]

        # Guidance = class mean (makes SORG behave like "robust herding")
        mu_c = np.mean(X_c, axis=0)

        model = SORG(
            k=int(current_k),
            alpha=float(alpha),
            p_norm=float(p_norm),
        )
        model.fit(X_c.T, r=mu_c)
        sel_local = np.asarray(model.get_support(), dtype=int)
        selected_indices.append(idx_global[sel_local])

    if not selected_indices:
        return np.zeros((0,), dtype=int)
    return np.concatenate(selected_indices).astype(int)


# ============================================================
# CIFAR-100 datasets for end-to-end training
# ============================================================
def get_cifar_datasets():
    """
    Standard CIFAR-100 training / test pipelines for end-to-end training.
    We use 224x224 resizing to match standard ResNet-18 configuration.
    """
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]

    train_transform = T.Compose(
        [
            T.Resize(256),
            T.RandomResizedCrop(224, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )

    test_transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )

    train_ds = torchvision.datasets.CIFAR100(
        TORCH_DATASETS, train=True, download=True, transform=train_transform
    )
    test_ds = torchvision.datasets.CIFAR100(
        TORCH_DATASETS, train=False, download=True, transform=test_transform
    )
    return train_ds, test_ds


# ============================================================
# ResNet18 model for end-to-end training
# ============================================================
def create_resnet18_cifar(num_classes: int, device: str) -> nn.Module:
    """
    Standard torchvision ResNet-18, trained from scratch on CIFAR-100.
    Input size is 224x224 due to the transforms above.
    """
    model = torchvision.models.resnet18(weights=None)
    in_feat = model.fc.in_features
    model.fc = nn.Linear(in_feat, num_classes)
    model.to(device)
    return model


# ============================================================
# Training & evaluation
# ============================================================
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float]:
    """
    Return (avg_loss, accuracy) on the loader.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * targets.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += int((preds == targets).sum().item())
            total_samples += targets.size(0)

    avg_loss = total_loss / max(1, total_samples)
    acc = total_correct / max(1, total_samples)
    return avg_loss, acc


def train_model(
    method: str,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    cfg: E9Config,
    device: str,
) -> List[dict]:
    """
    Train model from scratch, log epoch-wise train/test stats and elapsed time.
    Returns: list of log rows (dict) for each epoch.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.base_lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_epochs
    )

    logs: List[dict] = []
    t_start = time.time()

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * targets.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += int((preds == targets).sum().item())
            total_samples += targets.size(0)

        train_loss = total_loss / max(1, total_samples)
        train_acc = total_correct / max(1, total_samples)

        # Evaluate on test set
        test_loss, test_acc = evaluate(model, test_loader, device)

        scheduler.step()
        elapsed = time.time() - t_start

        log_row = {
            "method": method,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "elapsed_time_sec": elapsed,
        }
        logs.append(log_row)

        print(
            f"[{method:8s}] Epoch {epoch:3d}/{cfg.num_epochs:3d} | "
            f"TrainAcc={train_acc*100:5.2f}% | TestAcc={test_acc*100:5.2f}% | "
            f"Time={elapsed/60.0:6.2f} min"
        )

    return logs


# ============================================================
# Optional plotting helpers
# ============================================================
def plot_time_vs_acc(logs_df: pd.DataFrame, out_dir: str) -> None:
    """
    Plot Test Accuracy vs Elapsed Time for each method.
    This directly visualizes the "speedup" story.
    """
    plt.figure(figsize=(8, 6))
    for method in sorted(logs_df["method"].unique()):
        sub = logs_df[logs_df["method"] == method]
        plt.plot(
            sub["elapsed_time_sec"] / 60.0,
            sub["test_acc"] * 100.0,
            marker="o",
            label=method,
        )
    plt.xlabel("Elapsed Time (minutes)")
    plt.ylabel("Test Accuracy (%)")
    plt.title("CIFAR-100 End-to-End Training: Test Acc vs Time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "E9_time_vs_testacc.png"), dpi=200)
    plt.close()


def plot_epoch_vs_acc(logs_df: pd.DataFrame, out_dir: str) -> None:
    """
    Plot Test Accuracy vs Epoch for each method.
    """
    plt.figure(figsize=(8, 6))
    for method in sorted(logs_df["method"].unique()):
        sub = logs_df[logs_df["method"] == method]
        plt.plot(
            sub["epoch"],
            sub["test_acc"] * 100.0,
            marker="o",
            label=method,
        )
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.title("CIFAR-100 End-to-End Training: Test Acc vs Epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "E9_epoch_vs_testacc.png"), dpi=200)
    plt.close()


# ============================================================
# Main
# ============================================================
def main():
    exp_dir = os.path.join(os.getcwd(), f"{CFG.out_prefix}_{now_stamp()}")
    ensure_dir(exp_dir)

    # Save config
    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(CFG), f, indent=2)

    print("============================================================")
    print("E9: End-to-End Training Speedup (CIFAR-100 | ResNet18)")
    print("------------------------------------------------------------")
    print(f"Device:        {CFG.device}")
    print(f"Budget (frac): {CFG.budget_frac:.2f}")
    print(f"Epochs:        {CFG.num_epochs}")
    print(f"Batch size:    {CFG.batch_size}")
    print(f"SORG:          alpha={CFG.sorg_alpha}, p={CFG.sorg_p_norm}")
    print(f"Outputs:       {exp_dir}")
    print("============================================================")

    # 1) Selection phase using frozen features
    seed_everything(CFG.seed)
    Xt, Yt, Xte, Yte = get_feature_data(CFG)

    scaler = StandardScaler()
    Xt_s = scaler.fit_transform(Xt)

    N = len(Yt)
    k = int(N * CFG.budget_frac)
    print(f"Total train N={N}, budget={CFG.budget_frac:.3f} -> k={k}")

    subsets: Dict[str, np.ndarray] = {}

    # Full data: 100% baseline
    subsets["full"] = np.arange(N, dtype=int)

    # Random 20% (using Baselines.select_random_group if available)
    print("\n[Selection] Random-20% subset...")
    subsets["random20"] = select_random_group(Yt, k, seed=CFG.seed + 123)

    # Group Herding 20%
    print("[Selection] Group Herding-20% subset...")
    subsets["herding20"] = select_herding_group(Xt_s, Yt, k)

    # SORG Group 20%
    print("[Selection] SORG-20% subset (Group + Guided)...")
    subsets["sorg20"] = select_sorg_group(
        X=Xt_s,
        y=Yt,
        k=k,
        alpha=CFG.sorg_alpha,
        p_norm=CFG.sorg_p_norm,
    )

    # Save selection indices for reproducibility
    sel_save_path = os.path.join(exp_dir, "selection_indices.npz")
    np.savez(sel_save_path, **subsets)
    print(f"[Selection] Saved index subsets to {sel_save_path}")

    # 2) End-to-end training on CIFAR-100
    print("\n[Data] Loading CIFAR-100 datasets for end-to-end training...")
    train_ds, test_ds = get_cifar_datasets()
    num_classes = 100
    device = CFG.device

    test_loader = DataLoader(
        test_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    logs_all: List[dict] = []
    summary_rows: List[dict] = []

    # Training order: full -> random20 -> herding20 -> sorg20
    for method_name in ["full", "random20", "herding20", "sorg20"]:
        idx = subsets[method_name]
        print(
            f"\n[Training] Method={method_name} | "
            f"train_size={len(idx)} ({len(idx)/N*100:.1f}% of data)"
        )

        # Reset seed for fair initialization & data shuffling
        seed_everything(CFG.seed)

        if method_name == "full":
            train_subset = train_ds  # use entire dataset
        else:
            train_subset = Subset(train_ds, idx.tolist())

        train_loader = DataLoader(
            train_subset,
            batch_size=CFG.batch_size,
            shuffle=True,
            num_workers=CFG.num_workers,
            pin_memory=True,
        )

        model = create_resnet18_cifar(num_classes=num_classes, device=device)
        logs = train_model(
            method=method_name,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            cfg=CFG,
            device=device,
        )
        logs_all.extend(logs)

        # Final stats
        final_test_acc = logs[-1]["test_acc"]
        total_time = logs[-1]["elapsed_time_sec"]
        summary_rows.append(
            {
                "method": method_name,
                "train_size": len(idx),
                "train_frac": len(idx) / N,
                "final_test_acc": final_test_acc,
                "total_time_sec": total_time,
            }
        )

    # 3) Save logs & summaries
    logs_df = pd.DataFrame(logs_all)
    logs_path = os.path.join(exp_dir, "training_logs.csv")
    logs_df.to_csv(logs_path, index=False)
    print(f"\n[Result] Saved per-epoch logs to {logs_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(exp_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"[Result] Saved final summary to {summary_path}")
    print(summary_df)

    # 4) Optional plots
    if CFG.save_plots:
        print("\n[Plot] Generating curves...")
        plot_time_vs_acc(logs_df, exp_dir)
        plot_epoch_vs_acc(logs_df, exp_dir)
        print("[Plot] Saved E9_time_vs_testacc.png and E9_epoch_vs_testacc.png")

    print("\nDONE. E9 finished. Results saved to:", exp_dir)


if __name__ == "__main__":
    main()
