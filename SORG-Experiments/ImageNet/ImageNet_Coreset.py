# Recommended filename: E7_ImageNetCoreset.py
# E7_ImageNet.py (ICML-SA-ready: ImageNet-1K | All Strong Baselines | SORG)
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
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ============================================================
# [Rule #1] Force caches/data under /nfs/hpc/share/baekji
# ============================================================
BASE_NFS = "/nfs/hpc/share/baekji"
TORCH_HOME = os.path.join(BASE_NFS, "torch_home")
# [CRITICAL] Verify this path points to extracted ImageNet (train/val folders)
IMAGENET_ROOT = os.path.join(BASE_NFS, "vision_datasets", "imagenet")
FEATURE_CACHE = os.path.join(BASE_NFS, "E7_imagenet_cache")

os.makedirs(TORCH_HOME, exist_ok=True)
os.makedirs(FEATURE_CACHE, exist_ok=True)

os.environ["TORCH_HOME"] = TORCH_HOME
os.environ["HF_HOME"] = os.path.join(BASE_NFS, "hf_cache")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.models as models
import torchvision.transforms as T

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Clean Logs
warnings.filterwarnings("ignore")

# Cap torch CPU threads too (align with OMP_NUM_THREADS)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))

try:
    from SORG import SORG
except Exception as e:
    raise ImportError(f"SORG import failed: {e}")


# ============================================================
# Config
# ============================================================
@dataclass
class E7Config:
    dataset: str = "imagenet1k"

    # Budgets: 10%, 20% (Standard Coreset benchmarks)
    # 10% is the most critical comparison point.
    budgets: Tuple[float, ...] = (0.10, 0.20)

    # Methods: The Full Lineup (SORG version)
    methods: Tuple[str, ...] = (
        "random",
        "kcenter",  # Baseline 1: Geometry
        "herding",  # Baseline 2: Distribution (Mean)
        "sorg",     # Ours: Geometry + Distribution (SORG)
    )

    # Feature Extraction
    encoder: str = "resnet50"
    batch_size: int = 256
    num_workers: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True

    # Linear Probe Training
    head_epochs: int = 40
    head_lr: float = 0.1
    head_momentum: float = 0.9
    head_wd: float = 1e-4
    head_bs: int = 256

    seed: int = 0
    out_prefix: str = "E7_outputs"


CFG = E7Config()


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


def save_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")


def l2_normalize(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


# ============================================================
# Data & Feature Extraction (Cached)
# ============================================================
def get_resnet50(device):
    print("Loading ResNet50 (Pretrained)...")
    weights = torchvision.models.ResNet50_Weights.DEFAULT
    model = torchvision.models.resnet50(weights=weights)
    feat_model = nn.Sequential(*list(model.children())[:-1])
    feat_model.to(device)
    feat_model.eval()
    return feat_model, weights.transforms()


@torch.no_grad()
def extract_features(model, loader, device, amp):
    model.eval()
    feats, labels = [], []
    for i, (images, target) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        if amp and device.startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(images).flatten(1)
        else:
            output = model(images).flatten(1)
        feats.append(output.cpu().numpy().astype(np.float32))
        labels.append(target.numpy().astype(np.int64))
        if i % 100 == 0:
            print(f"Extracting: {i}/{len(loader)} batches...", end="\r")
    print("\nExtraction Done.")
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


def get_or_create_features(cfg):
    # Check Cache
    paths = {
        "train_X": os.path.join(FEATURE_CACHE, "train_X.npy"),
        "train_y": os.path.join(FEATURE_CACHE, "train_y.npy"),
        "test_X": os.path.join(FEATURE_CACHE, "test_X.npy"),
        "test_y": os.path.join(FEATURE_CACHE, "test_y.npy"),
    }

    if all(os.path.exists(p) for p in paths.values()):
        print("Loading cached ImageNet features...")
        X_tr = np.load(paths["train_X"])
        y_tr = np.load(paths["train_y"])
        X_te = np.load(paths["test_X"])
        y_te = np.load(paths["test_y"])
        return X_tr, y_tr, X_te, y_te

    print("Feature cache not found. Extracting from scratch...")
    model, preprocess = get_resnet50(cfg.device)

    traindir = os.path.join(IMAGENET_ROOT, "train")
    valdir = os.path.join(IMAGENET_ROOT, "val")

    if not os.path.exists(traindir):
        raise FileNotFoundError(f"ImageNet not found at {traindir}")

    train_ds = torchvision.datasets.ImageFolder(traindir, preprocess)
    val_ds = torchvision.datasets.ImageFolder(valdir, preprocess)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    X_tr, y_tr = extract_features(model, train_loader, cfg.device, cfg.amp)
    X_te, y_te = extract_features(model, val_loader, cfg.device, cfg.amp)

    print("Saving features to cache...")
    np.save(paths["train_X"], X_tr)
    np.save(paths["train_y"], y_tr)
    np.save(paths["test_X"], X_te)
    np.save(paths["test_y"], y_te)

    return X_tr, y_tr, X_te, y_te


# ============================================================
# Selection Logic (Class-wise Parallel)
# ============================================================
def select_random(N: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.choice(N, k, replace=False)


def run_classwise_selection(
    method: str,
    X: np.ndarray,
    y: np.ndarray,
    k_total: int,
    n_classes: int,
    seed: int,
) -> np.ndarray:
    """
    Apply selection independently per class to handle scale.
    - 'herding': mean-matching greedy (Frank-Wolfe style)
    - 'kcenter': class-wise k-center
    - 'sorg'   : SORG with class-mean guidance (r = mu_c)
    """
    k_per_class = k_total // n_classes
    selected_indices: List[np.ndarray] = []

    print(f"[{method}] Running Class-wise Selection (Target k/class={k_per_class})...")

    for c in range(n_classes):
        indices_c = np.where(y == c)[0]
        X_c = X[indices_c]
        N_c = X_c.shape[0]
        k_c = min(k_per_class, N_c)

        if k_c <= 0 or N_c == 0:
            continue

        if method == "herding":
            # Frank-Wolfe / Greedy Mean Matching
            mu = np.mean(X_c, axis=0)
            current_sum = np.zeros_like(mu)
            local_sel = []
            mask = np.ones(N_c, dtype=bool)

            for _ in range(k_c):
                target = (len(local_sel) + 1) * mu - current_sum
                scores = X_c @ target
                scores[~mask] = -np.inf
                best = int(np.argmax(scores))
                if not mask[best]:
                    break
                local_sel.append(best)
                mask[best] = False
                current_sum += X_c[best]

            local_sel = np.asarray(local_sel, dtype=int)

        elif method == "kcenter":
            # Greedy K-Center
            rng = np.random.RandomState(seed + c)
            centers = [rng.randint(N_c)]
            dists = np.sum((X_c - X_c[centers[0]]) ** 2, axis=1)
            for _ in range(1, k_c):
                new_c = int(np.argmax(dists))
                centers.append(new_c)
                new_dists = np.sum((X_c - X_c[new_c]) ** 2, axis=1)
                dists = np.minimum(dists, new_dists)
            local_sel = np.asarray(centers, dtype=int)

        elif method == "sorg":
            # SORG: Geometry + Distribution, guided by class mean
            mu = np.mean(X_c, axis=0)
            sel_model = SORG(k=int(k_c), alpha=1.0, p_norm=2.0)
            sel_model.fit(X_c.T, r=mu)
            local_sel = np.asarray(sel_model.get_support(), dtype=int)

        else:
            raise ValueError(f"Unknown method: {method}")

        selected_indices.append(indices_c[local_sel])

        if c % 100 == 0:
            print(f"  Processed {c}/{n_classes} classes...", end="\r")

    print(f"  Processed {n_classes}/{n_classes} classes. Done.")
    if not selected_indices:
        return np.zeros((0,), dtype=int)
    return np.concatenate(selected_indices, axis=0)


# ============================================================
# Evaluation: Linear Probe
# ============================================================
def train_linear_head(X_tr, y_tr, n_classes, cfg: E7Config) -> nn.Module:
    device = cfg.device
    X_t = torch.from_numpy(X_tr).float().to(device)
    y_t = torch.from_numpy(y_tr).long().to(device)

    head = nn.Linear(X_tr.shape[1], n_classes).to(device)

    # Standard SGD for ImageNet Linear Probe
    optimizer = torch.optim.SGD(
        head.parameters(),
        lr=cfg.head_lr,
        momentum=cfg.head_momentum,
        weight_decay=cfg.head_wd,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.head_epochs
    )
    criterion = nn.CrossEntropyLoss()

    bs = cfg.head_bs
    n = X_tr.shape[0]

    head.train()
    for epoch in range(cfg.head_epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            optimizer.zero_grad()
            out = head(X_t[idx])
            loss = criterion(out, y_t[idx])
            loss.backward()
            optimizer.step()
        scheduler.step()

    head.eval()
    return head


@torch.no_grad()
def evaluate(head: nn.Module, X_te: np.ndarray, y_te: np.ndarray, device: str) -> float:
    head.eval()
    bs = 1024
    total = int(len(y_te))
    correct = 0

    X_t = torch.from_numpy(X_te).float()
    y_t = torch.from_numpy(y_te).long()

    for i in range(0, total, bs):
        xb = X_t[i : i + bs].to(device)
        yb = y_t[i : i + bs].to(device)
        out = head(xb)
        pred = out.argmax(dim=1)
        correct += int((pred == yb).sum().item())

    return 100.0 * correct / float(total)


# ============================================================
# Plot helpers
# ============================================================
def plot_metric_vs_budget(
    agg: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    out_path: str,
) -> None:
    plt.figure(figsize=(8, 6))
    for m in CFG.methods:
        sub = agg[agg["method"] == m]
        if sub.empty:
            continue
        plt.plot(sub["budget"], sub[metric], marker="o", linewidth=2, label=m)
    plt.xlabel("Subset Size (Fraction)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ============================================================
# Main
# ============================================================
def main():
    exp_dir = os.path.join(os.getcwd(), f"{CFG.out_prefix}_{now_stamp()}")
    ensure_dir(exp_dir)

    # Save config as JSON (paper-ready)
    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(CFG), f, indent=2)

    print("============================================================")
    print("E7: ImageNet-1K Coreset | All Strong Baselines (SORG)")
    print("------------------------------------------------------------")
    print(f"Budgets: {CFG.budgets}")
    print(f"Methods: {CFG.methods}")
    print(f"Device:  {CFG.device}")
    print(f"Outputs: {exp_dir}")
    print("============================================================")

    # 1. Load Data
    X_tr, y_tr, X_te, y_te = get_or_create_features(CFG)

    # Normalize features
    X_tr = l2_normalize(X_tr)
    X_te = l2_normalize(X_te)

    n_classes = int(np.max(y_tr) + 1)
    seed_everything(CFG.seed)

    rows: List[dict] = []

    for b in CFG.budgets:
        k = int(len(X_tr) * b)
        print(f"\n=== Budget {b*100:.0f}% (k={k}) ===")

        for m in CFG.methods:
            t0 = time.time()

            if m == "random":
                sel_idx = select_random(len(X_tr), k, CFG.seed)
            else:
                # Class-wise selection for scalability
                sel_idx = run_classwise_selection(
                    method=m,
                    X=X_tr,
                    y=y_tr,
                    k_total=k,
                    n_classes=n_classes,
                    seed=CFG.seed,
                )

            sel_time = float(time.time() - t0)
            print(f"  [{m}] Selection Done: {sel_time:.2f}s (selected={len(sel_idx)})")

            # Train
            print(f"  [{m}] Training Linear Probe...")
            t_train_0 = time.time()
            head = train_linear_head(
                X_tr[sel_idx],
                y_tr[sel_idx],
                n_classes,
                CFG,
            )
            train_time = float(time.time() - t_train_0)

            # Eval
            acc = evaluate(head, X_te, y_te, CFG.device)
            print(f"  [{m}] Result: Top-1 Acc = {acc:.2f}%")

            rows.append(
                {
                    "budget": float(b),
                    "k": int(k),
                    "method": str(m),
                    "top1_acc": float(acc),
                    "sel_time_sec": sel_time,
                    "train_time_sec": train_time,
                }
            )

    # Save Results
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(exp_dir, "results.csv"), index=False)

    # Aggregate over repeats (here it's just 1 seed, but keep API)
    agg = df.groupby(["method", "budget"], as_index=False).mean()

    # Plots: accuracy + time curves (완벽 플롯 세트)
    plot_metric_vs_budget(
        agg,
        metric="top1_acc",
        ylabel="ImageNet Top-1 Accuracy (%)",
        title="ImageNet-1K Coreset Selection (ResNet50 Linear Probe)",
        out_path=os.path.join(exp_dir, "imagenet_top1_vs_budget.png"),
    )
    plot_metric_vs_budget(
        agg,
        metric="sel_time_sec",
        ylabel="Selection Time (sec)",
        title="ImageNet-1K: Selection Time vs Budget",
        out_path=os.path.join(exp_dir, "imagenet_sel_time_vs_budget.png"),
    )
    plot_metric_vs_budget(
        agg,
        metric="train_time_sec",
        ylabel="Linear Head Train Time (sec)",
        title="ImageNet-1K: Train Time vs Budget",
        out_path=os.path.join(exp_dir, "imagenet_train_time_vs_budget.png"),
    )

    print("\nDONE. Results saved to:", exp_dir)


if __name__ == "__main__":
    main()
