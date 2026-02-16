# Recommended filename: E6_BackdoorRobustness.py
# E6.py (ICML-SA-ready: Backdoor Robustness | Guided SORG | Smart Caching)
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
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

# ============================================================
# [Rule #1] Force caches/data under /nfs/hpc/share/baekji
# ============================================================
BASE_NFS = "/nfs/hpc/share/baekji"
TORCH_HOME = os.path.join(BASE_NFS, "torch_home")
TORCH_DATASETS = os.path.join(BASE_NFS, "torch_datasets")
FEATURE_CACHE = os.path.join(BASE_NFS, "E1_feature_cache")  # Reuse E1 cache

os.makedirs(TORCH_HOME, exist_ok=True)
os.makedirs(TORCH_DATASETS, exist_ok=True)
os.makedirs(FEATURE_CACHE, exist_ok=True)

os.environ["TORCH_HOME"] = TORCH_HOME
os.environ["HF_HOME"] = os.path.join(BASE_NFS, "hf_cache")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.models as models
import torchvision.transforms as T

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Hard cap torch CPU threads too
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))

try:
    from SORG import SORG
except Exception as e:
    raise ImportError(f"SORG import failed: {e}")


# ============================================================
# Config
# ============================================================
@dataclass
class E6Config:
    dataset: str = "cifar100"

    # Budgets (1%, 2%, 5%, 10%)
    budgets: Tuple[float, ...] = (0.01, 0.02, 0.05, 0.10)

    # Methods
    methods: Tuple[str, ...] = (
        "random",
        "herding",
        "kcenter",
        "kmeans_mb",
        "sorg",         # Guided SORG (r=Mean) + Gate ON
        "sorg_nogate",  # Guided SORG (r=Mean) + Gate OFF
    )

    # Threat Model
    # 20% Noise (Wrong Labels), 5% Backdoor (Trojan)
    label_noise_ratio: float = 0.20
    backdoor_ratio: float = 0.05

    backdoor_target_class: int = 0
    patch_size: int = 4

    encoder: str = "resnet18"
    batch_size: int = 256
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    head_epochs: int = 30
    head_lr: float = 1e-2
    head_wd: float = 5e-4
    head_bs: int = 512

    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    out_prefix: str = "E6_outputs"


CFG = E6Config()


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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
# Smart Caching Logic (Mix & Match)
# ============================================================
def get_resnet18(device):
    try:
        w = torchvision.models.ResNet18_Weights.DEFAULT
        m = torchvision.models.resnet18(weights=w)
        transform = w.transforms()
    except Exception:
        m = torchvision.models.resnet18(pretrained=True)
        transform = T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    feat = nn.Sequential(*list(m.children())[:-1]).to(device)
    feat.eval()
    return feat, transform


@torch.no_grad()
def extract_all(model, loader, device):
    model.eval()
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        z = model(x).flatten(1)
        feats.append(z.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


def _apply_patch(img, patch_size):
    # Apply patch to PIL image directly
    w, h = img.size
    ps = patch_size
    img_np = np.array(img)
    img_np[h - ps:h, w - ps:w, :] = 255  # White square bottom-right
    return torchvision.transforms.functional.to_pil_image(img_np)


class PatchWrapper(torch.utils.data.Dataset):
    def __init__(self, ds, patch_size, transform):
        self.ds = ds
        self.patch_size = patch_size
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, y = self.ds[idx]
        img = _apply_patch(img, self.patch_size)
        if self.transform:
            img = self.transform(img)
        return img, y


def get_cached_features(cfg):
    """
    Load or create: Clean Train, Patched Train, Clean Test, Backdoor Test
    """
    prefix = f"{cfg.dataset}_{cfg.encoder}"
    paths = {
        "Xc": os.path.join(FEATURE_CACHE, f"{prefix}_train_clean_X.npy"),
        "Xp": os.path.join(FEATURE_CACHE, f"{prefix}_train_patched_X.npy"),
        "Y": os.path.join(FEATURE_CACHE, f"{prefix}_train_y.npy"),
        "Xte": os.path.join(FEATURE_CACHE, f"{prefix}_test_X.npy"),
        "Yte": os.path.join(FEATURE_CACHE, f"{prefix}_test_y.npy"),
        "Xte_bd": os.path.join(FEATURE_CACHE, f"{prefix}_test_bd_X.npy"),
    }

    if all(os.path.exists(p) for p in paths.values()):
        print(">>> Loading cached features (Instant Start)...")
        return (
            np.load(paths["Xc"]),
            np.load(paths["Xp"]),
            np.load(paths["Y"]),
            np.load(paths["Xte"]),
            np.load(paths["Yte"]),
            np.load(paths["Xte_bd"]),
        )

    print(">>> Cache miss. Extracting features (This happens ONCE)...")

    model, transform = get_resnet18(cfg.device)
    if cfg.dataset == "cifar10":
        train_raw = torchvision.datasets.CIFAR10(TORCH_DATASETS, train=True, download=True)
        test_raw = torchvision.datasets.CIFAR10(TORCH_DATASETS, train=False, download=True)
        train_for_feats = torchvision.datasets.CIFAR10(TORCH_DATASETS, train=True, transform=transform)
        test_for_feats = torchvision.datasets.CIFAR10(TORCH_DATASETS, train=False, transform=transform)
    else:
        train_raw = torchvision.datasets.CIFAR100(TORCH_DATASETS, train=True, download=True)
        test_raw = torchvision.datasets.CIFAR100(TORCH_DATASETS, train=False, download=True)
        train_for_feats = torchvision.datasets.CIFAR100(TORCH_DATASETS, train=True, transform=transform)
        test_for_feats = torchvision.datasets.CIFAR100(TORCH_DATASETS, train=False, transform=transform)

    # 1. Clean Train
    print("   Extracting Clean Train...")
    Xc, Y = extract_all(
        model,
        DataLoader(train_for_feats, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True),
        cfg.device,
    )

    # 2. Patched Train (All images backdoored)
    print("   Extracting Patched Train...")
    ds_patched = PatchWrapper(train_raw, cfg.patch_size, transform)
    Xp, _ = extract_all(
        model,
        DataLoader(ds_patched, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True),
        cfg.device,
    )

    # 3. Test Clean
    print("   Extracting Clean Test...")
    Xte, Yte = extract_all(
        model,
        DataLoader(test_for_feats, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True),
        cfg.device,
    )

    # 4. Test Backdoor
    print("   Extracting Backdoor Test...")
    ds_te_bd = PatchWrapper(test_raw, cfg.patch_size, transform)
    Xte_bd, _ = extract_all(
        model,
        DataLoader(ds_te_bd, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True),
        cfg.device,
    )

    # Save
    print(">>> Saving to cache...")
    np.save(paths["Xc"], Xc)
    np.save(paths["Xp"], Xp)
    np.save(paths["Y"], Y)
    np.save(paths["Xte"], Xte)
    np.save(paths["Yte"], Yte)
    np.save(paths["Xte_bd"], Xte_bd)

    return Xc, Xp, Y, Xte, Yte, Xte_bd


# ============================================================
# Selection Logic (SORG + Herding + K-center/K-means)
# ============================================================
def select_kcenter(X, k, seed):
    N = X.shape[0]
    rng = np.random.RandomState(seed)
    centers = [rng.randint(N)]
    dists = np.sum((X - X[centers[0]]) ** 2, axis=1)
    for _ in range(1, k):
        new_c = int(np.argmax(dists))
        centers.append(new_c)
        new_dists = np.sum((X - X[new_c]) ** 2, axis=1)
        dists = np.minimum(dists, new_dists)
    return np.array(centers, dtype=int)


def select_kmeans_mb(X, k, seed):
    km = MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        batch_size=4096,
        n_init=1,
    ).fit(X)
    closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, X)
    return np.unique(closest).astype(int)


def select_herding(X: np.ndarray, k: int) -> np.ndarray:
    """
    Global herding baseline.
    Greedy approximation to the global mean.
    """
    n, d = X.shape
    k = int(min(k, n))
    if k <= 0:
        return np.array([], dtype=int)

    Xf = X.astype(np.float32, copy=False)
    m = np.mean(Xf, axis=0).astype(np.float32)
    selected = []
    selected_mask = np.zeros(n, dtype=bool)
    running = np.zeros_like(m, dtype=np.float32)

    for t in range(k):
        w = running - t * m
        scores = -(Xf @ w)
        scores[selected_mask] = -np.inf
        j = int(np.argmax(scores))
        if selected_mask[j]:
            break
        selected.append(j)
        selected_mask[j] = True
        running += Xf[j]

    return np.asarray(selected, dtype=int)


def select_sorg_guided(
    X: np.ndarray,
    k: int,
    groups: np.ndarray,
    p_norm: float,
) -> np.ndarray:
    """
    Guided SORG:
      - Run SORG per class with r = class-mean guidance.
      - Balanced budget: k_per_class = floor(k / #classes).
      - No top-up: effective k may be <= requested k if some classes are very small.
    """
    unique_groups = np.unique(groups)
    if unique_groups.size == 0 or k <= 0:
        return np.zeros((0,), dtype=int)

    selected_indices = []
    k_per_class = k // unique_groups.size
    if k_per_class <= 0:
        # If k is too small, just fallback to one global SORG
        model = SORG(k=int(min(k, X.shape[0])), alpha=1.0, p_norm=float(p_norm))
        # Global mean guidance
        mu_global = np.mean(X, axis=0)
        model.fit(X.T, r=mu_global)
        sel_global = model.get_support()
        return np.asarray(sel_global, dtype=int)

    for c in unique_groups:
        idx_c = np.where(groups == c)[0]
        X_c = X[idx_c]
        N_c = X_c.shape[0]
        k_c = min(k_per_class, N_c)
        if k_c <= 0:
            continue

        mu_c = np.mean(X_c, axis=0)

        model = SORG(k=int(k_c), alpha=1.0, p_norm=float(p_norm))
        model.fit(X_c.T, r=mu_c)

        sel_c = model.get_support()
        selected_indices.append(idx_c[sel_c])

    if not selected_indices:
        return np.zeros((0,), dtype=int)

    return np.concatenate(selected_indices).astype(int)


# ============================================================
# Training & Eval
# ============================================================
def train_head(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    device: str,
    epochs: int,
    lr: float,
    wd: float,
    bs: int,
) -> nn.Module:
    head = nn.Linear(X.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)

    X_t = torch.from_numpy(X).float().to(device)
    y_t = torch.from_numpy(y).long().to(device)
    head.train()
    n = X_t.shape[0]

    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            out = head(X_t[idx])
            loss = F.cross_entropy(out, y_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    head.eval()
    return head


def plot_metric_vs_budget(
    agg: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    out_path: str,
) -> None:
    plt.figure(figsize=(7, 5))
    for m in CFG.methods:
        sub = agg[agg["method"] == m]
        if sub.empty:
            continue
        plt.plot(sub["budget"], sub[metric], marker="o", label=m)
    plt.xlabel("Budget (fraction of training set)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    exp_dir = os.path.join(os.getcwd(), f"{CFG.out_prefix}_{now_stamp()}")
    ensure_dir(exp_dir)

    # Save config as JSON
    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(CFG), f, indent=2)

    print("============================================================")
    print("E6: Backdoor Robustness (Cached + Guided SORG) | Smart Caching")
    print("------------------------------------------------------------")
    print(f"Device: {CFG.device}")
    print(f"Outputs: {exp_dir}")
    print(f"Threads: OMP={os.environ.get('OMP_NUM_THREADS','?')} Torch={torch.get_num_threads()}")
    print("============================================================")

    # 1. Load All Features
    Xc, Xp, Y, Xte, Yte, Xte_bd = get_cached_features(CFG)

    # [FAIRNESS] Normalize ALL features for selection AND training
    print(">>> Normalizing features (Fair comparison)...")
    Xc_norm = l2_normalize(Xc)
    Xp_norm = l2_normalize(Xp)
    Xte_norm = l2_normalize(Xte)
    Xte_bd_norm = l2_normalize(Xte_bd)

    N = len(Y)
    n_classes = int(np.max(Y) + 1)
    rows: List[dict] = []

    for seed in CFG.seeds:
        seed_everything(seed)

        # 2. Construct Poisoned Dataset
        idx_perm = np.random.RandomState(seed).permutation(N)
        n_bd = int(N * CFG.backdoor_ratio)
        n_noise = int(N * CFG.label_noise_ratio)

        bd_indices = idx_perm[:n_bd]
        noise_indices = idx_perm[n_bd : n_bd + n_noise]

        # Mix Clean and Patched Features (Using Normalized)
        X_train = Xc_norm.copy()
        X_train[bd_indices] = Xp_norm[bd_indices]

        # Labels
        y_train = Y.copy()
        y_train[bd_indices] = CFG.backdoor_target_class

        noise_rng = np.random.RandomState(seed + 123)
        for i in noise_indices:
            orig = y_train[i]
            new_l = noise_rng.randint(0, n_classes)
            if new_l == orig:
                new_l = (new_l + 1) % n_classes
            y_train[i] = new_l

        # Ground Truth Masks
        mask_bd = np.zeros(N, dtype=bool)
        mask_bd[bd_indices] = True
        mask_noise = np.zeros(N, dtype=bool)
        mask_noise[noise_indices] = True

        print(f"\n[Seed {seed}] Poison Constructed: BD={n_bd}, Noise={n_noise}")

        for b in CFG.budgets:
            k = int(N * b)
            if k < n_classes:
                k = n_classes

            for m in CFG.methods:
                t0 = time.time()

                if m == "random":
                    # Desynchronized seed for random baseline
                    sel = np.random.RandomState(seed + 99999).choice(N, k, replace=False)

                elif m == "herding":
                    sel = select_herding(X_train, k)

                elif m == "kcenter":
                    sel = select_kcenter(X_train, k, seed)

                elif m == "kmeans_mb":
                    sel = select_kmeans_mb(X_train, k, seed)

                elif m == "sorg":
                    # Guided SORG (r=Mean) + Gate ON (p=4.0)
                    sel = select_sorg_guided(X_train, k, groups=y_train, p_norm=4.0)

                elif m == "sorg_nogate":
                    # Guided SORG (r=Mean) + Gate OFF (p=0.0)
                    sel = select_sorg_guided(X_train, k, groups=y_train, p_norm=0.0)

                else:
                    raise ValueError(f"Unknown method: {m}")

                t_sel = float(time.time() - t0)

                sel = np.asarray(sel, dtype=int)
                if sel.size == 0:
                    # Degenerate (should not really happen, but guard anyway)
                    bd_rate = 0.0
                    noise_rate = 0.0
                    acc = 0.0
                    asr = 0.0
                    train_time = 0.0
                else:
                    # Metrics on selected set
                    bd_rate = float(mask_bd[sel].mean())
                    noise_rate = float(mask_noise[sel].mean())

                    # Train Head
                    head_t0 = time.time()
                    head = train_head(
                        X_train[sel],
                        y_train[sel],
                        n_classes,
                        CFG.device,
                        CFG.head_epochs,
                        CFG.head_lr,
                        CFG.head_wd,
                        CFG.head_bs,
                    )
                    train_time = float(time.time() - head_t0)

                    with torch.no_grad():
                        # Clean Acc
                        pc = head(torch.from_numpy(Xte_norm).float().to(CFG.device)).argmax(1).cpu().numpy()
                        acc = float((pc == Yte).mean())
                        # ASR (Attack Success Rate)
                        pb = head(torch.from_numpy(Xte_bd_norm).float().to(CFG.device)).argmax(1).cpu().numpy()
                        asr = float((pb == CFG.backdoor_target_class).mean())

                print(
                    f"  [{m:12s}] b={b:<4} | "
                    f"Acc={acc:.4f} ASR={asr:.4f} | "
                    f"BD%={bd_rate*100:.1f}% Noise%={noise_rate*100:.1f}% | "
                    f"sel={t_sel:.3f}s train={train_time:.3f}s"
                )

                rows.append({
                    "seed": int(seed),
                    "budget": float(b),
                    "k": int(k),
                    "method": str(m),
                    "acc": float(acc),
                    "asr": float(asr),
                    "bd_rate": float(bd_rate),
                    "noise_rate": float(noise_rate),
                    "sel_time_sec": float(t_sel),
                    "train_time_sec": float(train_time),
                })

    # Save & Plot
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(exp_dir, "results.csv"), index=False)

    # Aggregate over seeds (mean)
    agg = df.groupby(["method", "budget"], as_index=False).mean()

    # --- Plots vs Budget (headline story) ---
    plot_metric_vs_budget(
        agg,
        metric="asr",
        ylabel="Backdoor ASR (lower is better)",
        title="E6: Safety against Backdoor (Guided SORG)",
        out_path=os.path.join(exp_dir, "plot_asr_vs_budget.png"),
    )
    plot_metric_vs_budget(
        agg,
        metric="acc",
        ylabel="Clean Test Accuracy",
        title="E6: Clean Accuracy vs Budget",
        out_path=os.path.join(exp_dir, "plot_acc_vs_budget.png"),
    )
    plot_metric_vs_budget(
        agg,
        metric="bd_rate",
        ylabel="Selected Backdoor Fraction",
        title="E6: Backdoor Fraction in Selected Set vs Budget",
        out_path=os.path.join(exp_dir, "plot_bdfrac_vs_budget.png"),
    )
    plot_metric_vs_budget(
        agg,
        metric="noise_rate",
        ylabel="Selected Label-Noise Fraction",
        title="E6: Label-Noise Fraction in Selected Set vs Budget",
        out_path=os.path.join(exp_dir, "plot_noisefrac_vs_budget.png"),
    )
    plot_metric_vs_budget(
        agg,
        metric="sel_time_sec",
        ylabel="Selection Time (sec)",
        title="E6: Selection Time vs Budget",
        out_path=os.path.join(exp_dir, "plot_sel_time_vs_budget.png"),
    )
    plot_metric_vs_budget(
        agg,
        metric="train_time_sec",
        ylabel="Head Train Time (sec)",
        title="E6: Head Training Time vs Budget",
        out_path=os.path.join(exp_dir, "plot_train_time_vs_budget.png"),
    )

    print("\nDONE. Results saved to:", exp_dir)


if __name__ == "__main__":
    main()
