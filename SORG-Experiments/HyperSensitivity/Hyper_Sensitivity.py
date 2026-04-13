# (Final Defense: Hyperparameter Sensitivity | SORG Group Mode)
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
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.models as models
import torchvision.transforms as T

from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

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

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))

# ============================================================
# Core algorithm: SORG
# ============================================================
try:
    from SORG import SORG
except Exception as e:
    raise ImportError(f"SORG import failed: {e}")

# ============================================================
# Config
# ============================================================
@dataclass
class E8Config:
    dataset: str = "cifar100"
    encoder: str = "resnet18"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Base Settings
    budget: float = 0.05  # Fix budget at 5% (k ≈ 2500)
    seed: int = 0

    # Sensitivity Sweep for SORG
    # p_norms: Gate strength
    p_norms: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0, float("inf"))
    
    # alphas: Correction rate
    # 0.0 = Pure Greedy (No Update) -> Baseline check
    # 1.0 = Full Orthogonal -> Theoretical optimum
    alphas: Tuple[float, ...] = (0.0, 0.3, 0.5, 0.8, 1.0)

    # Environments (Check both Clean and Noisy)
    scenarios: Tuple[str, ...] = ("clean", "noisy")
    noise_ratio: float = 0.2
    noise_scale: float = 5.0  # Strong noise to test Gate

    # Ridge head
    ridge_alpha: float = 1.0

    # Encoder / loader
    batch_size_enc: int = 512
    num_workers_enc: int = 4

    out_prefix: str = "E8_Sensitivity"
    base_nfs: str = BASE_NFS


CFG = E8Config()


# ============================================================
# Helpers
# ============================================================
def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_resnet18(device: str):
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


def cifar_feat_paths(dataset: str, encoder: str) -> Dict[str, str]:
    tag = f"{dataset}_{encoder}"
    return {
        "train_X": os.path.join(FEATURE_CACHE, f"{tag}_train_X.npy"),
        "train_y": os.path.join(FEATURE_CACHE, f"{tag}_train_y_fine.npy"),
        "test_X": os.path.join(FEATURE_CACHE, f"{tag}_test_X.npy"),
        "test_y": os.path.join(FEATURE_CACHE, f"{tag}_test_y_fine.npy"),
        "meta": os.path.join(FEATURE_CACHE, f"{tag}_meta.json"),
    }


def get_data(cfg: E8Config):
    """
    Reuse E1 feature cache if available; otherwise extract once.
    """
    paths = cifar_feat_paths(cfg.dataset, cfg.encoder)
    needed = ("train_X", "train_y", "test_X", "test_y", "meta")
    if all(os.path.exists(paths[k]) for k in needed):
        print(">>> Loading cached CIFAR100 features (E1 cache)...")
        Xt = np.load(paths["train_X"])
        Yt = np.load(paths["train_y"])
        Xte = np.load(paths["test_X"])
        Yte = np.load(paths["test_y"])
        return Xt, Yt, Xte, Yte

    print(">>> Cache miss. Extracting CIFAR100 features for E8 (once)...")
    model, transform = get_resnet18(cfg.device)

    train_ds = torchvision.datasets.CIFAR100(
        TORCH_DATASETS, train=True, download=True, transform=transform
    )
    test_ds = torchvision.datasets.CIFAR100(
        TORCH_DATASETS, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size_enc,
        num_workers=cfg.num_workers_enc,
        shuffle=False,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size_enc,
        num_workers=cfg.num_workers_enc,
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
# Selection methods
# ============================================================
def select_random(rng: np.random.RandomState, n: int, k: int) -> np.ndarray:
    """Random baseline (Global)"""
    k = int(min(k, n))
    return rng.choice(n, size=k, replace=False).astype(int)


def select_herding_group(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """
    Group Herding (Class-wise Mean Matching).
    Fair baseline for Group SORG.
    """
    classes = np.unique(y)
    n_classes = len(classes)
    k_per_class = k // n_classes
    remainder = k % n_classes
    
    selected_indices = []
    
    for i, c in enumerate(classes):
        current_k = k_per_class + (1 if i < remainder else 0)
        if current_k == 0: continue
        
        idx = np.where(y == c)[0]
        X_c = X[idx]
        
        # Herding Logic per class
        mu = np.mean(X_c, axis=0)
        current_sum = np.zeros_like(mu)
        mask = np.ones(len(idx), dtype=bool)
        
        for t in range(current_k):
            target = (t + 1) * mu - current_sum
            scores = X_c @ target
            scores[~mask] = -np.inf
            best = int(np.argmax(scores))
            
            selected_indices.append(idx[best])
            mask[best] = False
            current_sum += X_c[best]
            
    return np.array(selected_indices, dtype=int)


def select_sorg_group(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    alpha: float,
    p_norm: float,
) -> np.ndarray:
    """
    [CRITICAL] Group-wise SORG with Explicit Guidance (r=Mean).
    
    To match Herding's objective on clean data, we must provide r=class_mean.
    Without 'r', SORG defaults to Diversity (Variance Max), which is suboptimal
    for clean accuracy compared to Mean Matching.
    """
    classes = np.unique(y)
    n_classes = len(classes)
    k_per_class = k // n_classes
    remainder = k % n_classes
    
    selected_indices = []
    
    for i, c in enumerate(classes):
        current_k = k_per_class + (1 if i < remainder else 0)
        if current_k == 0: continue
        
        # 1. Get class data
        idx_global = np.where(y == c)[0]
        X_c = X[idx_global]
        
        # 2. Calculate Guidance (Target = Mean)
        # This makes SORG act as "Robust Herding"
        mu_c = np.mean(X_c, axis=0)
        
        # 3. Run SORG on this class
        # No scaling needed (handled externally)
        model = SORG(
            k=int(current_k),
            alpha=float(alpha),
            p_norm=float(p_norm),
        )
        
        # Pass transposed X (d, N) and target r (d,)
        model.fit(X_c.T, r=mu_c)
        
        # 4. Map back to global indices
        sel_local = model.get_support()
        selected_indices.append(idx_global[sel_local])
            
    return np.concatenate(selected_indices).astype(int)


# ============================================================
# Train / Eval (Ridge on standardized features)
# ============================================================
def train_eval_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
) -> float:
    clf = RidgeClassifier(alpha=float(alpha))
    clf.fit(X_train, y_train)
    acc = float(clf.score(X_test, y_test))
    return acc


# ============================================================
# Plot helpers
# ============================================================
def plot_heatmaps(df: pd.DataFrame, exp_dir: str) -> None:
    """
    p_norm x alpha heatmaps for SORG only, per scenario.
    """
    for scen in CFG.scenarios:
        sub = df[(df["scenario"] == scen) & (df["method"] == "sorg")]
        if sub.empty:
            continue

        p_labels = [("inf" if p == float("inf") else f"{p:g}") for p in CFG.p_norms]
        sub = sub.copy()
        sub["p_label"] = sub["p_norm"].apply(
            lambda v: "inf" if float(v) == float("inf") else f"{float(v):g}"
        )

        pivot = sub.pivot(index="p_label", columns="alpha", values="acc")
        pivot = pivot.reindex(index=p_labels)

        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
        plt.title(f"SORG Sensitivity ({scen.upper()})")
        plt.ylabel("SoftNorm Power (p)")
        plt.xlabel("Correction Rate (alpha)")
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, f"heatmap_sorg_{scen}.png"), dpi=200)
        plt.close()


def plot_best_vs_baselines(df: pd.DataFrame, exp_dir: str) -> None:
    """
    For each scenario, compare best SORG config vs Random / Group Herding.
    """
    for scen in CFG.scenarios:
        sub = df[df["scenario"] == scen]
        if sub.empty:
            continue

        # Baselines
        rand_acc = sub[sub["method"] == "random"]["acc"].mean()
        herd_acc = sub[sub["method"] == "herding"]["acc"].mean()

        # Best SORG over (p, alpha)
        sorg_sub = sub[sub["method"] == "sorg"]
        if not sorg_sub.empty:
            best_idx = sorg_sub["acc"].idxmax()
            best_acc = float(sorg_sub.loc[best_idx, "acc"])
            best_p = sorg_sub.loc[best_idx, "p_norm"]
            best_alpha = sorg_sub.loc[best_idx, "alpha"]
        else:
            best_acc = float("nan")
            best_p = None
            best_alpha = None

        labels = ["Random", "Group Herding", f"SORG (Best)"]
        values = [rand_acc, herd_acc, best_acc]
        colors = ['gray', 'orange', 'green']

        plt.figure(figsize=(7, 5))
        bars = plt.bar(labels, values, color=colors)
        plt.ylabel("Test Accuracy")
        plt.title(f"Performance Comparison ({scen.upper()})")
        
        # Add values on bars
        for bar, v in zip(bars, values):
            if np.isfinite(v):
                plt.text(bar.get_x() + bar.get_width()/2, v + 0.005, 
                         f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight='bold')
        
        # Adjust Y limit
        ymax = max([v for v in values if np.isfinite(v)] + [0.5])
        plt.ylim(0.0, ymax * 1.1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, f"best_vs_baselines_{scen}.png"), dpi=200)
        plt.close()


# ============================================================
# Main Logic
# ============================================================
def main():
    exp_dir = os.path.join(os.getcwd(), f"{CFG.out_prefix}_{now_stamp()}")
    ensure_dir(exp_dir)

    print("============================================================")
    print("E8: Hyperparameter Sensitivity Analysis (SORG Group Mode)")
    print("------------------------------------------------------------")

    # Save config
    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(CFG), f, indent=2)

    # Load / build features
    Xt, Yt, Xte, Yte = get_data(CFG)

    # Global standardization (External Preprocessing)
    print(">>> Applying StandardScaler (External Preprocessing)...")
    scaler = StandardScaler()
    Xt_s = scaler.fit_transform(Xt)
    Xte_s = scaler.transform(Xte)

    N = len(Yt)
    k = int(N * CFG.budget)
    print(f"Total train N={N}, budget={CFG.budget:.3f} -> k={k}")

    rows: List[dict] = []

    for scenario in CFG.scenarios:
        print(f"\n>>> Scenario: {scenario}")
        seed_everything(CFG.seed)

        if scenario == "clean":
            X_curr = Xt_s.copy()
            y_curr = Yt.copy()
        else:
            # Noisy: add feature noise (Outlier-like)
            print(f"   Injecting noise (ratio={CFG.noise_ratio}, scale={CFG.noise_scale})...")
            rng = np.random.RandomState(CFG.seed)
            noise_idx = rng.choice(N, int(N * CFG.noise_ratio), replace=False)
            X_curr = Xt_s.copy()
            noise_vec = rng.randn(len(noise_idx), Xt_s.shape[1]).astype(np.float32)
            noise_vec *= CFG.noise_scale
            X_curr[noise_idx] += noise_vec
            y_curr = Yt.copy()

        # ---------------------------
        # Baselines: Random / Group Herding
        # ---------------------------
        # 1. Random
        rng = np.random.RandomState(CFG.seed + 999)
        sel_rand = select_random(rng, N, k)
        acc_rand = train_eval_ridge(
            X_curr[sel_rand], y_curr[sel_rand], Xte_s, Yte, alpha=CFG.ridge_alpha
        )
        print(f"  [Baseline: Random       ] acc={acc_rand:.4f}")
        rows.append({
            "scenario": scenario,
            "method": "random",
            "p_norm": "NA",
            "alpha": "NA",
            "acc": acc_rand,
        })

        # 2. Group Herding (Strong Baseline)
        sel_herd = select_herding_group(X_curr, y_curr, k)
        acc_herd = train_eval_ridge(
            X_curr[sel_herd], y_curr[sel_herd], Xte_s, Yte, alpha=CFG.ridge_alpha
        )
        print(f"  [Baseline: Group Herding] acc={acc_herd:.4f}")
        rows.append({
            "scenario": scenario,
            "method": "herding",
            "p_norm": "NA",
            "alpha": "NA",
            "acc": acc_herd,
        })

        # ---------------------------
        # SORG Hyperparameter Sweep (Group Mode)
        # ---------------------------
        for p in CFG.p_norms:
            p_display = "inf" if p == float("inf") else float(p)
            for alpha in CFG.alphas:
                t0 = time.time()
                try:
                    # [KEY] Use Group SORG (Class-wise) with Guidance
                    sel_sorg = select_sorg_group(
                        X=X_curr,
                        y=y_curr,
                        k=k,
                        alpha=alpha,
                        p_norm=p,
                    )
                    
                    acc = train_eval_ridge(
                        X_curr[sel_sorg],
                        y_curr[sel_sorg],
                        Xte_s,
                        Yte,
                        alpha=CFG.ridge_alpha,
                    )
                except Exception as e:
                    print(f"  [SORG ERROR] p={p} alpha={alpha}: {e}")
                    acc = 0.0

                dt = time.time() - t0
                print(
                    f"  [SORG] p={str(p_display):>4} alpha={alpha:.2f} | acc={acc:.4f} | time={dt:.2f}s"
                )

                rows.append({
                    "scenario": scenario,
                    "method": "sorg",
                    "p_norm": p_display,
                    "alpha": alpha,
                    "acc": acc,
                })

    # Save CSV
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(exp_dir, "sensitivity_results.csv"), index=False)

    # Plot Results
    plot_heatmaps(df, exp_dir)
    plot_best_vs_baselines(df, exp_dir)

    print("\nDONE. Results saved to:", exp_dir)


if __name__ == "__main__":
    main()