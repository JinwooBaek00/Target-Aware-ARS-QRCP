# E5_Outlier_Robustness.py
# E5: Outlier Robustness | SORG (NoScale) vs Baselines | Camera-ready
# -*- coding: utf-8 -*-

import os

# ============================================================
# [Critical] Stabilize CPU thread usage (HPC-safe)
# ============================================================
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import json
import time
import zlib
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ============================================================
# [Rule #1] Path Config (HPC share only)
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
torch.set_num_threads(4)

try:
    from SORG import SORG
    import Baselines as BL
except Exception as e:
    raise ImportError(f"Import failed: {e}")


# ============================================================
# Config
# ============================================================
@dataclass
class E5Config:
    # Data / encoder
    dataset: str = "cifar100"
    encoder: str = "resnet18"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Seeds / budgets
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    k_list: Tuple[int, ...] = (100, 200, 500)

    # Pool / contamination
    pool_size: int = 6000
    contam_rates: Tuple[float, ...] = (0.0, 0.1, 0.3)

    # Outlier generation
    outlier_multiplier: float = 50.0  # Norm multiplier vs clean median
    outlier_df: float = 2.0          # t-distribution degrees of freedom

    # Methods
    methods: Tuple[str, ...] = (
        "random",
        "herding",
        "kcenter",
        "kmeans_mb",
        "sorg",         # SORG (SoftNorm gate ON, p=2.0)
        "sorg_nogate",  # Ablation (gate OFF, p=0.0)
    )

    # Baseline parameters
    kmeans_max_iter: int = 80
    ridge_alpha: float = 1.0

    # Output
    out_prefix: str = "E5_outputs"
    base_nfs: str = BASE_NFS

    # Plot config
    main_k_index: int = 1  # index in k_list to use as "main" k for paper plots


CFG = E5Config()


# ============================================================
# Small Helpers
# ============================================================
def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stable_int(s: str) -> int:
    """Stable string-to-int (for rng seeds)."""
    return int(zlib.adler32(s.encode("utf-8")) & 0xFFFFFFFF)


def save_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ============================================================
# Feature Extraction (Cached CIFAR-100 + ResNet-18)
# ============================================================
def get_resnet18(device: str):
    """Load ResNet-18 encoder and transforms (ImageNet-pretrained)."""
    try:
        w = torchvision.models.ResNet18_Weights.DEFAULT
        m = torchvision.models.resnet18(weights=w)
        t = w.transforms()
    except Exception:
        # Fallback for older torchvision versions
        m = torchvision.models.resnet18(pretrained=True)
        t = T.Compose(
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
    return feat, t


@torch.no_grad()
def extract_all(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        z = model(x).flatten(1)
        feats.append(z.cpu().numpy().astype(np.float32))
        labels.append(y.numpy().astype(np.int64))
    X = np.concatenate(feats, axis=0)
    y = np.concatenate(labels, axis=0)
    return X, y


def load_or_build_features(cfg: E5Config):
    """Load cached CIFAR-100 features or build and cache them."""
    tag = f"{cfg.dataset}_{cfg.encoder}"
    paths = {
        "train_X": os.path.join(FEATURE_CACHE, f"{tag}_train_X.npy"),
        "train_y": os.path.join(FEATURE_CACHE, f"{tag}_train_y_fine.npy"),
        "test_X": os.path.join(FEATURE_CACHE, f"{tag}_test_X.npy"),
        "test_y": os.path.join(FEATURE_CACHE, f"{tag}_test_y_fine.npy"),
        "meta": os.path.join(FEATURE_CACHE, f"{tag}_meta.json"),
    }

    # Try loading cache
    if all(os.path.exists(p) for p in paths.values()):
        print(">>> Loading cached features...")
        with open(paths["meta"], "r", encoding="utf-8") as f:
            meta = json.load(f)
        Xt = np.load(paths["train_X"])
        Yt = np.load(paths["train_y"])
        Xte = np.load(paths["test_X"])
        Yte = np.load(paths["test_y"])
        return Xt, Yt, Xte, Yte, int(meta["feat_dim"])

    # Cache miss: extract
    print(">>> Extracting CIFAR-100 features (ResNet-18)...")
    model, transform = get_resnet18(cfg.device)

    tr_ds = torchvision.datasets.CIFAR100(
        TORCH_DATASETS,
        train=True,
        download=True,
        transform=transform,
    )
    te_ds = torchvision.datasets.CIFAR100(
        TORCH_DATASETS,
        train=False,
        download=True,
        transform=transform,
    )

    Xt, Yt = extract_all(model, DataLoader(tr_ds, batch_size=512, num_workers=4), cfg.device)
    Xte, Yte = extract_all(model, DataLoader(te_ds, batch_size=512, num_workers=4), cfg.device)

    np.save(paths["train_X"], Xt)
    np.save(paths["train_y"], Yt)
    np.save(paths["test_X"], Xte)
    np.save(paths["test_y"], Yte)
    save_json(
        paths["meta"],
        {"feat_dim": Xt.shape[1], "dataset": cfg.dataset, "encoder": cfg.encoder},
    )

    return Xt, Yt, Xte, Yte, Xt.shape[1]


# ============================================================
# Outlier Generation (Targeted Norm Attack)
# ============================================================
def generate_outliers(
    rng: np.random.RandomState,
    n_out: int,
    d: int,
    pool_median: float,
    multiplier: float,
    df: float,
) -> np.ndarray:
    """Generate heavy-tailed, large-norm outliers."""
    if n_out <= 0:
        return np.zeros((0, d), dtype=np.float32)

    # Heavy-tailed t-distribution in R^d
    X = rng.standard_t(df=df, size=(n_out, d)).astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12

    # Scale norms to be 'multiplier' times the clean median norm
    target_norm = float(multiplier) * float(pool_median)
    X = (X / norms) * target_norm
    return X


def generate_outlier_labels(rng: np.random.RandomState, n_out: int, n_classes: int) -> np.ndarray:
    """Random labels for outliers (can be changed to targeted labels for stronger attacks)."""
    if n_out <= 0:
        return np.zeros((0,), dtype=np.int64)
    return rng.randint(0, n_classes, size=(n_out,), dtype=np.int64)


# ============================================================
# SORG Selection (prefix-based, single run per scenario)
# ============================================================
def select_sorg_prefixes(
    Xp: np.ndarray,
    k_list: Tuple[int, ...],
    p_norm: float,
) -> Dict[int, np.ndarray]:
    """
    Run SORG once with k_max and return prefixes for each k in k_list.

    This makes the SORG subsets across different k consistent as prefixes
    of a single greedy path, matching our theoretical analysis.
    """
    if len(k_list) == 0:
        return {}

    k_unique = sorted(set(int(k) for k in k_list if k > 0))
    if not k_unique:
        return {}

    k_max = max(k_unique)
    if k_max > Xp.shape[0]:
        # Caller is responsible for filtering invalid k's beforehand
        k_max = max(k for k in k_unique if k <= Xp.shape[0])

    model = SORG(k=int(k_max), alpha=1.0, p_norm=float(p_norm))
    # SORG expects shape (d, N)
    model.fit(Xp.T)
    full_support = np.asarray(model.get_support(), dtype=int)

    if full_support.shape[0] < k_max:
        raise RuntimeError(
            f"SORG returned only {full_support.shape[0]} indices, "
            f"but k_max={k_max} was requested."
        )

    prefix_dict: Dict[int, np.ndarray] = {}
    for k in k_unique:
        if k <= full_support.shape[0]:
            prefix_dict[k] = full_support[:k].copy()
    return prefix_dict


# ============================================================
# Eval (linear probe)
# ============================================================
def train_eval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float = 1.0,
) -> Tuple[float, float]:
    """
    Train a RidgeClassifier on standardized features and return (acc, train_time).
    """
    t0 = time.time()

    scaler = StandardScaler()
    Xt_s = scaler.fit_transform(X_train)
    Xte_s = scaler.transform(X_test)

    clf = RidgeClassifier(alpha=alpha)
    clf.fit(Xt_s, y_train)
    acc = float(clf.score(Xte_s, y_test))

    return acc, time.time() - t0


# ============================================================
# Plotting Utilities
# ============================================================
def make_plots(df: pd.DataFrame, exp_dir: str, cfg: E5Config) -> None:
    """
    Generate camera-ready plots:
      - Main: contamination sweep (0, 0.1, 0.3) at a fixed k (default: middle of k_list),
              showing accuracy and outlier ratio.
      - Appendix-style: same sweeps for all k in k_list.
    """
    if df.empty:
        print("[WARN] Empty DataFrame, skipping plots.")
        return

    ensure_dir(exp_dir)
    sns.set(style="whitegrid", font_scale=1.2)

    # Save aggregated stats for convenience
    summary = (
        df.groupby(["method", "contam", "k"])
        .agg(
            acc_mean=("acc", "mean"),
            acc_std=("acc", "std"),
            out_mean=("outlier_ratio", "mean"),
            out_std=("outlier_ratio", "std"),
        )
        .reset_index()
    )
    summary.to_csv(os.path.join(exp_dir, "summary.csv"), index=False)

    # Choose main k (for paper figures)
    k_list_sorted = sorted(cfg.k_list)
    main_k_idx = min(max(cfg.main_k_index, 0), len(k_list_sorted) - 1)
    main_k = k_list_sorted[main_k_idx]
    print(f"[Plot] Using k={main_k} as main budget for contamination sweeps.")

    df_main = df[df["k"] == main_k].copy()

    # 1) Main figure: accuracy vs contamination (k fixed)
    plt.figure(figsize=(7, 5))
    sns.lineplot(
        data=df_main,
        x="contam",
        y="acc",
        hue="method",
        style="method",
        marker="o",
        errorbar="sd",
    )
    plt.title(f"E5: CIFAR-100 Outlier Robustness (Accuracy, k={main_k})")
    plt.xlabel("Contamination rate")
    plt.ylabel("Test accuracy")
    plt.ylim(0.0, max(0.35, df_main["acc"].max() * 1.05))
    plt.legend(title="Method", loc="upper right", bbox_to_anchor=(1.02, 1.0))
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, f"plot_main_acc_k{main_k}.png"), dpi=300)
    plt.close()

    # 2) Main figure: outlier_ratio vs contamination (k fixed)
    plt.figure(figsize=(7, 5))
    sns.lineplot(
        data=df_main,
        x="contam",
        y="outlier_ratio",
        hue="method",
        style="method",
        marker="o",
        errorbar="sd",
    )
    plt.title(f"E5: CIFAR-100 Outlier Robustness (Outlier Fraction, k={main_k})")
    plt.xlabel("Contamination rate")
    plt.ylabel("Outlier fraction in subset")
    plt.ylim(-0.02, 1.02)
    plt.legend(title="Method", loc="upper left", bbox_to_anchor=(1.02, 1.0))
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, f"plot_main_outlier_k{main_k}.png"), dpi=300)
    plt.close()

    # 3) Appendix-style: small multiples over k for both metrics
    metrics = [("acc", "Test accuracy", (0.0, max(0.35, df["acc"].max() * 1.05))),
               ("outlier_ratio", "Outlier fraction", (-0.02, 1.02))]

    for metric, ylabel, ylim in metrics:
        plt.figure(figsize=(4 * len(k_list_sorted), 4))
        for i, k in enumerate(k_list_sorted):
            plt.subplot(1, len(k_list_sorted), i + 1)
            sub = df[df["k"] == k]
            if sub.empty:
                continue
            sns.lineplot(
                data=sub,
                x="contam",
                y=metric,
                hue="method",
                marker="o",
                errorbar="sd",
            )
            plt.title(f"k={k}")
            plt.xlabel("Contamination rate")
            plt.ylabel(ylabel)
            plt.ylim(*ylim)
            if i == len(k_list_sorted) - 1:
                plt.legend(title="Method", bbox_to_anchor=(1.05, 1.0), loc="upper left")
            else:
                plt.legend([], [], frameon=False)
        plt.suptitle(f"E5: {ylabel} vs Contamination (All k)", y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, f"plot_{metric}_allk.png"), dpi=300)
        plt.close()


# ============================================================
# Main
# ============================================================
def main():
    exp_dir = os.path.join(os.getcwd(), f"{CFG.out_prefix}_{now_stamp()}")
    ensure_dir(exp_dir)
    save_json(os.path.join(exp_dir, "config.json"), asdict(CFG))

    print("============================================================")
    print("E5: Outlier Robustness (SORG NoScale) | Camera-ready")
    print("------------------------------------------------------------")

    # Load features
    Xt_raw, Yt, Xte_raw, Yte, feat_dim = load_or_build_features(CFG)
    n_classes = int(np.max(Yt) + 1)

    rows: List[Dict] = []

    # ========================================================
    # Main experiment loop
    # ========================================================
    for seed in CFG.seeds:
        seed_everything(seed)
        rng = np.random.RandomState(seed)

        # Subsample clean pool
        P = min(CFG.pool_size, len(Xt_raw))
        pool_idx = rng.choice(len(Xt_raw), size=P, replace=False)
        X_pool = Xt_raw[pool_idx]
        y_pool = Yt[pool_idx]

        # Median norm of clean pool (for outlier scaling)
        pool_norms = np.linalg.norm(X_pool, axis=1)
        pool_median = float(np.median(pool_norms))

        for contam in CFG.contam_rates:
            n_out = int(P * contam)
            rng_out = np.random.RandomState(seed + stable_int(f"out_{contam}"))

            # Generate outliers
            X_out = generate_outliers(
                rng_out,
                n_out=n_out,
                d=feat_dim,
                pool_median=pool_median,
                multiplier=CFG.outlier_multiplier,
                df=CFG.outlier_df,
            )
            y_out = generate_outlier_labels(rng_out, n_out=n_out, n_classes=n_classes)

            # Combine clean + outliers
            X_all = np.concatenate([X_pool, X_out], axis=0)
            y_all = np.concatenate([y_pool, y_out], axis=0)
            is_outlier = np.zeros(len(X_all), dtype=bool)
            if n_out > 0:
                is_outlier[P:] = True

            print(
                f"\n[Seed {seed}] Contam={contam:.1f} | "
                f"Pool={len(X_pool)} Out={n_out} Total={len(X_all)}"
            )

            # Valid budgets for this scenario
            k_valid = [k for k in CFG.k_list if k <= len(X_all)]
            if not k_valid:
                print("  [WARN] No valid k for this pool size, skipping.")
                continue

            # Precompute SORG paths (gate ON/OFF) once per (seed, contam)
            try:
                sorg_prefix = select_sorg_prefixes(X_all, tuple(k_valid), p_norm=2.0)
            except Exception as e:
                print(f"  [ERROR] SORG (gate ON) failed: {e}")
                sorg_prefix = {}

            try:
                sorg_nogate_prefix = select_sorg_prefixes(X_all, tuple(k_valid), p_norm=0.0)
            except Exception as e:
                print(f"  [ERROR] SORG (gate OFF) failed: {e}")
                sorg_nogate_prefix = {}

            # Inner loop: iterate over k and methods
            for k in k_valid:
                for method in CFG.methods:
                    t0 = time.time()

                    try:
                        # ------------------------
                        # Baselines
                        # ------------------------
                        if method == "random":
                            sel = BL.random_selection(X_all, k, random_state=seed)
                        elif method == "herding":
                            sel = BL.herding_selection(X_all, k)
                        elif method == "kcenter":
                            sel = BL.kcenter_selection(X_all, k, random_state=seed)
                        elif method == "kmeans_mb":
                            sel = BL.kmeans_coreset(
                                X_all,
                                k,
                                max_iter=CFG.kmeans_max_iter,
                                random_state=seed,
                            )

                        # ------------------------
                        # SORG (SoftNorm ON / OFF)
                        # ------------------------
                        elif method == "sorg":
                            sel = sorg_prefix.get(k, np.array([], dtype=int))
                        elif method == "sorg_nogate":
                            sel = sorg_nogate_prefix.get(k, np.array([], dtype=int))

                        else:
                            sel = np.array([], dtype=int)

                    except Exception as e:
                        print(f"  [ERROR] {method}: {e}")
                        sel = np.array([], dtype=int)

                    t_sel = time.time() - t0

                    if sel.size == 0:
                        # Failed selection or method not available
                        continue

                    # Metrics
                    outlier_ratio = float(np.mean(is_outlier[sel])) if sel.size > 0 else 0.0
                    acc, t_train = train_eval(
                        X_train=X_all[sel],
                        y_train=y_all[sel],
                        X_test=Xte_raw,
                        y_test=Yte,
                        alpha=CFG.ridge_alpha,
                    )

                    print(
                        f"  [{method:15s}] k={k:<4} | "
                        f"Acc={acc:.4f} | OutRatio={outlier_ratio:.3f}"
                    )

                    rows.append(
                        {
                            "seed": seed,
                            "contam": contam,
                            "k": k,
                            "method": method,
                            "acc": acc,
                            "outlier_ratio": outlier_ratio,
                            "sel_time": t_sel,
                            "train_time": t_train,
                        }
                    )

    # ========================================================
    # Save results & plots
    # ========================================================
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(exp_dir, "results.csv"), index=False)
    print(f"\n[Info] Saved raw results to {os.path.join(exp_dir, 'results.csv')}")

    make_plots(df, exp_dir, CFG)

    print("\nDONE.")


if __name__ == "__main__":
    main()
