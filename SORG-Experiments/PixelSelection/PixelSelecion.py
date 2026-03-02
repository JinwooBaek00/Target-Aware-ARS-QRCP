# PixelSelection.py
# Feature Selection on Image Pixels
#      SORG vs Random vs Variance vs L1-Logistic (oracle)
#
# Dataset:
#   - MNIST (28 x 28 = 784 pixels).
#
# Methods:
#   - Random      : uniform random subset of pixels (label-free).
#   - Variance    : highest-variance pixels (label-free).
#   - L1_LogReg   : supervised oracle; pixels ranked by L1-logistic coefficients.
#   - SORG        : unsupervised spectral subset selector (single primitive).
#
# Evaluation:
#   - Train multinomial logistic regression ONLY on selected pixels.
#   - Labels are used only in the evaluation classifier, except for L1_LogReg
#     which uses labels to build an oracle feature ranking.
#
# Outputs:
#   - results.csv / summary.csv
#   - full_baseline.json (784-pixel logistic accuracy)
#   - E10_Accuracy.png       (Acc vs k, with error bands)
#   - E10_SelectionTime.png  (Selection time vs k, log-scale)
#   - E10_PixelMasks.png     (Random / Variance / L1_LogReg / SORG masks for k=50)
#
# -*- coding: utf-8 -*-

import os
import time
import json
import warnings
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision import transforms
from sklearn.linear_model import LogisticRegression

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# [Critical] Environment Setup
# ============================================================
os.environ["OMP_NUM_THREADS"] = "4"
warnings.filterwarnings("ignore")

BASE_NFS = "/nfs/hpc/share/limjoy"
os.makedirs(BASE_NFS, exist_ok=True)

try:
    from SORG import SORG
except ImportError:
    raise ImportError("SORG.py not found. Please ensure it is in the same directory.")


# ============================================================
# Config
# ============================================================
@dataclass
class E10Config:
    # Data: MNIST (28x28 = 784 features)
    n_samples_train: int = 5000
    n_samples_test: int = 1000

    # Pixel budgets k
    k_list: tuple = (10, 20, 50, 100, 200)

    # SORG params (unsupervised mode: r=None)
    # Here we use a “NoGate” configuration (p=0.0).
    sorg_alpha: float = 1.0
    sorg_p_norm: float = 0.0

    # L1-logistic (oracle) hyperparams
    l1_C: float = 0.1          # sparsity strength (smaller = more sparse)
    l1_max_iter: int = 500

    # Robustness over selection randomness (affects only Random)
    base_seed: int = 42
    n_seeds: int = 3

    out_prefix: str = "E10_PixelSelection"


CFG = E10Config()


# ============================================================
# Utils
# ============================================================
def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# Data Loader
# ============================================================
def load_mnist_flat():
    """
    Load MNIST and return flattened pixel arrays.

    Returns:
        X_train_full: (N_train_full, 784), float32 in [0, 1]
        y_train_full: (N_train_full,)
        X_test_full:  (N_test_full, 784), float32 in [0, 1]
        y_test_full:  (N_test_full,)
    """
    print(">>> Loading MNIST...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    data_dir = os.path.join(BASE_NFS, "torch_datasets")
    train_set = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    # Use raw data (0-255) and normalize to [0, 1] as features
    X_train_full = train_set.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_train_full = train_set.targets.numpy()

    X_test_full = test_set.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_test_full = test_set.targets.numpy()

    return X_train_full, y_train_full, X_test_full, y_test_full


# ============================================================
# Selection Methods
# ============================================================
def select_random(n_features: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.choice(n_features, k, replace=False)


def select_variance(X: np.ndarray, k: int) -> np.ndarray:
    """
    Select pixels with highest variance across the training set (label-free).
    X: (N, d)
    """
    variances = np.var(X, axis=0)
    return np.argsort(variances)[-k:]


def l1_feature_ranking(
    X: np.ndarray,
    y: np.ndarray,
    C: float,
    max_iter: int,
    random_state: int,
) -> np.ndarray:
    """
    Supervised oracle baseline:
      - Train L1-regularized multinomial logistic regression on ALL pixels.
      - Rank pixels by ||coef_j|| across classes.
      - Return an ordering of pixel indices from lowest to highest score.
    """
    clf = LogisticRegression(
        penalty="l1",
        solver="saga",
        multi_class="multinomial",
        C=C,
        max_iter=max_iter,
        n_jobs=1,
        random_state=random_state,
    )
    clf.fit(X, y)
    # coef_: (n_classes, d); use L2 norm across classes as importance score
    scores = np.linalg.norm(clf.coef_, axis=0)
    order = np.argsort(scores)   # ascending
    return order  # smallest -> largest; top-k = order[-k:]


def select_sorg_pixel(X: np.ndarray, k: int, cfg) -> np.ndarray:
    """
    Unsupervised SORG: Select columns (pixels) that best reconstruct X.
    X: (N, d) where d = 784.
    """
    selector = SORG(k=k, alpha=cfg["alpha"], p_norm=cfg["p"])
    # This SORG implementation is column-selection on features (N x d).
    selector.fit(X, r=None)
    idx = selector.get_support()
    return np.array(idx, dtype=int)


# ============================================================
# Main Experiment
# ============================================================
def main():
    exp_dir = os.path.join(os.getcwd(), f"{CFG.out_prefix}_{now_stamp()}")
    ensure_dir(exp_dir)

    print("============================================================")
    print("E10: Pixel Selection on MNIST")
    print("     (SORG vs Random vs Variance vs L1-Logistic Oracle)")
    print("------------------------------------------------------------")
    print(json.dumps(asdict(CFG), indent=2))
    print("============================================================")

    # --------------------------------------------------------
    # 1. Load & Subsample Data (fixed across all seeds)
    # --------------------------------------------------------
    X_train_full, y_train_full, X_test_full, y_test_full = load_mnist_flat()

    rng_data = np.random.RandomState(CFG.base_seed)
    train_idx = rng_data.choice(len(X_train_full), CFG.n_samples_train, replace=False)
    test_idx = rng_data.choice(len(X_test_full), CFG.n_samples_test, replace=False)

    X_train = X_train_full[train_idx]
    y_train = y_train_full[train_idx]
    X_test = X_test_full[test_idx]
    y_test = y_test_full[test_idx]

    print(f"[INFO] Train subset: {X_train.shape}, Test subset: {X_test.shape}")

    # --------------------------------------------------------
    # 2. Full-feature baseline (784 pixels)
    # --------------------------------------------------------
    print("\n[Baseline] Training full-feature logistic regression...")
    clf_full = LogisticRegression(
        solver="lbfgs",
        max_iter=200,
        multi_class="multinomial",
        n_jobs=1,
    )
    t0_full = time.time()
    clf_full.fit(X_train, y_train)
    full_acc = clf_full.score(X_test, y_test)
    t_full = time.time() - t0_full
    print(f"[Baseline] Full 784-pixel accuracy: {full_acc:.4f} (train time={t_full:.2f}s)")

    # --------------------------------------------------------
    # 3. Selection & Evaluation Loop
    # --------------------------------------------------------
    methods = ["Random", "Variance", "L1_LogReg", "SORG"]
    results = []

    vis_k = 50  # visualize this k
    masks = {}

    run_seeds = [CFG.base_seed + i * 100 for i in range(CFG.n_seeds)]

    for seed in run_seeds:
        seed_everything(seed)
        print(f"\n[Seed {seed}] Evaluation loop start...")

        # L1-logistic ranking (oracle) is computed once per seed
        print("  [Oracle] Fitting L1-logistic for feature ranking...")
        t0_l1 = time.time()
        l1_order = l1_feature_ranking(
            X_train,
            y_train,
            C=CFG.l1_C,
            max_iter=CFG.l1_max_iter,
            random_state=seed,
        )
        l1_time = time.time() - t0_l1
        print(f"  [Oracle] L1 ranking done in {l1_time:.2f}s")

        for k in CFG.k_list:
            for m in methods:
                t0 = time.time()

                if m == "Random":
                    sel_idx = select_random(784, k, seed)
                elif m == "Variance":
                    sel_idx = select_variance(X_train, k)
                elif m == "L1_LogReg":
                    # Use the same ranking; top-k are most important
                    sel_idx = l1_order[-k:]
                elif m == "SORG":
                    sel_idx = select_sorg_pixel(
                        X_train, k, {"alpha": CFG.sorg_alpha, "p": CFG.sorg_p_norm}
                    )
                else:
                    raise ValueError(f"Unknown method: {m}")

                sel_time = time.time() - t0

                # Train multinomial logistic regression on selected pixels
                clf = LogisticRegression(
                    solver="lbfgs",
                    max_iter=200,
                    multi_class="multinomial",
                    n_jobs=1,
                )
                clf.fit(X_train[:, sel_idx], y_train)
                acc = clf.score(X_test[:, sel_idx], y_test)

                results.append(
                    {
                        "k": k,
                        "Method": m,
                        "Seed": seed,
                        "Accuracy": acc,
                        "SelTime": sel_time,
                    }
                )

                # Store masks from the first seed for visualization
                if seed == run_seeds[0] and k == vis_k:
                    masks[m] = np.array(sel_idx, dtype=int)

                # Only print detailed logs for the first seed
                if seed == run_seeds[0]:
                    print(
                        f"  k={k:<3} | {m:10s} | Acc={acc:.4f} "
                        f"| SelTime={sel_time:.4f}s"
                    )

    # --------------------------------------------------------
    # 4. Save & Summarize
    # --------------------------------------------------------
    df = pd.DataFrame(results)
    df_path = os.path.join(exp_dir, "results.csv")
    df.to_csv(df_path, index=False)
    print(f"\n[Result] Saved raw results to {df_path}")

    summary = (
        df.groupby(["k", "Method"])
        .agg(
            mean_acc=("Accuracy", "mean"),
            std_acc=("Accuracy", "std"),
            mean_sel_time=("SelTime", "mean"),
        )
        .reset_index()
    )
    summary["std_acc"] = summary["std_acc"].fillna(0.0)

    print("\n=== Final Summary (Accuracy) ===")
    print(summary[["k", "Method", "mean_acc", "std_acc"]])

    summary_path = os.path.join(exp_dir, "summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"[Result] Saved summary to {summary_path}")

    # Save full-feature baseline
    baseline_path = os.path.join(exp_dir, "full_baseline.json")
    with open(baseline_path, "w") as f:
        json.dump(
            {
                "full_pixels_acc": float(full_acc),
                "n_train": int(X_train.shape[0]),
                "n_test": int(X_test.shape[0]),
            },
            f,
            indent=2,
        )
    print(f"[Result] Saved full-feature baseline to {baseline_path}")

    # --------------------------------------------------------
    # 5. Plots
    # --------------------------------------------------------
    sns.set(style="whitegrid")

    # 5.1 Accuracy vs k
    plt.figure(figsize=(8, 6))
    sns.lineplot(
        data=df,
        x="k",
        y="Accuracy",
        hue="Method",
        style="Method",
        markers=True,
        dashes=False,
        linewidth=2.5,
        errorbar="sd",
    )
    plt.axhline(full_acc, color="black", linestyle="--", label="Full (784 pixels)")
    plt.title(f"MNIST Pixel Selection Accuracy (N={CFG.n_samples_train})")
    plt.ylabel("Test Accuracy")
    plt.xlabel("Number of Pixels Selected (k)")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    acc_fig_path = os.path.join(exp_dir, "E10_Accuracy.png")
    plt.savefig(acc_fig_path, dpi=200)
    plt.close()
    print(f"[Plot] Saved accuracy plot to {acc_fig_path}")

    # 5.2 Selection Time vs k (log-scale)
    plt.figure(figsize=(8, 6))
    sns.lineplot(
        data=df,
        x="k",
        y="SelTime",
        hue="Method",
        style="Method",
        markers=True,
        dashes=False,
        linewidth=2.0,
        errorbar="sd",
    )
    plt.yscale("log")
    plt.title("Selection Time vs Number of Pixels (log-scale)")
    plt.ylabel("Selection Time (s, log scale)")
    plt.xlabel("k (Number of selected pixels)")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    time_fig_path = os.path.join(exp_dir, "E10_SelectionTime.png")
    plt.savefig(time_fig_path, dpi=200)
    plt.close()
    print(f"[Plot] Saved selection time plot to {time_fig_path}")

    # 5.3 Pixel Mask Visualization (k=vis_k)
    masks_dir = os.path.join(exp_dir, "masks")
    ensure_dir(masks_dir)

    def idx_to_img(idx: np.ndarray) -> np.ndarray:
        img = np.zeros(784, dtype=np.float32)
        img[idx] = 1.0
        return img.reshape(28, 28)

    methods_order = ["Random", "Variance", "L1_LogReg", "SORG"]

    plt.figure(figsize=(12, 4))
    for i, m in enumerate(methods_order):
        if m not in masks:
            continue
        sel_idx = np.array(masks[m], dtype=int)
        np.save(os.path.join(masks_dir, f"mask_{m}_k{vis_k}.npy"), sel_idx)

        plt.subplot(1, len(methods_order), i + 1)
        plt.imshow(idx_to_img(sel_idx), cmap="gray_r", vmin=0, vmax=1)
        plt.title(f"{m}\n(k={vis_k})")
        plt.axis("off")

    plt.suptitle(f"Selected Pixel Maps (k={vis_k})", fontsize=14)
    plt.tight_layout()
    mask_fig_path = os.path.join(exp_dir, "E10_PixelMasks.png")
    plt.savefig(mask_fig_path, dpi=200)
    plt.close()
    print(f"[Plot] Saved pixel mask visualization to {mask_fig_path}")

    print(f"\nDONE. All results saved to {exp_dir}")


if __name__ == "__main__":
    main()
