# E0_LethalMode_SORG.py
# -*- coding: utf-8 -*-
#
# ==============================================================================
# Title: E0_LethalMode_SORG.py - SoftNorm Gate (SORG) vs Greedy Baselines
# Author: Jinwoo Baek
# Date: 2025-12-30
#
# Description:
#   Synthetic "Lethal Mode" experiment to verify:
#
#     1) SORG with SoftNorm Gate (p_norm > 0) can
#        statistically suppress massive-norm outliers and
#        recover the union-of-subspaces structure.
#
#     2) SORG without the gate (p_norm = 0, "sorg_nogate")
#        collapses to greedy behavior (similar to k-center/herding)
#        and catastrophically over-selects outliers.
#
#   Compared methods:
#     - random       : uniform baseline
#     - kcenter      : greedy farthest-first (no gate)
#     - kmeans       : centroid-based coreset (no gate)
#     - herding      : mean-matching greedy (no gate)
#     - sorg         : SORG with SoftNorm Gate (canonical)
#     - sorg_nogate  : SORG with gate disabled (p_norm = 0.0)
#
#   Outputs:
#     - results_budget.csv      : metrics over k (fixed outlier scale)
#     - results_scale.csv       : metrics over outlier scales (fixed k)
#     - curve_outlier_frac.png  : Outlier fraction vs k
#     - curve_coverage.png      : Coverage vs k
#     - curve_phase_transition.png : Outlier fraction vs scale
#     - rep_counts.png          : Group/outlier counts for a representative run
#     - viz_norm_hist.png       : Norm histograms (full vs selections)
#     - story.txt               : ICML-style textual snapshot
# ==============================================================================

import os
import json
import time
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Suppress minor warnings for clean output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------
# Dependencies: SORG + Baselines
# ---------------------------------------------------------------------
try:
    from SORG import SORG
except Exception as e:
    raise ImportError(
        "\n[CRITICAL ERROR] Failed to import SORG.\n"
        "Please ensure 'SORG.py' is in the same directory as E0_LethalMode_SORG.py.\n"
        f"Error details: {e}\n"
    )

try:
    from Baselines import (
        random_selection,
        kcenter_selection,
        kmeans_coreset,
        herding_selection,
    )
except Exception as e:
    raise ImportError(
        "\n[CRITICAL ERROR] Failed to import Baselines.\n"
        "Please ensure 'Baselines.py' is in the same directory as E0_LethalMode_SORG.py.\n"
        f"Error details: {e}\n"
    )


# ============================================================
# Configuration: Lethal Union-of-Subspaces
# ============================================================
@dataclass
class E0Config:
    # 1. Data Generation (Union of Subspaces + Massive Outliers)
    d: int = 256              # ambient dimension
    n_groups: int = 5         # #subspaces
    subspace_dim: int = 5     # intrinsic dimension per subspace

    n_per_group: int = 40     # inliers per group -> total inliers = 200
    n_outliers: int = 100     # outliers = 100 (≈33% contamination)

    # Outlier norm scale:
    #   target_norm ≈ outlier_scale * sqrt(d)
    outlier_scale: float = 100.0

    noise_std: float = 0.05
    group_scale_spread: float = 0.0  # >0 ⇒ different variance per group

    # 2. Part 1: Budget Sweep (fixed scale)
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    k_list: Tuple[int, ...] = (10, 20, 40, 60, 80)

    # 3. Part 2: Scale Sweep (Phase Transition)
    do_scale_sweep: bool = True
    sweep_scales: Tuple[float, ...] = (5.0, 10.0, 30.0, 60.0, 100.0, 200.0)
    sweep_k: int = 40  # fixed budget for scale sweep

    # 4. Methods & SORG hyperparams
    methods: Tuple[str, ...] = (
        "random",
        "kcenter",
        "kmeans",
        "herding",
        "sorg",
        "sorg_nogate",
    )

    sorg_alpha: float = 1.0
    sorg_p_norm: float = 2.0   # gate ON if > 0

    # 5. Output
    representative_seed: int = 0
    representative_k: int = 40

    out_prefix: str = "E0_Lethal_outputs"


CFG = E0Config()


# ============================================================
# Helper Functions
# ============================================================
def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_rng(seed: int) -> np.random.RandomState:
    return np.random.RandomState(int(seed))


def l2_norms_cols(M: np.ndarray) -> np.ndarray:
    """
    Column-wise L2 norms.
    M: (d, N) → (N,)
    """
    return np.sqrt(np.sum(M * M, axis=0) + 1e-12).astype(np.float32)


def compute_cov(M: np.ndarray) -> np.ndarray:
    """
    Empirical second moment (covariance proxy, assuming mean≈0).
    M: (d, N)
    """
    d, N = M.shape
    if N <= 0:
        return np.zeros((d, d), dtype=np.float32)
    return (M @ M.T) / float(N)


def rel_fro_err(A: np.ndarray, B: np.ndarray, eps: float = 1e-12) -> float:
    """
    Relative Frobenius error: ||A-B||_F / (||A||_F + eps)
    """
    diff = A - B
    num = float(np.linalg.norm(diff, ord="fro"))
    den = float(np.linalg.norm(A, ord="fro")) + eps
    return num / den


def save_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            line = [str(r[k]) for k in keys]
            f.write(",".join(line) + "\n")


# ============================================================
# Data Generation: Union-of-Subspaces + Lethal Outliers
# ============================================================
def generate_union_of_subspaces(
    rng: np.random.RandomState,
    d: int,
    n_groups: int,
    subspace_dim: int,
    n_per_group: int,
    n_outliers: int,
    outlier_scale: float,
    noise_std: float,
    group_scale_spread: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    M: (d, N)  with N = n_groups * n_per_group + n_outliers
    y: (N,)    labels in {0..n_groups-1} for inliers, -1 for outliers
    """
    blocks: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    # Group-wise scale variation (optional)
    if group_scale_spread > 0:
        base = np.linspace(-1.0, 1.0, n_groups, dtype=np.float32)
        group_scales = np.exp(group_scale_spread * base).astype(np.float32)
    else:
        group_scales = np.ones(n_groups, dtype=np.float32)

    # Inliers: union of subspaces
    for g in range(n_groups):
        A = rng.normal(size=(d, subspace_dim)).astype(np.float32)
        U, _ = np.linalg.qr(A)
        U = U.astype(np.float32)

        coeff = rng.normal(size=(subspace_dim, n_per_group)).astype(np.float32)
        X = U @ coeff

        if noise_std > 0:
            X += (noise_std * rng.normal(size=X.shape)).astype(np.float32)

        X *= float(group_scales[g])

        blocks.append(X)
        labels.append(np.full(n_per_group, g, dtype=np.int64))

    # Outliers: isotropic random directions with huge norm
    if n_outliers > 0:
        V = rng.normal(size=(d, n_outliers)).astype(np.float32)
        Vn = np.sqrt(np.sum(V * V, axis=0) + 1e-12).astype(np.float32)
        V = V / Vn.reshape(1, -1)

        target_norm = float(outlier_scale) * np.sqrt(float(d))
        V = V * target_norm

        blocks.append(V)
        labels.append(np.full(n_outliers, -1, dtype=np.int64))

    M = np.concatenate(blocks, axis=1).astype(np.float32)
    y = np.concatenate(labels, axis=0).astype(np.int64)

    perm = rng.permutation(M.shape[1])
    M = M[:, perm]
    y = y[perm]

    return M, y


# ============================================================
# SORG wrapper (with sanity checks)
# ============================================================
def select_sorg(
    M: np.ndarray,
    k: int,
    alpha: float,
    p_norm: float,
    method_name: str,
) -> np.ndarray:
    """
    Runs SORG on M (d, N) with given p_norm.
    p_norm > 0  → gate ON (canonical SORG)
    p_norm <= 0 → gate OFF (ablation: sorg_nogate)
    """
    d, N = M.shape
    k = int(min(max(1, k), N))

    model = SORG(
        k=int(k),
        alpha=float(alpha),
        p_norm=float(p_norm),
    )
    model.fit(M)

    sel = model.get_support().astype(int)

    # Sanity checks
    if sel.size > k:
        raise ValueError(f"[{method_name}] SORG selected {sel.size} > {k}!")

    if sel.size > 0:
        if sel.min() < 0:
            raise ValueError(f"[{method_name}] Negative index: {sel.min()}")
        if sel.max() >= N:
            raise ValueError(f"[{method_name}] Index out of bounds: {sel.max()} >= {N}")
        if len(np.unique(sel)) != len(sel):
            raise ValueError(f"[{method_name}] Duplicate indices selected!")

    return sel


# ============================================================
# Metrics
# ============================================================
def calculate_metrics(
    M: np.ndarray,
    labels: np.ndarray,
    sel: np.ndarray,
    n_groups: int,
) -> Dict[str, float]:
    """
    Returns:
      - outlier_frac   : fraction of selected points with label -1
      - coverage_ratio : #covered groups / n_groups
      - cov_relerr     : covariance reconstruction error (inliers only)
      - mean_norm      : mean norm of selected points
      - max_norm       : max norm of selected points
    """
    d, N = M.shape
    k = int(sel.size)

    if k <= 0:
        return {
            "outlier_frac": 0.0,
            "coverage_ratio": 0.0,
            "cov_relerr": 1.0,
            "mean_norm": 0.0,
            "max_norm": 0.0,
        }

    sel_labels = labels[sel]
    n_outliers_sel = np.sum(sel_labels == -1)
    outlier_frac = float(n_outliers_sel) / float(k)

    unique_groups = np.unique(sel_labels)
    n_covered = np.sum(unique_groups != -1)
    coverage_ratio = float(n_covered) / float(n_groups)

    # Covariance reconstruction (inliers only)
    inlier_all = np.where(labels != -1)[0]
    inlier_sel = sel[sel_labels != -1]

    if inlier_all.size > 0:
        Cov_full = compute_cov(M[:, inlier_all])
        if inlier_sel.size > 1:
            Cov_sel = compute_cov(M[:, inlier_sel])
            cov_err = rel_fro_err(Cov_full, Cov_sel)
        else:
            cov_err = 1.0
    else:
        cov_err = 0.0

    norms = l2_norms_cols(M[:, sel])
    mean_norm = float(np.mean(norms))
    max_norm = float(np.max(norms))

    return {
        "outlier_frac": outlier_frac,
        "coverage_ratio": coverage_ratio,
        "cov_relerr": cov_err,
        "mean_norm": mean_norm,
        "max_norm": max_norm,
    }


def get_group_counts(
    labels: np.ndarray,
    sel: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    """
    Returns counts per (group + outlier):
      index 0..n_groups-1 : group counts
      index n_groups       : outlier count
    """
    cnt = np.zeros(n_groups + 1, dtype=int)
    if sel.size == 0:
        return cnt

    s_labels = labels[sel]
    for g in range(n_groups):
        cnt[g] = np.sum(s_labels == g)
    cnt[-1] = np.sum(s_labels == -1)
    return cnt


# ============================================================
# Plotting helpers
# ============================================================
def aggregate_metrics(
    rows: List[dict],
    methods: List[str],
    x_key: str,
    x_list: List[float],
    y_key: str,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Aggregates metric 'y_key' over seeds, indexed by 'x_key'.
    """
    res: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for m in methods:
        means, stds = [], []
        for x in x_list:
            vals = [
                r[y_key] for r in rows
                if (r["method"] == m) and (abs(float(r[x_key]) - float(x)) < 1e-9)
            ]
            if len(vals) > 0:
                arr = np.asarray(vals, dtype=np.float32)
                means.append(float(np.mean(arr)))
                stds.append(float(np.std(arr)))
            else:
                means.append(np.nan)
                stds.append(np.nan)
        res[m] = (np.asarray(means, dtype=np.float32),
                  np.asarray(stds, dtype=np.float32))
    return res


def plot_curves(
    out_path: str,
    x_vals: List[float],
    agg_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    x_label: str,
    y_label: str,
    title: str,
) -> None:
    plt.figure(figsize=(7.0, 5.0))
    xs = np.asarray(x_vals, dtype=np.float32)

    for m, (mu, sd) in agg_data.items():
        mu = np.asarray(mu, dtype=np.float32)
        sd = np.asarray(sd, dtype=np.float32)
        mask = np.isfinite(mu)
        if not np.any(mask):
            continue
        x = xs[mask]
        y = mu[mask]
        e = sd[mask]
        plt.plot(x, y, marker="o", label=m)
        plt.fill_between(x, y - e, y + e, alpha=0.2)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_histogram(
    out_path: str,
    M: np.ndarray,
    selections: Dict[str, np.ndarray],
    title: str,
) -> None:
    plt.figure(figsize=(8.0, 5.0))
    all_norms = l2_norms_cols(M)

    # Full data
    if all_norms.size > 0:
        try:
            plt.hist(
                np.log10(all_norms + 1e-9),
                bins=50,
                color="gray",
                alpha=0.3,
                label="Full Data",
                density=True,
            )
        except Exception:
            pass

    # Selected sets
    for name, sel in selections.items():
        if sel.size == 0:
            continue
        sel_norms = l2_norms_cols(M[:, sel])
        try:
            plt.hist(
                np.log10(sel_norms + 1e-9),
                bins=50,
                alpha=0.7,
                histtype="step",
                linewidth=2,
                label=name,
                density=True,
            )
        except Exception:
            pass

    plt.xlabel("log10(||x||)")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_group_bar(
    out_path: str,
    counts_dict: Dict[str, np.ndarray],
    n_groups: int,
    title: str,
) -> None:
    labels = [f"G{i}" for i in range(n_groups)] + ["Out"]
    x = np.arange(len(labels))
    width = 0.8 / max(1, len(counts_dict))

    plt.figure(figsize=(10.0, 5.0))
    for i, (name, cnt) in enumerate(counts_dict.items()):
        offset = (i - len(counts_dict) / 2.0) * width + width / 2.0
        plt.bar(x + offset, cnt, width=width, label=name)

    plt.xticks(x, labels)
    plt.ylabel("Selected Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ============================================================
# Main
# ============================================================
def main():
    # ------------------------------------------------------------------
    # Output dir
    # ------------------------------------------------------------------
    out_dir = os.path.join(os.getcwd(), f"{CFG.out_prefix}_{now_stamp()}")
    ensure_dir(out_dir)
    save_json(os.path.join(out_dir, "config.json"), asdict(CFG))

    total_inliers = CFG.n_groups * CFG.n_per_group
    total_points = total_inliers + CFG.n_outliers

    print("============================================================")
    print("E0: LETHAL MODE + PHASE TRANSITION (SORG + Ablations)")
    print("------------------------------------------------------------")
    print(f"d = {CFG.d}, N = {total_points} (Inliers={total_inliers}, Outliers={CFG.n_outliers})")
    print(f"Structure    : {CFG.n_groups} subspaces (dim={CFG.subspace_dim}) + lethal outliers")
    print(f"Lethal Scale : {CFG.outlier_scale}x (vs sqrt(d))")
    print(f"Methods      : {CFG.methods}")
    print(f"Output dir   : {out_dir}")
    print("============================================================")

    # ------------------------------------------------------------------
    # Method spec: baselines + SORG variants
    # ------------------------------------------------------------------
    methods_def = {
        "random": {
            "type": "baseline",
            "baseline": "random",
        },
        "kcenter": {
            "type": "baseline",
            "baseline": "kcenter",
        },
        "kmeans": {
            "type": "baseline",
            "baseline": "kmeans",
        },
        "herding": {
            "type": "baseline",
            "baseline": "herding",
        },
        # SORG with gate (canonical)
        "sorg": {
            "type": "sorg",
            "p_norm": CFG.sorg_p_norm,
        },
        # SORG without gate (ablation)
        "sorg_nogate": {
            "type": "sorg",
            "p_norm": 0.0,
        },
    }
    method_names = list(methods_def.keys())

    # =======================================================
    # Part 1: Budget Sweep (fixed outlier_scale)
    # =======================================================
    print("\n[Part 1] Budget Sweep at fixed lethal scale "
          f"(outlier_scale={CFG.outlier_scale})")

    rows_budget: List[dict] = []
    rep_M, rep_y = None, None
    rep_selections: Dict[str, np.ndarray] = {}
    rep_k = int(CFG.representative_k)

    for seed in CFG.seeds:
        print(f"\n[Seed {seed}] Generating lethal data...")
        rng = set_rng(seed)
        M, y = generate_union_of_subspaces(
            rng=rng,
            d=CFG.d,
            n_groups=CFG.n_groups,
            subspace_dim=CFG.subspace_dim,
            n_per_group=CFG.n_per_group,
            n_outliers=CFG.n_outliers,
            outlier_scale=CFG.outlier_scale,
            noise_std=CFG.noise_std,
            group_scale_spread=CFG.group_scale_spread,
        )

        if int(seed) == int(CFG.representative_seed):
            rep_M, rep_y = M, y

        d, N = M.shape
        X = M.T  # (N, d) for Baselines

        for k in CFG.k_list:
            k = int(min(max(1, k), N))

            for name, spec in methods_def.items():
                t0 = time.perf_counter()

                if spec["type"] == "baseline":
                    # Baselines operate on X: (N, d)
                    if spec["baseline"] == "random":
                        sel = random_selection(
                            X,
                            k,
                            random_state=seed * 1000 + k,
                        )

                    elif spec["baseline"] == "kcenter":
                        sel = kcenter_selection(
                            X,
                            k,
                            metric="euclidean",
                            random_state=seed * 1000 + k,
                        )

                    elif spec["baseline"] == "kmeans":
                        sel = kmeans_coreset(
                            X,
                            k,
                            random_state=seed * 1000 + k,
                        )

                    elif spec["baseline"] == "herding":
                        sel = herding_selection(X, k)

                    else:
                        raise ValueError(f"Unknown baseline: {spec['baseline']}")

                elif spec["type"] == "sorg":
                    sel = select_sorg(
                        M,
                        k,
                        alpha=CFG.sorg_alpha,
                        p_norm=spec["p_norm"],
                        method_name=name,
                    )
                else:
                    raise ValueError(f"Unknown method type: {spec['type']}")

                sel = np.asarray(sel, dtype=int)
                t1 = time.perf_counter()

                metrics = calculate_metrics(M, y, sel, CFG.n_groups)
                metrics.update({
                    "seed": int(seed),
                    "k": int(k),
                    "method": name,
                    "time": float(t1 - t0),
                })
                rows_budget.append(metrics)

                print(
                    f"  > [{name:12s}] k={k:3d} | "
                    f"Outlier%={metrics['outlier_frac']:.1%} | "
                    f"Coverage={metrics['coverage_ratio']:.2f} | "
                    f"Time={metrics['time']:.4f}s"
                )

                if int(seed) == int(CFG.representative_seed) and int(k) == rep_k:
                    rep_selections[name] = sel

    # Save Part 1 results
    print("\n[Part 1] Saving results and plots...")
    save_csv(os.path.join(out_dir, "results_budget.csv"), rows_budget)

    k_list_f = [float(k) for k in CFG.k_list]
    agg_out = aggregate_metrics(rows_budget, method_names, "k", k_list_f, "outlier_frac")
    agg_cov = aggregate_metrics(rows_budget, method_names, "k", k_list_f, "coverage_ratio")

    plot_curves(
        os.path.join(out_dir, "curve_outlier_frac.png"),
        k_list_f,
        agg_out,
        "k (budget)",
        "Selected Outlier Fraction",
        "E0: Outlier Selection vs Budget (Lethal Scale)",
    )
    plot_curves(
        os.path.join(out_dir, "curve_coverage.png"),
        k_list_f,
        agg_cov,
        "k (budget)",
        "Coverage Ratio",
        "E0: Group Coverage vs Budget (Lethal Scale)",
    )

    if rep_M is not None:
        rep_counts = {
            name: get_group_counts(rep_y, sel, CFG.n_groups)
            for name, sel in rep_selections.items()
        }
        plot_group_bar(
            os.path.join(out_dir, "rep_counts.png"),
            rep_counts,
            CFG.n_groups,
            f"Selection Counts (k={rep_k}, scale={CFG.outlier_scale})",
        )
        plot_histogram(
            os.path.join(out_dir, "viz_norm_hist.png"),
            rep_M,
            rep_selections,
            f"Norm Distribution (k={rep_k}, scale={CFG.outlier_scale})",
        )

    # =======================================================
    # Part 2: Scale Sweep (Phase Transition)
    # =======================================================
    rows_scale: List[dict] = []
    if CFG.do_scale_sweep:
        print("\n============================================================")
        print("[Part 2] Scale Sweep (Phase Transition of Gate vs Greedy)")
        print("============================================================")

        k_sweep = int(CFG.sweep_k)
        subset_methods = ["sorg", "sorg_nogate", "kcenter", "herding"]

        for scale in CFG.sweep_scales:
            print(f"\n>>> Outlier Scale = {scale:.1f}, k={k_sweep}")
            for seed in CFG.seeds:
                rng = set_rng(seed + 1000)
                M, y = generate_union_of_subspaces(
                    rng=rng,
                    d=CFG.d,
                    n_groups=CFG.n_groups,
                    subspace_dim=CFG.subspace_dim,
                    n_per_group=CFG.n_per_group,
                    n_outliers=CFG.n_outliers,
                    outlier_scale=float(scale),
                    noise_std=CFG.noise_std,
                    group_scale_spread=CFG.group_scale_spread,
                )

                d, N = M.shape
                k_eff = int(min(max(1, k_sweep), N))
                X = M.T

                for name in subset_methods:
                    spec = methods_def[name]

                    if spec["type"] == "baseline":
                        if spec["baseline"] == "kcenter":
                            sel = kcenter_selection(
                                X,
                                k_eff,
                                metric="euclidean",
                                random_state=seed * 1000 + int(scale * 10),
                            )
                        elif spec["baseline"] == "herding":
                            sel = herding_selection(X, k_eff)
                        else:
                            # For scale sweep we only use kcenter/herding as baselines
                            continue

                    elif spec["type"] == "sorg":
                        sel = select_sorg(
                            M,
                            k_eff,
                            alpha=CFG.sorg_alpha,
                            p_norm=spec["p_norm"],
                            method_name=name,
                        )
                    else:
                        raise ValueError(f"Unknown method type: {spec['type']}")

                    sel = np.asarray(sel, dtype=int)
                    metrics = calculate_metrics(M, y, sel, CFG.n_groups)
                    metrics.update({
                        "seed": int(seed),
                        "scale": float(scale),
                        "k": int(k_eff),
                        "method": name,
                    })
                    rows_scale.append(metrics)

                    # Only log seed=0 for brevity
                    if seed == CFG.seeds[0]:
                        print(
                            f"  [{name:12s}] Outlier%={metrics['outlier_frac']:.1%} "
                            f"(mean_norm={metrics['mean_norm']:.2f})"
                        )

        save_csv(os.path.join(out_dir, "results_scale.csv"), rows_scale)

        agg_phase = aggregate_metrics(
            rows_scale,
            ["sorg", "sorg_nogate", "kcenter", "herding"],
            "scale",
            [float(s) for s in CFG.sweep_scales],
            "outlier_frac",
        )
        plot_curves(
            os.path.join(out_dir, "curve_phase_transition.png"),
            [float(s) for s in CFG.sweep_scales],
            agg_phase,
            "Outlier Scale",
            "Selected Outlier Fraction",
            f"E0: Phase Transition (k={k_sweep})",
        )

    # =======================================================
    # Story snapshot (text)
    # =======================================================
    story: List[str] = []
    story.append("E0 LETHAL MODE - ICML Story Snapshot (SORG vs Greedy)")
    story.append("-" * 60)
    story.append("Sanity Check: Indices valid, no duplicates, shapes consistent.")
    story.append(f"Lethal Regime: outlier_scale={CFG.outlier_scale}x, "
                 f"d={CFG.d}, inliers={total_inliers}, outliers={CFG.n_outliers}")
    story.append("Compared methods: " + ", ".join(method_names))
    story.append("-" * 60)

    def get_mean(
        rows: List[dict],
        method: str,
        k_val: Optional[int] = None,
        scale_val: Optional[float] = None,
        key: str = "outlier_frac",
    ) -> float:
        vals = []
        for r in rows:
            if r["method"] != method:
                continue
            if k_val is not None and int(r.get("k", -1)) != int(k_val):
                continue
            if scale_val is not None:
                if abs(float(r.get("scale", -1.0)) - float(scale_val)) > 1e-9:
                    continue
            vals.append(r[key])
        return float(np.mean(vals)) if vals else 0.0

    # Metric 1: Outlier fraction at a headline k
    headline_k = CFG.k_list[2]  # typically 40
    story.append(f"Metric 1: Outlier fraction at k={headline_k} (lethal scale {CFG.outlier_scale}):")
    for m in method_names:
        val = get_mean(rows_budget, m, k_val=headline_k, key="outlier_frac")
        story.append(f"  * {m:12s}: {val:.1%} outliers")

    # Metric 2: Scale phase transition for gate vs no-gate vs greedy
    if CFG.do_scale_sweep and rows_scale:
        story.append("")
        story.append("Metric 2: Phase transition (Outlier scale 10 → 100):")
        for m in ["sorg", "sorg_nogate", "kcenter", "herding"]:
            v_low = get_mean(rows_scale, m, scale_val=10.0, key="outlier_frac")
            v_high = get_mean(rows_scale, m, scale_val=100.0, key="outlier_frac")
            story.append(
                f"  * {m:12s}: {v_low:.1%} → {v_high:.1%} (selected outliers)"
            )

    with open(os.path.join(out_dir, "story.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(story) + "\n")

    print("\n" + "\n".join(story))
    print("\nDONE. Results saved.")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
