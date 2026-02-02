# E1_VisionCoreset_SORG.py (Publish-ready: CIFAR-100 Vision Coreset | SORG + Baselines)
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
from typing import Dict, List, Tuple

import numpy as np

# ------------------------------
# [Rule #1] Force caches to /nfs/hpc/share/baekji
# ------------------------------
BASE_NFS = "/nfs/hpc/share/baekji"
TORCH_HOME = os.path.join(BASE_NFS, "torch_home")
TORCH_DATASETS = os.path.join(BASE_NFS, "torch_datasets")
FEATURE_CACHE = os.path.join(BASE_NFS, "E1_feature_cache")

os.makedirs(TORCH_HOME, exist_ok=True)
os.makedirs(TORCH_DATASETS, exist_ok=True)
os.makedirs(FEATURE_CACHE, exist_ok=True)

os.environ["TORCH_HOME"] = TORCH_HOME
os.environ["HF_HOME"] = os.path.join(BASE_NFS, "hf_cache")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(BASE_NFS, "hf_cache", "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(BASE_NFS, "hf_cache", "datasets")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.models as models

# Hard cap torch CPU threads too (prevents weird slowdowns on shared nodes)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))

# [Standard] Robust sklearn solver
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Suppress noisy warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ============================================================
# External algorithms: SORG + Baselines
# ============================================================
try:
    from SORG import SORG
except Exception as e:
    raise ImportError(
        "Failed to import SORG. Make sure SORG.py is in the same directory.\n"
        f"Import error: {e}"
    )

try:
    from Baselines import (
        random_selection,
        herding_selection,
        kcenter_selection,
        kmeans_coreset,
    )
except Exception as e:
    raise ImportError(
        "Failed to import Baselines. Make sure Baselines.py is in the same directory.\n"
        f"Import error: {e}"
    )


# ============================================================
# Config
# ============================================================
@dataclass
class E1Config:
    dataset: str = "cifar100"
    encoder: str = "resnet18"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True

    batch_size: int = 256
    num_workers_cap: int = 6

    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)

    # Budgets (fraction of train set)
    budgets: Tuple[float, ...] = (0.005, 0.01, 0.02, 0.05, 0.10)

    # Full Rank (No projection) by default
    proj_dim: int = 0
    proj_seed: int = 0

    # Common ICML-SA baselines + SORG
    methods: Tuple[str, ...] = ("random", "kcenter", "kmeans", "herding", "sorg")

    # ---- k-center runtime guard ----
    kcenter_pool: int = 10000       # run greedy on a random pool of this size
    kcenter_greedy_cap: int = 2000  # greedy farthest-first only up to this many
    kcenter_fill_random: bool = True  # fill remaining with random from pool

    # ---- k-means runtime guard ----
    kmeans_max_k: int = 1000           # skip kmeans if k > this
    kmeans_max_iter: int = 100
    kmeans_batch_size: int = 4096

    # SORG params (clean features -> no scaling inside SORG)
    sorg_alpha: float = 1.0
    sorg_p_norm: float = 2.0

    # Probe params (keep stable + not too slow)
    probe_max_iter: int = 800          # 1000 -> 800 is usually enough
    probe_n_jobs: int = 1              # avoid CPU thrash on shared nodes

    out_prefix: str = "E1_outputs"


CFG = E1Config()


# ============================================================
# Helpers
# ============================================================
def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_num_workers(cap: int) -> int:
    try:
        n = os.cpu_count() or 4
        return int(max(0, min(cap, n)))
    except Exception:
        return int(cap)


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
            f.write(",".join(str(r[k]) for k in keys) + "\n")


def l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.sqrt(np.sum(X * X, axis=1, keepdims=True) + eps).astype(np.float32)
    return (X / n).astype(np.float32)


def make_deterministic_projection(d_in: int, d_out: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    W = rng.normal(size=(d_in, d_out)).astype(np.float32)
    W /= np.sqrt(float(d_out))
    return W


def project_features(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    return (X @ W).astype(np.float32)


def budget_to_id(b: float) -> int:
    # stable key for joining/aggregation
    if b >= 1.0:
        return 1_000_000
    return int(round(float(b) * 1e6))


# ============================================================
# Feature extraction (cached in NFS)
# ============================================================
def build_encoder(name: str):
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        preprocess = weights.transforms()
        feat_dim = 512
    elif name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        preprocess = weights.transforms()
        feat_dim = 2048
    else:
        raise ValueError(f"Unknown encoder: {name}")

    backbone = nn.Sequential(*list(model.children())[:-1])
    backbone.eval()
    return backbone, preprocess, feat_dim


@torch.no_grad()
def extract_features(backbone: nn.Module, loader: DataLoader, device: str, amp: bool) -> Tuple[np.ndarray, np.ndarray]:
    feats, labels = [], []
    backbone = backbone.to(device)

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)

        if amp and device.startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                fb = backbone(xb).flatten(1)
        else:
            fb = backbone(xb).flatten(1)

        feats.append(fb.detach().cpu().float().numpy())
        labels.append(yb.detach().cpu().long().numpy())

    X = np.concatenate(feats, axis=0).astype(np.float32)
    y = np.concatenate(labels, axis=0).astype(np.int64)
    return X, y


def feat_cache_paths(cfg: E1Config) -> Dict[str, str]:
    tag = f"{cfg.dataset}_{cfg.encoder}"
    return {
        "train_X": os.path.join(FEATURE_CACHE, f"{tag}_train_X.npy"),
        "train_y": os.path.join(FEATURE_CACHE, f"{tag}_train_y.npy"),
        "test_X": os.path.join(FEATURE_CACHE, f"{tag}_test_X.npy"),
        "test_y": os.path.join(FEATURE_CACHE, f"{tag}_test_y.npy"),
        "meta": os.path.join(FEATURE_CACHE, f"{tag}_meta.json"),
    }


def load_or_build_features(cfg: E1Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    paths = feat_cache_paths(cfg)
    if all(os.path.exists(paths[k]) for k in ("train_X", "train_y", "test_X", "test_y", "meta")):
        train_X = np.load(paths["train_X"])
        train_y = np.load(paths["train_y"])
        test_X = np.load(paths["test_X"])
        test_y = np.load(paths["test_y"])
        with open(paths["meta"], "r", encoding="utf-8") as f:
            meta = json.load(f)
        return train_X, train_y, test_X, test_y, int(meta["feat_dim"])

    backbone, preprocess, feat_dim = build_encoder(cfg.encoder)

    train_set = torchvision.datasets.CIFAR100(root=TORCH_DATASETS, train=True, download=True, transform=preprocess)
    test_set = torchvision.datasets.CIFAR100(root=TORCH_DATASETS, train=False, download=True, transform=preprocess)

    nw = compute_num_workers(cfg.num_workers_cap)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=False, num_workers=nw, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=nw, pin_memory=True)

    train_X, train_y = extract_features(backbone, train_loader, cfg.device, cfg.amp)
    test_X, test_y = extract_features(backbone, test_loader, cfg.device, cfg.amp)

    np.save(paths["train_X"], train_X)
    np.save(paths["train_y"], train_y)
    np.save(paths["test_X"], test_X)
    np.save(paths["test_y"], test_y)
    save_json(paths["meta"], {"feat_dim": feat_dim, "dataset": cfg.dataset, "encoder": cfg.encoder})

    return train_X, train_y, test_X, test_y, feat_dim


# ============================================================
# Robust Probe (sklearn Logistic Regression)
# ============================================================
def train_linear_probe_sklearn(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    seed: int,
    max_iter: int,
    n_jobs: int,
) -> Tuple[float, float]:
    t0 = time.perf_counter()

    clf = LogisticRegression(
        solver="lbfgs",
        multi_class="multinomial",
        max_iter=int(max_iter),
        random_state=int(seed),
        C=1.0,
        n_jobs=int(n_jobs),
    )
    clf.fit(train_X, train_y)
    acc = accuracy_score(test_y, clf.predict(test_X))

    t1 = time.perf_counter()
    return float(acc), float(t1 - t0)


# ============================================================
# Plotting / Aggregation
# ============================================================
def plot_curve(out_path: str, xs: List[float], curves: Dict[str, Tuple[np.ndarray, np.ndarray]], ylab: str, title: str) -> None:
    plt.figure(figsize=(7.2, 4.8))
    xs_arr = np.asarray(xs, dtype=np.float32)

    for name, (mean, std) in curves.items():
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        mask = np.isfinite(mean) & np.isfinite(std)
        if not np.any(mask):
            continue
        x = xs_arr[mask]
        m = mean[mask]
        s = std[mask]
        plt.plot(x, m, marker="o", linewidth=2, label=name)
        plt.fill_between(x, m - s, m + s, alpha=0.2)

    plt.xlabel("Budget (fraction of train set)")
    plt.ylabel(ylab)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def aggregate(rows: List[dict], methods: List[str], budgets: List[float], key: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    curves = {}
    budget_ids = [budget_to_id(b) for b in budgets]
    for m in methods:
        means, stds = [], []
        for bid in budget_ids:
            vals = [r[key] for r in rows if r["method"] == m and int(r["budget_id"]) == int(bid)]
            v = np.asarray(vals, dtype=np.float32)
            means.append(float(np.nanmean(v)) if v.size else np.nan)
            stds.append(float(np.nanstd(v)) if v.size else np.nan)
        curves[m] = (np.asarray(means, dtype=np.float32), np.asarray(stds, dtype=np.float32))
    return curves


# ============================================================
# k-center with runtime guard (uses Baselines.kcenter_selection)
# ============================================================
def select_kcenter_pool_guard(
    Xp: np.ndarray,
    k: int,
    rng: np.random.RandomState,
    pool: int,
    greedy_cap: int,
    fill_random: bool,
    seed: int,
) -> np.ndarray:
    """
    Run k-center (farthest-first) on a random pool for stability.

    - Xp: (N, d_sel) selection-space features
    - k : target budget
    - rng: RandomState for pool sampling
    - pool: max pool size
    - greedy_cap: run greedy at most this many steps, then optionally fill with random
    """
    N, d = Xp.shape
    k = int(min(max(1, k), N))
    if k <= 0:
        return np.array([], dtype=int)

    pool = int(min(pool, N))
    pool_idx = rng.choice(N, size=pool, replace=False)
    X_pool = Xp[pool_idx].astype(np.float32, copy=False)

    # Greedy phase on the pool
    k_greedy = int(min(k, greedy_cap, pool))
    if k_greedy > 0:
        sel_local = kcenter_selection(
            X_pool,
            k_greedy,
            metric="euclidean",
            random_state=seed,
        )
        sel = pool_idx[np.asarray(sel_local, dtype=int)]
    else:
        sel = np.array([], dtype=int)

    # Fill remaining with random points from the pool (or full set) if requested
    if k > sel.size and fill_random:
        remaining = np.setdiff1d(
            np.arange(N, dtype=int),
            sel,
            assume_unique=False,
        )
        need = k - sel.size
        if remaining.size > 0:
            fill_rng = np.random.RandomState(seed + 123)
            fill = remaining[
                fill_rng.choice(
                    remaining.size,
                    size=min(need, remaining.size),
                    replace=False,
                )
            ]
            sel = np.concatenate([sel, fill.astype(int)], axis=0)

    if sel.size > k:
        sel = sel[:k]
    return sel.astype(int)


# ============================================================
# Main
# ============================================================
def main():
    out_dir = os.path.join(os.getcwd(), f"{CFG.out_prefix}_{now_stamp()}")
    ensure_dir(out_dir)
    save_json(os.path.join(out_dir, "config.json"), asdict(CFG))

    print("============================================================")
    print("E1: CIFAR-100 Vision Coreset | Balanced SORG (class groups)")
    print("------------------------------------------------------------")
    print(f"Device : {CFG.device}")
    print(f"Budgets: {CFG.budgets}")
    print(f"Outputs: {out_dir}")
    print(f"Threads: OMP={os.environ.get('OMP_NUM_THREADS','?')} Torch={torch.get_num_threads()}")
    print("============================================================")

    # --------------------------------------------------------
    # 1) Load or build cached features
    # --------------------------------------------------------
    train_X, train_y, test_X, test_y, feat_dim = load_or_build_features(CFG)
    n_train = int(train_X.shape[0])

    print(f"[Features] train={train_X.shape} test={test_X.shape} feat_dim={feat_dim}")

    # Normalize features for probe & selection
    train_Xn = l2_normalize_rows(train_X)
    test_Xn = l2_normalize_rows(test_X)

    # Selection space (optionally projected)
    if 0 < CFG.proj_dim < feat_dim:
        W = make_deterministic_projection(feat_dim, CFG.proj_dim, CFG.proj_seed)
        train_Xp = project_features(train_Xn, W)
        print(f"[Selection Space] projected: {train_Xp.shape}")
    else:
        train_Xp = train_Xn.astype(np.float32, copy=False)
        print(f"[Selection Space] using original: {train_Xp.shape}")

    # SORG expects candidates as columns: M shape (d, N)
    train_Xp_T = train_Xp.T.copy()

    budgets = list(CFG.budgets)
    methods = list(CFG.methods)
    rows: List[dict] = []

    # --------------------------------------------------------
    # 2) Full Data Reference (compute once, duplicate per seed)
    # --------------------------------------------------------
    print("Running Full Data Probe...")
    acc_full, t_full = train_linear_probe_sklearn(
        train_Xn, train_y, test_Xn, test_y,
        seed=0,
        max_iter=CFG.probe_max_iter,
        n_jobs=CFG.probe_n_jobs,
    )
    for seed in CFG.seeds:
        rows.append({
            "seed": int(seed),
            "budget": 1.0,
            "budget_id": budget_to_id(1.0),
            "k": n_train,
            "method": "full",
            "test_acc": float(acc_full),
            "sel_time_sec": 0.0,
            "train_time_sec": float(t_full),
        })
    print(f"[Full] Acc: {acc_full:.4f}")
    print("------------------------------------------------------------")

    # --------------------------------------------------------
    # 3) Subset selection + Probes
    # --------------------------------------------------------
    for seed in CFG.seeds:
        set_seed(seed)
        rng = np.random.RandomState(seed)

        for b in budgets:
            bid = budget_to_id(b)
            k = int(max(1, round(float(b) * n_train)))
            k = int(min(k, n_train))

            for m in methods:
                t0 = time.perf_counter()

                # ---------------- Selection ----------------
                if m == "random":
                    sel = random_selection(
                        train_Xp,
                        k,
                        random_state=seed * 100000 + bid,
                    )

                elif m == "kcenter":
                    sel = select_kcenter_pool_guard(
                        train_Xp,
                        k,
                        rng,
                        pool=CFG.kcenter_pool,
                        greedy_cap=CFG.kcenter_greedy_cap,
                        fill_random=CFG.kcenter_fill_random,
                        seed=seed * 100000 + bid,
                    )

                elif m == "kmeans":
                    if k > CFG.kmeans_max_k:
                        # record as skipped (keeps CSV schema consistent)
                        rows.append({
                            "seed": int(seed),
                            "budget": float(b),
                            "budget_id": int(bid),
                            "k": int(k),
                            "method": str(m),
                            "test_acc": float("nan"),
                            "sel_time_sec": float("nan"),
                            "train_time_sec": float("nan"),
                        })
                        print(f"[{m:10s}] seed={seed} b={b:.3f} k={k:4d} | SKIP (k>{CFG.kmeans_max_k})")
                        continue

                    sel = kmeans_coreset(
                        train_Xp,
                        k,
                        batch_size=CFG.kmeans_batch_size,
                        max_iter=CFG.kmeans_max_iter,
                        random_state=seed * 13 + bid,
                    )

                elif m == "herding":
                    sel = herding_selection(train_Xp, k)

                elif m == "sorg":
                    model = SORG(
                        k=int(k),
                        alpha=CFG.sorg_alpha,
                        p_norm=CFG.sorg_p_norm,
                    )
                    # Class-balanced SORG via groups = labels
                    model.fit(train_Xp_T, groups=train_y)
                    sel = model.get_support().astype(int)

                else:
                    raise ValueError(f"Unknown method: {m}")

                sel_time = float(time.perf_counter() - t0)

                # ---------------- Probe training ----------------
                acc, train_time = train_linear_probe_sklearn(
                    train_Xn[sel], train_y[sel],
                    test_Xn, test_y,
                    seed=seed * 100 + bid,
                    max_iter=CFG.probe_max_iter,
                    n_jobs=CFG.probe_n_jobs,
                )

                rows.append({
                    "seed": int(seed),
                    "budget": float(b),
                    "budget_id": int(bid),
                    "k": int(k),
                    "method": str(m),
                    "test_acc": float(acc),
                    "sel_time_sec": float(sel_time),
                    "train_time_sec": float(train_time),
                })

                print(f"[{m:10s}] seed={seed} b={b:.3f} k={k:4d} | acc={acc:.4f} | sel={sel_time:.3f}s")

        print("------------------------------------------------------------")

    # Save raw results
    save_csv(os.path.join(out_dir, "results.csv"), rows)

    # --------------------------------------------------------
    # 4) Curves
    # --------------------------------------------------------
    curve_acc = aggregate(rows, methods + ["full"], budgets + [1.0], "test_acc")
    curve_sel = aggregate(rows, methods, budgets, "sel_time_sec")
    curve_train = aggregate(rows, methods, budgets, "train_time_sec")

    plot_curve(
        os.path.join(out_dir, "curve_acc.png"),
        budgets + [1.0], curve_acc,
        "Test Accuracy", "E1: CIFAR-100 | Budget vs Accuracy (SORG vs Baselines)"
    )
    plot_curve(
        os.path.join(out_dir, "curve_sel_time.png"),
        budgets, curve_sel,
        "Selection Time (sec)", "E1: Selection Time vs Budget"
    )
    plot_curve(
        os.path.join(out_dir, "curve_train_time.png"),
        budgets, curve_train,
        "Probe Train Time (sec)", "E1: Probe Train Time vs Budget"
    )

    # --------------------------------------------------------
    # 5) Story snapshot
    # --------------------------------------------------------
    headline_b = budgets[0]
    headline_bid = budget_to_id(headline_b)
    story = []
    story.append("E1 ICML-story snapshot (CIFAR-100 Vision Coreset)")
    story.append(f"- Dataset: CIFAR-100, Encoder: {CFG.encoder}")
    story.append(f"- Headline budget: b={headline_b:.3f} (k={int(round(headline_b*n_train))})")
    story.append("")
    story.append("Headline accuracy (mean±std over seeds):")
    for m in methods:
        vals = [r["test_acc"] for r in rows if r["method"] == m and int(r["budget_id"]) == int(headline_bid)]
        v = np.asarray(vals, dtype=np.float32)
        if v.size == 0 or not np.isfinite(v).any():
            story.append(f"  * {m:12s}: acc=nan ± nan (skipped or missing)")
        else:
            story.append(f"  * {m:12s}: acc={float(np.nanmean(v)):.4f} ± {float(np.nanstd(v)):.4f}")

    with open(os.path.join(out_dir, "story.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(story) + "\n")

    print("============================================================")
    print("DONE. Saved results.")
    print(f"Output dir: {out_dir}")
    print("============================================================")


if __name__ == "__main__":
    main()
