# E2_RobustnessSuite_SORG.py
# E2: Robustness Stress Test (Scaling + Outliers) | SORG (NoScale) vs Baselines
# -*- coding: utf-8 -*-

import os

# ============================================================
# [Critical] Stabilize CPU thread usage
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
# [Rule #1] Path Config & Imports
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
class E2Config:
    dataset: str = "cifar100"
    encoder: str = "resnet18"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    budgets: Tuple[float, ...] = (0.001, 0.002, 0.005, 0.01)
    
    pool_size: int = 8000
    
    # Methods
    methods: Tuple[str, ...] = (
        "random",
        "kcenter",
        "kmeans_mb",
        "herding",
        "sorg",         # Gate ON (p=2.0)
        "sorg_nogate",  # Gate OFF (p=0.0)
    )
    
    # Attacks
    scale_dim_frac: float = 0.10
    scale_factors: Tuple[float, ...] = (10.0, 100.0)
    outlier_fracs: Tuple[float, ...] = (0.01, 0.05)
    outlier_scale: float = 50.0
    
    ridge_alpha: float = 1.0
    kmeans_max_iter: int = 120
    
    out_prefix: str = "E2_outputs"
    base_nfs: str = BASE_NFS

CFG = E2Config()

# ============================================================
# Helpers
# ============================================================
def now_stamp() -> str: return time.strftime("%Y%m%d_%H%M%S")
def ensure_dir(p: str) -> None: os.makedirs(p, exist_ok=True)
def seed_everything(seed: int):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
def stable_int_hash(s: str) -> int:
    return int(zlib.adler32(s.encode("utf-8")) & 0xFFFFFFFF)
def seed_mix(*items: int) -> int:
    x = 2166136261
    for v in items:
        x = (x ^ (v & 0xFFFFFFFF)) * 16777619
        x &= 0xFFFFFFFF
    return int(x)

def save_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f: json.dump(obj, f, indent=2)

def save_csv(path: str, rows: List[dict]):
    if not rows: return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")

# ============================================================
# Feature Extraction (Cached)
# ============================================================
def get_resnet18(device):
    try:
        w = torchvision.models.ResNet18_Weights.DEFAULT
        m = torchvision.models.resnet18(weights=w)
        t = w.transforms()
    except:
        m = torchvision.models.resnet18(pretrained=True)
        t = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), 
                       T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    feat = nn.Sequential(*list(m.children())[:-1]).to(device)
    feat.eval()
    return feat, t

@torch.no_grad()
def extract_all(model, loader, device):
    model.eval()
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        z = model(x).flatten(1)
        feats.append(z.cpu().numpy().astype(np.float32))
        labels.append(y.numpy().astype(np.int64))
    return np.concatenate(feats), np.concatenate(labels)

def load_or_build_features(cfg):
    tag = f"{cfg.dataset}_{cfg.encoder}"
    paths = {
        "train_X": os.path.join(FEATURE_CACHE, f"{tag}_train_X.npy"),
        "train_y": os.path.join(FEATURE_CACHE, f"{tag}_train_y.npy"),
        "test_X": os.path.join(FEATURE_CACHE, f"{tag}_test_X.npy"),
        "test_y": os.path.join(FEATURE_CACHE, f"{tag}_test_y.npy"),
        "meta": os.path.join(FEATURE_CACHE, f"{tag}_meta.json"),
    }
    
    if all(os.path.exists(p) for p in paths.values()):
        print(">>> Loading cached features...")
        with open(paths["meta"], "r") as f: meta = json.load(f)
        return (np.load(paths["train_X"]), np.load(paths["train_y"]), 
                np.load(paths["test_X"]), np.load(paths["test_y"]), 
                int(meta["feat_dim"]))

    print(">>> Extracting features...")
    model, transform = get_resnet18(cfg.device)
    
    tr_ds = torchvision.datasets.CIFAR100(TORCH_DATASETS, train=True, download=True, transform=transform)
    te_ds = torchvision.datasets.CIFAR100(TORCH_DATASETS, train=False, download=True, transform=transform)
    
    Xt, Yt = extract_all(model, DataLoader(tr_ds, 256, num_workers=4), cfg.device)
    Xte, Yte = extract_all(model, DataLoader(te_ds, 256, num_workers=4), cfg.device)
    
    np.save(paths["train_X"], Xt); np.save(paths["train_y"], Yt)
    np.save(paths["test_X"], Xte); np.save(paths["test_y"], Yte)
    save_json(paths["meta"], {"feat_dim": Xt.shape[1], "dataset": cfg.dataset, "encoder": cfg.encoder})
    
    return Xt, Yt, Xte, Yte, Xt.shape[1]

# ============================================================
# Attack Logic
# ============================================================
def apply_outlier_attack(X, rng, frac, multiplier):
    """
    Replace a frac of rows in X with large-norm random outliers.

    Outlier norm is set to (multiplier * median_norm_of_current_pool),
    so it is always 'large' relative to the current data scale.
    """
    X2 = X.copy()
    n, d = X2.shape
    m = int(max(1, round(frac * n)))
    idx = rng.choice(n, size=m, replace=False)
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True

    # Compute current pool median norm (after any scaling attack)
    norms = np.linalg.norm(X2, axis=1)
    median_norm = float(np.median(norms) + 1e-12)

    # Generate random directions and scale them
    V = rng.normal(size=(m, d)).astype(np.float32)
    vn = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    target_norm = float(multiplier) * median_norm
    V = (V / vn) * target_norm

    X2[idx] = V
    return X2, mask

# ============================================================
# Selection Wrapper
# ============================================================
def select_sorg(Xp, k, p_norm):
    # SORG: NoScale, Alpha=1.0 (Orthogonal)
    model = SORG(k=int(k), alpha=1.0, p_norm=float(p_norm))
    model.fit(Xp.T)
    return model.get_support().astype(int)

# ============================================================
# Eval
# ============================================================
def train_eval(X_train, y_train, X_test, y_test, alpha=1.0):
    t0 = time.time()
    scaler = StandardScaler()
    Xt_s = scaler.fit_transform(X_train)
    Xte_s = scaler.transform(X_test)
    
    clf = RidgeClassifier(alpha=alpha)
    clf.fit(Xt_s, y_train)
    acc = float(clf.score(Xte_s, y_test))
    return acc, time.time() - t0

# ============================================================
# Main
# ============================================================
def main():
    exp_dir = os.path.join(os.getcwd(), f"{CFG.out_prefix}_{now_stamp()}")
    ensure_dir(exp_dir)
    save_json(os.path.join(exp_dir, "config.json"), asdict(CFG))
    
    print("============================================================")
    print("E2: Robustness Stress Suite | SORG NoScale")
    print("------------------------------------------------------------")
    
    Xt_raw, Yt, Xte_raw, Yte, feat_dim = load_or_build_features(CFG)
    N = len(Xt_raw)
    
    # Attack Specs
    attacks = [("clean", None)]
    for f in CFG.scale_factors: attacks.append((f"scale_{int(f)}", ("scale", float(f))))
    for p in CFG.outlier_fracs: attacks.append((f"outlier_{int(round(p*100))}pct", ("outlier", float(p))))
    attacks.append(("combo_scale100_outlier5pct", ("combo", None)))
    
    rows = []
    clean_acc_cache = {}
    
    for seed in CFG.seeds:
        seed_everything(seed)
        
        # Subsample Pool
        P = min(CFG.pool_size, N)
        rng_pool = np.random.RandomState(seed_mix(seed, 12345))
        pool_idx = rng_pool.choice(N, P, replace=False)
        X_pool_clean = Xt_raw[pool_idx]
        y_pool = Yt[pool_idx]
        
        # Scale Dims (Fixed per seed)
        rng_scale = np.random.RandomState(seed_mix(seed, 2027))
        d = X_pool_clean.shape[1]
        scale_dims = rng_scale.choice(d, int(max(1, round(CFG.scale_dim_frac * d))), replace=False)
        
        for b in CFG.budgets:
            k = int(min(int(b * N), P))
            
            for atk_key, spec in attacks:
                X_pool_att = X_pool_clean.copy()
                outlier_mask = np.zeros(P, dtype=bool)
                rng_att = np.random.RandomState(seed_mix(seed, int(b*1e6), stable_int_hash(atk_key)))
                
                # Apply Attacks
                if atk_key.startswith("scale_"):
                    X_pool_att[:, scale_dims] *= float(spec[1])
                elif atk_key.startswith("outlier_"):
                    X_pool_att, outlier_mask = apply_outlier_attack(X_pool_att, rng_att, float(spec[1]), CFG.outlier_scale)
                elif atk_key.startswith("combo_"):
                    X_pool_att[:, scale_dims] *= 100.0
                    X_pool_att, outlier_mask = apply_outlier_attack(X_pool_att, rng_att, 0.05, CFG.outlier_scale)
                    
                print(f"[Seed {seed}] Budget={b:.3f} | Attack={atk_key}")
                
                for method in CFG.methods:
                    t0 = time.time()
                    sel_seed = seed_mix(seed, int(b*1e6), stable_int_hash(atk_key), stable_int_hash(method))
                    
                    try:
                        # --- Baselines ---
                        if method == "random":
                            sel_local = BL.random_selection(X_pool_att, k, random_state=sel_seed)
                        elif method == "kcenter":
                            sel_local = BL.kcenter_selection(X_pool_att, k, random_state=sel_seed)
                        elif method == "kmeans_mb":
                            sel_local = BL.kmeans_coreset(X_pool_att, k, max_iter=CFG.kmeans_max_iter, random_state=sel_seed)
                        elif method == "herding":
                            sel_local = BL.herding_selection(X_pool_att, k)
                            
                        # --- SORG ---
                        elif method == "sorg":
                            sel_local = select_sorg(X_pool_att, k, p_norm=2.0)
                        elif method == "sorg_nogate":
                            sel_local = select_sorg(X_pool_att, k, p_norm=0.0)
                            
                        else:
                            sel_local = np.array([], dtype=int)
                            
                    except Exception as e:
                        print(f"Error {method}: {e}")
                        sel_local = np.array([], dtype=int)
                    
                    t_sel = time.time() - t0
                    if len(sel_local) == 0: continue
                    
                    # Eval
                    # Important: Train on ATTACKED features (to simulate poisoned training set)
                    # But evaluate on CLEAN test set (standard robustness protocol)
                    acc, t_train = train_eval(X_pool_att[sel_local], y_pool[sel_local], Xte_raw, Yte)
                    out_frac = np.mean(outlier_mask[sel_local]) if np.any(outlier_mask) else 0.0
                    
                    # Calculate Delta
                    if atk_key == "clean":
                        clean_acc_cache[(seed, b, method)] = acc
                        delta = 0.0
                    else:
                        ref = clean_acc_cache.get((seed, b, method), np.nan)
                        delta = acc - ref
                        
                    print(f"  [{method:15s}] Acc={acc:.4f} (Δ={delta:+.4f}) | Out={out_frac:.2f}")
                    
                    rows.append({
                        "seed": seed, "budget": b, "k": k, "method": method, "attack": atk_key,
                        "test_acc": acc, "delta_acc": delta, "outlier_frac": out_frac,
                        "sel_time": t_sel
                    })

    # Save & Plot
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(exp_dir, "results.csv"), index=False)
    
    # Plot Delta for key attacks
    for atk in ["scale_100", "outlier_5pct", "combo_scale100_outlier5pct"]:
        sub = df[df.attack == atk]
        if sub.empty: continue
        plt.figure(figsize=(7, 5))
        sns.lineplot(data=sub, x="budget", y="delta_acc", hue="method", marker="o", errorbar='sd')
        plt.title(f"E2 Robustness: {atk}")
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, f"delta_{atk}.png"))
        plt.close()

    print("\nDONE.")

if __name__ == "__main__":
    main()
