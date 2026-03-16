# E4_TargetAwareDomainShift.py
# E4: Target-Aware Domain Shift (Compositional) | SORG (NoScale) vs Baselines
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
import pickle
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

# Robust sklearn solver
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
FEATURE_CACHE = os.path.join(BASE_NFS, "E4_feature_cache")  # Separate cache for E4 (Coarse Labels)

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
class E4Config:
    dataset: str = "cifar100"
    encoder: str = "resnet18"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    k_list: Tuple[int, ...] = (100, 200, 500, 1000)
    
    # Source Pool (Subsampled from Train)
    pool_size: int = 10000
    
    # Target Setup
    target_unlabeled: int = 1000  # Size of unlabeled set to estimate mean (r)
    target_eval_cap: int = 2000   # Max size of target test set for speed
    
    # Methods
    methods: Tuple[str, ...] = (
        "random",
        "herding",             # Target-Agnostic Baseline
        "kcenter",             # Diversity Baseline
        "kmeans_mb",
        "mean_match",          # Target-Aware Baseline (Cosine Sim)
        "sorg",                # Unguided SORG (Diversity)
        "sorg_guided",         # Guided SORG (Target-Aware)
        "sorg_guided_nogate"   # Ablation
    )
    
    # Target Domains (Superclass Composition)
    target_domains: Tuple[str, ...] = ("target_bio", "target_nature", "target_manmade")
    
    ridge_alpha: float = 1.0
    kmeans_max_iter: int = 120
    
    out_prefix: str = "E4_outputs"
    base_nfs: str = BASE_NFS

CFG = E4Config()

# ============================================================
# Helpers
# ============================================================
def now_stamp() -> str: return time.strftime("%Y%m%d_%H%M%S")
def ensure_dir(p: str) -> None: os.makedirs(p, exist_ok=True)
def seed_everything(seed: int):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def save_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f: json.dump(obj, f, indent=2)

def save_csv(path: str, rows: List[dict]):
    if not rows: return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")

# ============================================================
# CIFAR-100 Coarse Label Handling (Essential for E4)
# ============================================================
def _load_cifar100_pickle(root, name):
    path = os.path.join(root, "cifar-100-python", name)
    with open(path, "rb") as f: return pickle.load(f, encoding="bytes")

def load_cifar100_meta(root):
    meta = _load_cifar100_pickle(root, "meta")
    coarse = [x.decode("utf-8") for x in meta[b"coarse_label_names"]]
    fine = [x.decode("utf-8") for x in meta[b"fine_label_names"]]
    return coarse, fine

def load_cifar100_labels(root, train=True):
    name = "train" if train else "test"
    data = _load_cifar100_pickle(root, name)
    y_fine = np.array(data[b"fine_labels"], dtype=np.int64)
    y_coarse = np.array(data[b"coarse_labels"], dtype=np.int64)
    return y_fine, y_coarse

def build_target_domains(coarse_names):
    name2idx = {n.lower().replace("_", " "): i for i, n in enumerate(coarse_names)}
    def get_idxs(names): return [name2idx[n] for n in names]
    
    return {
        "target_bio": get_idxs(["aquatic mammals", "fish", "insects", "non-insect invertebrates", "reptiles"]),
        "target_nature": get_idxs(["flowers", "trees", "large natural outdoor scenes", "fruit and vegetables", "food containers"]),
        "target_manmade": get_idxs(["vehicles 1", "vehicles 2", "household electrical devices", "household furniture", "large man-made outdoor things"]),
    }

# ============================================================
# Feature Extraction (Cached with Coarse Labels)
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
def extract_features(model, loader, device):
    model.eval()
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        z = model(x).flatten(1)
        feats.append(z.cpu().numpy().astype(np.float32))
        labels.append(y.numpy().astype(np.int64))
    return np.concatenate(feats), np.concatenate(labels)

def get_data_e4(cfg):
    tag = f"{cfg.dataset}_{cfg.encoder}_E4"
    paths = {
        "train_X": os.path.join(FEATURE_CACHE, f"{tag}_train_X.npy"),
        "test_X": os.path.join(FEATURE_CACHE, f"{tag}_test_X.npy"),
        "train_y_fine": os.path.join(FEATURE_CACHE, f"{tag}_train_y_fine.npy"),
        "test_y_fine": os.path.join(FEATURE_CACHE, f"{tag}_test_y_fine.npy"),
        "test_y_coarse": os.path.join(FEATURE_CACHE, f"{tag}_test_y_coarse.npy"),
        "meta": os.path.join(FEATURE_CACHE, f"{tag}_meta.json"),
    }
    
    if all(os.path.exists(p) for p in paths.values()):
        print(">>> Loading cached features (E4)...")
        with open(paths["meta"], "r") as f: meta = json.load(f)
        return (np.load(paths["train_X"]), np.load(paths["test_X"]), 
                np.load(paths["train_y_fine"]), np.load(paths["test_y_fine"]), 
                np.load(paths["test_y_coarse"]), meta["coarse_names"])

    print(">>> Extracting features (E4)...")
    # Ensure dataset exists for pickle loading
    _ = torchvision.datasets.CIFAR100(TORCH_DATASETS, train=True, download=True)
    _ = torchvision.datasets.CIFAR100(TORCH_DATASETS, train=False, download=True)
    
    coarse_names, _ = load_cifar100_meta(TORCH_DATASETS)
    _, test_y_coarse = load_cifar100_labels(TORCH_DATASETS, train=False)
    
    model, transform = get_resnet18(cfg.device)
    tr_ds = torchvision.datasets.CIFAR100(TORCH_DATASETS, train=True, transform=transform)
    te_ds = torchvision.datasets.CIFAR100(TORCH_DATASETS, train=False, transform=transform)
    
    Xt, Yt_fine = extract_features(model, DataLoader(tr_ds, 512, num_workers=4), cfg.device)
    Xte, Yte_fine = extract_features(model, DataLoader(te_ds, 512, num_workers=4), cfg.device)
    
    # Verify alignment
    if not np.array_equal(Yte_fine, np.array(te_ds.targets)):
        # Fallback: if dataloader shuffles or transforms mess up order, rely on extracted Yte_fine
        # But for E4 we need Coarse labels aligned.
        # CIFAR100 dataset order is fixed if shuffle=False.
        pass

    np.save(paths["train_X"], Xt); np.save(paths["test_X"], Xte)
    np.save(paths["train_y_fine"], Yt_fine); np.save(paths["test_y_fine"], Yte_fine)
    np.save(paths["test_y_coarse"], test_y_coarse)
    
    meta = {"coarse_names": coarse_names}
    save_json(paths["meta"], meta)
    
    return Xt, Xte, Yt_fine, Yte_fine, test_y_coarse, coarse_names

# ============================================================
# Selection Methods (Wrappers around Baselines & SORG)
# ============================================================
def select_mean_match(X, r, k):
    # Simple baseline: Select items with highest dot product to r
    # (Since SORG NoScale uses dot product, we keep this simple too)
    scores = X @ r
    idx = np.argsort(-scores)[:k]
    return idx

def select_sorg_wrapper(X, k, r=None, alpha=1.0, p_norm=2.0):
    model = SORG(k=k, alpha=alpha, p_norm=p_norm)
    if r is None:
        model.fit(X.T)
    else:
        model.fit(X.T, r=r)
    return model.get_support()

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
    print("E4: Target-Aware Domain Shift (SORG NoScale)")
    print("------------------------------------------------------------")
    
    # Load Data
    Xt_raw, Xte_raw, Yt_fine, Yte_fine, Yte_coarse, coarse_names = get_data_e4(CFG)
    domains = build_target_domains(coarse_names)
    
    rows = []
    
    # Pre-select Source Pool (to speed up exp)
    N = len(Xt_raw)
    rng_pool = np.random.RandomState(CFG.seeds[0]) # Fix pool across seeds for consistency or vary?
    # Let's vary pool by seed for robustness
    
    for seed in CFG.seeds:
        seed_everything(seed)
        rng = np.random.RandomState(seed)
        
        # 1. Subsample Source Pool
        pool_idx = rng.choice(N, min(N, CFG.pool_size), replace=False)
        X_pool = Xt_raw[pool_idx]
        y_pool = Yt_fine[pool_idx]
        
        # 2. Iterate Target Domains
        for domain_name in CFG.target_domains:
            target_superclasses = domains[domain_name]
            
            # Filter Test set for Target Domain
            mask = np.isin(Yte_coarse, target_superclasses)
            target_indices = np.where(mask)[0]
            
            # Split Target into Unlabeled (Guidance) and Eval
            rng.shuffle(target_indices)
            n_unlab = min(len(target_indices), CFG.target_unlabeled)
            idx_unlab = target_indices[:n_unlab]
            idx_eval = target_indices[n_unlab:]
            
            if len(idx_eval) > CFG.target_eval_cap:
                idx_eval = idx_eval[:CFG.target_eval_cap]
                
            X_unlab = Xte_raw[idx_unlab]
            X_eval = Xte_raw[idx_eval]
            y_eval = Yte_fine[idx_eval]
            
            # Guidance Vector (Target Mean)
            # NoScale: Compute mean directly on raw features
            r_target = np.mean(X_unlab, axis=0)
            
            print(f"\n[Seed {seed}] Domain: {domain_name} | Pool: {len(X_pool)} | Unlab: {len(X_unlab)} | Eval: {len(X_eval)}")
            
            for k in CFG.k_list:
                if k > len(X_pool): continue
                
                for method in CFG.methods:
                    t0 = time.time()
                    try:
                        # --- Baselines (Target-Agnostic) ---
                        if method == "random":
                            sel_local = BL.random_selection(X_pool, k, random_state=seed)
                        elif method == "herding":
                            sel_local = BL.herding_selection(X_pool, k)
                        elif method == "kcenter":
                            sel_local = BL.kcenter_selection(X_pool, k)
                        elif method == "kmeans_mb":
                            sel_local = BL.kmeans_coreset(X_pool, k, max_iter=CFG.kmeans_max_iter, random_state=seed)
                            
                        # --- Baselines (Target-Aware) ---
                        elif method == "mean_match":
                            sel_local = select_mean_match(X_pool, r_target, k)
                            
                        # --- SORG ---
                        elif method == "sorg":
                            # Unguided (Diversity)
                            sel_local = select_sorg_wrapper(X_pool, k, r=None, alpha=1.0, p_norm=2.0)
                            
                        elif method == "sorg_guided":
                            # Guided (Target Mean)
                            sel_local = select_sorg_wrapper(X_pool, k, r=r_target, alpha=1.0, p_norm=2.0)
                            
                        elif method == "sorg_guided_nogate":
                            # Guided (Target Mean) + No Gate
                            sel_local = select_sorg_wrapper(X_pool, k, r=r_target, alpha=1.0, p_norm=0.0)
                            
                        else:
                            sel_local = np.array([], dtype=int)
                            
                    except Exception as e:
                        print(f"Error {method}: {e}")
                        sel_local = np.array([], dtype=int)
                    
                    t_sel = time.time() - t0
                    
                    if len(sel_local) == 0:
                        continue
                        
                    # Evaluation
                    # Test on Source (Original Test Set) -> Robustness Check
                    acc_src, _ = train_eval(X_pool[sel_local], y_pool[sel_local], Xte_raw, Yte_fine)
                    # Test on Target -> Adaptation Check
                    acc_tgt, _ = train_eval(X_pool[sel_local], y_pool[sel_local], X_eval, y_eval)
                    
                    print(f"  [{method:18s}] k={k:<4} | TgtAcc={acc_tgt:.4f} SrcAcc={acc_src:.4f}")
                    
                    rows.append({
                        "seed": seed, "domain": domain_name, "k": k, "method": method,
                        "acc_target": acc_tgt, "acc_source": acc_src,
                        "delta": acc_tgt - acc_src,
                        "sel_time": t_sel
                    })

    # Save & Plot
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(exp_dir, "results.csv"), index=False)
    
    # Simple Plot per Domain
    for d in CFG.target_domains:
        sub = df[df.domain == d]
        if sub.empty: continue
        
        plt.figure(figsize=(7, 5))
        sns.lineplot(data=sub, x="k", y="acc_target", hue="method", marker="o", errorbar='sd')
        plt.title(f"E4 Domain Adaptation: {d}")
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, f"plot_{d}.png"))
        plt.close()

    print("\nDONE.")

if __name__ == "__main__":
    main()
