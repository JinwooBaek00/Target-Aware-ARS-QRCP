# Baselines.py
# -*- coding: utf-8 -*-
"""
Common baseline selection algorithms used across experiments.

Conventions:
    - Input features X are row-major:
        X shape: (N, d)  where N is #candidates, d is feature dim.
    - This module NEVER does scaling / normalization.
      Any standardization must be done OUTSIDE (in the experiment files).

Provided baselines:
    - random_selection
    - herding_selection
    - kcenter_selection
    - kmeans_coreset
    - select_baseline        (dispatcher)
    - group_balanced_selection (optional label-aware wrapper)
"""

from typing import Optional, Dict, Any
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
import math

# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def _check_X(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (N, d).")
    return X


def _get_rng(random_state: Optional[int]) -> np.random.RandomState:
    if random_state is None:
        return np.random.RandomState()
    if isinstance(random_state, np.random.RandomState):
        return random_state
    return np.random.RandomState(int(random_state))


# ----------------------------------------------------------------------
# 1. Random
# ----------------------------------------------------------------------
def random_selection(X: np.ndarray,
                     k: int,
                     random_state: Optional[int] = None) -> np.ndarray:
    """
    Uniform random subset without replacement.

    Parameters
    ----------
    X : array-like, shape (N, d)
        Candidate features.
    k : int
        Target budget size.
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    indices : np.ndarray, shape (k,)
        Selected row indices into X.
    """
    X = _check_X(X)
    N = X.shape[0]
    if N == 0 or k <= 0:
        return np.array([], dtype=int)

    k = int(min(max(1, k), N))
    rng = _get_rng(random_state)
    idx = rng.choice(N, size=k, replace=False)
    return idx.astype(int)


# ----------------------------------------------------------------------
# 2. Herding
# ----------------------------------------------------------------------
def herding_selection(X: np.ndarray,
                      k: int) -> np.ndarray:
    """
    Classic herding to approximate the global mean with a small subset.

    Algorithm (row-major):
        - Let mu = mean of X over rows.
        - Initialize running mean r = 0.
        - For t = 1..k:
            * direction = mu - r
            * choose index j maximizing <x_j, direction>
            * update r <- r + (x_j - r) / t

    Deterministic given X.

    Parameters
    ----------
    X : array-like, shape (N, d)
        Candidate features.
    k : int
        Target budget size.

    Returns
    -------
    indices : np.ndarray, shape (<=k,)
        Selected row indices into X.
    """
    X = _check_X(X)
    N, d = X.shape
    if N == 0 or k <= 0:
        return np.array([], dtype=int)

    k = int(min(max(1, k), N))

    mu = X.mean(axis=0)                       # (d,)
    running = np.zeros_like(mu)               # (d,)
    used = np.zeros(N, dtype=bool)
    selected = []

    for t in range(k):
        direction = mu - running             # (d,)
        # score_i = <x_i, direction>
        scores = X @ direction               # (N,)
        scores[used] = -np.inf

        j = int(np.argmax(scores))
        if not np.isfinite(scores[j]):
            break

        selected.append(j)
        used[j] = True

        # Update running mean
        running += (X[j] - running) / float(t + 1)

    return np.asarray(selected, dtype=int)


# ----------------------------------------------------------------------
# 3. k-center (greedy)
# ----------------------------------------------------------------------
def kcenter_selection(X: np.ndarray,
                      k: int,
                      metric: str = "euclidean",
                      random_state: Optional[int] = None) -> np.ndarray:
    """
    Greedy k-center selection.

    Algorithm:
        1. Start from a random point.
        2. Repeatedly add the point whose minimum distance to the current
           centers is maximal.

    Complexity: O(k * N^2) in the naive form.

    Parameters
    ----------
    X : array-like, shape (N, d)
        Candidate features.
    k : int
        Target budget size.
    metric : str
        Distance metric passed to sklearn.metrics.pairwise_distances.
    random_state : int or None
        Seed for reproducibility (initial center).

    Returns
    -------
    indices : np.ndarray, shape (k,)
        Selected row indices into X.
    """
    X = _check_X(X)
    N, d = X.shape
    if N == 0 or k <= 0:
        return np.array([], dtype=int)

    k = int(min(max(1, k), N))
    rng = _get_rng(random_state)

    # 1) choose first center randomly
    first = int(rng.randint(0, N))
    centers = [first]

    # 2) initialize min distances to the first center
    dists = pairwise_distances(X, X[[first]], metric=metric).reshape(-1)
    min_dists = dists.copy()

    for _ in range(1, k):
        j = int(np.argmax(min_dists))
        centers.append(j)

        d_new = pairwise_distances(X, X[[j]], metric=metric).reshape(-1)
        min_dists = np.minimum(min_dists, d_new)

    return np.asarray(centers, dtype=int)


# ----------------------------------------------------------------------
# 4. k-means coreset
# ----------------------------------------------------------------------
def kmeans_coreset(X: np.ndarray,
                   k: int,
                   batch_size: int = 1024,
                   max_iter: int = 100,
                   random_state: Optional[int] = None) -> np.ndarray:
    """
    k-means-based coreset:
        1. Run (MiniBatch) K-Means with k clusters.
        2. For each cluster, pick the closest point to its center.

    Parameters
    ----------
    X : array-like, shape (N, d)
        Candidate features.
    k : int
        Target budget size.
    batch_size : int
        Mini-batch size for MiniBatchKMeans.
    max_iter : int
        Maximum number of iterations.
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    indices : np.ndarray, shape (<=k,)
        Selected row indices into X.
    """
    X = _check_X(X)
    N, d = X.shape
    if N == 0 or k <= 0:
        return np.array([], dtype=int)

    k = int(min(max(1, k), N))

    mbk = MiniBatchKMeans(
        n_clusters=k,
        batch_size=batch_size,
        max_iter=max_iter,
        random_state=random_state,
        n_init=10,
    )
    labels = mbk.fit_predict(X)
    centers = mbk.cluster_centers_       # (k, d)

    indices = []
    for c in range(k):
        idx_in_cluster = np.where(labels == c)[0]
        if idx_in_cluster.size == 0:
            # empty cluster -> skip or pick random?
            # Here we simply skip. You could also fallback to random.
            continue

        Xc = X[idx_in_cluster]          # (Nc, d)
        center = centers[c][None, :]    # (1, d)
        dist = np.linalg.norm(Xc - center, axis=1)  # (Nc,)
        j_local = int(np.argmin(dist))
        indices.append(int(idx_in_cluster[j_local]))

    if not indices:
        return np.array([], dtype=int)
    return np.asarray(indices, dtype=int)


# ----------------------------------------------------------------------
# 5. Unified dispatcher
# ----------------------------------------------------------------------
def select_baseline(X: np.ndarray,
                    k: int,
                    method: str = "random",
                    random_state: Optional[int] = None,
                    **kwargs: Dict[str, Any]) -> np.ndarray:
    """
    Unified entrypoint for baseline selection.

    Parameters
    ----------
    X : array-like, shape (N, d)
        Candidate features.
    k : int
        Target budget size.
    method : {'random', 'herding', 'kcenter', 'kmeans'}
        Baseline algorithm name.
    random_state : int or None
        Seed for methods that use randomness.
    **kwargs :
        Additional arguments forwarded to individual methods.

    Returns
    -------
    indices : np.ndarray
        Selected row indices into X.
    """
    method = method.lower()
    if method == "random":
        return random_selection(X, k, random_state=random_state)
    elif method == "herding":
        return herding_selection(X, k)
    elif method in ("kcenter", "k-center"):
        metric = kwargs.get("metric", "euclidean")
        return kcenter_selection(X, k, metric=metric, random_state=random_state)
    elif method in ("kmeans", "k-means"):
        batch_size = kwargs.get("batch_size", 1024)
        max_iter = kwargs.get("max_iter", 100)
        return kmeans_coreset(
            X, k,
            batch_size=batch_size,
            max_iter=max_iter,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown baseline method: {method}")


# ----------------------------------------------------------------------
# 6. Optional: group-balanced wrapper
# ----------------------------------------------------------------------
def group_balanced_selection(X: np.ndarray,
                             groups: np.ndarray,
                             k: int,
                             method: str = "random",
                             random_state: Optional[int] = None,
                             **kwargs: Dict[str, Any]) -> np.ndarray:
    """
    Group-balanced baseline selection (e.g., class-balanced random/herding).

    Strategy:
        - Compute per-group quotas (roughly k / #groups, then redistribute
          leftover capacity to larger groups).
        - For each group, run the chosen baseline on X_g with budget q_g.

    Parameters
    ----------
    X : array-like, shape (N, d)
        Candidate features.
    groups : array-like, shape (N,)
        Group or class labels for each row of X.
    k : int
        Total budget size across all groups.
    method : str
        Baseline name, passed to select_baseline.
    random_state : int or None
        Seed for random-based methods.
    **kwargs :
        Extra arguments forwarded to select_baseline.

    Returns
    -------
    indices : np.ndarray
        Selected row indices into X.
    """
    X = _check_X(X)
    groups = np.asarray(groups)
    if groups.ndim != 1:
        raise ValueError("groups must be a 1D array of length N.")

    N = X.shape[0]
    if groups.shape[0] != N:
        raise ValueError(f"groups length mismatch: got {groups.shape[0]}, expected {N}.")

    if N == 0 or k <= 0:
        return np.array([], dtype=int)

    k = int(min(max(1, k), N))
    uniq, counts = np.unique(groups, return_counts=True)
    n_groups = len(uniq)
    if n_groups == 0:
        return np.array([], dtype=int)

    # initial quotas
    base = k // n_groups
    rem = k % n_groups
    quotas = np.full(n_groups, base, dtype=int)
    quotas[:rem] += 1
    quotas = np.minimum(quotas, counts)

    used = int(np.sum(quotas))
    leftover = k - used

    if leftover > 0:
        capacity = counts - quotas
        order = np.argsort(-capacity)
        for j in order:
            if leftover <= 0:
                break
            give = int(min(capacity[j], leftover))
            quotas[j] += give
            leftover -= give

    rng = _get_rng(random_state)
    selected = []

    for i, g in enumerate(uniq):
        q = int(quotas[i])
        if q <= 0:
            continue
        mask = (groups == g)
        idx_g = np.where(mask)[0]
        X_g = X[idx_g]

        # To avoid coupling, use independent seeds per group
        seed_g = int(rng.randint(0, 2**31 - 1))
        idx_local = select_baseline(
            X_g,
            k=q,
            method=method,
            random_state=seed_g,
            **kwargs,
        )
        selected.append(idx_g[idx_local])

    if not selected:
        return np.array([], dtype=int)
    return np.concatenate(selected).astype(int)

# ----------------------------------------------------------------------
# 7. SubZeroCore 
# ----------------------------------------------------------------------  
def subzero_selection(X: np.ndarray, k: int, coverage_target: float=0.95) -> np.ndarray
    """
    SubZeroCore: A modern coreset via Density-Weighted Facility Location.
    
    Algorithm:
      1. Determine coverage radius K based on budget and coverage target.
      2. Compute K-NN radii for all points.
      3. Compute Gaussian density weights s_i based on radii.
      4. Select points via Lazy-Greedy Weighted Facility Location.
    
    Parameters
    ----------
    X : (N, d) Input features
    k : int Budget
    coverage_target : float, target probability (default 0.95)
    
    Returns
    -------
    indices : Selected indices
    """
    X = _check_X(X)
    N = X.shape[0]

    if k >= N:
        return np.arange(N)
    if k <= 0:
        return np.array([], dtype=int)
    
    # Step 1: Determine Optimal K (Coverage Radius)
    # Find min k such that Prob(failure) <= (1 - coverage)
    # Failure probability approx: product((N - k_budget - i)(N-i))
    target_error = 1.0 - coverage_target
    current_prod = 1.0
    K_val = 1 # Minimum 1 neighbor

    # Combinatorial probability loop
    for i in range(N):
        term = (N-k-i)/(N-i)
        if current_prod <= target_error:
            K_val = i+1
            break

    # Step 2: Compute radii
    # We need the distance to the K-th nearest neighbor
    nbrs = NearestNeighbors(n_neighbors=K_val + 1, algorithm='auto', n_jobs=-1).fit(X)
    distances, _ = nbrs.kneighbors(X)

    # r_i is the distance to the K-th neighbor (last column)
    r = distances[:, -1]
    
    # Step 3: Compute Density-Based Weights
    mu = np.mean(r)
    sigma = np.std(r)
    if sigma < 1e-9: sigma = 1.0  # Avoid division by zero
    
    # s_i: higher weight for points with "average" density
    weights = np.exp(-((r - mu)**2) / (2 * sigma**2))
    
    # Step 4: Lazy Greedy Weighted Facility Location
    # Objective: max_{S} sum_{i} max_{j in S} (weights[j] * sim(i, j))
    # We assume sim(i, j) is dot product. 
    
    # Current best score for each data point i: val_i = max_{j in S} (w_j * sim(i, j))
    # Initialize with -inf (not covered)
    current_values = np.full(N, -np.inf)
    
    # Marginal gain for adding candidate j is:
    # Gain(j) = sum_{i} max(current_values[i], w_j * sim(i, j)) - current_values[i]
    # simplified: sum_{i} max(0, w_j * sim(i, j) - current_values[i])
    
    selected = []
    candidates = list(range(N))
    
    # Fast calc for 1st element:
    # Total Score of j = sum_i (w_j * x_i.T x_j) = w_j * (sum_i x_i)^T x_j
    sum_X = np.sum(X, axis=0) # (d,)
    # Score vector (N,)
    scores_1 = weights * (X @ sum_X)
    
    best_idx = int(np.argmax(scores_1))
    selected.append(best_idx)
    
    # Update current_values based on first selection
    best_w = weights[best_idx]
    best_sims = X @ X[best_idx]
    current_values = best_w * best_sims # (N,)
    
    # --- Main Loop (Iterations 2 to k) ---
    # We maintain a list of 'marginal_gains' which are upper bounds.
    # But for dot-product, bounds are loose. 
    # We use the standard lazy greedy logic:
    # Keep candidates sorted by "last computed gain". 
    # Pop best, recompute. If still best, keep.
    
    # Initialize gains: strictly, we don't know them without computation.
    # We reset gains to infinity to force re-evaluation
    gains = np.full(N, np.inf)
    gains[best_idx] = -1.0 # Already selected
    
    for t in range(1, k):
        while True:
            # Pick candidate with highest estimate
            cand_idx = int(np.argmax(gains))
            
            # If we've already selected it (shouldn't happen with -1 logic but safety check)
            if gains[cand_idx] < 0:
                break
                
            # Recompute exact gain for this candidate
            # Gain = sum(max(0, new_val - old_val))
            w_cand = weights[cand_idx]
            sim_cand = X @ X[cand_idx] # (N,)
            new_vals = w_cand * sim_cand
            
            # Improvement vector
            improvements = np.maximum(0, new_vals - current_values)
            true_gain = np.sum(improvements)
            
            # Update the gain in our list
            gains[cand_idx] = true_gain
            
            # Check if it's still the winner (Lazy check)
            # Find the next best estimate
            # Mask out the current one to find 2nd best
            temp_gains = gains.copy()
            temp_gains[cand_idx] = -1.0
            second_best = np.max(temp_gains)
            
            if true_gain >= second_best:
                # Found the winner
                selected.append(cand_idx)
                
                # Update system state
                current_values = np.maximum(current_values, new_vals)
                
                # Mark as selected
                gains[cand_idx] = -1.0
                break
                
    return np.array(selected, dtype=int)


# ----------------------------------------------------------------------
# 7. ELFS 
# ----------------------------------------------------------------------  
def elfs():
    #TODO 
        


__all__ = [
    "random_selection",
    "herding_selection",
    "kcenter_selection",
    "kmeans_coreset",
    "select_baseline",
    "group_balanced_selection",
    "subzero_selection",
    "elfs"
]
