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


__all__ = [
    "random_selection",
    "herding_selection",
    "kcenter_selection",
    "kmeans_coreset",
    "select_baseline",
    "group_balanced_selection",
]
