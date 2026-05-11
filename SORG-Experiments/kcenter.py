from __future__ import annotations

import numpy as np


NAME = "kcenter"
DEFAULT_INIT = "mean_medoid"


def _validate_candidates(X: np.ndarray) -> np.ndarray:
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim != 2:
        raise ValueError(f"k-center expects a 2D candidate array, got shape {X_arr.shape}.")
    return X_arr


def _sq_l2_distances(X: np.ndarray, y: np.ndarray, x_norm_sq: np.ndarray | None = None) -> np.ndarray:
    x_sq = np.sum(X * X, axis=1) if x_norm_sq is None else x_norm_sq
    y_sq = float(np.dot(y, y))
    distances = x_sq + y_sq - 2.0 * (X @ y)
    return np.maximum(distances, 0.0)


def _initialize_first_center(
    X: np.ndarray,
    x_norm_sq: np.ndarray,
    config: dict,
    seed: int,
) -> int:
    init = str(config.get("init", DEFAULT_INIT))
    if init == "mean_medoid":
        mean_feature = np.mean(X, axis=0)
        return int(np.argmin(_sq_l2_distances(X, mean_feature, x_norm_sq=x_norm_sq)))
    if init == "max_norm":
        return int(np.argmax(x_norm_sq))
    if init == "random":
        rng = np.random.default_rng(seed)
        return int(rng.integers(0, X.shape[0]))
    raise ValueError(
        f"Unsupported k-center init={init!r}. Expected 'mean_medoid', 'max_norm', or 'random'."
    )


def select(
    X: np.ndarray,
    budget: int,
    guidance: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    config: dict | None = None,
    seed: int = 0,
) -> np.ndarray:
    del guidance, groups
    X_arr = _validate_candidates(X)
    cfg = config or {}
    n = X_arr.shape[0]
    if budget <= 0 or n == 0:
        return np.empty(0, dtype=np.int64)
    if budget > n:
        raise ValueError(f"Budget {budget} exceeds pool size {n}.")

    selected = np.empty(budget, dtype=np.int64)
    chosen = np.zeros(n, dtype=bool)
    x_norm_sq = np.sum(X_arr * X_arr, axis=1)

    first = _initialize_first_center(X_arr, x_norm_sq=x_norm_sq, config=cfg, seed=seed)
    selected[0] = first
    chosen[first] = True

    min_dist_sq = _sq_l2_distances(X_arr, X_arr[first], x_norm_sq=x_norm_sq)
    min_dist_sq[first] = -np.inf

    for step in range(1, budget):
        idx = int(np.argmax(min_dist_sq))
        selected[step] = idx
        chosen[idx] = True
        min_dist_sq[idx] = -np.inf
        candidate_dist_sq = _sq_l2_distances(X_arr, X_arr[idx], x_norm_sq=x_norm_sq)
        active = ~chosen
        min_dist_sq[active] = np.minimum(min_dist_sq[active], candidate_dist_sq[active])

    return selected
