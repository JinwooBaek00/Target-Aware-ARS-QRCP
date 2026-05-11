from __future__ import annotations

import numpy as np


NAME = "herding"
DEFAULT_EPS = 1e-12


def _validate_candidates(X: np.ndarray) -> np.ndarray:
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim != 2:
        raise ValueError(f"Herding expects a 2D candidate array, got shape {X_arr.shape}.")
    return X_arr


def _initialize_mean_feature(X: np.ndarray, config: dict) -> np.ndarray:
    mean_feature = np.mean(X, axis=0)
    if bool(config.get("normalize_mean", False)):
        norm = float(np.linalg.norm(mean_feature))
        if norm > float(config.get("eps", DEFAULT_EPS)):
            mean_feature = mean_feature / norm
    return mean_feature


def _score_candidates(
    X: np.ndarray,
    mean_feature: np.ndarray,
    running_sum: np.ndarray,
    step: int,
    chosen: np.ndarray,
) -> np.ndarray:
    target = (step + 1) * mean_feature - running_sum
    scores = X @ target
    scores[chosen] = -np.inf
    return scores


def select(
    X: np.ndarray,
    budget: int,
    guidance: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    config: dict | None = None,
    seed: int = 0,
) -> np.ndarray:
    del guidance, groups, seed
    X_arr = _validate_candidates(X)
    cfg = config or {}
    n = X_arr.shape[0]
    if budget <= 0 or n == 0:
        return np.empty(0, dtype=np.int64)
    if budget > n:
        raise ValueError(f"Budget {budget} exceeds pool size {n}.")

    mean_feature = _initialize_mean_feature(X_arr, cfg)
    running_sum = np.zeros(X_arr.shape[1], dtype=np.float64)
    selected = np.empty(budget, dtype=np.int64)
    chosen = np.zeros(n, dtype=bool)

    for step in range(budget):
        scores = _score_candidates(
            X=X_arr,
            mean_feature=mean_feature,
            running_sum=running_sum,
            step=step,
            chosen=chosen,
        )
        idx = int(np.argmax(scores))
        selected[step] = idx
        chosen[idx] = True
        running_sum += X_arr[idx]

    return selected
