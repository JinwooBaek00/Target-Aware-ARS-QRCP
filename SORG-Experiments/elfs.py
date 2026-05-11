from __future__ import annotations

import numpy as np


NAME = "elfs"
DEFAULT_BETA = None
DEFAULT_HIGHER_IS_HARDER = False
DEFAULT_BETA_SCHEDULE = {
    0.3: 0.1,
    0.5: 0.2,
    0.7: 0.4,
    0.8: 0.5,
    0.9: 0.6,
}


def _validate_inputs(X: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X_arr = np.asarray(X, dtype=np.float64)
    scores_arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    if X_arr.ndim != 2:
        raise ValueError(f"ELFS expects a 2D candidate array, got shape {X_arr.shape}.")
    if X_arr.shape[0] != scores_arr.shape[0]:
        raise ValueError(
            f"ELFS score length mismatch: expected {X_arr.shape[0]}, got {scores_arr.shape[0]}."
        )
    if not np.all(np.isfinite(scores_arr)):
        raise ValueError("ELFS requires all proxy difficulty scores to be finite.")
    return X_arr, scores_arr


def _resolve_scores(X: np.ndarray, config: dict) -> np.ndarray:
    scores = config.get("scores")
    if scores is None:
        raise ValueError(
            "ELFS requires pseudo-label-based training-dynamics scores. Provide "
            "config['scores'] computed from the ELFS pseudo-label pipeline."
        )
    return np.asarray(scores, dtype=np.float64).reshape(-1)


def _resolve_beta(
    n: int,
    budget: int,
    config: dict,
) -> float:
    beta = config.get("beta", DEFAULT_BETA)
    if beta is not None:
        beta_value = float(beta)
        if not 0.0 <= beta_value <= 1.0:
            raise ValueError(f"ELFS beta must lie in [0, 1], got {beta_value}.")
        return beta_value

    pruning_rate = 1.0 - (float(budget) / float(n))
    schedule = config.get("beta_schedule", DEFAULT_BETA_SCHEDULE)
    schedule_map = {round(float(key), 3): float(value) for key, value in dict(schedule).items()}
    rounded_rate = round(pruning_rate, 3)
    if rounded_rate in schedule_map:
        return schedule_map[rounded_rate]

    heuristic_beta = max(0.0, min(0.7, round(pruning_rate - 0.3, 1)))
    return heuristic_beta


def _hardness_order(scores: np.ndarray, higher_is_harder: bool) -> np.ndarray:
    indices = np.arange(scores.shape[0], dtype=np.int64)
    if higher_is_harder:
        return np.lexsort((indices, -scores))
    return np.lexsort((indices, scores))


def select(
    X: np.ndarray,
    budget: int,
    guidance: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    config: dict | None = None,
    seed: int = 0,
) -> np.ndarray:
    del guidance, groups, seed
    cfg = config or {}
    scores = _resolve_scores(X, cfg)
    X_arr, scores_arr = _validate_inputs(X, scores)

    n = X_arr.shape[0]
    if budget <= 0 or n == 0:
        return np.empty(0, dtype=np.int64)
    if budget > n:
        raise ValueError(f"Budget {budget} exceeds pool size {n}.")

    higher_is_harder = bool(cfg.get("higher_is_harder", DEFAULT_HIGHER_IS_HARDER))
    beta = _resolve_beta(n=n, budget=budget, config=cfg)
    hard_prune_count = int(np.floor(n * beta))
    if hard_prune_count >= n:
        raise ValueError(
            f"ELFS hard pruning removes the full pool: floor(n * beta)={hard_prune_count}, n={n}."
        )
    if n - hard_prune_count < budget:
        raise ValueError(
            "ELFS hard pruning leaves too few examples for the requested budget. "
            f"Remaining pool {n - hard_prune_count}, budget {budget}."
        )

    order = _hardness_order(scores_arr, higher_is_harder=higher_is_harder)
    selected = order[hard_prune_count : hard_prune_count + budget]
    if selected.size != budget:
        raise RuntimeError(f"ELFS selected {selected.size} examples, expected {budget}.")
    return np.sort(selected.astype(np.int64, copy=False))

