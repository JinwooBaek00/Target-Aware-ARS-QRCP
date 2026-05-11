from __future__ import annotations

import numpy as np


NAME = "ccs"
DEFAULT_NUM_STRATA = 50
DEFAULT_BETA = 0.0


def _validate_inputs(X: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X_arr = np.asarray(X, dtype=np.float64)
    scores_arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    if X_arr.ndim != 2:
        raise ValueError(f"CCS expects a 2D candidate array, got shape {X_arr.shape}.")
    if not np.all(np.isfinite(X_arr)):
        raise ValueError("CCS requires all candidate features to be finite.")
    if X_arr.shape[0] != scores_arr.shape[0]:
        raise ValueError(
            f"CCS score length mismatch: expected {X_arr.shape[0]}, got {scores_arr.shape[0]}."
        )
    if not np.all(np.isfinite(scores_arr)):
        raise ValueError("CCS requires all importance scores to be finite.")
    return X_arr, scores_arr


def _hard_prune_indices(
    scores: np.ndarray,
    hard_prune_count: int,
    higher_is_harder: bool,
) -> np.ndarray:
    order = np.argsort(scores, kind="stable")
    if hard_prune_count <= 0:
        return np.empty(0, dtype=np.int64)
    if higher_is_harder:
        return np.sort(order[-hard_prune_count:])
    return np.sort(order[:hard_prune_count])


def _build_strata(indices: np.ndarray, scores: np.ndarray, num_strata: int) -> list[np.ndarray]:
    if indices.size == 0:
        return []
    if num_strata <= 1:
        return [indices]

    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    if max_score <= min_score:
        return [indices]

    edges = np.linspace(min_score, max_score, num_strata + 1)
    bin_ids = np.searchsorted(edges[1:-1], scores, side="right")
    return [indices[bin_ids == bucket] for bucket in range(num_strata) if np.any(bin_ids == bucket)]


def _stratified_sample(
    strata: list[np.ndarray],
    budget: int,
    rng: np.random.Generator,
) -> np.ndarray:
    remaining_budget = int(budget)
    active = [stratum.copy() for stratum in strata if stratum.size > 0]
    selected: list[int] = []

    while active:
        sizes = np.asarray([stratum.size for stratum in active], dtype=np.int64)
        smallest_pos = int(np.argmin(sizes))
        stratum = active.pop(smallest_pos)

        remaining_strata = len(active) + 1
        stratum_budget = min(int(stratum.size), remaining_budget // remaining_strata)
        if stratum_budget > 0:
            if stratum_budget == stratum.size:
                sampled = stratum
            else:
                sampled = np.sort(rng.choice(stratum, size=stratum_budget, replace=False))
            selected.extend(sampled.tolist())
            remaining_budget -= int(stratum_budget)

        if remaining_budget <= 0:
            break

    return np.asarray(selected, dtype=np.int64)


def select(
    X: np.ndarray,
    budget: int,
    guidance: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    config: dict | None = None,
    seed: int = 0,
) -> np.ndarray:
    del guidance, groups
    cfg = config or {}
    scores = cfg.get("scores")
    if scores is None:
        raise ValueError(
            "CCS implements coverage-centric stratified sampling over paper-defined "
            "importance scores. Provide config['scores'] from training dynamics "
            "(for example AUM, forgetting, EL2N, entropy, or pseudo-label scores)."
        )

    X_arr, scores_arr = _validate_inputs(X, scores)
    n = X_arr.shape[0]
    if budget <= 0 or n == 0:
        return np.empty(0, dtype=np.int64)
    if budget > n:
        raise ValueError(f"Budget {budget} exceeds pool size {n}.")

    beta = float(cfg.get("beta", DEFAULT_BETA))
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"CCS beta must lie in [0, 1], got {beta}.")
    num_strata = int(cfg.get("num_strata", DEFAULT_NUM_STRATA))
    if num_strata <= 0:
        raise ValueError(f"CCS num_strata must be positive, got {num_strata}.")
    higher_is_harder = bool(cfg.get("higher_is_harder", True))

    hard_prune_count = int(np.floor(n * beta))
    if hard_prune_count > budget:
        raise ValueError(
            "CCS requires floor(n * beta) <= budget, matching Algorithm 1's hard-cutoff setting. "
            f"Got floor(n * beta)={hard_prune_count}, budget={budget}."
        )
    if n - hard_prune_count < budget:
        raise ValueError(
            "CCS hard cutoff removes too many examples for the requested budget. "
            f"Remaining pool {n - hard_prune_count}, budget {budget}."
        )

    prune_idx = _hard_prune_indices(scores_arr, hard_prune_count, higher_is_harder)
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[prune_idx] = False
    retained_idx = np.flatnonzero(keep_mask)
    retained_scores = scores_arr[retained_idx]
    strata = _build_strata(retained_idx, retained_scores, num_strata=num_strata)

    rng = np.random.default_rng(seed)
    selected = _stratified_sample(strata, budget=budget, rng=rng)
    if selected.size != budget:
        raise RuntimeError(
            f"CCS selected {selected.size} examples, expected {budget}. "
            "Check score bins and hard cutoff settings."
        )
    return np.sort(selected)
