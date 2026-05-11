from __future__ import annotations

import numpy as np


NAME = "d2pruning"
DEFAULT_K_NEIGHBORS = 10
DEFAULT_FORWARD_WEIGHT = 1.0
DEFAULT_REVERSE_WEIGHT = 0.3
DEFAULT_FORWARD_STEPS = 1
DEFAULT_GRAPH_BLOCK_SIZE = 256


def _validate_inputs(X: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X_arr = np.asarray(X, dtype=np.float64)
    scores_arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    if X_arr.ndim != 2:
        raise ValueError(f"D2Pruning expects a 2D candidate array, got shape {X_arr.shape}.")
    if X_arr.shape[0] != scores_arr.shape[0]:
        raise ValueError(
            f"D2Pruning score length mismatch: expected {X_arr.shape[0]}, got {scores_arr.shape[0]}."
        )
    if not np.all(np.isfinite(X_arr)):
        raise ValueError("D2Pruning requires all candidate features to be finite.")
    if not np.all(np.isfinite(scores_arr)):
        raise ValueError("D2Pruning requires all difficulty scores to be finite.")
    return X_arr, scores_arr


def _resolve_difficulty_scores(X: np.ndarray, config: dict) -> np.ndarray:
    scores = config.get("scores")
    if scores is None:
        if bool(config.get("self_supervised", False)):
            return np.ones(X.shape[0], dtype=np.float64)
        raise ValueError(
            "D2Pruning initializes graph node features with paper-defined difficulty "
            "scores. Provide config['scores'] from training dynamics, or set "
            "config['self_supervised']=True for the paper's uniform-score "
            "self-supervised mode."
        )
    return np.asarray(scores, dtype=np.float64).reshape(-1)


def _orient_scores(scores: np.ndarray, higher_is_harder: bool) -> np.ndarray:
    if higher_is_harder:
        difficulty = scores.astype(np.float64, copy=True)
        min_score = float(np.min(difficulty))
        if min_score < 0.0:
            difficulty -= min_score
        return difficulty
    return float(np.max(scores)) - scores


def _exact_knn(
    X: np.ndarray,
    k_neighbors: int,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    if n <= 1 or k_neighbors <= 0:
        return np.empty((n, 0), dtype=np.int64), np.empty((n, 0), dtype=np.float64)

    k_eff = min(int(k_neighbors), n - 1)
    block = max(1, int(block_size))
    norms = np.sum(X * X, axis=1)
    neighbor_idx = np.empty((n, k_eff), dtype=np.int64)
    neighbor_d2 = np.empty((n, k_eff), dtype=np.float64)

    for start in range(0, n, block):
        end = min(start + block, n)
        block_X = X[start:end]
        d2 = norms[start:end, None] + norms[None, :] - 2.0 * (block_X @ X.T)
        np.maximum(d2, 0.0, out=d2)
        local_rows = np.arange(end - start)
        d2[local_rows, np.arange(start, end)] = np.inf

        part = np.argpartition(d2, kth=k_eff - 1, axis=1)[:, :k_eff]
        part_d2 = np.take_along_axis(d2, part, axis=1)
        for row in range(end - start):
            order = np.lexsort((part[row], part_d2[row]))
            neighbor_idx[start + row] = part[row, order]
            neighbor_d2[start + row] = part_d2[row, order]

    return neighbor_idx, neighbor_d2


def _build_adjacency(
    neighbor_idx: np.ndarray,
    neighbor_d2: np.ndarray,
    symmetrize_graph: bool,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    n = neighbor_idx.shape[0]
    if neighbor_idx.shape[1] == 0:
        empty_idx = [np.empty(0, dtype=np.int64) for _ in range(n)]
        empty_d2 = [np.empty(0, dtype=np.float64) for _ in range(n)]
        return empty_idx, empty_d2

    if not symmetrize_graph:
        return [
            np.asarray(neighbor_idx[i], dtype=np.int64) for i in range(n)
        ], [
            np.asarray(neighbor_d2[i], dtype=np.float64) for i in range(n)
        ]

    adjacency: list[dict[int, float]] = [dict() for _ in range(n)]
    for i in range(n):
        for j, d2 in zip(neighbor_idx[i], neighbor_d2[i]):
            adjacency[i][int(j)] = float(d2)
            adjacency[int(j)][i] = float(d2)

    neighbor_lists: list[np.ndarray] = []
    distance_lists: list[np.ndarray] = []
    for i in range(n):
        if not adjacency[i]:
            neighbor_lists.append(np.empty(0, dtype=np.int64))
            distance_lists.append(np.empty(0, dtype=np.float64))
            continue
        idx = np.fromiter(adjacency[i].keys(), dtype=np.int64)
        d2 = np.fromiter((adjacency[i][int(j)] for j in idx), dtype=np.float64)
        order = np.lexsort((idx, d2))
        neighbor_lists.append(idx[order])
        distance_lists.append(d2[order])
    return neighbor_lists, distance_lists


def _compute_edge_weights(
    distances: list[np.ndarray],
    coefficient: float,
) -> list[np.ndarray]:
    weights: list[np.ndarray] = []
    for d2 in distances:
        if d2.size == 0:
            weights.append(d2)
            continue
        weights.append(np.exp(-coefficient * d2).astype(np.float64, copy=False))
    return weights


def _forward_message_passing(
    node_features: np.ndarray,
    neighbor_lists: list[np.ndarray],
    forward_weights: list[np.ndarray],
    num_steps: int,
) -> np.ndarray:
    current = node_features.astype(np.float64, copy=True)
    for _ in range(num_steps):
        updated = current.copy()
        for i, nbr_idx in enumerate(neighbor_lists):
            if nbr_idx.size == 0:
                continue
            updated[i] = current[i] + float(forward_weights[i] @ current[nbr_idx])
        current = updated
    return current


def _reverse_select(
    node_features: np.ndarray,
    neighbor_lists: list[np.ndarray],
    reverse_weights: list[np.ndarray],
    budget: int,
) -> np.ndarray:
    n = node_features.shape[0]
    current = node_features.astype(np.float64, copy=True)
    available = np.ones(n, dtype=bool)
    selected = np.empty(budget, dtype=np.int64)

    for step in range(budget):
        masked = current.copy()
        masked[~available] = -np.inf
        idx = int(np.argmax(masked))
        if not np.isfinite(masked[idx]):
            raise RuntimeError("D2Pruning exhausted all selectable samples before reaching the budget.")

        selected[step] = idx
        available[idx] = False
        selected_score = current[idx]
        nbr_idx = neighbor_lists[idx]
        if nbr_idx.size:
            nbr_mask = available[nbr_idx]
            if np.any(nbr_mask):
                current[nbr_idx[nbr_mask]] -= reverse_weights[idx][nbr_mask] * selected_score
        current[idx] = -np.inf

    return selected


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
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim != 2:
        raise ValueError(f"D2Pruning expects a 2D candidate array, got shape {X_arr.shape}.")
    scores = _resolve_difficulty_scores(X_arr, cfg)
    X_arr, scores_arr = _validate_inputs(X_arr, scores)
    n = X_arr.shape[0]
    if budget <= 0 or n == 0:
        return np.empty(0, dtype=np.int64)
    if budget > n:
        raise ValueError(f"Budget {budget} exceeds pool size {n}.")

    higher_is_harder = bool(cfg.get("higher_is_harder", True))
    difficulty = _orient_scores(scores_arr, higher_is_harder=higher_is_harder)

    k_neighbors = int(cfg.get("k_neighbors", DEFAULT_K_NEIGHBORS))
    if k_neighbors < 0:
        raise ValueError(f"D2Pruning k_neighbors must be non-negative, got {k_neighbors}.")

    forward_weight = float(cfg.get("forward_weight", DEFAULT_FORWARD_WEIGHT))
    reverse_weight = float(cfg.get("reverse_weight", DEFAULT_REVERSE_WEIGHT))
    if forward_weight < 0.0:
        raise ValueError(f"D2Pruning forward weight must be non-negative, got {forward_weight}.")
    if reverse_weight < 0.0:
        raise ValueError(f"D2Pruning reverse weight must be non-negative, got {reverse_weight}.")

    num_forward_steps = int(cfg.get("num_forward_steps", DEFAULT_FORWARD_STEPS))
    if num_forward_steps <= 0:
        raise ValueError(
            f"D2Pruning num_forward_steps must be positive, got {num_forward_steps}."
        )

    block_size = int(cfg.get("graph_block_size", DEFAULT_GRAPH_BLOCK_SIZE))
    if block_size <= 0:
        raise ValueError(f"D2Pruning graph_block_size must be positive, got {block_size}.")
    symmetrize_graph = bool(cfg.get("symmetrize_graph", True))

    neighbor_idx, neighbor_d2 = _exact_knn(X_arr, k_neighbors=k_neighbors, block_size=block_size)
    neighbor_lists, distance_lists = _build_adjacency(
        neighbor_idx=neighbor_idx,
        neighbor_d2=neighbor_d2,
        symmetrize_graph=symmetrize_graph,
    )
    forward_weights = _compute_edge_weights(distance_lists, coefficient=forward_weight)
    reverse_weights = _compute_edge_weights(distance_lists, coefficient=reverse_weight)
    updated_scores = _forward_message_passing(
        node_features=difficulty,
        neighbor_lists=neighbor_lists,
        forward_weights=forward_weights,
        num_steps=num_forward_steps,
    )
    return _reverse_select(
        node_features=updated_scores,
        neighbor_lists=neighbor_lists,
        reverse_weights=reverse_weights,
        budget=budget,
    )
