from __future__ import annotations

import numpy as np


NAME = "infomax"
DEFAULT_PAIRWISE_WEIGHT = 0.3
DEFAULT_SPARSE_RATE = 5
DEFAULT_NUM_ITERATIONS = 20
DEFAULT_PARTITIONS = 1
DEFAULT_EPS = 1e-12
DEFAULT_KMEANS_ITERS = 25
DEFAULT_KMEANS_RESTARTS = 3
DEFAULT_BLOCK_SIZE = 256
DEFAULT_NUM_CLUSTERS = 64


def _validate_candidates(X: np.ndarray) -> np.ndarray:
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim != 2:
        raise ValueError(f"InfoMax expects a 2D candidate array, got shape {X_arr.shape}.")
    if not np.all(np.isfinite(X_arr)):
        raise ValueError("InfoMax requires all candidate features to be finite.")
    return X_arr


def _l2_normalize_rows(X: np.ndarray, eps: float = DEFAULT_EPS) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(norms, eps, None)


def _minmax_scale(scores: np.ndarray, eps: float = DEFAULT_EPS) -> np.ndarray:
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    if max_score - min_score <= eps:
        return np.ones(scores.shape[0], dtype=np.float64)
    return (scores - min_score) / (max_score - min_score)


def _orient_scores(scores: np.ndarray, higher_is_better: bool) -> np.ndarray:
    if higher_is_better:
        return scores.astype(np.float64, copy=True)
    return float(np.max(scores)) - scores


def _resolve_scores(X: np.ndarray, budget: int, config: dict, seed: int) -> np.ndarray:
    scores = config.get("scores")
    if scores is not None:
        scores_arr = np.asarray(scores, dtype=np.float64).reshape(-1)
        if scores_arr.shape[0] != X.shape[0]:
            raise ValueError(
                f"InfoMax score length mismatch: expected {X.shape[0]}, got {scores_arr.shape[0]}."
            )
        if not np.all(np.isfinite(scores_arr)):
            raise ValueError("InfoMax requires all intra-sample information scores to be finite.")
        return scores_arr
    return _compute_ssp_scores(X=X, budget=budget, config=config, seed=seed)


def _kmeans_pp_init(X: np.ndarray, num_clusters: int, rng: np.random.Generator) -> np.ndarray:
    n = X.shape[0]
    centers = np.empty((num_clusters, X.shape[1]), dtype=np.float64)
    first_idx = int(rng.integers(n))
    centers[0] = X[first_idx]
    closest_d2 = np.sum((X - centers[0]) ** 2, axis=1)

    for center_idx in range(1, num_clusters):
        total = float(np.sum(closest_d2))
        if not np.isfinite(total) or total <= DEFAULT_EPS:
            backup_idx = np.flatnonzero(np.all(np.isfinite(X), axis=1))[center_idx % n]
            centers[center_idx] = X[backup_idx]
            continue
        probs = closest_d2 / total
        next_idx = int(rng.choice(n, p=probs))
        centers[center_idx] = X[next_idx]
        d2 = np.sum((X - centers[center_idx]) ** 2, axis=1)
        closest_d2 = np.minimum(closest_d2, d2)
    return centers


def _run_kmeans(
    X: np.ndarray,
    num_clusters: int,
    max_iters: int,
    num_restarts: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    best_inertia = np.inf
    best_labels: np.ndarray | None = None
    best_centers: np.ndarray | None = None

    for restart in range(max(1, num_restarts)):
        centers = _kmeans_pp_init(X, num_clusters=num_clusters, rng=np.random.default_rng(rng.integers(1 << 32)))
        labels = np.zeros(X.shape[0], dtype=np.int64)

        for _ in range(max(1, max_iters)):
            d2 = (
                np.sum(X * X, axis=1, keepdims=True)
                + np.sum(centers * centers, axis=1)[None, :]
                - 2.0 * (X @ centers.T)
            )
            np.maximum(d2, 0.0, out=d2)
            new_labels = np.argmin(d2, axis=1)
            if np.array_equal(new_labels, labels):
                labels = new_labels
                break
            labels = new_labels

            new_centers = centers.copy()
            for cluster_idx in range(num_clusters):
                mask = labels == cluster_idx
                if np.any(mask):
                    new_centers[cluster_idx] = np.mean(X[mask], axis=0)
                else:
                    new_centers[cluster_idx] = X[int(rng.integers(X.shape[0]))]
            centers = new_centers

        final_d2 = (
            np.sum(X * X, axis=1, keepdims=True)
            + np.sum(centers * centers, axis=1)[None, :]
            - 2.0 * (X @ centers.T)
        )
        np.maximum(final_d2, 0.0, out=final_d2)
        chosen_d2 = final_d2[np.arange(X.shape[0]), labels]
        inertia = float(np.sum(chosen_d2))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()

    if best_labels is None or best_centers is None:
        raise RuntimeError("InfoMax failed to compute SSP scores.")
    return best_labels, best_centers


def _compute_ssp_scores(
    X: np.ndarray,
    budget: int,
    config: dict,
    seed: int,
) -> np.ndarray:
    n = X.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64)

    num_clusters = int(
        config.get(
            "num_clusters",
            max(2, min(n, min(max(8, int(np.sqrt(max(n, 1)))), max(budget, DEFAULT_NUM_CLUSTERS)))),
        )
    )
    num_clusters = max(1, min(num_clusters, n))
    labels, centers = _run_kmeans(
        X=X,
        num_clusters=num_clusters,
        max_iters=int(config.get("kmeans_max_iters", DEFAULT_KMEANS_ITERS)),
        num_restarts=int(config.get("kmeans_restarts", DEFAULT_KMEANS_RESTARTS)),
        seed=seed,
    )
    deltas = X - centers[labels]
    return np.linalg.norm(deltas, axis=1)


def _build_sparse_similarity(
    X: np.ndarray,
    sparse_rate: int,
    *,
    nonnegative: bool,
    block_size: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    n = X.shape[0]
    if n == 0 or sparse_rate <= 0:
        empty_idx = [np.empty(0, dtype=np.int64) for _ in range(n)]
        empty_val = [np.empty(0, dtype=np.float64) for _ in range(n)]
        return empty_idx, empty_val

    k_eff = min(int(sparse_rate), max(0, n - 1))
    if k_eff == 0:
        empty_idx = [np.empty(0, dtype=np.int64) for _ in range(n)]
        empty_val = [np.empty(0, dtype=np.float64) for _ in range(n)]
        return empty_idx, empty_val

    block = max(1, int(block_size))
    all_indices: list[np.ndarray] = []
    all_values: list[np.ndarray] = []

    for start in range(0, n, block):
        end = min(start + block, n)
        sims = X[start:end] @ X.T
        rows = np.arange(end - start)
        sims[rows, np.arange(start, end)] = -np.inf
        if nonnegative:
            sims[sims < 0.0] = -np.inf

        part = np.argpartition(-sims, kth=k_eff - 1, axis=1)[:, :k_eff]
        part_vals = np.take_along_axis(sims, part, axis=1)
        for row in range(end - start):
            order = np.lexsort((part[row], -part_vals[row]))
            idx = part[row, order]
            vals = part_vals[row, order]
            keep = np.isfinite(vals)
            all_indices.append(idx[keep].astype(np.int64, copy=False))
            all_values.append(vals[keep].astype(np.float64, copy=False))
    return all_indices, all_values


def _sparse_matvec(
    neighbor_idx: list[np.ndarray],
    neighbor_val: list[np.ndarray],
    x: np.ndarray,
) -> np.ndarray:
    result = np.zeros(x.shape[0], dtype=np.float64)
    for i, idx in enumerate(neighbor_idx):
        if idx.size == 0:
            continue
        result[i] = float(neighbor_val[i] @ x[idx])
    return result


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp_logits = np.exp(shifted)
    total = float(np.sum(exp_logits))
    if not np.isfinite(total) or total <= DEFAULT_EPS:
        return np.full(logits.shape[0], 1.0 / max(1, logits.shape[0]), dtype=np.float64)
    return exp_logits / total


def _topk_indices(values: np.ndarray, budget: int) -> np.ndarray:
    indices = np.arange(values.shape[0], dtype=np.int64)
    order = np.lexsort((indices, -values))
    return np.sort(order[:budget].astype(np.int64, copy=False))


def _allocate_partition_budgets(lengths: list[int], budget: int) -> list[int]:
    total = int(sum(lengths))
    if total <= 0:
        return [0 for _ in lengths]
    raw = np.asarray([budget * length / total for length in lengths], dtype=np.float64)
    floors = np.floor(raw).astype(int)
    remainder = int(budget - np.sum(floors))
    if remainder > 0:
        frac = raw - floors
        order = np.lexsort((np.arange(len(lengths)), -frac))
        for idx in order[:remainder]:
            floors[int(idx)] += 1
    return floors.tolist()


def _partition_indices(n: int, partitions: int, seed: int) -> list[np.ndarray]:
    if partitions <= 1 or n == 0:
        return [np.arange(n, dtype=np.int64)]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return [chunk.astype(np.int64, copy=False) for chunk in np.array_split(perm, partitions) if chunk.size > 0]


def _solve_subset(
    X: np.ndarray,
    budget: int,
    *,
    scores: np.ndarray | None,
    config: dict,
    seed: int,
) -> np.ndarray:
    n = X.shape[0]
    if budget <= 0 or n == 0:
        return np.empty(0, dtype=np.int64)
    if budget > n:
        raise ValueError(f"InfoMax subset budget {budget} exceeds subset size {n}.")

    normalize_features = bool(config.get("normalize_features", True))
    features = _l2_normalize_rows(X) if normalize_features else X.astype(np.float64, copy=False)
    intra_scores = _resolve_scores(features, budget=budget, config={**config, "scores": scores} if scores is not None else config, seed=seed)
    intra_scores = _orient_scores(intra_scores, higher_is_better=bool(config.get("higher_is_better", True)))
    normalize_scores = bool(config.get("normalize_scores", True))
    info = _minmax_scale(intra_scores) if normalize_scores else intra_scores.astype(np.float64, copy=True)

    sparse_rate = int(config.get("sparse_rate", DEFAULT_SPARSE_RATE))
    pairwise_weight = float(config.get("pairwise_weight", DEFAULT_PAIRWISE_WEIGHT))
    num_iterations = int(config.get("num_iterations", DEFAULT_NUM_ITERATIONS))
    block_size = int(config.get("block_size", DEFAULT_BLOCK_SIZE))
    if sparse_rate < 0:
        raise ValueError(f"InfoMax sparse_rate must be non-negative, got {sparse_rate}.")
    if pairwise_weight < 0.0:
        raise ValueError(f"InfoMax pairwise_weight must be non-negative, got {pairwise_weight}.")
    if num_iterations <= 0:
        raise ValueError(f"InfoMax num_iterations must be positive, got {num_iterations}.")
    if block_size <= 0:
        raise ValueError(f"InfoMax block_size must be positive, got {block_size}.")

    neighbor_idx, neighbor_val = _build_sparse_similarity(
        features,
        sparse_rate=sparse_rate,
        nonnegative=bool(config.get("nonnegative_similarity", True)),
        block_size=block_size,
    )

    x = np.full(n, 1.0 / n, dtype=np.float64)
    for _ in range(num_iterations):
        redundancy = _sparse_matvec(neighbor_idx, neighbor_val, x)
        logits = budget * info - 2.0 * budget * pairwise_weight * redundancy
        x = _softmax(logits)

    return _topk_indices(x, budget=budget)


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
    n = X_arr.shape[0]
    if budget <= 0 or n == 0:
        return np.empty(0, dtype=np.int64)
    if budget > n:
        raise ValueError(f"Budget {budget} exceeds pool size {n}.")

    cfg = config or {}
    scores = cfg.get("scores")
    if scores is not None:
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if scores.shape[0] != n:
            raise ValueError(f"InfoMax score length mismatch: expected {n}, got {scores.shape[0]}.")

    partitions = int(cfg.get("partitions", DEFAULT_PARTITIONS))
    if partitions <= 1:
        return _solve_subset(X_arr, budget, scores=scores, config=cfg, seed=seed)

    chunks = _partition_indices(n, partitions=partitions, seed=seed)
    chunk_lengths = [int(chunk.size) for chunk in chunks]
    chunk_budgets = _allocate_partition_budgets(chunk_lengths, budget=budget)
    selected_parts: list[np.ndarray] = []

    for part_id, (chunk, part_budget) in enumerate(zip(chunks, chunk_budgets)):
        if part_budget <= 0:
            continue
        part_scores = None if scores is None else scores[chunk]
        local = _solve_subset(
            X_arr[chunk],
            budget=part_budget,
            scores=part_scores,
            config=cfg,
            seed=seed + part_id + 1,
        )
        selected_parts.append(chunk[local])

    if not selected_parts:
        return np.empty(0, dtype=np.int64)
    selected = np.concatenate(selected_parts).astype(np.int64, copy=False)
    if selected.size != budget:
        raise RuntimeError(f"InfoMax selected {selected.size} examples, expected {budget}.")
    return np.sort(selected)
