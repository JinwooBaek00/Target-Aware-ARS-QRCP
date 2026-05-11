from __future__ import annotations

import numpy as np


NAME = "drcs"
DEFAULT_SHIFT_RADIUS = 0.5
DEFAULT_MAX_ITERS = 200
DEFAULT_STEP_SIZE = 0.5
DEFAULT_TOL = 1e-10
DEFAULT_EPS = 1e-12


def _validate_candidates(X: np.ndarray) -> np.ndarray:
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim != 2:
        raise ValueError(f"DRCS expects a 2D candidate array, got shape {X_arr.shape}.")
    if not np.all(np.isfinite(X_arr)):
        raise ValueError("DRCS requires all candidate features to be finite.")
    return X_arr


def _validate_quadratic_inputs(n: int, config: dict) -> tuple[np.ndarray, np.ndarray, float]:
    if "A" not in config or "b" not in config:
        raise ValueError(
            "DRCS implements the paper's greedy selection over the quadratic upper-bound "
            "inputs A, b, c. Provide config['A'] and config['b'] computed from the DRCS "
            "duality-gap bound."
        )

    A = np.asarray(config["A"], dtype=np.float64)
    b = np.asarray(config["b"], dtype=np.float64).reshape(-1)
    c = float(config.get("c", 0.0))
    if A.shape != (n, n):
        raise ValueError(f"DRCS expects A to have shape {(n, n)}, got {A.shape}.")
    if b.shape[0] != n:
        raise ValueError(f"DRCS expects b to have length {n}, got {b.shape[0]}.")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)) or not np.isfinite(c):
        raise ValueError("DRCS requires finite A, b, and c.")
    return 0.5 * (A + A.T), b, c


def _project_to_ball(w: np.ndarray, radius: float, nonnegative: bool) -> np.ndarray:
    center = np.ones_like(w)
    out = np.maximum(w, 0.0) if nonnegative else w.astype(np.float64, copy=True)
    diff = out - center
    norm = float(np.linalg.norm(diff))
    if norm > radius > 0.0:
        out = center + (radius / norm) * diff
        if nonnegative:
            out = np.maximum(out, 0.0)
    return out


def _maximize_on_ball_exact(A: np.ndarray, b: np.ndarray, radius: float) -> np.ndarray:
    n = b.shape[0]
    center = np.ones(n, dtype=np.float64)
    if n == 0 or radius <= 0.0:
        return center

    eigvals, eigvecs = np.linalg.eigh(A)
    lambda_max = float(eigvals[-1])
    g = A @ center + 0.5 * b
    g_hat = eigvecs.T @ g
    g_norm = float(np.linalg.norm(g_hat))
    if g_norm <= DEFAULT_EPS:
        return center + radius * eigvecs[:, -1]

    top = np.isclose(eigvals, lambda_max, rtol=1e-10, atol=1e-12)
    if float(np.linalg.norm(g_hat[top])) <= DEFAULT_EPS:
        y_hat = np.zeros(n, dtype=np.float64)
        non_top = ~top
        denom = lambda_max - eigvals[non_top]
        y_hat[non_top] = g_hat[non_top] / np.clip(denom, DEFAULT_EPS, None)
        norm_y = float(np.linalg.norm(y_hat))
        if norm_y <= radius:
            spare = max(0.0, radius * radius - norm_y * norm_y)
            first_top = int(np.flatnonzero(top)[0])
            y_hat[first_top] = np.sqrt(spare)
            return center + eigvecs @ y_hat

    def secular_norm(mu: float) -> float:
        denom = np.clip(mu - eigvals, DEFAULT_EPS, None)
        return float(np.linalg.norm(g_hat / denom))

    low = lambda_max + DEFAULT_EPS
    high = lambda_max + max(1.0, abs(lambda_max), g_norm / max(radius, DEFAULT_EPS))
    while secular_norm(high) > radius:
        high = lambda_max + 2.0 * (high - lambda_max)

    for _ in range(100):
        mid = 0.5 * (low + high)
        if secular_norm(mid) > radius:
            low = mid
        else:
            high = mid

    mu = high
    y_hat = g_hat / np.clip(mu - eigvals, DEFAULT_EPS, None)
    return center + eigvecs @ y_hat


def _maximize_on_ball_projected(
    A: np.ndarray,
    b: np.ndarray,
    radius: float,
    *,
    max_iters: int,
    step_size: float,
    tol: float,
    nonnegative: bool,
) -> np.ndarray:
    w = np.ones(b.shape[0], dtype=np.float64)
    previous = float(w @ (A @ w) + b @ w)
    for step in range(max(1, max_iters)):
        grad = 2.0 * (A @ w) + b
        candidate = _project_to_ball(
            w + (step_size / np.sqrt(step + 1.0)) * grad,
            radius=radius,
            nonnegative=nonnegative,
        )
        current = float(candidate @ (A @ candidate) + b @ candidate)
        if abs(current - previous) <= tol:
            return candidate
        w = candidate
        previous = current
    return w


def _solve_worst_weights(
    A: np.ndarray,
    b: np.ndarray,
    radius: float,
    active: np.ndarray,
    config: dict,
) -> np.ndarray:
    active_idx = np.flatnonzero(active)
    weights = np.ones(b.shape[0], dtype=np.float64)
    if active_idx.size == 0:
        return weights

    A_sub = A[np.ix_(active_idx, active_idx)]
    b_sub = b[active_idx]
    nonnegative = bool(config.get("nonnegative_weights", False))
    if nonnegative:
        weights[active_idx] = _maximize_on_ball_projected(
            A=A_sub,
            b=b_sub,
            radius=radius,
            max_iters=int(config.get("max_iters", DEFAULT_MAX_ITERS)),
            step_size=float(config.get("step_size", DEFAULT_STEP_SIZE)),
            tol=float(config.get("tol", DEFAULT_TOL)),
            nonnegative=True,
        )
    else:
        weights[active_idx] = _maximize_on_ball_exact(A=A_sub, b=b_sub, radius=radius)
    return weights


def _objective(A: np.ndarray, b: np.ndarray, c: float, z: np.ndarray) -> float:
    return float(z @ (A @ z) + b @ z + c)


def _deletion_objectives(
    A: np.ndarray,
    b: np.ndarray,
    c: float,
    z: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    base = _objective(A, b, c, z)
    Az = A @ z
    scores = np.full(z.shape[0], np.inf, dtype=np.float64)
    idx = np.flatnonzero(active)
    scores[idx] = base - 2.0 * z[idx] * Az[idx] + np.diag(A)[idx] * (z[idx] ** 2) - b[idx] * z[idx]
    return scores


def _approach_1(A: np.ndarray, b: np.ndarray, c: float, budget: int, radius: float, config: dict) -> np.ndarray:
    n = b.shape[0]
    active = np.ones(n, dtype=bool)
    for _ in range(n - budget):
        scores = np.full(n, np.inf, dtype=np.float64)
        for i in np.flatnonzero(active):
            candidate_active = active.copy()
            candidate_active[i] = False
            w = _solve_worst_weights(A, b, radius, candidate_active, config)
            z = w * candidate_active.astype(np.float64)
            scores[i] = _objective(A, b, c, z)
        active[int(np.argmin(scores))] = False
    return np.flatnonzero(active).astype(np.int64, copy=False)


def _approach_2(A: np.ndarray, b: np.ndarray, c: float, budget: int, radius: float, config: dict) -> np.ndarray:
    n = b.shape[0]
    active = np.ones(n, dtype=bool)
    w = _solve_worst_weights(A, b, radius, active, config)
    z = w.copy()
    for _ in range(n - budget):
        scores = _deletion_objectives(A, b, c, z, active)
        delete_idx = int(np.argmin(scores))
        active[delete_idx] = False
        z[delete_idx] = 0.0
    return np.flatnonzero(active).astype(np.int64, copy=False)


def _approach_3(A: np.ndarray, b: np.ndarray, c: float, budget: int, radius: float, config: dict) -> np.ndarray:
    n = b.shape[0]
    active = np.ones(n, dtype=bool)
    w = _solve_worst_weights(A, b, radius, active, config)
    scores = _deletion_objectives(A, b, c, w, active)
    delete_idx = np.lexsort((np.arange(n, dtype=np.int64), scores))[: n - budget]
    active[delete_idx] = False
    return np.flatnonzero(active).astype(np.int64, copy=False)


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
    n = X_arr.shape[0]
    if budget <= 0 or n == 0:
        return np.empty(0, dtype=np.int64)
    if budget > n:
        raise ValueError(f"Budget {budget} exceeds pool size {n}.")

    cfg = config or {}
    A, b, c = _validate_quadratic_inputs(n, cfg)
    radius = float(cfg.get("shift_radius", DEFAULT_SHIFT_RADIUS))
    if radius < 0.0:
        raise ValueError(f"DRCS shift_radius must be non-negative, got {radius}.")

    approach = str(cfg.get("approach", "large_dynamic"))
    if approach == "small":
        selected = _approach_1(A, b, c, budget, radius, cfg)
    elif approach == "large_dynamic":
        selected = _approach_2(A, b, c, budget, radius, cfg)
    elif approach == "large_static":
        selected = _approach_3(A, b, c, budget, radius, cfg)
    else:
        raise ValueError(
            "Unsupported DRCS approach. Expected 'large_dynamic', 'large_static', or 'small'."
        )
    return np.sort(selected)
