from __future__ import annotations

import numpy as np


NAME = "sorg"
SUPPORTED_VARIANTS = ("core", "gate", "guided", "grouped", "full")
DEFAULT_ALPHA = 1.0
DEFAULT_P = 2.0
DEFAULT_EPS = 1e-12
DEFAULT_RHO_MIN = 1e-6
DEFAULT_SOFTNORM_C = 4.0
DEFAULT_GROUP_AXIS = "auto"


def _canonicalize_candidates(
    X: np.ndarray,
    guidance: np.ndarray | None,
    candidates_axis: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    if X.ndim != 2:
        raise ValueError(f"SORG expects a 2D candidate array, got shape {X.shape}.")

    guidance_vec = None if guidance is None else np.asarray(guidance, dtype=np.float64).reshape(-1)
    X_arr = np.asarray(X, dtype=np.float64)

    if candidates_axis == "rows":
        candidates = X_arr
    elif candidates_axis == "columns":
        candidates = X_arr.T
    elif candidates_axis == "auto":
        if guidance_vec is not None:
            if X_arr.shape[1] == guidance_vec.size and X_arr.shape[0] != guidance_vec.size:
                candidates = X_arr
            elif X_arr.shape[0] == guidance_vec.size and X_arr.shape[1] != guidance_vec.size:
                candidates = X_arr.T
            elif X_arr.shape[1] == guidance_vec.size:
                candidates = X_arr
            else:
                raise ValueError(
                    "Could not align guidance with candidates. "
                    f"Candidate shape {X_arr.shape}, guidance length {guidance_vec.size}."
                )
        else:
            candidates = X_arr if X_arr.shape[0] >= X_arr.shape[1] else X_arr.T
    else:
        raise ValueError(
            f"Unsupported candidates_axis={candidates_axis!r}. Expected 'rows', 'columns', or 'auto'."
        )

    if guidance_vec is not None and candidates.shape[1] != guidance_vec.size:
        raise ValueError(
            "Guidance dimension mismatch after canonicalization. "
            f"Candidate dimension {candidates.shape[1]}, guidance length {guidance_vec.size}."
        )
    return candidates, guidance_vec


def _softnorm_weights(
    base_norm_sq: np.ndarray,
    valid_mask: np.ndarray,
    p: float,
    rho_min: float,
    softnorm_c: float,
) -> np.ndarray:
    weights = np.ones_like(base_norm_sq, dtype=np.float64)
    if p <= 0:
        return weights

    delta = 1e-12
    valid_idx = np.flatnonzero(valid_mask)
    if valid_idx.size == 0:
        return weights

    norms = np.sqrt(np.maximum(base_norm_sq, 0.0))
    valid_norms = norms[valid_idx]
    base = float(np.median(valid_norms)) + delta
    z_valid = valid_norms / base
    center = float(np.median(z_valid))
    mad = float(np.median(np.abs(z_valid - center))) + delta
    threshold = center + softnorm_c * mad

    logits = np.clip(p * (z_valid - threshold), -60.0, 60.0)
    valid_weights = 1.0 / (1.0 + np.exp(logits))
    weights[valid_idx] = np.clip(valid_weights, rho_min, 1.0)
    return weights


def _project_residual(x: np.ndarray, basis: list[np.ndarray]) -> np.ndarray:
    if not basis:
        return x.copy()
    coeffs = np.asarray([np.dot(u, x) for u in basis], dtype=np.float64)
    return x - coeffs @ np.vstack(basis)


def _allocate_balanced_quotas(groups: np.ndarray, budget: int) -> dict[object, int]:
    unique_groups, counts = np.unique(groups, return_counts=True)
    capacities = {group: int(count) for group, count in zip(unique_groups.tolist(), counts.tolist())}
    quotas = {group: 0 for group in unique_groups.tolist()}

    remaining = min(int(budget), int(np.sum(counts)))
    active = [group for group in unique_groups.tolist() if capacities[group] > 0]

    while remaining > 0 and active:
        share = remaining // len(active)
        extra = remaining % len(active)
        granted_total = 0

        for offset, group in enumerate(active):
            target = share + (1 if offset < extra else 0)
            available = capacities[group] - quotas[group]
            granted = min(available, target)
            quotas[group] += granted
            granted_total += granted

        if granted_total == 0:
            break
        remaining -= granted_total
        active = [group for group in active if quotas[group] < capacities[group]]

    return quotas


def _resolve_group_quotas(groups: np.ndarray, budget: int, config: dict) -> dict[object, int]:
    explicit = config.get("group_quotas")
    if explicit is None:
        return _allocate_balanced_quotas(groups, budget)

    unique_groups, counts = np.unique(groups, return_counts=True)
    capacities = {group: int(count) for group, count in zip(unique_groups.tolist(), counts.tolist())}
    quotas = {group: 0 for group in unique_groups.tolist()}

    if isinstance(explicit, dict):
        for group, quota in explicit.items():
            if group in quotas:
                quotas[group] = max(0, min(int(quota), capacities[group]))
    else:
        explicit_array = np.asarray(explicit)
        if explicit_array.size != unique_groups.size:
            raise ValueError(
                f"group_quotas length mismatch: expected {unique_groups.size}, got {explicit_array.size}."
            )
        for group, quota in zip(unique_groups.tolist(), explicit_array.tolist()):
            quotas[group] = max(0, min(int(quota), capacities[group]))

    assigned = int(sum(quotas.values()))
    if assigned > int(budget):
        raise ValueError(
            f"Explicit group_quotas sum to {assigned}, which exceeds budget {budget}."
        )
    remaining = min(int(budget), int(np.sum(counts))) - assigned
    if remaining <= 0:
        return quotas

    active = [group for group in unique_groups.tolist() if quotas[group] < capacities[group]]
    while remaining > 0 and active:
        granted_any = False
        for group in active:
            if remaining <= 0:
                break
            if quotas[group] < capacities[group]:
                quotas[group] += 1
                remaining -= 1
                granted_any = True
        if not granted_any:
            break
        active = [group for group in active if quotas[group] < capacities[group]]
    return quotas


def _greedy_select(
    X: np.ndarray,
    budget: int,
    *,
    guidance: np.ndarray | None,
    alpha: float,
    p: float,
    eps: float,
    rho_min: float,
    softnorm_c: float,
) -> np.ndarray:
    n, _ = X.shape
    if budget <= 0 or n == 0:
        return np.empty(0, dtype=np.int64)
    if budget > n:
        raise ValueError(f"Budget {budget} exceeds pool size {n}.")

    a = float(np.clip(alpha, 0.0, 1.0))
    rho = a * (2.0 - a)
    selected_mask = np.zeros(n, dtype=bool)
    selected: list[int] = []
    basis: list[np.ndarray] = []

    base_norm_sq = np.sum(X * X, axis=1)
    gamma = base_norm_sq.astype(np.float64, copy=True)

    guidance_res = None if guidance is None else guidance.astype(np.float64, copy=True)
    corr = None if guidance_res is None else X @ guidance_res
    base_signal = base_norm_sq.astype(np.float64, copy=True)
    if guidance_res is not None:
        base_corr = X @ guidance_res
        base_signal = (base_corr * base_corr) / (base_norm_sq + eps)

    for _ in range(min(budget, n)):
        valid_mask = (~selected_mask) & (gamma > eps)
        if not np.any(valid_mask):
            remaining_idx = np.flatnonzero(~selected_mask)
            need = min(int(budget) - len(selected), int(remaining_idx.size))
            if need > 0:
                remaining_mask = np.zeros(n, dtype=bool)
                remaining_mask[remaining_idx] = True
                weights = _softnorm_weights(
                    base_norm_sq=base_norm_sq,
                    valid_mask=remaining_mask,
                    p=p,
                    rho_min=rho_min,
                    softnorm_c=softnorm_c,
                )
                completion_scores = base_signal[remaining_idx] * weights[remaining_idx]
                order = np.lexsort((remaining_idx, -completion_scores))
                fill = remaining_idx[order[:need]]
                selected.extend(fill.astype(int).tolist())
                selected_mask[fill] = True
            break

        weights = _softnorm_weights(
            base_norm_sq=base_norm_sq,
            valid_mask=valid_mask,
            p=p,
            rho_min=rho_min,
            softnorm_c=softnorm_c,
        )
        if corr is None:
            signal = gamma
        else:
            signal = (corr * corr) / (gamma + eps)

        valid_idx = np.flatnonzero(valid_mask)
        weighted_signal = signal[valid_idx] * weights[valid_idx]
        best_idx = int(valid_idx[int(np.argmax(weighted_signal))])

        residual = _project_residual(X[best_idx], basis)
        residual_norm_sq = float(np.dot(residual, residual))
        if residual_norm_sq <= eps:
            gamma[best_idx] = 0.0
            continue

        u = residual / np.sqrt(residual_norm_sq)
        selected.append(best_idx)
        selected_mask[best_idx] = True
        basis.append(u)

        d = X @ u
        gamma = np.maximum(0.0, gamma - rho * (d * d))
        if corr is not None and guidance_res is not None:
            s = float(np.dot(u, guidance_res))
            guidance_res = guidance_res - a * u * s
            corr = corr - a * s * d

    return np.asarray(selected, dtype=np.int64)


def select(
    X: np.ndarray,
    budget: int,
    guidance: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    config: dict | None = None,
    seed: int = 0,
) -> np.ndarray:
    cfg = config or {}
    del seed
    variant = cfg.get("variant", "core")
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(
            f"Unsupported SORG variant: {variant}. Expected one of {SUPPORTED_VARIANTS}."
        )

    X_rows, guidance_vec = _canonicalize_candidates(
        X=np.asarray(X),
        guidance=guidance,
        candidates_axis=cfg.get("candidates_axis", DEFAULT_GROUP_AXIS),
    )
    n = X_rows.shape[0]
    if budget > n:
        raise ValueError(f"Budget {budget} exceeds pool size {n}.")

    groups_arr = None if groups is None else np.asarray(groups)
    if groups_arr is not None and groups_arr.shape[0] != n:
        raise ValueError(
            f"Group array length mismatch: expected {n}, got {groups_arr.shape[0]}."
        )

    use_gate = bool(cfg.get("use_gate", variant in {"gate", "full"}))
    use_guidance = bool(cfg.get("use_guidance", variant in {"guided", "full"}))
    use_grouping = bool(cfg.get("use_grouping", variant in {"grouped", "full"}))

    if use_guidance and guidance_vec is None:
        raise ValueError(f"SORG variant {variant!r} requires a guidance vector.")
    if use_grouping and groups_arr is None:
        raise ValueError(f"SORG variant {variant!r} requires group labels.")

    alpha = float(cfg.get("alpha", DEFAULT_ALPHA))
    p = float(cfg.get("p", DEFAULT_P if use_gate else 0.0))
    eps = float(cfg.get("eps", DEFAULT_EPS))
    rho_min = float(cfg.get("rho_min", DEFAULT_RHO_MIN))
    softnorm_c = float(cfg.get("softnorm_c", DEFAULT_SOFTNORM_C))

    effective_guidance = guidance_vec if use_guidance else None
    effective_p = p if use_gate else 0.0

    if not use_grouping:
        return _greedy_select(
            X_rows,
            budget,
            guidance=effective_guidance,
            alpha=alpha,
            p=effective_p,
            eps=eps,
            rho_min=rho_min,
            softnorm_c=softnorm_c,
        )

    quotas = _resolve_group_quotas(groups_arr, budget, cfg)
    selected_groups: list[np.ndarray] = []

    for group in quotas:
        quota = quotas[group]
        if quota <= 0:
            continue
        group_idx = np.flatnonzero(groups_arr == group)
        group_selected_local = _greedy_select(
            X_rows[group_idx],
            quota,
            guidance=effective_guidance,
            alpha=alpha,
            p=effective_p,
            eps=eps,
            rho_min=rho_min,
            softnorm_c=softnorm_c,
        )
        selected_groups.append(group_idx[group_selected_local])

    if not selected_groups:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(selected_groups, axis=0)

