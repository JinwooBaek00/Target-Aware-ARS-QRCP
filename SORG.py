# SORG.py (Single Algorithm: Simple + Strong)  [NO do_scaling]
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional


class SORG(BaseEstimator, TransformerMixin):
    """
    One-and-only SORG (context-free):
      - SoftNorm Gate (deterministic; no exposed threshold)
      - Orthogonal Residual Greedy (implicit; no full matrix updates)
      - Optional Group-balanced selection via repeated same algorithm

    Canonical view:
      - Candidates are columns.
      - M shape: (d, N) where d is row-dim and N is #candidates.

    Typical usage:
      - Data coreset / Few-shot:
          SORG(k=...).fit(M=embeddings.T, groups=labels)
      - Token selection:
          SORG(k=...).fit(M=token_features.T, r=np.mean(token_features, axis=0))
      - Generic selection:
          SORG(k=...).fit(M, r=optional)

    Design constraints:
      - No context, no thr, no variants.
      - Deterministic single algorithm.
    """

    def __init__(
        self,
        k: int,
        alpha: float = 1.0,          # energy removal strength (1.0 = full)
        p_norm: float = 2.0,         # gate sharpness (<=0 disables gate)
        epsilon: float = 1e-9,
        min_multiplier: float = 1e-12,
    ):
        self.k = int(k)
        self.alpha = float(alpha)
        self.p_norm = float(p_norm)
        self.epsilon = float(epsilon)
        self.min_multiplier = float(min_multiplier)

        self.selected_indices_ = None
        self.n_candidates_ = None

    # -------------------------------------------------------------------------
    # Input canonicalization (minimal + safe)
    # -------------------------------------------------------------------------
    def _canonicalize(self, M, r, groups):
        M = np.asarray(M, dtype=np.float32)
        if M.ndim != 2:
            raise ValueError("M must be a 2D array.")

        d, N = M.shape

        if groups is not None:
            groups = np.asarray(groups)
            if groups.ndim != 1:
                raise ValueError("groups must be 1D.")
            # If user passed candidates along rows, transpose once
            if groups.shape[0] == d and groups.shape[0] != N:
                M = M.T
                d, N = M.shape
            if groups.shape[0] != N:
                raise ValueError(f"groups length mismatch: got {groups.shape[0]}, expected {N}")

        if r is not None:
            r = np.asarray(r, dtype=np.float32).reshape(-1)
            # If r matches columns but not rows and no other hints, transpose once
            if (r.shape[0] == N and r.shape[0] != d) and (groups is None):
                M = M.T
                d, N = M.shape
            if r.shape[0] != d:
                raise ValueError(f"r length mismatch: got {r.shape[0]}, expected {d}")
            r = r.reshape(-1, 1)

        return M, r, groups

    # -------------------------------------------------------------------------
    # Soft norm gate (deterministic; no exposed threshold)
    #   z = ||x|| / median(||x||)
    #   thr = median(z) + C * MAD(z)   (C fixed)
    #   w = sigmoid(p_norm * (thr - z))
    # -------------------------------------------------------------------------
    def _soft_norm_weight(self, norms_sq: np.ndarray, valid: np.ndarray) -> np.ndarray:
        n = norms_sq.shape[0]
        if self.p_norm <= 0:
            return np.ones(n, dtype=np.float32)

        norms = np.sqrt(np.maximum(norms_sq, 0.0)).astype(np.float32)
        v = norms[valid]
        if v.size == 0:
            return np.ones(n, dtype=np.float32)

        base = float(np.median(v) + 1e-12)
        z = norms / base

        z_v = z[valid]
        med = float(np.median(z_v))
        mad = float(np.median(np.abs(z_v - med)) + 1e-12)

        C = 4.0  # fixed constant (BN/LN-like)
        thr = med + C * mad

        beta = float(self.p_norm)
        t = beta * (z - thr)

        t = np.clip(t, -60.0, 60.0)
        w = 1.0 / (1.0 + np.exp(t))
        w = np.clip(w, self.min_multiplier, 1.0).astype(np.float32)
        return w

    # -------------------------------------------------------------------------
    # Core greedy selection (single run)
    #   - implicit orthogonalization: no full M updates
    #   - updates only:
    #       residual_norms_sq (N,)
    #       corr (N,) and r_res (d,1) if guided
    #       Q basis (d,t)
    # -------------------------------------------------------------------------
    def _fit_single(self, M: np.ndarray, r: Optional[np.ndarray]):
        d, N = M.shape
        self.n_candidates_ = int(N)

        if self.k <= 0 or N == 0:
            self.selected_indices_ = np.array([], dtype=int)
            return self

        k_target = min(self.k, N)

        # Fixed behavior: no scaling
        M_work = M.astype(np.float32, copy=True)
        r_work = None if r is None else np.asarray(r, dtype=np.float32).reshape(-1, 1)

        # Base norms (for gate) + residual norms (for greedy)
        norms_sq0 = np.sum(M_work * M_work, axis=0).astype(np.float32)
        residual_norms_sq = norms_sq0.copy()

        corr = None
        r_res = None
        if r_work is not None:
            r_res = r_work.astype(np.float32, copy=True)
            corr = (M_work.T @ r_res).reshape(-1).astype(np.float32)

        candidates = np.ones(N, dtype=bool)
        S = []

        # Orthonormal basis Q
        Q = np.zeros((d, 0), dtype=np.float32)

        a = float(np.clip(self.alpha, 0.0, 1.0))

        factor = a * (2.0 - a)

        while len(S) < k_target:
            valid = candidates & (residual_norms_sq > self.epsilon)
            if not np.any(valid):
                break

            if corr is not None:
                signal = (corr * corr) / (residual_norms_sq + self.epsilon)
            else:
                signal = residual_norms_sq

            scores = np.full(N, -np.inf, dtype=np.float32)
            scores[valid] = signal[valid]

            # SoftNorm gate (deterministic)
            w_norm = self._soft_norm_weight(norms_sq0, valid)
            scores[valid] *= w_norm[valid]

            best = int(np.argmax(scores))
            if not np.isfinite(scores[best]):
                break

            S.append(best)
            candidates[best] = False

            # Build new orthonormal direction u from best column after removing Q-span
            x = M_work[:, best].astype(np.float32, copy=False)
            if Q.shape[1] > 0:
                c = (Q.T @ x).astype(np.float32, copy=False)         # (t,)
                v = x - (Q @ c).astype(np.float32, copy=False)       # (d,)
            else:
                v = x

            nv2 = float(np.dot(v, v))
            if nv2 <= self.epsilon:
                break

            u = (v / (np.sqrt(nv2) + self.epsilon)).astype(np.float32)

            # Append to basis
            Q = np.concatenate([Q, u.reshape(-1, 1)], axis=1)

            # Projections of u onto all candidates
            proj = (u.reshape(1, -1) @ M_work).reshape(-1).astype(np.float32)

            # Update residual norms (implicit)
            if factor > 0.0:
                residual_norms_sq = residual_norms_sq - (factor * (proj * proj)).astype(np.float32)
                residual_norms_sq = np.maximum(residual_norms_sq, 0.0).astype(np.float32)

            # Update guided residual and corr (implicit)
            if r_res is not None and corr is not None and a > 0.0:
                s = float((u.reshape(1, -1) @ r_res).reshape(()))
                r_res = r_res - (a * u.reshape(-1, 1) * s).astype(np.float32)
                corr = corr - (a * s * proj).astype(np.float32)

        self.selected_indices_ = np.asarray(S, dtype=int)
        return self

    # -------------------------------------------------------------------------
    # Group-balanced selection (same algorithm per group)
    # -------------------------------------------------------------------------
    def _fit_grouped(self, M: np.ndarray, r: Optional[np.ndarray], groups: np.ndarray):
        d, N = M.shape
        groups = np.asarray(groups)
        uniq, counts = np.unique(groups, return_counts=True)
        n_groups = int(len(uniq))

        if self.k <= 0 or N == 0 or n_groups == 0:
            self.selected_indices_ = np.array([], dtype=int)
            self.n_candidates_ = int(N)
            return self

        # Balanced quota then redistribute leftovers by remaining capacity
        base = self.k // n_groups
        rem = self.k % n_groups

        quotas = np.array([base] * n_groups, dtype=int)
        quotas[:rem] += 1
        quotas = np.minimum(quotas, counts)

        used = int(np.sum(quotas))
        leftover = int(self.k - used)
        if leftover > 0:
            capacity = counts - quotas
            order = np.argsort(-capacity, kind="stable")
            for j in order:
                if leftover <= 0:
                    break
                give = min(int(capacity[j]), leftover)
                quotas[j] += give
                leftover -= give

        picks = []
        for g, q in zip(uniq, quotas):
            if q <= 0:
                continue
            mask = (groups == g)
            idx_global = np.where(mask)[0]
            Mg = M[:, mask]

            sub = SORG(
                k=int(q),
                alpha=self.alpha,
                p_norm=self.p_norm,
                epsilon=self.epsilon,
                min_multiplier=self.min_multiplier,
            )
            sub._fit_single(Mg, r=r)
            picks.append(idx_global[sub.selected_indices_])

        self.selected_indices_ = np.concatenate(picks) if picks else np.array([], dtype=int)
        self.n_candidates_ = int(N)
        return self

    # -------------------------------------------------------------------------
    # sklearn API
    # -------------------------------------------------------------------------
    def fit(
        self,
        M: np.ndarray,
        r: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ):
        M, r, groups = self._canonicalize(M, r, groups)

        if groups is not None:
            return self._fit_grouped(M, r, groups)

        return self._fit_single(M, r)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.selected_indices_ is None:
            raise RuntimeError("SORG is not fitted yet. Call fit() first.")

        idx = np.asarray(self.selected_indices_, dtype=int)
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if idx.size == 0:
            if self.n_candidates_ is not None and X.shape[1] == self.n_candidates_:
                return X[:, :0]
            if self.n_candidates_ is not None and X.shape[0] == self.n_candidates_:
                return X[:0, :]
            return X[:, :0]

        if self.n_candidates_ is not None:
            if X.shape[1] == self.n_candidates_:
                return X[:, idx]
            if X.shape[0] == self.n_candidates_:
                return X[idx, :]

        # fallback
        if X.shape[1] > int(np.max(idx)):
            return X[:, idx]
        raise ValueError("Shape mismatch: cannot apply selected indices to X.")

    def get_support(self) -> np.ndarray:
        if self.selected_indices_ is None:
            raise RuntimeError("SORG is not fitted yet. Call fit() first.")
        return np.asarray(self.selected_indices_, dtype=int)


__all__ = ["SORG"]
