"""Leakage-free baseline and mechanistic committor models (Tier 2 & Tier 3).

All estimators strictly fit internal scalers, spline knots, and feature selectors
exclusively inside ``fit()`` on the training fold.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import SplineTransformer, StandardScaler


def _fit_soft_logistic(
    Phi: np.ndarray,
    q_target: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    l1_reg: float = 0.0,
    l2_reg: float = 1e-4,
    max_iter: int = 500,
) -> Tuple[float, np.ndarray]:
    """Minimize weighted Bernoulli cross-entropy + L1 + L2 for soft targets q in [0, 1]."""
    X_mat = np.asarray(Phi, dtype=np.float64)
    y = np.clip(np.asarray(q_target, dtype=np.float64).reshape(-1), 0.0, 1.0)
    n, p = X_mat.shape

    if sample_weight is None:
        w = np.full(n, 1.0 / n, dtype=np.float64)
    else:
        w_raw = np.clip(np.asarray(sample_weight, dtype=np.float64).reshape(-1), 0.0, None)
        w_sum = float(np.sum(w_raw))
        if w_sum <= 0.0:
            raise ValueError("sample_weight sum must be strictly positive.")
        w = w_raw / w_sum

    y_bar = float(np.clip(np.sum(w * y), 1e-6, 1.0 - 1e-6))
    b0_init = float(np.log(y_bar / (1.0 - y_bar)))

    if p == 0:
        return b0_init, np.empty(0, dtype=np.float64)

    if l1_reg <= 0.0:
        theta0 = np.zeros(p + 1, dtype=np.float64)
        theta0[0] = b0_init

        def loss_and_grad(theta: np.ndarray) -> Tuple[float, np.ndarray]:
            b0 = theta[0]
            coef = theta[1:]
            z = b0 + X_mat @ coef
            prob = expit(z)
            ce = float(np.sum(w * (np.logaddexp(0.0, z) - y * z)))
            pen = 0.5 * l2_reg * float(np.dot(coef, coef))
            diff = w * (prob - y)
            grad_b0 = float(np.sum(diff))
            grad_c = X_mat.T @ diff + l2_reg * coef
            grad = np.empty(p + 1, dtype=np.float64)
            grad[0] = grad_b0
            grad[1:] = grad_c
            return ce + pen, grad

        res = minimize(loss_and_grad, theta0, jac=True, method="L-BFGS-B", options={"maxiter": max_iter})
        return float(res.x[0]), res.x[1:].astype(np.float64)

    # FISTA proximal-gradient solver for L1 + L2 composite objective
    # Lipschitz bound of smooth gradient: 0.25 * ||X||_op^2 + l2_reg
    col_sq = np.sum(w[:, None] * (X_mat ** 2), axis=0)
    L_lip = max(float(0.25 * np.sum(col_sq) + l2_reg + 0.25), 1e-2)
    step = 1.0 / L_lip

    b0 = b0_init
    coef = np.zeros(p, dtype=np.float64)
    yb0 = b0
    ycoef = coef.copy()
    t_fista = 1.0

    for _ in range(max_iter):
        z = yb0 + X_mat @ ycoef
        diff = w * (expit(z) - y)
        gb0 = float(np.sum(diff))
        gc = X_mat.T @ diff + l2_reg * ycoef

        b0_next = yb0 - step * gb0
        u = ycoef - step * gc
        thresh = step * l1_reg
        coef_next = np.sign(u) * np.maximum(np.abs(u) - thresh, 0.0)

        if max(abs(b0_next - b0), float(np.max(np.abs(coef_next - coef)))) < 1e-7:
            b0, coef = b0_next, coef_next
            break

        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_fista * t_fista))
        mom = (t_fista - 1.0) / t_next
        yb0 = b0_next + mom * (b0_next - b0)
        ycoef = coef_next + mom * (coef_next - coef)
        b0, coef, t_fista = b0_next, coef_next, t_next

    return float(b0), coef


class PriorBaselineB0:
    """B0 baseline: constant marginal mean (global or per-progress-bin)."""

    def __init__(
        self,
        progress_col: int = 0,
        bin_edges: Optional[np.ndarray] = None,
        eps: float = 1e-6,
    ) -> None:
        self.progress_col = int(progress_col)
        self.bin_edges = None if bin_edges is None else np.asarray(bin_edges, dtype=np.float64)
        self.eps = float(eps)
        self.global_mean_: float = 0.5
        self.bin_means_: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        q_target: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "PriorBaselineB0":
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        y = np.clip(np.asarray(q_target, dtype=np.float64).reshape(-1), 0.0, 1.0)
        w = np.ones_like(y) if sample_weight is None else np.clip(np.asarray(sample_weight, dtype=np.float64).reshape(-1), 0.0, None)
        self.global_mean_ = float(np.clip(np.sum(w * y) / max(np.sum(w), 1e-15), self.eps, 1.0 - self.eps))

        if self.bin_edges is not None:
            r = X_arr[:, self.progress_col]
            n_bins = self.bin_edges.size - 1
            b_idx = np.clip(np.digitize(r, self.bin_edges) - 1, 0, n_bins - 1)
            means = np.full(n_bins, self.global_mean_, dtype=np.float64)
            for b in range(n_bins):
                m = b_idx == b
                if np.any(m) and np.sum(w[m]) > 0:
                    means[b] = float(np.clip(np.sum(w[m] * y[m]) / np.sum(w[m]), self.eps, 1.0 - self.eps))
            self.bin_means_ = means
        return self

    def predict(
        self,
        X: np.ndarray,
        in_A: Optional[np.ndarray] = None,
        in_B: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        n = X_arr.shape[0]
        if self.bin_edges is None or self.bin_means_ is None:
            out = np.full(n, self.global_mean_, dtype=np.float64)
        else:
            r = X_arr[:, self.progress_col]
            n_bins = self.bin_edges.size - 1
            b_idx = np.clip(np.digitize(r, self.bin_edges) - 1, 0, n_bins - 1)
            out = self.bin_means_[b_idx].copy()
        if in_A is not None:
            out[np.asarray(in_A, dtype=bool).reshape(-1)] = 0.0
        if in_B is not None:
            out[np.asarray(in_B, dtype=bool).reshape(-1)] = 1.0
        return out


class SplineProgressBaselineB1:
    """B1 baseline: nonlinear 1D cubic spline on the progress coordinate + logistic link."""

    def __init__(
        self,
        progress_col: int = 0,
        n_knots: int = 5,
        degree: int = 3,
        l2_reg: float = 1e-4,
        eps: float = 1e-6,
    ) -> None:
        self.progress_col = int(progress_col)
        self.n_knots = int(n_knots)
        self.degree = int(degree)
        self.l2_reg = float(l2_reg)
        self.eps = float(eps)
        self.spline_: Optional[SplineTransformer] = None
        self.scaler_: Optional[StandardScaler] = None
        self.intercept_: float = 0.0
        self.coef_: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        q_target: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "SplineProgressBaselineB1":
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        r = X_arr[:, [self.progress_col]]
        self.spline_ = SplineTransformer(
            n_knots=self.n_knots,
            degree=self.degree,
            include_bias=False,
            extrapolation="linear",
        )
        Phi_raw = self.spline_.fit_transform(r)
        self.scaler_ = StandardScaler()
        Phi = self.scaler_.fit_transform(Phi_raw)
        self.intercept_, self.coef_ = _fit_soft_logistic(
            Phi, q_target, sample_weight=sample_weight, l1_reg=0.0, l2_reg=self.l2_reg
        )
        return self

    def predict(
        self,
        X: np.ndarray,
        in_A: Optional[np.ndarray] = None,
        in_B: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self.spline_ is None or self.scaler_ is None or self.coef_ is None:
            raise RuntimeError("SplineProgressBaselineB1 is not fitted.")
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        r = X_arr[:, [self.progress_col]]
        Phi = self.scaler_.transform(self.spline_.transform(r))
        out = np.clip(expit(self.intercept_ + Phi @ self.coef_), self.eps, 1.0 - self.eps)
        if in_A is not None:
            out[np.asarray(in_A, dtype=bool).reshape(-1)] = 0.0
        if in_B is not None:
            out[np.asarray(in_B, dtype=bool).reshape(-1)] = 1.0
        return out


class ProgressVelocityBaselineB2:
    """B2 baseline: 1D progress spline + finite-lag 1D progress displacement."""

    def __init__(
        self,
        progress_col: int = 0,
        n_knots: int = 5,
        degree: int = 3,
        l2_reg: float = 1e-4,
        eps: float = 1e-6,
    ) -> None:
        self.progress_col = int(progress_col)
        self.n_knots = int(n_knots)
        self.degree = int(degree)
        self.l2_reg = float(l2_reg)
        self.eps = float(eps)
        self.spline_: Optional[SplineTransformer] = None
        self.scaler_: Optional[StandardScaler] = None
        self.intercept_: float = 0.0
        self.coef_: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        dX: np.ndarray,
        q_target: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "ProgressVelocityBaselineB2":
        X_arr = np.asarray(X, dtype=np.float64)
        dX_arr = np.asarray(dX, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        if dX_arr.ndim == 1:
            dX_arr = dX_arr[:, None]

        r = X_arr[:, [self.progress_col]]
        dr = dX_arr[:, [self.progress_col]]
        self.spline_ = SplineTransformer(
            n_knots=self.n_knots,
            degree=self.degree,
            include_bias=False,
            extrapolation="linear",
        )
        Phi_raw = np.hstack([self.spline_.fit_transform(r), dr])
        self.scaler_ = StandardScaler()
        Phi = self.scaler_.fit_transform(Phi_raw)
        self.intercept_, self.coef_ = _fit_soft_logistic(
            Phi, q_target, sample_weight=sample_weight, l1_reg=0.0, l2_reg=self.l2_reg
        )
        return self

    def predict(
        self,
        X: np.ndarray,
        dX: np.ndarray,
        in_A: Optional[np.ndarray] = None,
        in_B: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self.spline_ is None or self.scaler_ is None or self.coef_ is None:
            raise RuntimeError("ProgressVelocityBaselineB2 is not fitted.")
        X_arr = np.asarray(X, dtype=np.float64)
        dX_arr = np.asarray(dX, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        if dX_arr.ndim == 1:
            dX_arr = dX_arr[:, None]
        r = X_arr[:, [self.progress_col]]
        dr = dX_arr[:, [self.progress_col]]
        Phi = self.scaler_.transform(np.hstack([self.spline_.transform(r), dr]))
        out = np.clip(expit(self.intercept_ + Phi @ self.coef_), self.eps, 1.0 - self.eps)
        if in_A is not None:
            out[np.asarray(in_A, dtype=bool).reshape(-1)] = 0.0
        if in_B is not None:
            out[np.asarray(in_B, dtype=bool).reshape(-1)] = 1.0
        return out


class PetersTroutCommittorModel:
    """Sparse Peters–Trout logistic committor model with optional bilinear epistasis.

    Fits q_hat(x) = sigma(w_0 + sum_i w_i xi_tilde_i + sum_{i<j} w_{ij} xi_tilde_i xi_tilde_j)
    strictly on standardized training features, with optional L1 sparsity + top-k refit.
    Exposes exact analytical logit gradients in raw physical coordinate units for H3.
    """

    def __init__(
        self,
        include_interactions: bool = True,
        l1_reg: float = 0.0,
        l2_reg: float = 1e-4,
        max_features: Optional[int] = None,
        feature_names: Optional[List[str]] = None,
        eps: float = 1e-6,
    ) -> None:
        self.include_interactions = bool(include_interactions)
        self.l1_reg = float(l1_reg)
        self.l2_reg = float(l2_reg)
        self.max_features = None if max_features is None else int(max_features)
        self.feature_names = feature_names
        self.eps = float(eps)

        self.scaler_: Optional[StandardScaler] = None
        self.pair_indices_: List[Tuple[int, int]] = []
        self.expanded_names_: List[str] = []
        self.active_mask_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.coef_: Optional[np.ndarray] = None
        self.linear_coef_scaled_: Optional[np.ndarray] = None
        self.linear_coef_raw_: Optional[np.ndarray] = None

    def _build_design(self, X_sc: np.ndarray) -> np.ndarray:
        if not self.include_interactions or len(self.pair_indices_) == 0:
            return X_sc
        inter_cols = [(X_sc[:, i] * X_sc[:, j])[:, None] for i, j in self.pair_indices_]
        return np.hstack([X_sc] + inter_cols)

    def fit(
        self,
        X: np.ndarray,
        q_target: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "PetersTroutCommittorModel":
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        _, d = X_arr.shape

        base_names = (
            list(self.feature_names)
            if self.feature_names is not None and len(self.feature_names) == d
            else [f"x{i}" for i in range(d)]
        )

        self.scaler_ = StandardScaler()
        X_sc = self.scaler_.fit_transform(X_arr)

        self.pair_indices_ = []
        self.expanded_names_ = list(base_names)
        if self.include_interactions and d > 1:
            for i in range(d):
                for j in range(i + 1, d):
                    self.pair_indices_.append((i, j))
                    self.expanded_names_.append(f"{base_names[i]}*{base_names[j]}")

        Phi = self._build_design(X_sc)
        b0, coef = _fit_soft_logistic(
            Phi, q_target, sample_weight=sample_weight, l1_reg=self.l1_reg, l2_reg=self.l2_reg
        )

        p_total = Phi.shape[1]
        if self.max_features is not None and 0 < self.max_features < p_total:
            # Select top-k terms by magnitude on training design and refit with L2 polish
            order = np.argsort(np.abs(coef))[::-1]
            keep_idx = np.sort(order[: self.max_features])
            active = np.zeros(p_total, dtype=bool)
            active[keep_idx] = True
            b0_refit, coef_sub = _fit_soft_logistic(
                Phi[:, active],
                q_target,
                sample_weight=sample_weight,
                l1_reg=0.0,
                l2_reg=self.l2_reg,
            )
            coef_full = np.zeros(p_total, dtype=np.float64)
            coef_full[active] = coef_sub
            b0, coef = b0_refit, coef_full
            self.active_mask_ = active
        else:
            self.active_mask_ = np.abs(coef) > 0.0

        self.intercept_ = float(b0)
        self.coef_ = coef
        self.linear_coef_scaled_ = coef[:d].copy()
        scale_safe = np.where(self.scaler_.scale_ > 0, self.scaler_.scale_, 1.0)
        self.linear_coef_raw_ = self.linear_coef_scaled_ / scale_safe
        return self

    def predict_logit(self, X: np.ndarray) -> np.ndarray:
        if self.scaler_ is None or self.coef_ is None:
            raise RuntimeError("PetersTroutCommittorModel is not fitted.")
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        X_sc = self.scaler_.transform(X_arr)
        Phi = self._build_design(X_sc)
        return self.intercept_ + Phi @ self.coef_

    def predict(
        self,
        X: np.ndarray,
        in_A: Optional[np.ndarray] = None,
        in_B: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        out = np.clip(expit(self.predict_logit(X)), self.eps, 1.0 - self.eps)
        if in_A is not None:
            out[np.asarray(in_A, dtype=bool).reshape(-1)] = 0.0
        if in_B is not None:
            out[np.asarray(in_B, dtype=bool).reshape(-1)] = 1.0
        return out

    def logit_gradient_raw(self, X: np.ndarray) -> np.ndarray:
        """Exact analytical gradient d(logit q_hat)/d(X_raw) at each point in X."""
        if self.scaler_ is None or self.coef_ is None:
            raise RuntimeError("PetersTroutCommittorModel is not fitted.")
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        n, d = X_arr.shape
        X_sc = self.scaler_.transform(X_arr)
        grad_sc = np.broadcast_to(self.coef_[:d], (n, d)).copy()

        if self.include_interactions and len(self.pair_indices_) > 0:
            inter_coefs = self.coef_[d:]
            for k, (i, j) in enumerate(self.pair_indices_):
                w_ij = inter_coefs[k]
                if w_ij != 0.0:
                    grad_sc[:, i] += w_ij * X_sc[:, j]
                    grad_sc[:, j] += w_ij * X_sc[:, i]

        scale_safe = np.where(self.scaler_.scale_ > 0, self.scaler_.scale_, 1.0)
        return grad_sc / scale_safe[None, :]

    def mean_logit_gradient_raw(
        self,
        X: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Weighted mean gradient vector in raw physical coordinates over region X."""
        grads = self.logit_gradient_raw(X)
        if weights is None:
            return np.mean(grads, axis=0)
        w = np.clip(np.asarray(weights, dtype=np.float64).reshape(-1), 0.0, None)
        return np.sum(grads * w[:, None], axis=0) / max(float(np.sum(w)), 1e-15)


class OracleHistGBRModel:
    """Tier-3 regularized nonparametric upper-bound estimator."""

    def __init__(
        self,
        max_iter: int = 150,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 50,
        l2_regularization: float = 1.0,
        eps: float = 1e-6,
        random_state: int = 42,
    ) -> None:
        self.max_iter = int(max_iter)
        self.max_leaf_nodes = int(max_leaf_nodes)
        self.min_samples_leaf = int(min_samples_leaf)
        self.l2_regularization = float(l2_regularization)
        self.eps = float(eps)
        self.random_state = int(random_state)
        self.est_: Optional[HistGradientBoostingRegressor] = None

    def fit(
        self,
        X: np.ndarray,
        q_target: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "OracleHistGBRModel":
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        y = np.clip(np.asarray(q_target, dtype=np.float64).reshape(-1), 0.0, 1.0)
        self.est_ = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=min(self.min_samples_leaf, max(5, X_arr.shape[0] // 10)),
            l2_regularization=self.l2_regularization,
            random_state=self.random_state,
        )
        self.est_.fit(X_arr, y, sample_weight=sample_weight)
        return self

    def predict(
        self,
        X: np.ndarray,
        in_A: Optional[np.ndarray] = None,
        in_B: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self.est_ is None:
            raise RuntimeError("OracleHistGBRModel is not fitted.")
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        out = np.clip(self.est_.predict(X_arr), self.eps, 1.0 - self.eps)
        if in_A is not None:
            out[np.asarray(in_A, dtype=bool).reshape(-1)] = 0.0
        if in_B is not None:
            out[np.asarray(in_B, dtype=bool).reshape(-1)] = 1.0
        return out
