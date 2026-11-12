"""Stopped Finite-Lag Dynamical Galerkin Approximation (DGA) & Discrete MSM TPT Solvers.

Implements exact horizon-safe forward and backward stopped committor solvers per:
- Thiede, Giannakis, Dinner & Weare (2019), J. Chem. Phys. 150, 244111 (DOI: 10.1063/1.5063730)
- Strahan, Antoszewski, Lorpaiboon, Vani, Weare & Dinner (2021), JCTC 17, 2948 (DOI: 10.1021/acs.jctc.0c00933)
- Metzner, Schütte & Vanden-Eijnden (2009), Multiscale Model. Simul. 7, 1192 (DOI: 10.1137/070699500)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy import linalg, sparse
from scipy.sparse import linalg as spla
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

from .splits import assert_contiguous_segments, assert_no_cross_seam_pairs


def _validate_inputs(
    X: np.ndarray,
    in_A: np.ndarray,
    in_B: np.ndarray,
    seg_id: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr[:, None]
    if X_arr.ndim != 2 or X_arr.shape[0] == 0:
        raise ValueError(f"Expected non-empty 2D feature array X, got shape {X_arr.shape}.")
    if not np.all(np.isfinite(X_arr)):
        raise ValueError("Feature matrix X contains NaN or Inf values.")

    n = X_arr.shape[0]
    mask_A = np.asarray(in_A, dtype=bool).reshape(-1)
    mask_B = np.asarray(in_B, dtype=bool).reshape(-1)
    if mask_A.size != n or mask_B.size != n:
        raise ValueError(f"in_A ({mask_A.size}) and in_B ({mask_B.size}) must match X ({n}).")
    if np.any(mask_A & mask_B):
        raise ValueError("Basins A and B must be strictly disjoint (in_A & in_B has True entries).")

    if seg_id is None:
        seg = np.zeros(n, dtype=np.int64)
    else:
        seg = np.asarray(seg_id, dtype=np.int64).reshape(-1)
        if seg.size != n:
            raise ValueError(f"seg_id ({seg.size}) must match X ({n}).")
        assert_contiguous_segments(seg)

    return X_arr, mask_A, mask_B, seg


def _segment_slices(seg_id: np.ndarray) -> Sequence[Tuple[int, int]]:
    """Return (start, end_exclusive) index pairs for each contiguous segment."""
    if seg_id.size == 0:
        return []
    cuts = np.flatnonzero(seg_id[1:] != seg_id[:-1]) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [seg_id.size]))
    return list(zip(starts.tolist(), ends.tolist()))


def build_forward_stopped_pairs(
    seg_id: np.ndarray,
    in_A: np.ndarray,
    in_B: np.ndarray,
    lag_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build horizon-safe interior forward stopped index pairs (t_n, t_n + (tau ^ tau_AB)).

    Strictly enforces:
    1. Starting frame lies in interior I = Omega \\ (A u B).
    2. Horizon safety: 0 <= t_n <= T_traj^{(m)} - 1 - lag_steps within segment m,
       eliminating trajectory-end survivorship bias.
    3. Exact first-hitting stopping at entry into A u B within [t_n, t_n + lag_steps].
    """
    if lag_steps < 1:
        raise ValueError(f"lag_steps must be >= 1, got {lag_steps}.")
    seg = np.asarray(seg_id, dtype=np.int64).reshape(-1)
    assert_contiguous_segments(seg)
    in_AB = np.asarray(in_A, dtype=bool).reshape(-1) | np.asarray(in_B, dtype=bool).reshape(-1)

    starts_out = []
    stops_out = []
    for s, e in _segment_slices(seg):
        T_m = e - s
        if T_m <= lag_steps:
            continue
        ab_seg = in_AB[s:e]
        local_idx = np.arange(T_m, dtype=np.int64)
        sentinel = T_m + lag_steps + 1
        hit_candidates = np.where(ab_seg, local_idx, sentinel)
        next_hit = np.minimum.accumulate(hit_candidates[::-1])[::-1]

        # Horizon-safe candidate starts: 0 <= t <= T_m - 1 - lag_steps, strictly in I
        valid_t = local_idx[: T_m - lag_steps]
        interior_t = valid_t[~ab_seg[: T_m - lag_steps]]
        if interior_t.size == 0:
            continue

        stop_t = np.minimum(interior_t + lag_steps, next_hit[interior_t])
        starts_out.append(s + interior_t)
        stops_out.append(s + stop_t)

    if not starts_out:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    idx_start = np.concatenate(starts_out)
    idx_stop = np.concatenate(stops_out)
    assert_no_cross_seam_pairs(seg, idx_start, idx_stop)
    return idx_start, idx_stop


def build_backward_stopped_pairs(
    seg_id: np.ndarray,
    in_A: np.ndarray,
    in_B: np.ndarray,
    lag_steps: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build horizon-safe backward stopped index triples (idx_t0, idx_tend, idx_back_stop).

    For each forward segment [t_0, t_0 + lag_steps] (0 <= t_0 <= T_m - 1 - lag_steps)
    whose terminal point t_end = t_0 + lag_steps lies in interior I:
    - ``idx_t0`` is the forward-initial index t_0 (where change-of-measure weight w(X_{t_0})
      must be evaluated per Thiede et al. 2019 Eqs. 20-24 & Strahan et al. 2021 Eq. 8).
    - ``idx_tend`` is the interior terminal index t_0 + lag_steps.
    - ``idx_back_stop`` is (t_0 + lag_steps) - (lag_steps ^ tau_AB^-), i.e. the forward
      last-exit index from A u B in [t_0, t_0 + lag_steps] (or t_0 if never in A u B).
    """
    if lag_steps < 1:
        raise ValueError(f"lag_steps must be >= 1, got {lag_steps}.")
    seg = np.asarray(seg_id, dtype=np.int64).reshape(-1)
    assert_contiguous_segments(seg)
    in_AB = np.asarray(in_A, dtype=bool).reshape(-1) | np.asarray(in_B, dtype=bool).reshape(-1)

    t0_out = []
    tend_out = []
    bstop_out = []
    for s, e in _segment_slices(seg):
        T_m = e - s
        if T_m <= lag_steps:
            continue
        ab_seg = in_AB[s:e]
        local_idx = np.arange(T_m, dtype=np.int64)
        sentinel = -lag_steps - 1
        hit_candidates = np.where(ab_seg, local_idx, sentinel)
        prev_hit = np.maximum.accumulate(hit_candidates)

        cand_t0 = local_idx[: T_m - lag_steps]
        cand_tend = cand_t0 + lag_steps
        keep = ~ab_seg[cand_tend]
        if not np.any(keep):
            continue

        t0_v = cand_t0[keep]
        tend_v = cand_tend[keep]
        bstop_v = np.maximum(t0_v, prev_hit[tend_v])

        t0_out.append(s + t0_v)
        tend_out.append(s + tend_v)
        bstop_out.append(s + bstop_v)

    if not t0_out:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty

    idx_t0 = np.concatenate(t0_out)
    idx_tend = np.concatenate(tend_out)
    idx_bstop = np.concatenate(bstop_out)
    assert_no_cross_seam_pairs(seg, idx_t0, idx_tend)
    assert_no_cross_seam_pairs(seg, idx_bstop, idx_tend)
    return idx_t0, idx_tend, idx_bstop


def build_unstopped_lag_pairs(
    seg_id: np.ndarray,
    lag_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return all horizon-safe unstopped pairs (t_0, t_0 + lag_steps) within segments."""
    if lag_steps < 1:
        raise ValueError(f"lag_steps must be >= 1, got {lag_steps}.")
    seg = np.asarray(seg_id, dtype=np.int64).reshape(-1)
    assert_contiguous_segments(seg)

    t0_list = []
    t1_list = []
    for s, e in _segment_slices(seg):
        T_m = e - s
        if T_m <= lag_steps:
            continue
        t0 = np.arange(s, e - lag_steps, dtype=np.int64)
        t0_list.append(t0)
        t1_list.append(t0 + lag_steps)

    if not t0_list:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    idx0 = np.concatenate(t0_list)
    idx1 = np.concatenate(t1_list)
    assert_no_cross_seam_pairs(seg, idx0, idx1)
    return idx0, idx1


def _compute_stationary_distribution(T: np.ndarray) -> np.ndarray:
    """Compute stationary row eigenvector pi = pi T of a row-stochastic matrix T."""
    evals, evecs = linalg.eig(T.T)
    idx = int(np.argmin(np.abs(evals - 1.0)))
    pi = np.real(evecs[:, idx])
    if np.sum(pi) < 0:
        pi = -pi
    pi = np.maximum(pi, 0.0)
    total = float(np.sum(pi))
    if total <= 0.0:
        raise ValueError("Failed to compute positive stationary distribution for T.")
    return pi / total


def estimate_reversible_msm_stationary_weights(
    X: np.ndarray,
    seg_id: np.ndarray,
    lag_steps: int,
    n_clusters: int = 80,
    prior_counts: float = 1e-6,
    random_state: int = 42,
) -> Tuple[np.ndarray, MiniBatchKMeans, np.ndarray]:
    """Estimate strictly positive change-of-measure weights w(x) = pi(x) / rho_data(x).

    Uses a k-means partition of scaled feature space, forms the row-normalized empirical
    transition matrix T_ij = C_ij / sum_k C_ik (which cancels non-Boltzmann seeding density
    rho_data), and solves for its true stationary eigenvector pi = pi T at ``lag_steps``
    (fit strictly on the supplied dataset, e.g. D_train).

    Returns
    -------
    weights_per_frame : np.ndarray of shape (N,)
        Normalized weights w(X_t) > 0 with empirical mean 1.0 on X.
    kmeans : MiniBatchKMeans
        Fitted clusterer (for predicting w on new points in the same fold if needed).
    cluster_weights : np.ndarray of shape (n_eff_clusters,)
        Ratio pi_k / rho_k per cluster.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr[:, None]
    seg = np.asarray(seg_id, dtype=np.int64).reshape(-1)
    idx0, idx1 = build_unstopped_lag_pairs(seg, lag_steps)
    if idx0.size == 0:
        raise ValueError(f"No valid lag pairs at lag_steps={lag_steps} to estimate weights.")

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_arr)
    k = max(4, min(int(n_clusters), idx0.size // 25))
    km = MiniBatchKMeans(
        n_clusters=k,
        random_state=random_state,
        batch_size=min(4096, X_sc.shape[0]),
        n_init=3,
    )
    labels = km.fit_predict(X_sc)

    C = np.full((k, k), prior_counts, dtype=np.float64)
    np.add.at(C, (labels[idx0], labels[idx1]), 1.0)

    # Row-normalize C_ij -> T_ij = C_ij / sum_m C_im to cancel non-Boltzmann seeding rho_i,
    # then extract the true stationary distribution pi = pi T via left Perron eigenvector.
    row_sums = np.sum(C, axis=1, keepdims=True)
    T_emp = C / np.maximum(row_sums, 1e-15)
    pi = _compute_stationary_distribution(T_emp)

    # Empirical initial-point distribution rho_data on t_0
    rho_counts = np.bincount(labels[idx0], minlength=k).astype(np.float64) + prior_counts * k
    rho = rho_counts / np.sum(rho_counts)

    cluster_w = np.clip(pi / np.maximum(rho, 1e-12), 1e-3, 1e3)
    frame_w = cluster_w[labels]
    frame_w = frame_w / np.mean(frame_w)
    # Attach scaler to kmeans for downstream transform convenience
    km._feature_scaler = scaler  # type: ignore[attr-defined]
    return frame_w, km, cluster_w


class StoppedDGACommittor:
    """Stopped finite-lag Dynamical Galerkin Approximation (DGA) solver for q+ and q-.

    Strictly fits all feature scalers, RBF centers, boundary distance interpolants,
    and linear operators on the supplied training trajectories only.
    """

    def __init__(
        self,
        lag_steps: int = 10,
        n_rbf: int = 80,
        include_poly2: bool = True,
        reg: float = 1e-5,
        guess_mode: str = "distance",
        clip_predictions: bool = True,
        random_state: int = 42,
    ) -> None:
        self.lag_steps = int(lag_steps)
        self.n_rbf = int(n_rbf)
        self.include_poly2 = bool(include_poly2)
        self.reg = float(reg)
        if guess_mode not in ("distance", "indicator"):
            raise ValueError("guess_mode must be 'distance' or 'indicator'.")
        self.guess_mode = guess_mode
        self.clip_predictions = bool(clip_predictions)
        self.random_state = int(random_state)

        self.scaler_: Optional[StandardScaler] = None
        self.rbf_centers_: Optional[np.ndarray] = None
        self.rbf_gammas_: Optional[np.ndarray] = None
        self.anchors_A_: Optional[np.ndarray] = None
        self.anchors_B_: Optional[np.ndarray] = None
        self.coef_plus_: Optional[np.ndarray] = None
        self.coef_minus_: Optional[np.ndarray] = None
        self.n_interior_pairs_: int = 0
        self.n_basis_: int = 0

    def _fit_basis_geometry(
        self,
        X_arr: np.ndarray,
        in_A: np.ndarray,
        in_B: np.ndarray,
    ) -> None:
        interior_mask = ~(in_A | in_B)
        if not np.any(interior_mask):
            raise ValueError("No interior frames (outside A u B) found in training data.")
        if not np.any(in_A) or not np.any(in_B):
            raise ValueError("Both basin A and basin B must have >=1 frame in training data.")

        self.scaler_ = StandardScaler()
        self.scaler_.fit(X_arr[interior_mask])
        X_sc = self.scaler_.transform(X_arr)

        X_int_sc = X_sc[interior_mask]
        k_rbf = min(self.n_rbf, max(4, X_int_sc.shape[0] // 20))
        if k_rbf > 0:
            km = MiniBatchKMeans(
                n_clusters=k_rbf,
                random_state=self.random_state,
                batch_size=min(4096, X_int_sc.shape[0]),
                n_init=3,
            )
            km.fit(X_int_sc)
            centers = km.cluster_centers_
            # Multi-scale bandwidths adapted from pairwise nearest-neighbor spacing of centers
            if centers.shape[0] > 1:
                dists = linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
                np.fill_diagonal(dists, np.inf)
                nn_dist = np.min(dists, axis=1)
                sigma = np.maximum(nn_dist * 1.25, 0.35)
            else:
                sigma = np.ones(1, dtype=np.float64)
            self.rbf_centers_ = centers
            self.rbf_gammas_ = 0.5 / (sigma ** 2)
        else:
            self.rbf_centers_ = np.empty((0, X_arr.shape[1]), dtype=np.float64)
            self.rbf_gammas_ = np.empty(0, dtype=np.float64)

        # Boundary anchor centroids in scaled coordinates for smooth guess g(x)
        rng = np.random.default_rng(self.random_state)
        X_A_sc = X_sc[in_A]
        X_B_sc = X_sc[in_B]
        n_anc_A = min(16, X_A_sc.shape[0])
        n_anc_B = min(16, X_B_sc.shape[0])
        idx_A = rng.choice(X_A_sc.shape[0], size=n_anc_A, replace=False)
        idx_B = rng.choice(X_B_sc.shape[0], size=n_anc_B, replace=False)
        self.anchors_A_ = X_A_sc[idx_A]
        self.anchors_B_ = X_B_sc[idx_B]

    def _eval_guess_and_basis(
        self,
        X_arr: np.ndarray,
        in_A: np.ndarray,
        in_B: np.ndarray,
        forward: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate boundary guess g(x) and interior basis Phi(x) vanishing on A u B."""
        if self.scaler_ is None or self.rbf_centers_ is None or self.rbf_gammas_ is None:
            raise RuntimeError("StoppedDGACommittor must be fitted before evaluation.")

        X_sc = self.scaler_.transform(X_arr)
        n, d = X_sc.shape
        in_AB = in_A | in_B
        interior = ~in_AB

        # Guess g(x): forward has g|_A=0, g|_B=1; backward has g_-|_A=1, g_-|_B=0
        if self.guess_mode == "indicator" or self.anchors_A_ is None or self.anchors_B_ is None:
            if forward:
                g = np.where(in_B, 1.0, 0.0)
            else:
                g = np.where(in_A, 1.0, 0.0)
        else:
            dA = np.min(linalg.norm(X_sc[:, None, :] - self.anchors_A_[None, :, :], axis=-1), axis=1)
            dB = np.min(linalg.norm(X_sc[:, None, :] - self.anchors_B_[None, :, :], axis=-1), axis=1)
            interp_plus = dA / np.maximum(dA + dB, 1e-12)
            if forward:
                g = np.where(in_A, 0.0, np.where(in_B, 1.0, interp_plus))
            else:
                g = np.where(in_A, 1.0, np.where(in_B, 0.0, 1.0 - interp_plus))

        cols = [np.ones((n, 1), dtype=np.float64), X_sc]
        if self.include_poly2:
            poly_terms = [X_sc ** 2]
            for i in range(d):
                for j in range(i + 1, d):
                    poly_terms.append((X_sc[:, i] * X_sc[:, j])[:, None])
            cols.extend(poly_terms)

        if self.rbf_centers_.shape[0] > 0:
            sq_dists = np.sum((X_sc[:, None, :] - self.rbf_centers_[None, :, :]) ** 2, axis=-1)
            rbf_vals = np.exp(-sq_dists * self.rbf_gammas_[None, :])
            cols.append(rbf_vals)

        Phi = np.hstack(cols)
        # Strictly vanish on A u B
        Phi[~interior, :] = 0.0
        return g, Phi

    def fit(
        self,
        X: np.ndarray,
        in_A: np.ndarray,
        in_B: np.ndarray,
        seg_id: Optional[np.ndarray] = None,
        sample_weights_t0: Optional[np.ndarray] = None,
    ) -> "StoppedDGACommittor":
        """Fit forward stopped-DGA committor q+ strictly on the provided trajectories."""
        X_arr, mask_A, mask_B, seg = _validate_inputs(X, in_A, in_B, seg_id)
        idx_start, idx_stop = build_forward_stopped_pairs(seg, mask_A, mask_B, self.lag_steps)
        if idx_start.size == 0:
            raise ValueError(f"Zero horizon-safe interior pairs found at lag_steps={self.lag_steps}.")

        self._fit_basis_geometry(X_arr, mask_A, mask_B)
        g, Phi = self._eval_guess_and_basis(X_arr, mask_A, mask_B, forward=True)

        Phi_t = Phi[idx_start]
        dPhi = Phi[idx_stop] - Phi_t
        dg = g[idx_stop] - g[idx_start]

        if sample_weights_t0 is not None:
            w = np.asarray(sample_weights_t0, dtype=np.float64).reshape(-1)[idx_start]
            w = np.clip(w, 0.0, None)
            w_sum = float(np.sum(w))
            if w_sum <= 0.0:
                raise ValueError("Sum of sample_weights_t0 on interior starts must be positive.")
            Phi_t_w = Phi_t * (w[:, None] / w_sum)
        else:
            Phi_t_w = Phi_t / float(idx_start.size)

        A_mat = Phi_t_w.T @ dPhi
        b_vec = Phi_t_w.T @ dg
        M = A_mat.shape[0]
        # Note: A_mat approximates negative-definite (S_tau - I), so subtract reg*I
        A_reg = A_mat - self.reg * np.eye(M, dtype=np.float64)
        self.coef_plus_ = linalg.solve(A_reg, -b_vec, assume_a="gen")
        self.n_interior_pairs_ = int(idx_start.size)
        self.n_basis_ = int(M)
        return self

    def fit_backward(
        self,
        X: np.ndarray,
        in_A: np.ndarray,
        in_B: np.ndarray,
        seg_id: Optional[np.ndarray] = None,
        weights_forward_initial: Optional[np.ndarray] = None,
        reweight: bool = True,
    ) -> "StoppedDGACommittor":
        """Fit backward stopped-DGA committor q- using forward-initial path reweighting w(X_{t_0}).

        Must be called after ``fit()`` (reuses the same interior basis geometry).
        If ``weights_forward_initial`` is None and ``reweight=True``, estimates stationary
        change-of-measure ratios w(X_{t_0}) = pi(X_{t_0}) / rho_data(X_{t_0}) automatically
        on the supplied dataset; if ``reweight=False``, uses uniform unit weights.
        """
        if self.scaler_ is None:
            raise RuntimeError("Call fit() before fit_backward().")
        X_arr, mask_A, mask_B, seg = _validate_inputs(X, in_A, in_B, seg_id)
        idx_t0, idx_tend, idx_bstop = build_backward_stopped_pairs(
            seg, mask_A, mask_B, self.lag_steps
        )
        if idx_tend.size == 0:
            raise ValueError(f"Zero horizon-safe backward interior segments at lag_steps={self.lag_steps}.")

        if weights_forward_initial is not None:
            w_all = np.asarray(weights_forward_initial, dtype=np.float64).reshape(-1)
            if w_all.size != X_arr.shape[0]:
                raise ValueError("weights_forward_initial length must match X.")
        elif reweight:
            w_all, _, _ = estimate_reversible_msm_stationary_weights(
                X_arr, seg, lag_steps=self.lag_steps, random_state=self.random_state
            )
        else:
            w_all = np.ones(X_arr.shape[0], dtype=np.float64)

        # CRITICAL: evaluate change-of-measure weight at forward-initial idx_t0!
        w_seg = np.clip(w_all[idx_t0], 0.0, None)
        w_sum = float(np.sum(w_seg))
        if w_sum <= 0.0:
            raise ValueError("Non-positive weight sum in fit_backward.")

        g_minus, Phi = self._eval_guess_and_basis(X_arr, mask_A, mask_B, forward=False)
        Phi_end = Phi[idx_tend]
        dPhi_back = Phi[idx_bstop] - Phi_end
        dg_back = g_minus[idx_bstop] - g_minus[idx_tend]

        Phi_end_w = Phi_end * (w_seg[:, None] / w_sum)
        A_minus = Phi_end_w.T @ dPhi_back
        b_minus = Phi_end_w.T @ dg_back
        M = A_minus.shape[0]
        A_minus_reg = A_minus - self.reg * np.eye(M, dtype=np.float64)
        self.coef_minus_ = linalg.solve(A_minus_reg, -b_minus, assume_a="gen")
        return self

    def predict(
        self,
        X: np.ndarray,
        in_A: Optional[np.ndarray] = None,
        in_B: Optional[np.ndarray] = None,
        clip: Optional[bool] = None,
    ) -> np.ndarray:
        """Predict forward committor q+ on configurations X."""
        if self.coef_plus_ is None:
            raise RuntimeError("StoppedDGACommittor is not fitted yet.")
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        n = X_arr.shape[0]
        mask_A = np.zeros(n, dtype=bool) if in_A is None else np.asarray(in_A, dtype=bool).reshape(-1)
        mask_B = np.zeros(n, dtype=bool) if in_B is None else np.asarray(in_B, dtype=bool).reshape(-1)

        g, Phi = self._eval_guess_and_basis(X_arr, mask_A, mask_B, forward=True)
        q = g + Phi @ self.coef_plus_
        q[mask_A] = 0.0
        q[mask_B] = 1.0
        do_clip = self.clip_predictions if clip is None else bool(clip)
        if do_clip:
            q = np.clip(q, 0.0, 1.0)
        return q

    def predict_backward(
        self,
        X: np.ndarray,
        in_A: Optional[np.ndarray] = None,
        in_B: Optional[np.ndarray] = None,
        clip: Optional[bool] = None,
    ) -> np.ndarray:
        """Predict backward committor q- on configurations X."""
        if self.coef_minus_ is None:
            raise RuntimeError("fit_backward() has not been called yet.")
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        n = X_arr.shape[0]
        mask_A = np.zeros(n, dtype=bool) if in_A is None else np.asarray(in_A, dtype=bool).reshape(-1)
        mask_B = np.zeros(n, dtype=bool) if in_B is None else np.asarray(in_B, dtype=bool).reshape(-1)

        g_minus, Phi = self._eval_guess_and_basis(X_arr, mask_A, mask_B, forward=False)
        q_m = g_minus + Phi @ self.coef_minus_
        q_m[mask_A] = 1.0
        q_m[mask_B] = 0.0
        do_clip = self.clip_predictions if clip is None else bool(clip)
        if do_clip:
            q_m = np.clip(q_m, 0.0, 1.0)
        return q_m


@dataclass
class MSMTPTResult:
    """Exact discrete-state Transition Path Theory (TPT) output."""

    q_plus: np.ndarray
    q_minus: np.ndarray
    stationary_distribution: np.ndarray
    gross_flux: np.ndarray
    net_flux: np.ndarray
    total_reactive_flux: float
    transition_rate_AB: float
    mfpt_AB: float
    reversibility_l2_error: float


def solve_msm_tpt_committor(
    T: np.ndarray,
    A_states: Sequence[int],
    B_states: Sequence[int],
    pi: Optional[np.ndarray] = None,
    lag_time: float = 1.0,
    reg: float = 1e-12,
) -> MSMTPTResult:
    """Solve exact discrete-MSM Transition Path Theory equations on (T, A, B).

    Solves (I - T_{II}) q_{+,I} = T_{IB} 1 for forward committor q_+, and the
    time-reversed adjoint system (I - T_rev_{II}) q_{-,I} = T_rev_{IA} 1 for q_-.
    """
    if sparse.issparse(T):
        T_arr = T.toarray().astype(np.float64)
    else:
        T_arr = np.asarray(T, dtype=np.float64)

    n = T_arr.shape[0]
    if T_arr.shape != (n, n):
        raise ValueError(f"Transition matrix T must be square, got {T_arr.shape}.")
    row_sums = np.sum(T_arr, axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        raise ValueError(
            f"Transition matrix T is not row-stochastic (min/max row sum: "
            f"{row_sums.min():.6f}, {row_sums.max():.6f})."
        )
    T_arr = T_arr / row_sums[:, None]

    mask_A = np.zeros(n, dtype=bool)
    mask_B = np.zeros(n, dtype=bool)
    mask_A[np.asarray(A_states, dtype=np.int64)] = True
    mask_B[np.asarray(B_states, dtype=np.int64)] = True
    if not np.any(mask_A) or not np.any(mask_B):
        raise ValueError("A_states and B_states must both be non-empty.")
    if np.any(mask_A & mask_B):
        raise ValueError("A_states and B_states must be strictly disjoint.")

    mask_I = ~(mask_A | mask_B)
    idx_I = np.flatnonzero(mask_I)
    idx_A = np.flatnonzero(mask_A)
    idx_B = np.flatnonzero(mask_B)

    if pi is None:
        pi_vec = _compute_stationary_distribution(T_arr)
    else:
        pi_vec = np.asarray(pi, dtype=np.float64).reshape(-1)
        if pi_vec.size != n or np.any(pi_vec < 0.0):
            raise ValueError("Invalid stationary distribution pi supplied.")
        pi_vec = pi_vec / np.sum(pi_vec)

    q_plus = np.zeros(n, dtype=np.float64)
    q_plus[idx_B] = 1.0
    if idx_I.size > 0:
        T_II = T_arr[np.ix_(idx_I, idx_I)]
        rhs_plus = np.sum(T_arr[np.ix_(idx_I, idx_B)], axis=1)
        lhs_plus = np.eye(idx_I.size, dtype=np.float64) - T_II + reg * np.eye(idx_I.size)
        q_plus[idx_I] = np.clip(spla.spsolve(sparse.csr_matrix(lhs_plus), rhs_plus), 0.0, 1.0)

    # Time-reversed transition matrix T_rev[i, j] = pi[j] * T[j, i] / pi[i]
    pi_safe = np.maximum(pi_vec, 1e-15)
    T_rev = (T_arr.T * pi_safe[None, :]) / pi_safe[:, None]
    rev_row_sums = np.sum(T_rev, axis=1)
    T_rev = np.where(rev_row_sums[:, None] > 0, T_rev / np.maximum(rev_row_sums[:, None], 1e-15), 0.0)

    q_minus = np.zeros(n, dtype=np.float64)
    q_minus[idx_A] = 1.0
    if idx_I.size > 0:
        T_rev_II = T_rev[np.ix_(idx_I, idx_I)]
        rhs_minus = np.sum(T_rev[np.ix_(idx_I, idx_A)], axis=1)
        lhs_minus = np.eye(idx_I.size, dtype=np.float64) - T_rev_II + reg * np.eye(idx_I.size)
        q_minus[idx_I] = np.clip(spla.spsolve(sparse.csr_matrix(lhs_minus), rhs_minus), 0.0, 1.0)

    # Detailed-balance deviation between pi_i T_ij and pi_j T_ji
    flux_eq = pi_vec[:, None] * T_arr
    rev_err = float(linalg.norm(flux_eq - flux_eq.T) / max(linalg.norm(flux_eq), 1e-15))

    # Reactive fluxes per Metzner, Schütte & Vanden-Eijnden (2009)
    gross_flux = (pi_vec * q_minus)[:, None] * T_arr * q_plus[None, :]
    np.fill_diagonal(gross_flux, 0.0)
    net_flux = np.maximum(0.0, gross_flux - gross_flux.T)

    # Total reactive A -> B flux per lag_time
    total_flux_per_step = float(np.sum(gross_flux[idx_A, :]))
    total_reactive_flux = total_flux_per_step / float(lag_time)
    prob_in_A_reactive = float(np.sum(pi_vec * q_minus))
    rate_AB = total_reactive_flux / max(prob_in_A_reactive, 1e-15)
    mfpt_AB = 1.0 / max(rate_AB, 1e-30)

    return MSMTPTResult(
        q_plus=q_plus,
        q_minus=q_minus,
        stationary_distribution=pi_vec,
        gross_flux=gross_flux,
        net_flux=net_flux,
        total_reactive_flux=total_reactive_flux,
        transition_rate_AB=rate_AB,
        mfpt_AB=mfpt_AB,
        reversibility_l2_error=rev_err,
    )
