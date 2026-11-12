"""Mori–Zwanzig non-Markovian memory tests, N1 group-permutation, N2a shuffle, and N2b reversibility."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

from .dga import StoppedDGACommittor, _segment_slices
from .metrics import (
    brier_skill_score,
    delta_log_likelihood,
    fast_additive_ratio_group_bootstrap_ci,
)
from .models import PetersTroutCommittorModel, SplineProgressBaselineB1
from .splits import LineageGroupSplitter, assert_contiguous_segments, assert_no_cross_seam_pairs


def compute_lagged_displacements(
    X: np.ndarray,
    seg_id: np.ndarray,
    lag_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute finite-lag displacements dX(t) = X(t) - X(t - lag_steps) strictly within segments.

    Returns
    -------
    dX : np.ndarray of shape (N, d)
        Backward finite differences (zeroed where ``valid_mask`` is False).
    valid_mask : np.ndarray of shape (N,), dtype bool
        True strictly for frames t >= lag_steps inside their continuous segment.
    """
    if lag_steps < 1:
        raise ValueError(f"lag_steps must be >= 1, got {lag_steps}.")
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr[:, None]
    seg = np.asarray(seg_id, dtype=np.int64).reshape(-1)
    if seg.size != X_arr.shape[0]:
        raise ValueError("seg_id length must match X.")
    assert_contiguous_segments(seg)

    dX = np.zeros_like(X_arr)
    valid = np.zeros(X_arr.shape[0], dtype=bool)
    past_idx_list = []
    now_idx_list = []

    for s, e in _segment_slices(seg):
        T_m = e - s
        if T_m <= lag_steps:
            continue
        now_slice = slice(s + lag_steps, e)
        past_slice = slice(s, e - lag_steps)
        dX[now_slice] = X_arr[now_slice] - X_arr[past_slice]
        valid[now_slice] = True
        past_idx_list.append(np.arange(s, e - lag_steps, dtype=np.int64))
        now_idx_list.append(np.arange(s + lag_steps, e, dtype=np.int64))

    if now_idx_list:
        assert_no_cross_seam_pairs(seg, np.concatenate(past_idx_list), np.concatenate(now_idx_list))
    return dX, valid


def within_cell_shuffle_history(
    X_static: np.ndarray,
    dX: np.ndarray,
    n_cells: int = 25,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """N2a Null: Permute history displacements dX strictly within fine static cells of X_static."""
    if rng is None:
        rng = np.random.default_rng(42)
    Xs = np.asarray(X_static, dtype=np.float64)
    if Xs.ndim == 1:
        Xs = Xs[:, None]
    dX_arr = np.asarray(dX, dtype=np.float64)
    if dX_arr.ndim == 1:
        dX_arr = dX_arr[:, None]
    n, d = Xs.shape
    k = max(2, min(int(n_cells), n // 15))

    if d == 1:
        qs = np.linspace(0.0, 1.0, k + 1)
        edges = np.unique(np.quantile(Xs[:, 0], qs))
        cells = np.digitize(Xs[:, 0], edges[1:-1])
    else:
        sc = StandardScaler().fit_transform(Xs)
        km = MiniBatchKMeans(
            n_clusters=k,
            random_state=int(rng.integers(0, 1_000_000)),
            batch_size=min(4096, n),
            n_init=2,
        )
        cells = km.fit_predict(sc)

    dX_shuf = dX_arr.copy()
    for c in np.unique(cells):
        idx = np.flatnonzero(cells == c)
        if idx.size > 1:
            dX_shuf[idx] = dX_arr[rng.permutation(idx)]
    return dX_shuf


def compute_memory_collapse_ratio(
    delta_ell_reduced: float,
    delta_ell_augmented: float,
) -> float:
    """Compute fractional Mori–Zwanzig memory collapse rho_mem = 1 - max(0, d_aug)/d_red."""
    if delta_ell_reduced <= 1e-12:
        return 0.0
    rem = max(0.0, float(delta_ell_augmented))
    return float(1.0 - rem / float(delta_ell_reduced))


def _extract_fold_train_targets(
    fold_q_entry: np.ndarray,
    tr_idx: np.ndarray,
    n_total: int,
) -> np.ndarray:
    """Map fold-specific training committor targets onto full array indices cleanly."""
    q_tr = np.clip(np.asarray(fold_q_entry, dtype=np.float64).reshape(-1), 0.0, 1.0)
    out = np.zeros(n_total, dtype=np.float64)
    if q_tr.size == tr_idx.size:
        out[tr_idx] = q_tr
    elif q_tr.size == n_total:
        out[tr_idx] = q_tr[tr_idx]
    else:
        raise ValueError(
            f"fold_q_train entry size ({q_tr.size}) must match either len(tr) ({tr_idx.size}) "
            f"or len(X) ({n_total})."
        )
    return out


def evaluate_mori_zwanzig_memory(
    X_static: np.ndarray,
    seg_id: np.ndarray,
    q_target: np.ndarray,
    groups: np.ndarray,
    lag_steps: int,
    eval_mask: Optional[np.ndarray] = None,
    in_A: Optional[np.ndarray] = None,
    in_B: Optional[np.ndarray] = None,
    fit_interior_only: bool = True,
    weights: Optional[np.ndarray] = None,
    include_interactions: bool = False,
    l2_reg: float = 1e-3,
    n_splits: int = 5,
    n_shuffles: int = 100,
    n_cells_shuffle: int = 25,
    random_state: int = 42,
    stratify_labels: Optional[np.ndarray] = None,
    folds: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]] = None,
    fold_q_train: Optional[Sequence[np.ndarray]] = None,
) -> Dict[str, object]:
    """Evaluate out-of-fold Mori–Zwanzig memory gain Delta_ell_mem(X_static; lag_steps) vs N2a.

    Fits static model q_hat(X(t)) and history model q_hat(X(t), dX(t; lag_steps))
    strictly inside training folds (never across segment boundaries), and evaluates
    on held-out test groups intersected with ``eval_mask`` (e.g., matched progress slice).
    When ``folds`` and ``fold_q_train`` are provided, each fold's predictors are trained
    exclusively on that fold's isolated training committor ``fold_q_train[j]`` and evaluated
    against ``q_target`` (e.g., test-fold-independent ``dga_te_j``), eliminating two-stage
    cross-fold target leakage.
    """
    Xs = np.asarray(X_static, dtype=np.float64)
    if Xs.ndim == 1:
        Xs = Xs[:, None]
    n_total = Xs.shape[0]
    y_all = np.clip(np.asarray(q_target, dtype=np.float64).reshape(-1), 0.0, 1.0)
    g_all = np.asarray(groups)
    dX_all, valid_hist = compute_lagged_displacements(Xs, seg_id, lag_steps=lag_steps)

    mask_A_all = None if in_A is None else np.asarray(in_A, dtype=bool).reshape(-1)
    mask_B_all = None if in_B is None else np.asarray(in_B, dtype=bool).reshape(-1)
    w_all = None if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)

    if eval_mask is None:
        emask = valid_hist.copy()
        if mask_A_all is not None and mask_B_all is not None:
            emask = emask & (~(mask_A_all | mask_B_all))
    else:
        emask = valid_hist & np.asarray(eval_mask, dtype=bool).reshape(-1)

    if not np.any(emask):
        raise ValueError(f"Zero valid evaluation frames at lag_steps={lag_steps}.")

    if folds is not None:
        oof_stat_full = np.zeros(n_total, dtype=np.float64)
        oof_hist_full = np.zeros(n_total, dtype=np.float64)
        fold_records_full = []
        for f_idx, (tr_raw, te_raw) in enumerate(folds):
            tr_full = np.asarray(tr_raw, dtype=np.int64)
            te_full = np.asarray(te_raw, dtype=np.int64)
            y_fold = (
                y_all
                if fold_q_train is None
                else _extract_fold_train_targets(fold_q_train[f_idx], tr_full, n_total)
            )
            tr_v = tr_full[valid_hist[tr_full]]
            te_v = te_full[valid_hist[te_full]]
            if te_v.size == 0:
                continue
            if fit_interior_only and mask_A_all is not None and mask_B_all is not None:
                int_tr = ~(mask_A_all[tr_v] | mask_B_all[tr_v])
                tr_fit = tr_v[int_tr] if np.sum(int_tr) >= 10 else tr_v
            else:
                tr_fit = tr_v
            w_tr = None if w_all is None else w_all[tr_fit]
            A_te = None if mask_A_all is None else mask_A_all[te_v]
            B_te = None if mask_B_all is None else mask_B_all[te_v]
            y_tr_fit = y_fold[tr_fit]

            m_stat = PetersTroutCommittorModel(
                include_interactions=include_interactions, l2_reg=l2_reg
            ).fit(Xs[tr_fit], y_tr_fit, sample_weight=w_tr)
            m_hist = PetersTroutCommittorModel(
                include_interactions=include_interactions, l2_reg=l2_reg
            ).fit(np.hstack([Xs[tr_fit], dX_all[tr_fit]]), y_tr_fit, sample_weight=w_tr)

            oof_stat_full[te_v] = m_stat.predict(Xs[te_v], in_A=A_te, in_B=B_te)
            oof_hist_full[te_v] = m_hist.predict(
                np.hstack([Xs[te_v], dX_all[te_v]]), in_A=A_te, in_B=B_te
            )
            fold_records_full.append((tr_fit, te_v, y_tr_fit, w_tr, A_te, B_te))

        y_eval = y_all[emask]
        p_stat_eval = oof_stat_full[emask]
        p_hist_eval = oof_hist_full[emask]
        g_eval = g_all[emask]
        w_eval = None if w_all is None else w_all[emask]

        obs_delta_ell = delta_log_likelihood(y_eval, p_hist_eval, p_stat_eval, weights=w_eval)
        obs_bss_mem = brier_skill_score(y_eval, p_hist_eval, p_stat_eval, weights=w_eval)

        eps = 1e-7
        p_h = np.clip(p_hist_eval, eps, 1.0 - eps)
        p_s = np.clip(p_stat_eval, eps, 1.0 - eps)
        w_unit = np.ones_like(y_eval) if w_eval is None else np.clip(w_eval, 0.0, None)
        ll_diff = w_unit * (
            (y_eval * np.log(p_h) + (1.0 - y_eval) * np.log(1.0 - p_h))
            - (y_eval * np.log(p_s) + (1.0 - y_eval) * np.log(1.0 - p_s))
        )
        boot_ci = fast_additive_ratio_group_bootstrap_ci(
            group_ids=g_eval,
            numerator_terms=ll_diff,
            denominator_terms=w_unit,
            n_boot=1000,
            random_state=random_state,
        )

        rng = np.random.default_rng(random_state)
        null_deltas = np.empty(int(n_shuffles), dtype=np.float64)
        for s_idx in range(int(n_shuffles)):
            oof_null_full = np.zeros(n_total, dtype=np.float64)
            for tr_fit, te_v, y_tr_fit, w_tr, A_te, B_te in fold_records_full:
                dX_tr_shuf = within_cell_shuffle_history(
                    Xs[tr_fit], dX_all[tr_fit], n_cells=n_cells_shuffle, rng=rng
                )
                dX_te_shuf = within_cell_shuffle_history(
                    Xs[te_v], dX_all[te_v], n_cells=n_cells_shuffle, rng=rng
                )
                m_null = PetersTroutCommittorModel(
                    include_interactions=include_interactions, l2_reg=l2_reg
                ).fit(np.hstack([Xs[tr_fit], dX_tr_shuf]), y_tr_fit, sample_weight=w_tr)
                oof_null_full[te_v] = m_null.predict(
                    np.hstack([Xs[te_v], dX_te_shuf]), in_A=A_te, in_B=B_te
                )
            null_deltas[s_idx] = delta_log_likelihood(
                y_eval, oof_null_full[emask], p_stat_eval, weights=w_eval
            )

        pval_n2a = float((1 + np.sum(null_deltas >= obs_delta_ell)) / (len(null_deltas) + 1))
        return {
            "lag_steps": int(lag_steps),
            "n_eval_frames": int(np.sum(emask)),
            "delta_ell_mem": float(obs_delta_ell),
            "bss_mem": float(obs_bss_mem),
            "delta_ell_ci": boot_ci,
            "n2a_null_mean": float(np.mean(null_deltas)),
            "n2a_null_q95": float(np.quantile(null_deltas, 0.95)),
            "p_value_vs_n2a": pval_n2a,
        }

    # Default path when precomputed full-array folds are not supplied
    idx_valid = np.flatnonzero(valid_hist)
    Xv = Xs[idx_valid]
    dXv = dX_all[idx_valid]
    yv = y_all[idx_valid]
    gv = g_all[idx_valid]
    ev = emask[idx_valid]
    wv = None if w_all is None else w_all[idx_valid]
    Av = None if mask_A_all is None else mask_A_all[idx_valid]
    Bv = None if mask_B_all is None else mask_B_all[idx_valid]
    st_v = None if stratify_labels is None else np.asarray(stratify_labels).reshape(-1)[idx_valid]

    splitter = LineageGroupSplitter(
        n_splits=n_splits, stratified=(st_v is not None), random_state=random_state
    )
    oof_stat = np.zeros(Xv.shape[0], dtype=np.float64)
    oof_hist = np.zeros(Xv.shape[0], dtype=np.float64)

    # Pre-store fitted history models per fold so N2a test-cell shuffles & refits run fast and cleanly
    fold_records = []
    for f_idx, (tr, te) in enumerate(splitter.split(Xv, st_v, gv)):
        y_fold_v = (
            yv
            if fold_q_train is None
            else _extract_fold_train_targets(fold_q_train[f_idx], tr, Xv.shape[0])
        )
        if fit_interior_only and Av is not None and Bv is not None:
            int_tr = ~(Av[tr] | Bv[tr])
            tr_fit = tr[int_tr] if np.sum(int_tr) >= 10 else tr
        else:
            tr_fit = tr
        w_tr = None if wv is None else wv[tr_fit]
        A_te = None if Av is None else Av[te]
        B_te = None if Bv is None else Bv[te]
        y_tr_fit = y_fold_v[tr_fit]
        m_stat = PetersTroutCommittorModel(
            include_interactions=include_interactions, l2_reg=l2_reg
        ).fit(Xv[tr_fit], y_tr_fit, sample_weight=w_tr)
        m_hist = PetersTroutCommittorModel(
            include_interactions=include_interactions, l2_reg=l2_reg
        ).fit(np.hstack([Xv[tr_fit], dXv[tr_fit]]), y_tr_fit, sample_weight=w_tr)

        oof_stat[te] = m_stat.predict(Xv[te], in_A=A_te, in_B=B_te)
        oof_hist[te] = m_hist.predict(np.hstack([Xv[te], dXv[te]]), in_A=A_te, in_B=B_te)
        fold_records.append((tr_fit, te, y_tr_fit, w_tr, A_te, B_te))

    y_eval = yv[ev]
    p_stat_eval = oof_stat[ev]
    p_hist_eval = oof_hist[ev]
    g_eval = gv[ev]
    w_eval = None if wv is None else wv[ev]

    obs_delta_ell = delta_log_likelihood(y_eval, p_hist_eval, p_stat_eval, weights=w_eval)
    obs_bss_mem = brier_skill_score(y_eval, p_hist_eval, p_stat_eval, weights=w_eval)

    # Group-cluster bootstrap CI on observed memory gain
    eps = 1e-7
    p_h = np.clip(p_hist_eval, eps, 1.0 - eps)
    p_s = np.clip(p_stat_eval, eps, 1.0 - eps)
    w_unit = np.ones_like(y_eval) if w_eval is None else np.clip(w_eval, 0.0, None)
    ll_diff = w_unit * (
        (y_eval * np.log(p_h) + (1.0 - y_eval) * np.log(1.0 - p_h))
        - (y_eval * np.log(p_s) + (1.0 - y_eval) * np.log(1.0 - p_s))
    )
    boot_ci = fast_additive_ratio_group_bootstrap_ci(
        group_ids=g_eval,
        numerator_terms=ll_diff,
        denominator_terms=w_unit,
        n_boot=1000,
        random_state=random_state,
    )

    # N2a Null: Refit & evaluate under within-static-cell history shuffles (strictly per fold!)
    rng = np.random.default_rng(random_state)
    null_deltas = np.empty(int(n_shuffles), dtype=np.float64)
    for s_idx in range(int(n_shuffles)):
        oof_null = np.zeros(Xv.shape[0], dtype=np.float64)
        for tr_fit, te, y_tr_fit, w_tr, A_te, B_te in fold_records:
            dX_tr_shuf = within_cell_shuffle_history(Xv[tr_fit], dXv[tr_fit], n_cells=n_cells_shuffle, rng=rng)
            dX_te_shuf = within_cell_shuffle_history(Xv[te], dXv[te], n_cells=n_cells_shuffle, rng=rng)
            m_null = PetersTroutCommittorModel(
                include_interactions=include_interactions, l2_reg=l2_reg
            ).fit(np.hstack([Xv[tr_fit], dX_tr_shuf]), y_tr_fit, sample_weight=w_tr)
            oof_null[te] = m_null.predict(np.hstack([Xv[te], dX_te_shuf]), in_A=A_te, in_B=B_te)
        null_deltas[s_idx] = delta_log_likelihood(y_eval, oof_null[ev], p_stat_eval, weights=w_eval)

    pval_n2a = float((1 + np.sum(null_deltas >= obs_delta_ell)) / (len(null_deltas) + 1))
    return {
        "lag_steps": int(lag_steps),
        "n_eval_frames": int(np.sum(ev)),
        "delta_ell_mem": float(obs_delta_ell),
        "bss_mem": float(obs_bss_mem),
        "delta_ell_ci": boot_ci,
        "n2a_null_mean": float(np.mean(null_deltas)),
        "n2a_null_q95": float(np.quantile(null_deltas, 0.95)),
        "p_value_vs_n2a": pval_n2a,
    }


# Alias for backward/cross-module convenience
evaluate_memory_gain_with_null = evaluate_mori_zwanzig_memory


def _permute_orthogonal_blocks_by_group(
    X: np.ndarray,
    groups: np.ndarray,
    progress_col: int,
    n_strata: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """N1 helper: circularly shift orthogonal feature streams across groups in same progress stratum."""
    X_perm = X.copy()
    orth_cols = [c for c in range(X.shape[1]) if c != progress_col]
    if not orth_cols:
        return X_perm

    g = np.asarray(groups)
    unique_g = np.unique(g)
    if unique_g.size < 2:
        return X_perm

    g_means = np.array([np.mean(X[g == ug, progress_col]) for ug in unique_g], dtype=np.float64)
    k_st = max(1, min(int(n_strata), unique_g.size // 3))
    if k_st == 1:
        strata = np.zeros(unique_g.size, dtype=np.int64)
    else:
        edges = np.unique(np.quantile(g_means, np.linspace(0.0, 1.0, k_st + 1)))
        strata = np.digitize(g_means, edges[1:-1])

    for st in np.unique(strata):
        st_groups = unique_g[strata == st]
        if st_groups.size < 2:
            st_groups = unique_g
        # Random group order + guaranteed non-zero shift >= max group length in stratum
        perm_groups = rng.permutation(st_groups)
        idx_concat = np.concatenate([np.flatnonzero(g == ug) for ug in perm_groups])
        max_g_len = max(int(np.sum(g == ug)) for ug in perm_groups)
        total_len = idx_concat.size
        if total_len > max_g_len + 1:
            shift = int(rng.integers(max_g_len, total_len - max_g_len + 1)) if total_len > 2 * max_g_len else max_g_len
            shifted_idx = np.roll(idx_concat, shift)
        else:
            shifted_idx = idx_concat[::-1]
        X_perm[np.ix_(idx_concat, orth_cols)] = X[np.ix_(shifted_idx, orth_cols)]
    return X_perm


def run_group_permutation_null_N1(
    X: np.ndarray,
    q_target: np.ndarray,
    groups: np.ndarray,
    progress_col: int = 0,
    slice_range: Optional[Tuple[float, float]] = None,
    in_A: Optional[np.ndarray] = None,
    in_B: Optional[np.ndarray] = None,
    fit_interior_only: bool = True,
    fit_on_slice: bool = False,
    weights: Optional[np.ndarray] = None,
    include_interactions: bool = True,
    l1_reg: float = 0.0,
    l2_reg: float = 1e-3,
    max_features: Optional[int] = None,
    n_splits: int = 5,
    n_perms: int = 100,
    random_state: int = 42,
    stratify_labels: Optional[np.ndarray] = None,
    folds: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]] = None,
    fold_q_train: Optional[Sequence[np.ndarray]] = None,
) -> Dict[str, object]:
    """Evaluate H1 model out-of-fold against B1 spline and N1 trajectory-block permutation null."""
    X_arr = np.asarray(X, dtype=np.float64)
    n_total = X_arr.shape[0]
    y = np.clip(np.asarray(q_target, dtype=np.float64).reshape(-1), 0.0, 1.0)
    g = np.asarray(groups)
    w = None if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
    mask_A = None if in_A is None else np.asarray(in_A, dtype=bool).reshape(-1)
    mask_B = None if in_B is None else np.asarray(in_B, dtype=bool).reshape(-1)

    r = X_arr[:, progress_col]
    if slice_range is None:
        in_slice = np.ones(n_total, dtype=bool)
        if mask_A is not None and mask_B is not None:
            in_slice = ~(mask_A | mask_B)
    else:
        in_slice = (r >= float(slice_range[0])) & (r <= float(slice_range[1]))

    oof_b1 = np.zeros_like(y)
    oof_h1 = np.zeros_like(y)
    if folds is None:
        st_lbl = None if stratify_labels is None else np.asarray(stratify_labels).reshape(-1)
        splitter = LineageGroupSplitter(
            n_splits=n_splits, stratified=(st_lbl is not None), random_state=random_state
        )
        fold_list = list(splitter.split(X_arr, st_lbl, g))
    else:
        fold_list = [(np.asarray(tr, dtype=np.int64), np.asarray(te, dtype=np.int64)) for tr, te in folds]

    fold_fit_info = []
    for f_idx, (tr, te) in enumerate(fold_list):
        y_fold = (
            y
            if fold_q_train is None
            else _extract_fold_train_targets(fold_q_train[f_idx], tr, n_total)
        )
        cand_mask = np.ones(tr.size, dtype=bool)
        if fit_interior_only and mask_A is not None and mask_B is not None:
            cand_mask &= ~(mask_A[tr] | mask_B[tr])
        if fit_on_slice:
            cand_mask &= in_slice[tr]
        tr_fit = tr[cand_mask] if np.sum(cand_mask) >= 10 else tr
        w_tr = None if w is None else w[tr_fit]
        A_te = None if mask_A is None else mask_A[te]
        B_te = None if mask_B is None else mask_B[te]
        y_tr_fit = y_fold[tr_fit]
        b1 = SplineProgressBaselineB1(progress_col=progress_col, l2_reg=l2_reg).fit(
            X_arr[tr_fit], y_tr_fit, sample_weight=w_tr
        )
        h1 = PetersTroutCommittorModel(
            include_interactions=include_interactions,
            l1_reg=l1_reg,
            l2_reg=l2_reg,
            max_features=max_features,
        ).fit(X_arr[tr_fit], y_tr_fit, sample_weight=w_tr)
        oof_b1[te] = b1.predict(X_arr[te], in_A=A_te, in_B=B_te)
        oof_h1[te] = h1.predict(X_arr[te], in_A=A_te, in_B=B_te)
        fold_fit_info.append((tr_fit, te, y_tr_fit, w_tr, A_te, B_te))

    w_sl = None if w is None else w[in_slice]
    obs_bss = brier_skill_score(y[in_slice], oof_h1[in_slice], oof_b1[in_slice], weights=w_sl)
    obs_dell = delta_log_likelihood(y[in_slice], oof_h1[in_slice], oof_b1[in_slice], weights=w_sl)

    rng = np.random.default_rng(random_state)
    null_bss = np.empty(int(n_perms), dtype=np.float64)
    null_dell = np.empty(int(n_perms), dtype=np.float64)

    for p_idx in range(int(n_perms)):
        X_perm = _permute_orthogonal_blocks_by_group(X_arr, g, progress_col=progress_col, n_strata=3, rng=rng)
        oof_perm = np.zeros_like(y)
        for tr_fit, te, y_tr_fit, w_tr, A_te, B_te in fold_fit_info:
            m_p = PetersTroutCommittorModel(
                include_interactions=include_interactions,
                l1_reg=l1_reg,
                l2_reg=l2_reg,
                max_features=max_features,
            ).fit(X_perm[tr_fit], y_tr_fit, sample_weight=w_tr)
            oof_perm[te] = m_p.predict(X_perm[te], in_A=A_te, in_B=B_te)
        null_bss[p_idx] = brier_skill_score(y[in_slice], oof_perm[in_slice], oof_b1[in_slice], weights=w_sl)
        null_dell[p_idx] = delta_log_likelihood(y[in_slice], oof_perm[in_slice], oof_b1[in_slice], weights=w_sl)

    pval_bss = float((1 + np.sum(null_bss >= obs_bss)) / (len(null_bss) + 1))
    pval_dell = float((1 + np.sum(null_dell >= obs_dell)) / (len(null_dell) + 1))

    return {
        "obs_bss_vs_b1": float(obs_bss),
        "obs_delta_ell_vs_b1": float(obs_dell),
        "n1_null_bss_mean": float(np.mean(null_bss)),
        "n1_null_bss_q95": float(np.quantile(null_bss, 0.95)),
        "n1_null_dell_mean": float(np.mean(null_dell)),
        "n1_null_dell_q95": float(np.quantile(null_dell, 0.95)),
        "p_value_bss_vs_n1": pval_bss,
        "p_value_dell_vs_n1": pval_dell,
        "oof_b1": oof_b1,
        "oof_h1": oof_h1,
    }


def run_time_reversal_diagnostic_N2b(
    dga_model: StoppedDGACommittor,
    X: np.ndarray,
    in_A: np.ndarray,
    in_B: np.ndarray,
    seg_id: np.ndarray,
    eval_mask: Optional[np.ndarray] = None,
    weights_forward_initial: Optional[np.ndarray] = None,
    threshold: float = 0.10,
    reweight: bool = True,
) -> Dict[str, float]:
    """Run N2b detailed-balance reversibility diagnostic ||q+ + q- - 1||_L2 on interior."""
    mask_A = np.asarray(in_A, dtype=bool).reshape(-1)
    mask_B = np.asarray(in_B, dtype=bool).reshape(-1)
    if dga_model.coef_minus_ is None or weights_forward_initial is not None or not reweight:
        dga_model.fit_backward(
            X,
            mask_A,
            mask_B,
            seg_id=seg_id,
            weights_forward_initial=weights_forward_initial,
            reweight=reweight,
        )
    q_plus = dga_model.predict(X, mask_A, mask_B)
    q_minus = dga_model.predict_backward(X, mask_A, mask_B)

    interior = ~(mask_A | mask_B)
    target_mask = interior if eval_mask is None else (interior & np.asarray(eval_mask, dtype=bool).reshape(-1))
    if not np.any(target_mask):
        target_mask = interior

    dev = q_plus[target_mask] + q_minus[target_mask] - 1.0
    l2_err = float(np.sqrt(np.mean(dev ** 2)))
    mae_err = float(np.mean(np.abs(dev)))
    rho = (
        float(stats.spearmanr(q_plus[target_mask], 1.0 - q_minus[target_mask]).statistic)
        if np.std(q_plus[target_mask]) > 1e-12 and np.std(q_minus[target_mask]) > 1e-12
        else 1.0
    )
    return {
        "n_eval": int(np.sum(target_mask)),
        "l2_reversibility_error": l2_err,
        "mae_reversibility_error": mae_err,
        "spearman_qplus_vs_1minus_qminus": rho,
        "passes_threshold": bool(l2_err <= float(threshold)),
        "threshold": float(threshold),
    }
