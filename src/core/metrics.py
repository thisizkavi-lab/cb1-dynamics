"""Out-of-sample operator residuals, matched-slice metrics, and lineage block bootstrap."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import linalg, stats
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

from .dga import build_forward_stopped_pairs


def projected_bellman_residual(
    q_pred_start: np.ndarray,
    q_pred_stop: np.ndarray,
    X_start_interior: np.ndarray,
    n_witness: int = 35,
    reg: float = 1e-6,
    weights: Optional[np.ndarray] = None,
    random_state: int = 42,
) -> float:
    """Compute Projected Bellman Residual E_PBR over a low-dim interior witness subspace V_L.

    Strictly enforces dim(V_L) << N_test_interior so conditional diffusion variance
    Var(q_hat(X') | X) averages out rather than collapsing pointwise (avoiding the
    unrestricted L^2 double-sampling bias).
    """
    q0 = np.asarray(q_pred_start, dtype=np.float64).reshape(-1)
    q1 = np.asarray(q_pred_stop, dtype=np.float64).reshape(-1)
    X_int = np.asarray(X_start_interior, dtype=np.float64)
    if X_int.ndim == 1:
        X_int = X_int[:, None]
    n, d = X_int.shape
    if q0.size != n or q1.size != n or n < 10:
        raise ValueError(f"Expected matching arrays with n >= 10, got n={n}.")

    if weights is None:
        w = np.full(n, 1.0 / n, dtype=np.float64)
    else:
        w_raw = np.clip(np.asarray(weights, dtype=np.float64).reshape(-1), 0.0, None)
        w = w_raw / max(float(np.sum(w_raw)), 1e-15)

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_int)

    # Enforce L = dim(V_L) <= max(4, n // 15) << n
    max_total_dim = max(4, min(int(n_witness), n // 15))
    n_rbf = max(0, max_total_dim - (1 + d))

    cols = [np.ones((n, 1), dtype=np.float64), X_sc]
    if n_rbf >= 2:
        km = MiniBatchKMeans(
            n_clusters=n_rbf,
            random_state=random_state,
            batch_size=min(4096, n),
            n_init=3,
        )
        km.fit(X_sc)
        centers = km.cluster_centers_
        dists = linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
        np.fill_diagonal(dists, np.inf)
        sigma = np.maximum(np.min(dists, axis=1) * 1.25, 0.4)
        gamma = 0.5 / (sigma ** 2)
        sq = np.sum((X_sc[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
        cols.append(np.exp(-sq * gamma[None, :]))

    Psi = np.hstack(cols)  # shape (n, L), defined on interior starts (psi|_{A u B} = 0)
    td_diff = q1 - q0
    r_vec = Psi.T @ (w * td_diff)
    G_mat = (Psi * w[:, None]).T @ Psi
    L_dim = G_mat.shape[0]
    G_reg = G_mat + float(reg) * np.eye(L_dim, dtype=np.float64)
    val = float(r_vec.T @ linalg.solve(G_reg, r_vec, assume_a="pos"))
    return float(np.sqrt(max(val, 0.0)))


def evaluate_committor_pbr(
    predict_interior_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    in_A: np.ndarray,
    in_B: np.ndarray,
    seg_id: np.ndarray,
    lag_steps: int,
    n_witness: int = 35,
    reg: float = 1e-6,
    weights: Optional[np.ndarray] = None,
    random_state: int = 42,
) -> float:
    """Evaluate out-of-sample Projected Bellman Residual of a predictor on test trajectories."""
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr[:, None]
    mask_A = np.asarray(in_A, dtype=bool).reshape(-1)
    mask_B = np.asarray(in_B, dtype=bool).reshape(-1)
    idx_start, idx_stop = build_forward_stopped_pairs(seg_id, mask_A, mask_B, lag_steps)
    if idx_start.size < 10:
        raise ValueError(f"Insufficient interior stopped test pairs ({idx_start.size}) for PBR.")

    q_all = np.asarray(predict_interior_fn(X_arr), dtype=np.float64).reshape(-1).copy()
    q_all[mask_A] = 0.0
    q_all[mask_B] = 1.0
    w_sub = None if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)[idx_start]
    return projected_bellman_residual(
        q_pred_start=q_all[idx_start],
        q_pred_stop=q_all[idx_stop],
        X_start_interior=X_arr[idx_start],
        n_witness=n_witness,
        reg=reg,
        weights=w_sub,
        random_state=random_state,
    )


def brier_score(
    q_ref: np.ndarray,
    q_pred: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    y = np.asarray(q_ref, dtype=np.float64).reshape(-1)
    p = np.asarray(q_pred, dtype=np.float64).reshape(-1)
    sq = (p - y) ** 2
    if weights is None:
        return float(np.mean(sq))
    w = np.clip(np.asarray(weights, dtype=np.float64).reshape(-1), 0.0, None)
    return float(np.sum(w * sq) / max(float(np.sum(w)), 1e-15))


def brier_skill_score(
    q_ref: np.ndarray,
    q_pred: np.ndarray,
    q_baseline: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    bs_model = brier_score(q_ref, q_pred, weights=weights)
    bs_base = brier_score(q_ref, q_baseline, weights=weights)
    if bs_base <= 1e-15:
        return 0.0
    return float(1.0 - bs_model / bs_base)


def committor_log_likelihood(
    q_ref: np.ndarray,
    q_pred: np.ndarray,
    weights: Optional[np.ndarray] = None,
    eps: float = 1e-7,
) -> float:
    """Mean Bernoulli / Peters-Trout committor log-likelihood (nats per sample)."""
    y = np.clip(np.asarray(q_ref, dtype=np.float64).reshape(-1), 0.0, 1.0)
    p = np.clip(np.asarray(q_pred, dtype=np.float64).reshape(-1), eps, 1.0 - eps)
    ll = y * np.log(p) + (1.0 - y) * np.log(1.0 - p)
    if weights is None:
        return float(np.mean(ll))
    w = np.clip(np.asarray(weights, dtype=np.float64).reshape(-1), 0.0, None)
    return float(np.sum(w * ll) / max(float(np.sum(w)), 1e-15))


def delta_log_likelihood(
    q_ref: np.ndarray,
    q_pred: np.ndarray,
    q_baseline: np.ndarray,
    weights: Optional[np.ndarray] = None,
    eps: float = 1e-7,
) -> float:
    """Mean log-likelihood gain (nats/sample) of q_pred over q_baseline."""
    return committor_log_likelihood(q_ref, q_pred, weights=weights, eps=eps) - committor_log_likelihood(
        q_ref, q_baseline, weights=weights, eps=eps
    )


def calibration_curve_soft(
    q_ref: np.ndarray,
    q_pred: np.ndarray,
    n_bins: int = 10,
    weights: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Reliability curve and Expected Calibration Error (ECE) for soft or hard targets."""
    y = np.asarray(q_ref, dtype=np.float64).reshape(-1)
    p = np.clip(np.asarray(q_pred, dtype=np.float64).reshape(-1), 0.0, 1.0)
    w = np.ones_like(y) if weights is None else np.clip(np.asarray(weights, dtype=np.float64).reshape(-1), 0.0, None)
    w_tot = max(float(np.sum(w)), 1e-15)

    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    bin_idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    pred_means: List[float] = []
    obs_means: List[float] = []
    counts: List[int] = []
    ece = 0.0

    for b in range(n_bins):
        m = bin_idx == b
        cnt = int(np.sum(m))
        if cnt == 0 or np.sum(w[m]) <= 0:
            continue
        wb = w[m]
        wb_sum = float(np.sum(wb))
        pm = float(np.sum(wb * p[m]) / wb_sum)
        om = float(np.sum(wb * y[m]) / wb_sum)
        pred_means.append(pm)
        obs_means.append(om)
        counts.append(cnt)
        ece += (wb_sum / w_tot) * abs(pm - om)

    return {
        "pred_mean": pred_means,
        "obs_mean": obs_means,
        "counts": counts,
        "ece": float(ece),
    }


def matched_slice_metrics(
    progress_coord: np.ndarray,
    q_ref: np.ndarray,
    q_pred: np.ndarray,
    q_baseline: np.ndarray,
    slice_range: Tuple[float, float],
    bin_width: float = 0.5,
    weights: Optional[np.ndarray] = None,
    eps: float = 1e-7,
) -> Dict[str, object]:
    """Evaluate committor predictions strictly on matched progress-coordinate slices."""
    r = np.asarray(progress_coord, dtype=np.float64).reshape(-1)
    y = np.asarray(q_ref, dtype=np.float64).reshape(-1)
    p = np.asarray(q_pred, dtype=np.float64).reshape(-1)
    b = np.asarray(q_baseline, dtype=np.float64).reshape(-1)
    w_all = None if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)

    lo, hi = float(slice_range[0]), float(slice_range[1])
    in_slice = (r >= lo) & (r <= hi)
    n_slice = int(np.sum(in_slice))
    if n_slice == 0:
        raise ValueError(f"Zero frames lie inside slice_range=[{lo}, {hi}].")

    w_s = None if w_all is None else w_all[in_slice]
    rho_s, pval_s = stats.spearmanr(y[in_slice], p[in_slice])
    rho_b, _ = stats.spearmanr(y[in_slice], b[in_slice])

    edges = np.arange(lo, hi + 1e-9, float(bin_width))
    if edges[-1] < hi - 1e-9:
        edges = np.append(edges, hi)

    per_bin: List[Dict[str, float]] = []
    for k in range(edges.size - 1):
        b_lo, b_hi = float(edges[k]), float(edges[k + 1])
        if k < edges.size - 2:
            m = (r >= b_lo) & (r < b_hi)
        else:
            m = (r >= b_lo) & (r <= b_hi)
        cnt = int(np.sum(m))
        if cnt < 5:
            continue
        wb = None if w_all is None else w_all[m]
        sp_rho = float(stats.spearmanr(y[m], p[m]).statistic) if np.std(y[m]) > 1e-12 and np.std(p[m]) > 1e-12 else 0.0
        # Compare against constant per-bin mean AND against q_baseline inside bin
        bin_mean_val = float(np.average(y[m], weights=wb)) if wb is not None else float(np.mean(y[m]))
        bin_const = np.full(cnt, bin_mean_val, dtype=np.float64)
        per_bin.append(
            {
                "bin_low": b_lo,
                "bin_high": b_hi,
                "n_frames": cnt,
                "cond_var_q_ref": float(np.var(y[m])),
                "cond_std_q_ref": float(np.std(y[m])),
                "spearman_rho": sp_rho,
                "bss_vs_b1": brier_skill_score(y[m], p[m], b[m], weights=wb),
                "bss_vs_bin_mean": brier_skill_score(y[m], p[m], bin_const, weights=wb),
                "delta_ell_vs_b1": delta_log_likelihood(y[m], p[m], b[m], weights=wb, eps=eps),
            }
        )

    weighted_mean_within_bin_var = (
        float(np.average([row["cond_var_q_ref"] for row in per_bin], weights=[row["n_frames"] for row in per_bin]))
        if per_bin
        else float(np.var(y[in_slice]))
    )

    return {
        "slice_low": lo,
        "slice_high": hi,
        "n_frames": n_slice,
        "bss_int": brier_skill_score(y[in_slice], p[in_slice], b[in_slice], weights=w_s),
        "delta_ell_int": delta_log_likelihood(y[in_slice], p[in_slice], b[in_slice], weights=w_s, eps=eps),
        "spearman_rho_model": 0.0 if np.isnan(rho_s) else float(rho_s),
        "spearman_pval_model": 1.0 if np.isnan(pval_s) else float(pval_s),
        "spearman_rho_b1": 0.0 if np.isnan(rho_b) else float(rho_b),
        "mean_within_bin_var_q_ref": weighted_mean_within_bin_var,
        "calibration": calibration_curve_soft(y[in_slice], p[in_slice], n_bins=8, weights=w_s),
        "per_bin": per_bin,
    }


def separatrix_purity(
    q_ref: np.ndarray,
    q_pred: np.ndarray,
    band: Tuple[float, float] = (0.45, 0.55),
) -> Dict[str, float]:
    """Evaluate true committor concentration on predicted separatrix q_hat in [0.45, 0.55]."""
    y = np.asarray(q_ref, dtype=np.float64).reshape(-1)
    p = np.asarray(q_pred, dtype=np.float64).reshape(-1)
    m = (p >= float(band[0])) & (p <= float(band[1]))
    cnt = int(np.sum(m))
    if cnt < 5:
        return {
            "n_separatrix": cnt,
            "mean_q_ref": float("nan"),
            "abs_bias_from_half": float("nan"),
            "std_q_ref": float("nan"),
            "iqr_q_ref": float("nan"),
            "bimodality_coeff": float("nan"),
            "frac_within_030_070": float("nan"),
        }

    ys = y[m]
    mu = float(np.mean(ys))
    sd = float(np.std(ys))
    q25, q75 = np.percentile(ys, [25.0, 75.0])
    # Sarle's bimodality coefficient BC = (skew^2 + 1) / (kurt_excess + 3 * (n-1)^2/((n-2)(n-3)))
    # Uniform distribution has BC = 5/9 ≈ 0.555; bimodal > 0.555; unimodal Gaussian = 1/3 ≈ 0.333
    if cnt >= 6 and sd > 1e-10:
        sk = float(stats.skew(ys, bias=False))
        ku = float(stats.kurtosis(ys, fisher=True, bias=False))
        denom = ku + 3.0 * ((cnt - 1) ** 2) / ((cnt - 2) * (cnt - 3))
        bc = float((sk * sk + 1.0) / denom) if denom > 1e-12 else 0.0
    else:
        bc = 0.0

    return {
        "n_separatrix": cnt,
        "mean_q_ref": mu,
        "abs_bias_from_half": abs(mu - 0.5),
        "std_q_ref": sd,
        "iqr_q_ref": float(q75 - q25),
        "bimodality_coeff": bc,
        "frac_within_030_070": float(np.mean((ys >= 0.30) & (ys <= 0.70))),
    }


def scale_invariant_gradient_cosine(
    w1: np.ndarray,
    w2: np.ndarray,
    Sigma_pool: np.ndarray,
    reg: float = 1e-12,
) -> float:
    """Compute affine-scale-invariant pooled-covariance cosine S_cos(L1, L2)."""
    v1 = np.asarray(w1, dtype=np.float64).reshape(-1)
    v2 = np.asarray(w2, dtype=np.float64).reshape(-1)
    S = np.asarray(Sigma_pool, dtype=np.float64)
    d = v1.size
    if v2.size != d or S.shape != (d, d):
        raise ValueError(f"Dimension mismatch: w1={v1.shape}, w2={v2.shape}, Sigma={S.shape}.")
    S_sym = 0.5 * (S + S.T) + float(reg) * np.eye(d, dtype=np.float64)
    n1 = float(v1.T @ S_sym @ v1)
    n2 = float(v2.T @ S_sym @ v2)
    if n1 <= 1e-24 or n2 <= 1e-24:
        return 0.0
    cos_val = float((v1.T @ S_sym @ v2) / np.sqrt(n1 * n2))
    return float(np.clip(cos_val, -1.0, 1.0))


def fast_additive_ratio_group_bootstrap_ci(
    group_ids: np.ndarray,
    numerator_terms: np.ndarray,
    denominator_terms: np.ndarray,
    transform: Callable[[float], float] = lambda r: r,
    n_boot: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> Dict[str, float]:
    """Exact whole-group cluster bootstrap for ratio statistics sum(num)/sum(den).

    Pre-aggregates per unique group ID so 1,000 resamples execute in <1 ms even for
    hundreds of thousands of frames.
    """
    g = np.asarray(group_ids)
    num = np.asarray(numerator_terms, dtype=np.float64).reshape(-1)
    den = np.asarray(denominator_terms, dtype=np.float64).reshape(-1)
    if g.size != num.size or g.size != den.size:
        raise ValueError("group_ids, numerator_terms, and denominator_terms must have equal length.")

    _, inv = np.unique(g, return_inverse=True)
    n_groups = int(inv.max()) + 1
    if n_groups < 2:
        pt = float(transform(float(np.sum(num) / max(float(np.sum(den)), 1e-15))))
        return {"point": pt, "ci_low": pt, "ci_high": pt, "std_err": 0.0, "n_groups": n_groups}

    g_num = np.bincount(inv, weights=num, minlength=n_groups)
    g_den = np.bincount(inv, weights=den, minlength=n_groups)

    pt_raw = float(np.sum(g_num) / max(float(np.sum(g_den)), 1e-15))
    point = float(transform(pt_raw))

    rng = np.random.default_rng(random_state)
    draws = rng.integers(0, n_groups, size=(int(n_boot), n_groups))
    b_num = np.sum(g_num[draws], axis=1)
    b_den = np.maximum(np.sum(g_den[draws], axis=1), 1e-15)
    ratios = b_num / b_den
    boot_vals = np.array([transform(float(r)) for r in ratios], dtype=np.float64)

    alpha = 0.5 * (1.0 - float(ci))
    lo, hi = np.quantile(boot_vals, [alpha, 1.0 - alpha])
    return {
        "point": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "std_err": float(np.std(boot_vals, ddof=1)),
        "n_groups": n_groups,
    }


def bootstrap_bss_and_delta_ell_ci(
    group_ids: np.ndarray,
    q_ref: np.ndarray,
    q_pred: np.ndarray,
    q_baseline: np.ndarray,
    weights: Optional[np.ndarray] = None,
    n_boot: int = 1000,
    ci: float = 0.95,
    eps: float = 1e-7,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Compute 1,000-group cluster bootstrap 95% CIs for BSS_int and Delta_ell_int."""
    y = np.clip(np.asarray(q_ref, dtype=np.float64).reshape(-1), 0.0, 1.0)
    p = np.clip(np.asarray(q_pred, dtype=np.float64).reshape(-1), eps, 1.0 - eps)
    b = np.clip(np.asarray(q_baseline, dtype=np.float64).reshape(-1), eps, 1.0 - eps)
    w = np.ones_like(y) if weights is None else np.clip(np.asarray(weights, dtype=np.float64).reshape(-1), 0.0, None)

    sq_mod = w * ((p - y) ** 2)
    sq_bas = w * ((b - y) ** 2)
    bss_ci = fast_additive_ratio_group_bootstrap_ci(
        group_ids=group_ids,
        numerator_terms=sq_mod,
        denominator_terms=sq_bas,
        transform=lambda ratio: 1.0 - ratio,
        n_boot=n_boot,
        ci=ci,
        random_state=random_state,
    )

    ll_diff = w * ((y * np.log(p) + (1.0 - y) * np.log(1.0 - p)) - (y * np.log(b) + (1.0 - y) * np.log(1.0 - b)))
    dell_ci = fast_additive_ratio_group_bootstrap_ci(
        group_ids=group_ids,
        numerator_terms=ll_diff,
        denominator_terms=w,
        transform=lambda r: r,
        n_boot=n_boot,
        ci=ci,
        random_state=random_state + 1,
    )
    return {"bss": bss_ci, "delta_ell": dell_ci}


def group_block_bootstrap_ci(
    group_ids: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    n_boot: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> Dict[str, float]:
    """General whole-group cluster bootstrap for arbitrary statistics stat_fn(sample_indices)."""
    g = np.asarray(group_ids)
    unique_g, inv = np.unique(g, return_inverse=True)
    n_groups = unique_g.size
    all_idx = np.arange(g.size, dtype=np.int64)
    point = float(stat_fn(all_idx))
    if n_groups < 2:
        return {"point": point, "ci_low": point, "ci_high": point, "std_err": 0.0, "n_groups": int(n_groups)}

    group_buckets = [np.flatnonzero(inv == k) for k in range(n_groups)]
    rng = np.random.default_rng(random_state)
    boot_vals = np.empty(int(n_boot), dtype=np.float64)
    for b in range(int(n_boot)):
        picked = rng.integers(0, n_groups, size=n_groups)
        idx_b = np.concatenate([group_buckets[k] for k in picked])
        boot_vals[b] = float(stat_fn(idx_b))

    alpha = 0.5 * (1.0 - float(ci))
    lo, hi = np.quantile(boot_vals, [alpha, 1.0 - alpha])
    return {
        "point": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "std_err": float(np.std(boot_vals, ddof=1)),
        "n_groups": int(n_groups),
    }
