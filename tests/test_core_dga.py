"""Comprehensive analytical & synthetic unit tests for src.core.

Validates:
1. Exact 2D switchback Langevin PDE ground-truth recovery via StoppedDGACommittor
   on short non-Boltzmann-seeded trajectories.
2. Initial-state path-reweighted backward committor N2b identity (q+ + q- ≈ 1).
3. Out-of-sample Projected Bellman Residual (E_PBR) ordering without double-sampling bias.
4. Orthogonal committor variance & H1 detection at matched x-slices (pure-x target A/B).
5. Two-step Mori–Zwanzig memory detection (y hidden) & collapse (y revealed, rho_mem > 0.6).
6. Scale-invariant gradient cosine affine invariance, discrete MSM TPT, and leakage guards.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse import linalg as spla

from src.core import (
    LineageGroupSplitter,
    PetersTroutCommittorModel,
    SplineProgressBaselineB1,
    StoppedDGACommittor,
    bootstrap_bss_and_delta_ell_ci,
    build_contiguous_macro_groups,
    build_forward_stopped_pairs,
    build_lineage_connected_components,
    compute_memory_collapse_ratio,
    evaluate_committor_pbr,
    evaluate_mori_zwanzig_memory,
    estimate_reversible_msm_stationary_weights,
    matched_slice_metrics,
    run_group_permutation_null_N1,
    run_time_reversal_diagnostic_N2b,
    scale_invariant_gradient_cosine,
    segment_concatenated_array,
    separatrix_purity,
    solve_msm_tpt_committor,
)


# ---------------------------------------------------------------------------
# Analytical 2D Berezhkovskii–Szabo Switchback Potential & Exact Grid PDE Solver
# ---------------------------------------------------------------------------

BETA = 2.5
DX_DIFF = 1.0
DY_DIFF = 0.35  # Slower orthogonal gating mode y
X_BOUND_A = -0.95
X_BOUND_B = 0.95


def switchback_potential_and_grad(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smooth 2D two-channel switchback potential V(x, y) and analytical (dV/dx, dV/dy)."""
    uA = 1.35 * ((x + 0.45) ** 2) + 4.2 * ((y + 0.80) ** 2)
    uB = 1.35 * ((x - 0.45) ** 2) + 4.2 * ((y - 0.80) ** 2)
    duA_dx = 2.70 * (x + 0.45)
    duA_dy = 8.40 * (y + 0.80)
    duB_dx = 2.70 * (x - 0.45)
    duB_dy = 8.40 * (y - 0.80)

    # Numerically stable log-sum-exp for -1/beta * log(exp(-beta uA) + exp(-beta uB))
    m = np.minimum(uA, uB)
    eA = np.exp(-BETA * (uA - m))
    eB = np.exp(-BETA * (uB - m))
    denom = eA + eB
    wA = eA / denom
    wB = eB / denom

    conf = 0.15 * (x ** 4 + y ** 4)
    V = m - (1.0 / BETA) * np.log(denom) + conf
    dV_dx = wA * duA_dx + wB * duB_dx + 0.60 * (x ** 3)
    dV_dy = wA * duA_dy + wB * duB_dy + 0.60 * (y ** 3)
    return V, dV_dx, dV_dy


def solve_exact_2d_switchback_committor(
    nx: int = 91,
    ny: int = 91,
    xlim: Tuple[float, float] = (-1.6, 1.6),
    ylim: Tuple[float, float] = (-1.6, 1.6),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve div(pi(x,y) D grad q+) = 0 on a 2D finite-volume grid with pure-x A/B boundaries."""
    xs = np.linspace(xlim[0], xlim[1], nx)
    ys = np.linspace(ylim[0], ylim[1], ny)
    hx = xs[1] - xs[0]
    hy = ys[1] - ys[0]
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    V, _, _ = switchback_potential_and_grad(XX, YY)
    pi = np.exp(-BETA * (V - np.min(V)))

    N = nx * ny
    rows, cols, vals = [], [], []
    rhs = np.zeros(N, dtype=np.float64)

    def flat(i: int, j: int) -> int:
        return i * ny + j

    for i in range(nx):
        for j in range(ny):
            k = flat(i, j)
            if xs[i] <= X_BOUND_A:
                rows.append(k)
                cols.append(k)
                vals.append(1.0)
                rhs[k] = 0.0
                continue
            if xs[i] >= X_BOUND_B:
                rows.append(k)
                cols.append(k)
                vals.append(1.0)
                rhs[k] = 1.0
                continue

            diag = 0.0
            # x-neighbors (harmonic/midpoint flux weight)
            for di in (-1, 1):
                ni = i + di
                if 0 <= ni < nx:
                    pi_mid = 0.5 * (pi[i, j] + pi[ni, j])
                    c = DX_DIFF * pi_mid / (hx * hx)
                    rows.append(k)
                    cols.append(flat(ni, j))
                    vals.append(c)
                    diag -= c
            # y-neighbors (reflecting Neumann at y boundaries)
            for dj in (-1, 1):
                nj = j + dj
                if 0 <= nj < ny:
                    pi_mid = 0.5 * (pi[i, j] + pi[i, nj])
                    c = DY_DIFF * pi_mid / (hy * hy)
                    rows.append(k)
                    cols.append(flat(i, nj))
                    vals.append(c)
                    diag -= c

            rows.append(k)
            cols.append(k)
            vals.append(diag)

    mat = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))
    q_grid = spla.spsolve(mat, rhs).reshape((nx, ny))
    return xs, ys, np.clip(q_grid, 0.0, 1.0)


def interpolate_2d_grid(
    xs: np.ndarray,
    ys: np.ndarray,
    q_grid: np.ndarray,
    pts: np.ndarray,
) -> np.ndarray:
    """Bilinear interpolation of exact PDE committor at trajectory points."""
    x = np.clip(pts[:, 0], xs[0], xs[-1] - 1e-9)
    y = np.clip(pts[:, 1], ys[0], ys[-1] - 1e-9)
    hx = xs[1] - xs[0]
    hy = ys[1] - ys[0]
    fx = (x - xs[0]) / hx
    fy = (y - ys[0]) / hy
    ix = np.clip(np.floor(fx).astype(int), 0, xs.size - 2)
    iy = np.clip(np.floor(fy).astype(int), 0, ys.size - 2)
    tx = fx - ix
    ty = fy - iy
    q00 = q_grid[ix, iy]
    q10 = q_grid[ix + 1, iy]
    q01 = q_grid[ix, iy + 1]
    q11 = q_grid[ix + 1, iy + 1]
    q = (1 - tx) * (1 - ty) * q00 + tx * (1 - ty) * q10 + (1 - tx) * ty * q01 + tx * ty * q11
    q = np.where(pts[:, 0] <= X_BOUND_A, 0.0, np.where(pts[:, 0] >= X_BOUND_B, 1.0, q))
    return q


def simulate_adaptive_short_trajectories(
    n_traj: int = 320,
    n_steps_saved: int = 60,
    substeps: int = 5,
    dt: float = 0.004,
    seed: int = 12345,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate short non-Boltzmann (uniformly seeded) overdamped Langevin trajectories."""
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-1.25, 1.25, size=n_traj)
    y0 = rng.uniform(-1.25, 1.25, size=n_traj)

    traj = np.empty((n_traj, n_steps_saved, 2), dtype=np.float64)
    x = x0.copy()
    y = y0.copy()
    sig_x = np.sqrt(2.0 * DX_DIFF * dt)
    sig_y = np.sqrt(2.0 * DY_DIFF * dt)

    for step in range(n_steps_saved):
        traj[:, step, 0] = x
        traj[:, step, 1] = y
        for _ in range(substeps):
            _, dVx, dVy = switchback_potential_and_grad(x, y)
            x = x - BETA * DX_DIFF * dVx * dt + sig_x * rng.standard_normal(n_traj)
            y = y - BETA * DY_DIFF * dVy * dt + sig_y * rng.standard_normal(n_traj)

    X_flat = traj.reshape(-1, 2)
    seg_id = np.repeat(np.arange(n_traj, dtype=np.int64), n_steps_saved)
    return X_flat, seg_id


@pytest.fixture(scope="module")
def switchback_benchmark():
    xs, ys, q_grid = solve_exact_2d_switchback_committor()
    X, seg_id = simulate_adaptive_short_trajectories()
    in_A = X[:, 0] <= X_BOUND_A
    in_B = X[:, 0] >= X_BOUND_B
    q_exact = interpolate_2d_grid(xs, ys, q_grid, X)
    return {
        "xs": xs,
        "ys": ys,
        "q_grid": q_grid,
        "X": X,
        "seg_id": seg_id,
        "in_A": in_A,
        "in_B": in_B,
        "q_exact": q_exact,
    }


def test_analytical_switchback_dga_and_n2b_reversibility(switchback_benchmark):
    X = switchback_benchmark["X"]
    seg_id = switchback_benchmark["seg_id"]
    in_A = switchback_benchmark["in_A"]
    in_B = switchback_benchmark["in_B"]
    q_exact = switchback_benchmark["q_exact"]

    # Verify analytical ground truth itself exhibits strong orthogonal committor variance at x ≈ 0
    mid_slice = (X[:, 0] >= -0.15) & (X[:, 0] <= 0.15)
    assert np.std(q_exact[mid_slice]) > 0.25

    # Split strictly by trajectory ID into train and test
    train_mask = seg_id < 220
    test_mask = seg_id >= 220

    dga = StoppedDGACommittor(lag_steps=6, n_rbf=70, reg=1e-4, random_state=42)
    dga.fit(X[train_mask], in_A[train_mask], in_B[train_mask], seg_id=seg_id[train_mask])

    q_pred_test = dga.predict(X[test_mask], in_A[test_mask], in_B[test_mask])
    interior_test = ~(in_A[test_mask] | in_B[test_mask])

    rmse = float(np.sqrt(np.mean((q_pred_test[interior_test] - q_exact[test_mask][interior_test]) ** 2)))
    corr = float(np.corrcoef(q_pred_test[interior_test], q_exact[test_mask][interior_test])[0, 1])
    assert rmse < 0.075, f"Stopped DGA RMSE vs analytical PDE too high: {rmse:.4f}"
    assert corr > 0.975, f"Stopped DGA correlation vs analytical PDE too low: {corr:.4f}"

    # Verify N2b backward committor with initial-state path reweighting w(X_{t_0})
    n2b = run_time_reversal_diagnostic_N2b(
        dga, X[train_mask], in_A[train_mask], in_B[train_mask], seg_id=seg_id[train_mask]
    )
    assert n2b["passes_threshold"], f"N2b reversibility failed on reversible Langevin: {n2b}"
    assert n2b["l2_reversibility_error"] < 0.085


def test_projected_bellman_residual_avoids_double_sampling_bias(switchback_benchmark):
    X = switchback_benchmark["X"]
    seg_id = switchback_benchmark["seg_id"]
    in_A = switchback_benchmark["in_A"]
    in_B = switchback_benchmark["in_B"]
    xs, ys, q_grid = switchback_benchmark["xs"], switchback_benchmark["ys"], switchback_benchmark["q_grid"]

    test_mask = seg_id >= 200
    X_te, A_te, B_te, seg_te = X[test_mask], in_A[test_mask], in_B[test_mask], seg_id[test_mask]

    # Exact 2D committor vs naive 1D linear-x guess
    pbr_exact = evaluate_committor_pbr(
        lambda pts: interpolate_2d_grid(xs, ys, q_grid, pts),
        X_te,
        A_te,
        B_te,
        seg_te,
        lag_steps=5,
        n_witness=25,
    )
    pbr_1d_naive = evaluate_committor_pbr(
        lambda pts: np.clip((pts[:, 0] - X_BOUND_A) / (X_BOUND_B - X_BOUND_A), 0.0, 1.0),
        X_te,
        A_te,
        B_te,
        seg_te,
        lag_steps=5,
        n_witness=25,
    )
    assert pbr_exact < 0.035
    assert pbr_exact < 0.30 * pbr_1d_naive


def test_h1_matched_slice_and_two_step_mori_zwanzig_memory_collapse(switchback_benchmark):
    X = switchback_benchmark["X"]
    seg_id = switchback_benchmark["seg_id"]
    in_A = switchback_benchmark["in_A"]
    in_B = switchback_benchmark["in_B"]
    q_exact = switchback_benchmark["q_exact"]

    # Group trajectories into 20 lineage groups
    groups = seg_id // 16
    in_mid = (X[:, 0] >= -0.45) & (X[:, 0] <= 0.45)

    # 1. H1 test at matched intermediate x in [-0.45, +0.45]
    res_h1 = run_group_permutation_null_N1(
        X=X,
        q_target=q_exact,
        groups=groups,
        progress_col=0,
        slice_range=(-0.45, 0.45),
        in_A=in_A,
        in_B=in_B,
        fit_on_slice=True,
        include_interactions=True,
        n_splits=5,
        n_perms=40,
        random_state=42,
    )
    assert res_h1["obs_bss_vs_b1"] > 0.85
    assert res_h1["obs_delta_ell_vs_b1"] > 0.30
    assert res_h1["p_value_bss_vs_n1"] <= 0.05

    slice_stats = matched_slice_metrics(
        progress_coord=X[:, 0],
        q_ref=q_exact,
        q_pred=res_h1["oof_h1"],
        q_baseline=res_h1["oof_b1"],
        slice_range=(-0.45, 0.45),
        bin_width=0.30,
    )
    assert slice_stats["bss_int"] > 0.85
    sep_h1 = separatrix_purity(q_exact[in_mid], res_h1["oof_h1"][in_mid])
    sep_b1 = separatrix_purity(q_exact[in_mid], res_h1["oof_b1"][in_mid])
    assert sep_h1["std_q_ref"] < 0.50 * sep_b1["std_q_ref"]
    assert sep_h1["bimodality_coeff"] < sep_b1["bimodality_coeff"]

    boot = bootstrap_bss_and_delta_ell_ci(
        group_ids=groups[in_mid],
        q_ref=q_exact[in_mid],
        q_pred=res_h1["oof_h1"][in_mid],
        q_baseline=res_h1["oof_b1"][in_mid],
        n_boot=500,
    )
    assert boot["bss"]["ci_low"] > 0.75

    # 2. H2 Two-Step Mori–Zwanzig Memory Detection & Collapse:
    # Step A: Observe ONLY x(t) (omitting slow orthogonal gate y(t)) -> positive memory!
    eval_slice = (X[:, 0] >= -0.55) & (X[:, 0] <= 0.55)
    mem_hidden = evaluate_mori_zwanzig_memory(
        X_static=X[:, [0]],
        seg_id=seg_id,
        q_target=q_exact,
        groups=groups,
        lag_steps=8,
        eval_mask=eval_slice,
        in_A=in_A,
        in_B=in_B,
        include_interactions=True,
        n_splits=5,
        n_shuffles=40,
        random_state=42,
    )
    assert mem_hidden["delta_ell_mem"] > 0.04, f"Expected positive MZ memory when y hidden: {mem_hidden}"
    assert mem_hidden["p_value_vs_n2a"] <= 0.05

    # Step B: Reveal full Markovian coordinates (x(t), y(t)) -> memory collapses toward zero!
    mem_revealed = evaluate_mori_zwanzig_memory(
        X_static=X,
        seg_id=seg_id,
        q_target=q_exact,
        groups=groups,
        lag_steps=8,
        eval_mask=eval_slice,
        in_A=in_A,
        in_B=in_B,
        include_interactions=True,
        n_splits=5,
        n_shuffles=30,
        random_state=42,
    )
    rho_collapse = compute_memory_collapse_ratio(
        mem_hidden["delta_ell_mem"], mem_revealed["delta_ell_mem"]
    )
    assert rho_collapse > 0.85, f"Expected >=85% memory collapse upon revealing y, got {rho_collapse:.3f}"


def test_msm_tpt_and_scale_invariant_cosine_and_split_guards():
    # 1. Exact 4-state reversible birth-death chain A={0} <-> 1 <-> 2 <-> B={3}
    T = np.array(
        [
            [0.80, 0.20, 0.00, 0.00],
            [0.25, 0.50, 0.25, 0.00],
            [0.00, 0.25, 0.50, 0.25],
            [0.00, 0.00, 0.20, 0.80],
        ],
        dtype=np.float64,
    )
    res = solve_msm_tpt_committor(T, A_states=[0], B_states=[3], lag_time=2.0)
    assert np.allclose(res.q_plus, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], atol=1e-6)
    assert np.allclose(res.q_plus + res.q_minus, 1.0, atol=1e-6)
    assert res.reversibility_l2_error < 1e-12
    assert res.total_reactive_flux > 0.0

    # 2. Affine unit-rescaling invariance of S_cos(L1, L2)
    rng = np.random.default_rng(7)
    X_pool = rng.normal(size=(500, 3))
    Sigma = np.cov(X_pool, rowvar=False)
    w1 = np.array([1.2, -0.8, 0.4])
    w2 = np.array([0.3, 1.1, 0.9])
    cos_orig = scale_invariant_gradient_cosine(w1, w2, Sigma)

    # Rescale coordinates X_tilde = X @ diag(scales) -> Sigma_tilde = D Sigma D, w_tilde = D^{-1} w
    scales = np.array([10.0, 0.05, 3.7])
    Sigma_scaled = np.diag(scales) @ Sigma @ np.diag(scales)
    w1_scaled = w1 / scales
    w2_scaled = w2 / scales
    cos_scaled = scale_invariant_gradient_cosine(w1_scaled, w2_scaled, Sigma_scaled)
    assert np.isclose(cos_orig, cos_scaled, atol=1e-10)

    # 3. Concatenation jump segmentation & horizon-safe pair guard
    seg1 = np.cumsum(rng.normal(0.0, 0.01, size=(30, 3)), axis=0)
    seg2 = np.cumsum(rng.normal(0.0, 0.01, size=(25, 3)), axis=0) + np.array([1.5, -1.2, 0.8])
    X_cat = np.vstack([seg1, seg2])
    seg_id, jumps = segment_concatenated_array(X_cat, jump_threshold=0.10)
    assert jumps.tolist() == [30]
    in_A = np.zeros(55, dtype=bool)
    in_B = np.zeros(55, dtype=bool)
    in_A[0] = True
    in_B[-1] = True
    s_idx, e_idx = build_forward_stopped_pairs(seg_id, in_A, in_B, lag_steps=5)
    assert np.all(seg_id[s_idx] == seg_id[e_idx])
    # Verify frames 25..29 in seg1 (within lag=5 of seam at 30) are excluded by horizon filter!
    assert not np.any((s_idx >= 25) & (s_idx < 30))

    # 4. Lineage connected components + group split disjointness
    rounds = ["pr_1", "pr_1", "pr_2", "pr_3", "pr_4", "pr_5", "pr_6"]
    rmsd = np.full((7, 7), 3.5)
    np.fill_diagonal(rmsd, 0.0)
    rmsd[2, 3] = rmsd[3, 2] = 0.65  # connects pr_2 and pr_3
    comp = build_lineage_connected_components(rounds, rmsd, rmsd_cutoff=1.0)
    assert comp[0] == comp[1]
    assert comp[2] == comp[3]
    assert np.unique(comp).size == 5

    g_macro, vmask = build_contiguous_macro_groups(seg_id, n_macro_blocks=2, buffer_segments=0)
    assert np.all(vmask)
    splitter = LineageGroupSplitter(n_splits=5)
    frame_groups = np.repeat(comp, 10)
    for tr, te in splitter.split(np.zeros((70, 2)), None, frame_groups):
        assert set(frame_groups[tr]).isdisjoint(set(frame_groups[te]))


def test_asymmetric_90_10_seeding_stationary_reweighting_and_n2b():
    """Verify stationary eigenvector reweighting recovers Boltzmann balance under 90/10 seeding."""
    rng = np.random.default_rng(2026)
    n_left = 450   # 90% seeded in left basin/channel
    n_right = 50   # 10% seeded in right basin/channel
    n_traj = n_left + n_right
    n_steps_saved = 20
    substeps = 4
    dt = 0.004

    x0 = np.concatenate([
        rng.uniform(-1.35, -0.05, size=n_left),
        rng.uniform(0.05, 1.35, size=n_right),
    ])
    y0 = np.concatenate([
        rng.uniform(-1.20, 0.15, size=n_left),
        rng.uniform(-0.15, 1.20, size=n_right),
    ])

    traj = np.empty((n_traj, n_steps_saved, 2), dtype=np.float64)
    x = x0.copy()
    y = y0.copy()
    sig_x = np.sqrt(2.0 * DX_DIFF * dt)
    sig_y = np.sqrt(2.0 * DY_DIFF * dt)
    for step in range(n_steps_saved):
        traj[:, step, 0] = x
        traj[:, step, 1] = y
        for _ in range(substeps):
            _, dVx, dVy = switchback_potential_and_grad(x, y)
            x = x - BETA * DX_DIFF * dVx * dt + sig_x * rng.standard_normal(n_traj)
            y = y - BETA * DY_DIFF * dVy * dt + sig_y * rng.standard_normal(n_traj)

    X_asym = traj.reshape(-1, 2)
    seg_asym = np.repeat(np.arange(n_traj, dtype=np.int64), n_steps_saved)
    in_A = X_asym[:, 0] <= X_BOUND_A
    in_B = X_asym[:, 0] >= X_BOUND_B

    raw_right_frac = float(np.mean(X_asym[:, 0] > 0.0))
    assert raw_right_frac < 0.25, f"Expected strongly skewed raw right-half occupancy, got {raw_right_frac:.3f}"

    w_stat, _, cluster_w = estimate_reversible_msm_stationary_weights(
        X_asym, seg_asym, lag_steps=5, n_clusters=35, random_state=42
    )
    reweighted_right_mass = float(np.sum(w_stat[X_asym[:, 0] > 0.0]) / np.sum(w_stat))
    # Right-half stationary weight should jump substantially (>2x) from ~0.19 toward symmetric 0.50
    assert 0.36 <= reweighted_right_mass <= 0.64, (
        f"Stationary reweighting failed to recover Boltzmann balance: "
        f"raw={raw_right_frac:.3f}, reweighted={reweighted_right_mass:.3f}"
    )
    assert float(np.max(cluster_w) / np.min(cluster_w)) > 4.0

    dga_rw = StoppedDGACommittor(lag_steps=5, n_rbf=50, reg=1e-4, random_state=42).fit(
        X_asym, in_A, in_B, seg_id=seg_asym
    )
    n2b_rw = run_time_reversal_diagnostic_N2b(
        dga_rw, X_asym, in_A, in_B, seg_id=seg_asym, reweight=True
    )

    dga_unw = StoppedDGACommittor(lag_steps=5, n_rbf=50, reg=1e-4, random_state=42).fit(
        X_asym, in_A, in_B, seg_id=seg_asym
    )
    n2b_unw = run_time_reversal_diagnostic_N2b(
        dga_unw, X_asym, in_A, in_B, seg_id=seg_asym, reweight=False
    )

    assert n2b_rw["passes_threshold"], f"Reweighted N2b failed on reversible 90/10 Langevin: {n2b_rw}"
    assert not n2b_unw["passes_threshold"], f"Unweighted N2b unexpectedly passed under 90/10 skew: {n2b_unw}"
    assert n2b_rw["l2_reversibility_error"] < 0.35 * n2b_unw["l2_reversibility_error"], (
        f"Expected reweight=True ({n2b_rw['l2_reversibility_error']:.4f}) to slash "
        f"reweight=False ({n2b_unw['l2_reversibility_error']:.4f}) error under 90/10 seeding!"
    )
