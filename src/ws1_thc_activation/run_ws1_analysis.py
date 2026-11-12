"""Complete leakage-free Phase 2 Workstream 1 execution script on Tier 1C (ShuklaGroup/THC_binding).

Executes:
1. Data audit, nm->Angstrom conversion, and multi-threshold segmentation sensitivity
   (z_step > 5.0 primary L2 standardized norm, 2-CV 0.15 nm, and 3-CV {0.10, 0.12, 0.15, 0.20} nm).
2. Strict 5-fold stratified buffered macro-block Stopped DGA committor estimation across
   lags tau in {2, 5, 10, 15, 25} steps, with strict fold isolation:
   - Training fold j fits ``dga_tr_j`` strictly on ``tr_j`` alone; all fold-j parametric/GBR/memory
     predictors train strictly on ``q_tr_j = dga_tr_j.predict(X[tr_j])`` (zero dependency on ``te_j``).
   - Held-out test fold j fits an independent ``dga_te_j`` strictly on ``te_j`` alone (zero dependency
     on ``tr_j``) for committor evaluation alongside model-free test-fold Projected Bellman Residual
     (E_PBR) and path-reweighted N2b time-reversal diagnostics (||q+ + q- - 1||_L2).
3. Three complementary non-tautological committor regimes:
   - Regime I (Global Full-Activation): A_TM6 (R_TM6 <= 11.8 A) -> B_TM6 (R_TM6 >= 15.2 A)
     on the pooled 3-CV coordinate space (278,196 frames, zero ligand-identity input, macro-blocks
     stratified by ligand ensemble), plus dwell-3 and 3-CV crystal-anchored core-boundary sensitivity.
   - Regime II (Within-Ligand Natural Transitions):
     * Full Agonist alone: A_ago (R_TM6 <= 13.2 A) -> B_ago (R_TM6 >= 15.2 A), slice [13.7, 14.7] A.
     * Delta9-THC alone: A_thc (R_TM6 <= 11.8 A) -> B_thc (R_TM6 >= 13.5 A), slice [12.2, 13.1] A.
   - Regime III (Within-Ligand Shared-Boundary Window):
     * Identical boundaries A_sh (R_TM6 <= 12.8 A) -> B_sh (R_TM6 >= 13.8 A), slice [13.0, 13.6] A,
       estimated independently within Full Agonist and within Delta9-THC.
4. Hypothesis H1 (Matched-TM6 Orthogonal Microswitch Gating & Bilinear Epistasis) vs B0, B1, B2,
   OracleHistGBR, and N1 group-permutation null with 1,000-macro-group bootstrap 95% CIs.
5. Hypothesis H2 Step A (Mori-Zwanzig Memory Detection across lags {2, 5, 10, 20, 35} vs N2a
   within-static-cell history permutation null, plus 1-CV -> 3-CV partial memory collapse check).
6. Hypothesis H3 (Scale-invariant pooled-covariance committor gradient cosine S_cos between
   Full Agonist and Delta9-THC vs operator-independent within-ligand split-half reliability ceiling
   and fold-isolated MWC shared-gradient null).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.core import (
    LineageGroupSplitter,
    OracleHistGBRModel,
    PetersTroutCommittorModel,
    PriorBaselineB0,
    ProgressVelocityBaselineB2,
    SplineProgressBaselineB1,
    StoppedDGACommittor,
    assert_disjoint_group_split,
    bootstrap_bss_and_delta_ell_ci,
    brier_skill_score,
    compute_lagged_displacements,
    compute_memory_collapse_ratio,
    delta_log_likelihood,
    evaluate_committor_pbr,
    evaluate_mori_zwanzig_memory,
    matched_slice_metrics,
    run_group_permutation_null_N1,
    run_time_reversal_diagnostic_N2b,
    scale_invariant_gradient_cosine,
    separatrix_purity,
)
from src.ws1_thc_activation.load_and_segment import (
    PROJECT_ROOT,
    load_and_build_ws1_dataset,
)


RESULTS_WS1_DIR = PROJECT_ROOT / "results" / "ws1"
FEATURE_NAMES = ["r_tm6_angstrom", "d_toggle_angstrom", "d_npxxy_angstrom"]


@dataclass
class FoldIsolatedDGABundle:
    """Strictly fold-isolated train/test DGA committors and operator diagnostics."""

    folds: List[Tuple[np.ndarray, np.ndarray]]
    fold_q_train: List[np.ndarray]
    q_te_indep: np.ndarray
    q_tr_oof_pred: np.ndarray
    lag_sweep: Dict[str, object]


def _resegment_subset_indices(orig_indices: np.ndarray, seg_id_subset: np.ndarray) -> np.ndarray:
    """Ensure contiguous segment IDs whenever frame subsetting introduces index gaps."""
    idx = np.asarray(orig_indices, dtype=np.int64)
    seg = np.asarray(seg_id_subset, dtype=np.int64)
    if idx.size <= 1:
        return np.zeros(idx.size, dtype=np.int64)
    breaks = (np.diff(idx) > 1) | (np.diff(seg) != 0)
    return np.r_[0, np.cumsum(breaks, dtype=np.int64)]


def solve_fold_isolated_dga_bundle(
    X: np.ndarray,
    in_A: np.ndarray,
    in_B: np.ndarray,
    seg_id: np.ndarray,
    groups: np.ndarray,
    stratify_labels: Optional[np.ndarray] = None,
    candidate_lags: Tuple[int, ...] = (2, 5, 10, 15, 25),
    primary_lag: int = 15,
    n_rbf: int = 65,
    reg: float = 1e-3,
    n_splits: int = 5,
    random_state: int = 42,
) -> FoldIsolatedDGABundle:
    """Fit train-fold DGA on tr_j alone and independent test-fold DGA on te_j alone."""
    y_strata = (
        (np.asarray(in_A, dtype=int) + 2 * np.asarray(in_B, dtype=int))
        if stratify_labels is None
        else np.asarray(stratify_labels)
    )
    splitter = LineageGroupSplitter(
        n_splits=n_splits, stratified=True, random_state=random_state
    )
    folds = [(np.asarray(tr, dtype=np.int64), np.asarray(te, dtype=np.int64)) for tr, te in splitter.split(X, y_strata, groups)]

    lag_sweep_results: Dict[str, Dict[str, float]] = {}
    fold_q_train_primary: List[np.ndarray] = []
    q_te_indep_primary = np.zeros(X.shape[0], dtype=np.float64)
    q_tr_oof_primary = np.zeros(X.shape[0], dtype=np.float64)

    for lag in candidate_lags:
        oof_q_from_tr = np.zeros(X.shape[0], dtype=np.float64)
        pbr_dga_folds: List[float] = []
        pbr_b1_folds: List[float] = []
        pbr_pt_folds: List[float] = []
        n2b_folds: List[float] = []
        n2b_te_folds: List[float] = []
        n_pairs_folds: List[int] = []
        fold_q_tr_cur: List[np.ndarray] = []
        q_te_indep_cur = np.zeros(X.shape[0], dtype=np.float64)

        for f_idx, (tr, te) in enumerate(folds):
            assert_disjoint_group_split(groups, tr, te)
            seg_tr = _resegment_subset_indices(tr, seg_id[tr])
            seg_te = _resegment_subset_indices(te, seg_id[te])

            # 1. Fit training-fold DGA strictly on tr alone (zero access to te!)
            dga_tr = StoppedDGACommittor(
                lag_steps=lag, n_rbf=n_rbf, reg=reg, random_state=random_state + f_idx
            ).fit(X[tr], in_A[tr], in_B[tr], seg_id=seg_tr)

            q_tr = dga_tr.predict(X[tr], in_A[tr], in_B[tr])
            q_te_pred = dga_tr.predict(X[te], in_A[te], in_B[te])
            oof_q_from_tr[te] = q_te_pred
            n_pairs_folds.append(dga_tr.n_interior_pairs_)

            # Path-reweighted N2b reversibility on training fold
            n2b = run_time_reversal_diagnostic_N2b(
                dga_tr, X[tr], in_A[tr], in_B[tr], seg_id=seg_tr, reweight=True
            )
            n2b_folds.append(n2b["l2_reversibility_error"])

            # Held-out test-fold Projected Bellman Residual (model-free operator validation)
            int_tr = ~(in_A[tr] | in_B[tr])
            b1_int = SplineProgressBaselineB1(progress_col=0, l2_reg=1e-3).fit(
                X[tr][int_tr], q_tr[int_tr]
            )
            pt_int = PetersTroutCommittorModel(include_interactions=True, l2_reg=1e-3).fit(
                X[tr][int_tr], q_tr[int_tr]
            )

            pbr_dga_folds.append(
                evaluate_committor_pbr(
                    lambda pts, m=dga_tr: m.predict(pts),
                    X[te],
                    in_A[te],
                    in_B[te],
                    seg_te,
                    lag_steps=lag,
                    n_witness=35,
                )
            )
            pbr_b1_folds.append(
                evaluate_committor_pbr(
                    lambda pts, m=b1_int: m.predict(pts),
                    X[te],
                    in_A[te],
                    in_B[te],
                    seg_te,
                    lag_steps=lag,
                    n_witness=35,
                )
            )
            pbr_pt_folds.append(
                evaluate_committor_pbr(
                    lambda pts, m=pt_int: m.predict(pts),
                    X[te],
                    in_A[te],
                    in_B[te],
                    seg_te,
                    lag_steps=lag,
                    n_witness=35,
                )
            )

            if lag == primary_lag:
                fold_q_tr_cur.append(q_tr)
                # 2. Fit independent test-fold DGA strictly on te alone (zero access to tr!)
                dga_te = StoppedDGACommittor(
                    lag_steps=lag,
                    n_rbf=min(n_rbf, 50),
                    reg=reg,
                    random_state=random_state + 100 + f_idx,
                ).fit(X[te], in_A[te], in_B[te], seg_id=seg_te)
                q_te_indep_cur[te] = dga_te.predict(X[te], in_A[te], in_B[te])
                n2b_te = run_time_reversal_diagnostic_N2b(
                    dga_te, X[te], in_A[te], in_B[te], seg_id=seg_te, reweight=True
                )
                n2b_te_folds.append(n2b_te["l2_reversibility_error"])

        entry: Dict[str, object] = {
            "lag_steps": int(lag),
            "mean_train_interior_pairs": float(np.mean(n_pairs_folds)),
            "mean_test_pbr_dga_3cv": float(np.mean(pbr_dga_folds)),
            "std_test_pbr_dga_3cv": float(np.std(pbr_dga_folds, ddof=1)),
            "mean_test_pbr_pt_3cv": float(np.mean(pbr_pt_folds)),
            "mean_test_pbr_b1_1cv": float(np.mean(pbr_b1_folds)),
            "pbr_reduction_ratio_dga_vs_b1": float(
                1.0 - np.mean(pbr_dga_folds) / max(float(np.mean(pbr_b1_folds)), 1e-12)
            ),
            "pbr_reduction_ratio_pt_vs_b1": float(
                1.0 - np.mean(pbr_pt_folds) / max(float(np.mean(pbr_b1_folds)), 1e-12)
            ),
            "mean_n2b_reversibility_l2_err": float(np.mean(n2b_folds)),
            "max_n2b_reversibility_l2_err": float(np.max(n2b_folds)),
            "passes_n2b_threshold_0p10": bool(np.mean(n2b_folds) <= 0.10),
        }

        if lag == primary_lag:
            fold_q_train_primary = fold_q_tr_cur
            q_te_indep_primary = q_te_indep_cur
            q_tr_oof_primary = oof_q_from_tr.copy()
            int_all = ~(in_A | in_B)
            entry["mean_test_fold_indep_n2b_l2_err"] = float(np.mean(n2b_te_folds))
            entry["train_oof_vs_test_indep_interior_pearson_r"] = float(
                np.corrcoef(q_tr_oof_primary[int_all], q_te_indep_primary[int_all])[0, 1]
            )
            entry["train_oof_vs_test_indep_interior_rmse"] = float(
                np.sqrt(np.mean((q_tr_oof_primary[int_all] - q_te_indep_primary[int_all]) ** 2))
            )

        lag_sweep_results[str(lag)] = entry

    return FoldIsolatedDGABundle(
        folds=folds,
        fold_q_train=fold_q_train_primary,
        q_te_indep=q_te_indep_primary,
        q_tr_oof_pred=q_tr_oof_primary,
        lag_sweep={
            "primary_lag_steps": int(primary_lag),
            "lags": lag_sweep_results,
        },
    )


def evaluate_h1_benchmark_suite(
    X: np.ndarray,
    bundle: FoldIsolatedDGABundle,
    in_A: np.ndarray,
    in_B: np.ndarray,
    seg_id: np.ndarray,
    groups: np.ndarray,
    slice_range: Tuple[float, float],
    bin_width: float = 0.5,
    b2_lag_steps: int = 10,
    n_perms: int = 80,
    n_boot: int = 1000,
    random_state: int = 42,
) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    """Run strictly fold-isolated H1 model suite (trained on dga_tr_j, evaluated on dga_te_j)."""
    folds = bundle.folds
    fold_q_train = bundle.fold_q_train
    q_eval_ref = bundle.q_te_indep

    lo, hi = float(slice_range[0]), float(slice_range[1])
    in_slice = (X[:, 0] >= lo) & (X[:, 0] <= hi)
    bin_edges = np.arange(lo, hi + 1e-9, float(bin_width))
    if bin_edges[-1] < hi - 1e-9:
        bin_edges = np.append(bin_edges, hi)

    dX_b2, valid_b2 = compute_lagged_displacements(X[:, [0]], seg_id, lag_steps=b2_lag_steps)

    oof_b0 = np.zeros_like(q_eval_ref)
    oof_b1 = np.zeros_like(q_eval_ref)
    oof_b2 = np.zeros_like(q_eval_ref)
    oof_pt_lin = np.zeros_like(q_eval_ref)
    oof_pt_epi = np.zeros_like(q_eval_ref)
    oof_gbr = np.zeros_like(q_eval_ref)

    fold_epi_coefs: List[np.ndarray] = []
    fold_epi_names: List[str] = []

    for f_idx, (tr, te) in enumerate(folds):
        q_tr_f = fold_q_train[f_idx]
        sl_tr = in_slice[tr]
        X_tr_sl = X[tr][sl_tr]
        q_tr_sl = q_tr_f[sl_tr]
        A_te, B_te = in_A[te], in_B[te]

        b0 = PriorBaselineB0(progress_col=0, bin_edges=bin_edges).fit(X_tr_sl, q_tr_sl)
        b1 = SplineProgressBaselineB1(progress_col=0, n_knots=5, l2_reg=1e-3).fit(X_tr_sl, q_tr_sl)
        pt_lin = PetersTroutCommittorModel(
            include_interactions=False, l2_reg=1e-3, feature_names=FEATURE_NAMES
        ).fit(X_tr_sl, q_tr_sl)
        pt_epi = PetersTroutCommittorModel(
            include_interactions=True, l2_reg=1e-3, feature_names=FEATURE_NAMES
        ).fit(X_tr_sl, q_tr_sl)
        gbr = OracleHistGBRModel(min_samples_leaf=50, random_state=random_state + f_idx).fit(
            X_tr_sl, q_tr_sl
        )

        oof_b0[te] = b0.predict(X[te], in_A=A_te, in_B=B_te)
        oof_b1[te] = b1.predict(X[te], in_A=A_te, in_B=B_te)
        oof_pt_lin[te] = pt_lin.predict(X[te], in_A=A_te, in_B=B_te)
        oof_pt_epi[te] = pt_epi.predict(X[te], in_A=A_te, in_B=B_te)
        oof_gbr[te] = gbr.predict(X[te], in_A=A_te, in_B=B_te)

        sl_b2 = in_slice[tr] & valid_b2[tr]
        b2 = ProgressVelocityBaselineB2(progress_col=0, n_knots=5, l2_reg=1e-3).fit(
            X[tr][sl_b2, :1], dX_b2[tr][sl_b2], q_tr_f[sl_b2]
        )
        pred_b2_te = b2.predict(X[te, :1], dX_b2[te], in_A=A_te, in_B=B_te)
        oof_b2[te] = np.where(valid_b2[te], pred_b2_te, oof_b1[te])

        fold_epi_coefs.append(pt_epi.coef_.copy())
        fold_epi_names = list(pt_epi.expanded_names_)

    # Evaluate matched-slice metrics against test-fold-independent q_eval_ref
    ms_pt_lin = matched_slice_metrics(
        X[:, 0], q_eval_ref, oof_pt_lin, oof_b1, slice_range=slice_range, bin_width=bin_width
    )
    ms_pt_epi = matched_slice_metrics(
        X[:, 0], q_eval_ref, oof_pt_epi, oof_b1, slice_range=slice_range, bin_width=bin_width
    )
    ms_b2 = matched_slice_metrics(
        X[:, 0], q_eval_ref, oof_b2, oof_b1, slice_range=slice_range, bin_width=bin_width
    )
    ms_gbr = matched_slice_metrics(
        X[:, 0], q_eval_ref, oof_gbr, oof_b1, slice_range=slice_range, bin_width=bin_width
    )

    g_sl = groups[in_slice]
    y_sl = q_eval_ref[in_slice]
    ci_pt_lin = bootstrap_bss_and_delta_ell_ci(
        g_sl, y_sl, oof_pt_lin[in_slice], oof_b1[in_slice], n_boot=n_boot, random_state=random_state
    )
    ci_pt_epi = bootstrap_bss_and_delta_ell_ci(
        g_sl, y_sl, oof_pt_epi[in_slice], oof_b1[in_slice], n_boot=n_boot, random_state=random_state + 10
    )
    ci_epistasis_vs_linear = bootstrap_bss_and_delta_ell_ci(
        g_sl, y_sl, oof_pt_epi[in_slice], oof_pt_lin[in_slice], n_boot=n_boot, random_state=random_state + 20
    )
    ci_gbr = bootstrap_bss_and_delta_ell_ci(
        g_sl, y_sl, oof_gbr[in_slice], oof_b1[in_slice], n_boot=n_boot, random_state=random_state + 30
    )
    ci_b2 = bootstrap_bss_and_delta_ell_ci(
        g_sl, y_sl, oof_b2[in_slice], oof_b1[in_slice], n_boot=n_boot, random_state=random_state + 40
    )

    r_comp = (
        float(ms_pt_epi["bss_int"] / ms_gbr["bss_int"])
        if ms_gbr["bss_int"] > 1e-6
        else 0.0
    )

    sep_b1 = separatrix_purity(y_sl, oof_b1[in_slice])
    sep_pt_epi = separatrix_purity(y_sl, oof_pt_epi[in_slice])
    sep_gbr = separatrix_purity(y_sl, oof_gbr[in_slice])

    # Fold-isolated N1 Trajectory-Group Block Permutation Null
    n1_res = run_group_permutation_null_N1(
        X=X,
        q_target=q_eval_ref,
        groups=groups,
        progress_col=0,
        slice_range=slice_range,
        in_A=in_A,
        in_B=in_B,
        fit_on_slice=True,
        include_interactions=True,
        l2_reg=1e-3,
        n_splits=len(folds),
        n_perms=n_perms,
        random_state=random_state,
        folds=folds,
        fold_q_train=fold_q_train,
    )

    coef_mat = np.vstack(fold_epi_coefs)
    coef_summary = {
        name: {
            "mean_standardized_coef": float(np.mean(coef_mat[:, j])),
            "std_across_folds": float(np.std(coef_mat[:, j], ddof=1)),
        }
        for j, name in enumerate(fold_epi_names)
    }

    summary = {
        "slice_range_angstrom": [lo, hi],
        "n_slice_frames": int(np.sum(in_slice)),
        "b0_vs_b1_note": {
            "bss_b1_vs_b0": brier_skill_score(y_sl, oof_b1[in_slice], oof_b0[in_slice]),
            "delta_ell_b1_vs_b0": delta_log_likelihood(y_sl, oof_b1[in_slice], oof_b0[in_slice]),
        },
        "b2_velocity_1cv": {
            "lag_steps": int(b2_lag_steps),
            "bss_vs_b1": ci_b2["bss"],
            "delta_ell_vs_b1": ci_b2["delta_ell"],
            "spearman_rho": ms_b2["spearman_rho_model"],
        },
        "pt_linear_3cv": {
            "bss_vs_b1": ci_pt_lin["bss"],
            "delta_ell_vs_b1": ci_pt_lin["delta_ell"],
            "spearman_rho": ms_pt_lin["spearman_rho_model"],
            "calibration_ece": ms_pt_lin["calibration"]["ece"],
        },
        "pt_epistasis_3cv": {
            "bss_vs_b1": ci_pt_epi["bss"],
            "delta_ell_vs_b1": ci_pt_epi["delta_ell"],
            "epistasis_gain_over_linear": ci_epistasis_vs_linear,
            "spearman_rho": ms_pt_epi["spearman_rho_model"],
            "spearman_rho_b1": ms_pt_epi["spearman_rho_b1"],
            "mean_within_bin_var_q_ref": ms_pt_epi["mean_within_bin_var_q_ref"],
            "mean_within_bin_std_q_ref": float(np.sqrt(ms_pt_epi["mean_within_bin_var_q_ref"])),
            "calibration": ms_pt_epi["calibration"],
            "per_bin": ms_pt_epi["per_bin"],
            "standardized_coefficients": coef_summary,
        },
        "oracle_hist_gbr_3cv": {
            "bss_vs_b1": ci_gbr["bss"],
            "delta_ell_vs_b1": ci_gbr["delta_ell"],
            "spearman_rho": ms_gbr["spearman_rho_model"],
            "calibration_ece": ms_gbr["calibration"]["ece"],
            "parametric_completeness_ratio_pt_over_gbr": r_comp,
        },
        "separatrix_purity": {
            "b1_1cv": sep_b1,
            "pt_epistasis_3cv": sep_pt_epi,
            "oracle_gbr_3cv": sep_gbr,
        },
        "n1_permutation_null": {
            "n_perms": int(n_perms),
            "obs_bss_vs_b1": n1_res["obs_bss_vs_b1"],
            "obs_delta_ell_vs_b1": n1_res["obs_delta_ell_vs_b1"],
            "null_bss_mean": n1_res["n1_null_bss_mean"],
            "null_bss_q95": n1_res["n1_null_bss_q95"],
            "null_dell_mean": n1_res["n1_null_dell_mean"],
            "null_dell_q95": n1_res["n1_null_dell_q95"],
            "p_value_bss_vs_n1": n1_res["p_value_bss_vs_n1"],
            "p_value_dell_vs_n1": n1_res["p_value_dell_vs_n1"],
        },
    }

    preds = {
        "oof_b0": oof_b0,
        "oof_b1": oof_b1,
        "oof_b2": oof_b2,
        "oof_pt_lin": oof_pt_lin,
        "oof_pt_epi": oof_pt_epi,
        "oof_gbr": oof_gbr,
    }
    return summary, preds


def run_h2_memory_profile(
    X: np.ndarray,
    seg_id: np.ndarray,
    bundle: FoldIsolatedDGABundle,
    groups: np.ndarray,
    in_A: np.ndarray,
    in_B: np.ndarray,
    slice_range: Tuple[float, float],
    lags: Tuple[int, ...] = (2, 5, 10, 20, 35),
    n_shuffles: int = 50,
    random_state: int = 42,
) -> Dict[str, object]:
    """Evaluate strictly fold-isolated Mori-Zwanzig memory gain across lags (1-CV vs 3-CV)."""
    eval_mask = (X[:, 0] >= float(slice_range[0])) & (X[:, 0] <= float(slice_range[1]))
    out_lags: Dict[str, object] = {}

    for lag in lags:
        mem_1cv = evaluate_mori_zwanzig_memory(
            X_static=X[:, [0]],
            seg_id=seg_id,
            q_target=bundle.q_te_indep,
            groups=groups,
            lag_steps=lag,
            eval_mask=eval_mask,
            in_A=in_A,
            in_B=in_B,
            include_interactions=False,
            l2_reg=1e-3,
            n_splits=len(bundle.folds),
            n_shuffles=n_shuffles,
            random_state=random_state + lag,
            folds=bundle.folds,
            fold_q_train=bundle.fold_q_train,
        )
        mem_3cv = evaluate_mori_zwanzig_memory(
            X_static=X,
            seg_id=seg_id,
            q_target=bundle.q_te_indep,
            groups=groups,
            lag_steps=lag,
            eval_mask=eval_mask,
            in_A=in_A,
            in_B=in_B,
            include_interactions=True,
            l2_reg=1e-3,
            n_splits=len(bundle.folds),
            n_shuffles=n_shuffles,
            random_state=random_state + 100 + lag,
            folds=bundle.folds,
            fold_q_train=bundle.fold_q_train,
        )
        rho_1_to_3 = compute_memory_collapse_ratio(
            mem_1cv["delta_ell_mem"], mem_3cv["delta_ell_mem"]
        )
        out_lags[str(lag)] = {
            "lag_steps": int(lag),
            "memory_1cv_rtm6_only": mem_1cv,
            "memory_3cv_protein_subspace": mem_3cv,
            "partial_collapse_1cv_to_3cv": rho_1_to_3,
        }

    return {"slice_range_angstrom": list(slice_range), "lags": out_lags}


def _operator_independent_split_half_orth_cosines(
    X_lig: np.ndarray,
    in_A_lig: np.ndarray,
    in_B_lig: np.ndarray,
    seg_lig: np.ndarray,
    groups_lig: np.ndarray,
    slice_lig: Tuple[float, float],
    Sigma_orth: np.ndarray,
    lag_steps: int,
    n_splits_half: int = 40,
    random_state: int = 42,
) -> np.ndarray:
    """Estimate within-ligand orthogonal gradient reliability via independent half-sample DGA fits."""
    rng = np.random.default_rng(random_state)
    u_g = np.unique(groups_lig)
    orig_all = np.arange(X_lig.shape[0], dtype=np.int64)
    in_sl = (X_lig[:, 0] >= slice_lig[0]) & (X_lig[:, 0] <= slice_lig[1])
    cos_vals: List[float] = []

    attempts = 0
    while len(cos_vals) < n_splits_half and attempts < n_splits_half * 4:
        attempts += 1
        perm = rng.permutation(u_g)
        g_h1 = set(perm[: u_g.size // 2].tolist())
        m1 = np.array([g in g_h1 for g in groups_lig], dtype=bool)
        m2 = ~m1
        if (
            np.sum(in_A_lig[m1]) < 3
            or np.sum(in_B_lig[m1]) < 3
            or np.sum(in_A_lig[m2]) < 3
            or np.sum(in_B_lig[m2]) < 3
            or np.sum(in_sl & m1) < 25
            or np.sum(in_sl & m2) < 25
        ):
            continue
        idx1 = orig_all[m1]
        idx2 = orig_all[m2]
        seg1 = _resegment_subset_indices(idx1, seg_lig[m1])
        seg2 = _resegment_subset_indices(idx2, seg_lig[m2])

        dga1 = StoppedDGACommittor(lag_steps=lag_steps, n_rbf=45, reg=1e-3, random_state=random_state + attempts).fit(
            X_lig[m1], in_A_lig[m1], in_B_lig[m1], seg_id=seg1
        )
        dga2 = StoppedDGACommittor(lag_steps=lag_steps, n_rbf=45, reg=1e-3, random_state=random_state + 500 + attempts).fit(
            X_lig[m2], in_A_lig[m2], in_B_lig[m2], seg_id=seg2
        )
        q1 = dga1.predict(X_lig[m1], in_A_lig[m1], in_B_lig[m1])
        q2 = dga2.predict(X_lig[m2], in_A_lig[m2], in_B_lig[m2])

        sl1 = in_sl[m1]
        sl2 = in_sl[m2]
        pt1 = PetersTroutCommittorModel(include_interactions=True, l2_reg=1e-3).fit(X_lig[m1][sl1], q1[sl1])
        pt2 = PetersTroutCommittorModel(include_interactions=True, l2_reg=1e-3).fit(X_lig[m2][sl2], q2[sl2])
        cos_vals.append(
            scale_invariant_gradient_cosine(
                pt1.mean_logit_gradient_raw(X_lig[m1][sl1])[1:],
                pt2.mean_logit_gradient_raw(X_lig[m2][sl2])[1:],
                Sigma_orth,
            )
        )

    return np.asarray(cos_vals, dtype=np.float64)


def run_h3_ligand_gradient_rotation(
    X_ago: np.ndarray,
    in_A_ago: np.ndarray,
    in_B_ago: np.ndarray,
    seg_ago: np.ndarray,
    bundle_ago: FoldIsolatedDGABundle,
    groups_ago: np.ndarray,
    slice_ago: Tuple[float, float],
    X_thc: np.ndarray,
    in_A_thc: np.ndarray,
    in_B_thc: np.ndarray,
    seg_thc: np.ndarray,
    bundle_thc: FoldIsolatedDGABundle,
    groups_thc: np.ndarray,
    slice_thc: Tuple[float, float],
    X_pool_shared: np.ndarray,
    lag_steps: int = 15,
    n_boot: int = 1000,
    random_state: int = 42,
) -> Dict[str, object]:
    """Compute fold-isolated S_cos(Ago, THC) + operator-independent reliability + MWC OOF test."""
    Sigma_pool = np.cov(X_pool_shared, rowvar=False)
    Sigma_orth = Sigma_pool[1:, 1:]

    q_ago_indep = bundle_ago.q_te_indep
    q_thc_indep = bundle_thc.q_te_indep

    m_ago = (X_ago[:, 0] >= slice_ago[0]) & (X_ago[:, 0] <= slice_ago[1])
    m_thc = (X_thc[:, 0] >= slice_thc[0]) & (X_thc[:, 0] <= slice_thc[1])

    Xa_s, qa_s, ga_s = X_ago[m_ago], q_ago_indep[m_ago], groups_ago[m_ago]
    Xp_s, qp_s, gp_s = X_thc[m_thc], q_thc_indep[m_thc], groups_thc[m_thc]

    pt_ago_lin = PetersTroutCommittorModel(include_interactions=False, l2_reg=1e-3, feature_names=FEATURE_NAMES).fit(Xa_s, qa_s)
    pt_thc_lin = PetersTroutCommittorModel(include_interactions=False, l2_reg=1e-3, feature_names=FEATURE_NAMES).fit(Xp_s, qp_s)

    pt_ago_epi = PetersTroutCommittorModel(include_interactions=True, l2_reg=1e-3, feature_names=FEATURE_NAMES).fit(Xa_s, qa_s)
    pt_thc_epi = PetersTroutCommittorModel(include_interactions=True, l2_reg=1e-3, feature_names=FEATURE_NAMES).fit(Xp_s, qp_s)

    w_ago_lin = pt_ago_lin.mean_logit_gradient_raw(Xa_s)
    w_thc_lin = pt_thc_lin.mean_logit_gradient_raw(Xp_s)
    s_cos_lin = scale_invariant_gradient_cosine(w_ago_lin, w_thc_lin, Sigma_pool)

    w_ago_epi = pt_ago_epi.mean_logit_gradient_raw(Xa_s)
    w_thc_epi = pt_thc_epi.mean_logit_gradient_raw(Xp_s)
    s_cos_epi = scale_invariant_gradient_cosine(w_ago_epi, w_thc_epi, Sigma_pool)

    s_cos_orth_lin = scale_invariant_gradient_cosine(w_ago_lin[1:], w_thc_lin[1:], Sigma_orth)
    s_cos_orth_epi = scale_invariant_gradient_cosine(w_ago_epi[1:], w_thc_epi[1:], Sigma_orth)

    # Joint 1,000-group cluster bootstrap across strictly disjoint Agonist and THC macro-blocks
    u_ga, inv_ga = np.unique(ga_s, return_inverse=True)
    u_gp, inv_gp = np.unique(gp_s, return_inverse=True)
    buckets_a = [np.flatnonzero(inv_ga == k) for k in range(u_ga.size)]
    buckets_p = [np.flatnonzero(inv_gp == k) for k in range(u_gp.size)]

    rng = np.random.default_rng(random_state)
    boot_scos_all = np.empty(n_boot, dtype=np.float64)
    boot_scos_orth = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        pa = rng.integers(0, u_ga.size, size=u_ga.size)
        pp = rng.integers(0, u_gp.size, size=u_gp.size)
        ia = np.concatenate([buckets_a[k] for k in pa])
        ip = np.concatenate([buckets_p[k] for k in pp])

        ma_b = PetersTroutCommittorModel(include_interactions=True, l2_reg=1e-3).fit(Xa_s[ia], qa_s[ia])
        mp_b = PetersTroutCommittorModel(include_interactions=True, l2_reg=1e-3).fit(Xp_s[ip], qp_s[ip])
        wa = ma_b.mean_logit_gradient_raw(Xa_s[ia])
        wp = mp_b.mean_logit_gradient_raw(Xp_s[ip])
        boot_scos_all[b] = scale_invariant_gradient_cosine(wa, wp, Sigma_pool)
        boot_scos_orth[b] = scale_invariant_gradient_cosine(wa[1:], wp[1:], Sigma_orth)

    # Operator-independent within-ligand split-half reliability ceiling (separate DGA on each half!)
    boot_within_ago = _operator_independent_split_half_orth_cosines(
        X_ago, in_A_ago, in_B_ago, seg_ago, groups_ago, slice_ago, Sigma_orth, lag_steps=lag_steps, n_splits_half=40, random_state=random_state
    )
    boot_within_thc = _operator_independent_split_half_orth_cosines(
        X_thc, in_A_thc, in_B_thc, seg_thc, groups_thc, slice_thc, Sigma_orth, lag_steps=lag_steps, n_splits_half=40, random_state=random_state + 200
    )

    # Strictly Fold-Isolated Stratified MWC Shared-Direction vs Ligand-Specific OOF Test:
    # Pair fold j of Agonist (trained strictly on dga_tr_ago_j) with fold j of THC (trained strictly on dga_tr_thc_j),
    # and evaluate on (te_ago_j, te_thc_j) against independent test-fold (dga_te_ago_j, dga_te_thc_j)!
    oof_mwc_ago = np.zeros(X_ago.shape[0], dtype=np.float64)
    oof_mwc_thc = np.zeros(X_thc.shape[0], dtype=np.float64)
    oof_spec_ago = np.zeros(X_ago.shape[0], dtype=np.float64)
    oof_spec_thc = np.zeros(X_thc.shape[0], dtype=np.float64)

    for f_idx in range(len(bundle_ago.folds)):
        tr_a, te_a = bundle_ago.folds[f_idx]
        tr_p, te_p = bundle_thc.folds[f_idx]
        qa_tr = bundle_ago.fold_q_train[f_idx]
        qp_tr = bundle_thc.fold_q_train[f_idx]

        sla_tr = m_ago[tr_a]
        slp_tr = m_thc[tr_p]
        Xa_tr_sl, ya_tr_sl = X_ago[tr_a][sla_tr], qa_tr[sla_tr]
        Xp_tr_sl, yp_tr_sl = X_thc[tr_p][slp_tr], qp_tr[slp_tr]

        X_tr_mwc = np.vstack([
            np.hstack([Xa_tr_sl, np.zeros((Xa_tr_sl.shape[0], 1))]),
            np.hstack([Xp_tr_sl, np.ones((Xp_tr_sl.shape[0], 1))]),
        ])
        y_tr_mwc = np.r_[ya_tr_sl, yp_tr_sl]
        m_mwc = PetersTroutCommittorModel(include_interactions=False, l2_reg=1e-3).fit(X_tr_mwc, y_tr_mwc)

        oof_mwc_ago[te_a] = m_mwc.predict(np.hstack([X_ago[te_a], np.zeros((te_a.size, 1))]))
        oof_mwc_thc[te_p] = m_mwc.predict(np.hstack([X_thc[te_p], np.ones((te_p.size, 1))]))

        ma = PetersTroutCommittorModel(include_interactions=True, l2_reg=1e-3).fit(Xa_tr_sl, ya_tr_sl)
        mp = PetersTroutCommittorModel(include_interactions=True, l2_reg=1e-3).fit(Xp_tr_sl, yp_tr_sl)
        oof_spec_ago[te_a] = ma.predict(X_ago[te_a])
        oof_spec_thc[te_p] = mp.predict(X_thc[te_p])

    g_comb = np.r_[ga_s, gp_s]
    q_comb_indep = np.r_[qa_s, qp_s]
    oof_spec_comb = np.r_[oof_spec_ago[m_ago], oof_spec_thc[m_thc]]
    oof_mwc_comb = np.r_[oof_mwc_ago[m_ago], oof_mwc_thc[m_thc]]

    mwc_vs_spec_ci = bootstrap_bss_and_delta_ell_ci(
        g_comb, q_comb_indep, oof_spec_comb, oof_mwc_comb, n_boot=n_boot, random_state=random_state + 99
    )

    return {
        "w_agonist_raw_per_angstrom": {FEATURE_NAMES[j]: float(w_ago_epi[j]) for j in range(3)},
        "w_thc_raw_per_angstrom": {FEATURE_NAMES[j]: float(w_thc_epi[j]) for j in range(3)},
        "s_cos_full_3cv_linear": float(s_cos_lin),
        "s_cos_full_3cv_epistasis": {
            "point": float(s_cos_epi),
            "ci_low": float(np.quantile(boot_scos_all, 0.025)),
            "ci_high": float(np.quantile(boot_scos_all, 0.975)),
            "std_err": float(np.std(boot_scos_all, ddof=1)),
        },
        "s_cos_orthogonal_2cv_toggle_npxxy": {
            "point_linear": float(s_cos_orth_lin),
            "point_epistasis": float(s_cos_orth_epi),
            "ci_low": float(np.quantile(boot_scos_orth, 0.025)),
            "ci_high": float(np.quantile(boot_scos_orth, 0.975)),
            "std_err": float(np.std(boot_scos_orth, ddof=1)),
        },
        "within_ligand_split_half_orth_reliability": {
            "full_agonist_median": float(np.median(boot_within_ago)),
            "full_agonist_ci": [
                float(np.quantile(boot_within_ago, 0.025)),
                float(np.quantile(boot_within_ago, 0.975)),
            ],
            "delta9_thc_median": float(np.median(boot_within_thc)),
            "delta9_thc_ci": [
                float(np.quantile(boot_within_thc, 0.025)),
                float(np.quantile(boot_within_thc, 0.975)),
            ],
        },
        "ligand_specific_vs_mwc_shared_gradient_oof": mwc_vs_spec_ci,
    }


def run_segmentation_and_endpoint_sensitivity(df: pd.DataFrame) -> Dict[str, object]:
    """Test sensitivity of Global Committor H1 BSS_int & Test PBR across jump thresholds & endpoint defs."""
    valid_df = df[df["macro_valid_mask"]].reset_index(drop=True)
    orig_idx = df.index[df["macro_valid_mask"]].to_numpy()
    X = valid_df[FEATURE_NAMES].to_numpy(dtype=np.float64)
    groups = valid_df["macro_group_id"].to_numpy(dtype=np.int64)
    is_ago_valid = (valid_df["ligand"].to_numpy() == "full_agonist").astype(int)
    in_A = valid_df["in_A_global"].to_numpy(dtype=bool)
    in_B = valid_df["in_B_global"].to_numpy(dtype=bool)
    slice_range = (12.5, 14.5)
    in_sl = (X[:, 0] >= slice_range[0]) & (X[:, 0] <= slice_range[1])

    seg_results: Dict[str, object] = {}
    for seg_label, seg_col, lag_use in [
        ("z5p0_primary", "seg_id_z5p0", 15),
        ("tm6_2cv_0p15nm", "seg_id_tm6_0p15nm", 10),
        ("3cv_0p20nm", "seg_id_3cv_0p20nm", 10),
        ("3cv_0p15nm", "seg_id_3cv_0p15nm", 5),
        ("3cv_0p12nm", "seg_id_3cv_0p12nm", 3),
        ("3cv_0p10nm", "seg_id_3cv_0p10nm", 2),
    ]:
        seg_sub = _resegment_subset_indices(orig_idx, valid_df[seg_col].to_numpy(dtype=np.int64))
        bundle = solve_fold_isolated_dga_bundle(
            X,
            in_A,
            in_B,
            seg_sub,
            groups,
            stratify_labels=is_ago_valid,
            candidate_lags=(lag_use,),
            primary_lag=lag_use,
            n_rbf=60,
            reg=1e-3,
            n_splits=5,
        )
        oof_b1 = np.zeros_like(bundle.q_te_indep)
        oof_pt = np.zeros_like(bundle.q_te_indep)
        for f_idx, (tr, te) in enumerate(bundle.folds):
            q_tr_f = bundle.fold_q_train[f_idx]
            sl_tr = in_sl[tr]
            b1 = SplineProgressBaselineB1(0, l2_reg=1e-3).fit(X[tr][sl_tr], q_tr_f[sl_tr])
            pt = PetersTroutCommittorModel(include_interactions=True, l2_reg=1e-3).fit(X[tr][sl_tr], q_tr_f[sl_tr])
            oof_b1[te] = b1.predict(X[te], in_A[te], in_B[te])
            oof_pt[te] = pt.predict(X[te], in_A[te], in_B[te])
        ms = matched_slice_metrics(
            X[:, 0], bundle.q_te_indep, oof_pt, oof_b1, slice_range=slice_range, bin_width=0.5
        )
        seg_results[seg_label] = {
            "lag_steps_used": lag_use,
            "bss_int_pt_vs_b1": ms["bss_int"],
            "delta_ell_int_pt_vs_b1": ms["delta_ell_int"],
            "spearman_rho_pt": ms["spearman_rho_model"],
            "test_pbr_dga_3cv": bundle.lag_sweep["lags"][str(lag_use)]["mean_test_pbr_dga_3cv"],
            "test_pbr_b1_1cv": bundle.lag_sweep["lags"][str(lag_use)]["mean_test_pbr_b1_1cv"],
            "n2b_reversibility_l2_err": bundle.lag_sweep["lags"][str(lag_use)]["mean_n2b_reversibility_l2_err"],
        }

    # Endpoint sensitivity on primary z5p0 segmentation (lag=15, stratified fold-isolated DGA):
    seg_z5 = _resegment_subset_indices(orig_idx, valid_df["seg_id_z5p0"].to_numpy(dtype=np.int64))
    in_A_dwell = valid_df["in_A_global_dwell3"].to_numpy(dtype=bool)
    in_B_dwell = valid_df["in_B_global_dwell3"].to_numpy(dtype=bool)
    bundle_dwell = solve_fold_isolated_dga_bundle(
        X, in_A_dwell, in_B_dwell, seg_z5, groups, stratify_labels=is_ago_valid, candidate_lags=(15,), primary_lag=15, n_rbf=60
    )

    in_A_core = (X[:, 0] <= 11.8) & (X[:, 1] <= 9.0) & (X[:, 2] <= 12.8)
    in_B_core = (X[:, 0] >= 15.2) & (X[:, 1] >= 10.0) & (X[:, 2] >= 15.0)
    bundle_core = solve_fold_isolated_dga_bundle(
        X, in_A_core, in_B_core, seg_z5, groups, stratify_labels=is_ago_valid, candidate_lags=(15,), primary_lag=15, n_rbf=60
    )

    bundle_prim = solve_fold_isolated_dga_bundle(
        X, in_A, in_B, seg_z5, groups, stratify_labels=is_ago_valid, candidate_lags=(15,), primary_lag=15, n_rbf=65
    )
    q_prim = bundle_prim.q_te_indep
    q_dwell = bundle_dwell.q_te_indep
    q_core = bundle_core.q_te_indep

    endpoint_sensitivity = {
        "dwell3_vs_primary_slice_pearson_r": float(np.corrcoef(q_prim[in_sl], q_dwell[in_sl])[0, 1]),
        "dwell3_vs_primary_slice_rmse": float(np.sqrt(np.mean((q_prim[in_sl] - q_dwell[in_sl]) ** 2))),
        "core3cv_vs_primary_slice_pearson_r": float(np.corrcoef(q_prim[in_sl], q_core[in_sl])[0, 1]),
        "core3cv_vs_primary_slice_rmse": float(np.sqrt(np.mean((q_prim[in_sl] - q_core[in_sl]) ** 2))),
        "n_A_core": int(np.sum(in_A_core)),
        "n_B_core": int(np.sum(in_B_core)),
    }

    return {
        "jump_threshold_sweep": seg_results,
        "endpoint_definition_sensitivity": endpoint_sensitivity,
    }


def main() -> Dict[str, object]:
    t0 = time.time()
    RESULTS_WS1_DIR.mkdir(parents=True, exist_ok=True)

    print("[WS1] Step 1/6: Loading Tier 1C arrays, auditing units/NaNs, and building buffered macro-blocks...")
    df, data_audit = load_and_build_ws1_dataset()

    valid_mask = df["macro_valid_mask"].to_numpy(dtype=bool)
    orig_idx = np.flatnonzero(valid_mask)
    df_val = df.iloc[orig_idx].reset_index(drop=True)

    X_pool = df_val[FEATURE_NAMES].to_numpy(dtype=np.float64)
    seg_pool = _resegment_subset_indices(orig_idx, df_val["seg_id_z5p0"].to_numpy(dtype=np.int64))
    groups_pool = df_val["macro_group_id"].to_numpy(dtype=np.int64)
    in_A_pool = df_val["in_A_global"].to_numpy(dtype=bool)
    in_B_pool = df_val["in_B_global"].to_numpy(dtype=bool)
    is_ago = df_val["ligand"].to_numpy() == "full_agonist"
    is_ago_int = is_ago.astype(int)

    print("[WS1] Step 2/6: Running Regime I (Global Full-Activation 11.8 -> 15.2 A, stratified fold-isolated DGA)...")
    bundle_global = solve_fold_isolated_dga_bundle(
        X_pool,
        in_A_pool,
        in_B_pool,
        seg_pool,
        groups_pool,
        stratify_labels=is_ago_int,
        candidate_lags=(2, 5, 10, 15, 25),
        primary_lag=15,
        n_rbf=65,
        reg=1e-3,
    )
    h1_global, preds_global = evaluate_h1_benchmark_suite(
        X_pool,
        bundle_global,
        in_A_pool,
        in_B_pool,
        seg_pool,
        groups_pool,
        slice_range=(12.5, 14.5),
        bin_width=0.5,
        b2_lag_steps=10,
        n_perms=80,
        n_boot=1000,
    )

    print("[WS1] Step 3/6: Running Regime II (Strictly Within-Ligand Natural Sub-Transitions)...")
    # 3A: Full Agonist alone (13.2 -> 15.2 A, slice [13.7, 14.7] A)
    idx_ago = orig_idx[is_ago]
    X_ago = X_pool[is_ago]
    seg_ago = _resegment_subset_indices(idx_ago, df_val.loc[is_ago, "seg_id_z5p0"].to_numpy(dtype=np.int64))
    g_ago = groups_pool[is_ago]
    in_A_ago = X_ago[:, 0] <= 13.2
    in_B_ago = X_ago[:, 0] >= 15.2

    bundle_ago = solve_fold_isolated_dga_bundle(
        X_ago, in_A_ago, in_B_ago, seg_ago, g_ago, candidate_lags=(5, 10, 15, 25), primary_lag=15, n_rbf=60
    )
    h1_ago, preds_ago = evaluate_h1_benchmark_suite(
        X_ago, bundle_ago, in_A_ago, in_B_ago, seg_ago, g_ago, slice_range=(13.7, 14.7), bin_width=0.5, n_perms=60
    )

    # 3B: Delta9-THC alone (11.8 -> 13.5 A, slice [12.2, 13.1] A)
    is_thc = ~is_ago
    idx_thc = orig_idx[is_thc]
    X_thc = X_pool[is_thc]
    seg_thc = _resegment_subset_indices(idx_thc, df_val.loc[is_thc, "seg_id_z5p0"].to_numpy(dtype=np.int64))
    g_thc = groups_pool[is_thc]
    in_A_thc = X_thc[:, 0] <= 11.8
    in_B_thc = X_thc[:, 0] >= 13.5

    bundle_thc = solve_fold_isolated_dga_bundle(
        X_thc, in_A_thc, in_B_thc, seg_thc, g_thc, candidate_lags=(5, 10, 15, 25), primary_lag=15, n_rbf=60
    )
    h1_thc, preds_thc = evaluate_h1_benchmark_suite(
        X_thc, bundle_thc, in_A_thc, in_B_thc, seg_thc, g_thc, slice_range=(12.2, 13.1), bin_width=0.45, n_perms=60
    )

    # 3C: Regime III Shared-Boundary Within-Ligand Window (12.8 -> 13.8 A, slice [13.0, 13.6] A)
    in_A_ago_sh, in_B_ago_sh = X_ago[:, 0] <= 12.8, X_ago[:, 0] >= 13.8
    in_A_thc_sh, in_B_thc_sh = X_thc[:, 0] <= 12.8, X_thc[:, 0] >= 13.8
    bundle_ago_sh = solve_fold_isolated_dga_bundle(
        X_ago, in_A_ago_sh, in_B_ago_sh, seg_ago, g_ago, candidate_lags=(10,), primary_lag=10, n_rbf=45
    )
    bundle_thc_sh = solve_fold_isolated_dga_bundle(
        X_thc, in_A_thc_sh, in_B_thc_sh, seg_thc, g_thc, candidate_lags=(10,), primary_lag=10, n_rbf=45
    )

    print("[WS1] Step 4/6: Running Hypothesis H2 Step A (Fold-Isolated Mori-Zwanzig Memory Profiles vs N2a)...")
    h2_global = run_h2_memory_profile(
        X_pool, seg_pool, bundle_global, groups_pool, in_A_pool, in_B_pool, (12.5, 14.5), n_shuffles=40
    )
    h2_ago = run_h2_memory_profile(
        X_ago, seg_ago, bundle_ago, g_ago, in_A_ago, in_B_ago, (13.7, 14.7), n_shuffles=40
    )
    h2_thc = run_h2_memory_profile(
        X_thc, seg_thc, bundle_thc, g_thc, in_A_thc, in_B_thc, (12.2, 13.1), n_shuffles=40
    )

    print("[WS1] Step 5/6: Running Hypothesis H3 (Full Agonist vs Delta9-THC Bottleneck Rotation)...")
    in_shared_pool = (X_pool[:, 0] >= 12.5) & (X_pool[:, 0] <= 14.5)
    h3_natural = run_h3_ligand_gradient_rotation(
        X_ago,
        in_A_ago,
        in_B_ago,
        seg_ago,
        bundle_ago,
        g_ago,
        (13.7, 14.7),
        X_thc,
        in_A_thc,
        in_B_thc,
        seg_thc,
        bundle_thc,
        g_thc,
        (12.2, 13.1),
        X_pool[in_shared_pool],
        lag_steps=15,
        n_boot=1000,
    )
    h3_shared_window = run_h3_ligand_gradient_rotation(
        X_ago,
        in_A_ago_sh,
        in_B_ago_sh,
        seg_ago,
        bundle_ago_sh,
        g_ago,
        (13.0, 13.6),
        X_thc,
        in_A_thc_sh,
        in_B_thc_sh,
        seg_thc,
        bundle_thc_sh,
        g_thc,
        (13.0, 13.6),
        X_pool[(X_pool[:, 0] >= 13.0) & (X_pool[:, 0] <= 13.6)],
        lag_steps=10,
        n_boot=1000,
    )

    print("[WS1] Step 6/6: Running Segmentation & Endpoint Sensitivity Sweeps and saving OOF Parquet + JSON...")
    sensitivity = run_segmentation_and_endpoint_sensitivity(df)

    # Save strictly fold-isolated committor predictions to processed parquet for figure generation & audit
    df_oof = df_val[
        ["ligand", "ligand_code", "frame_in_ensemble", "macro_group_id"] + FEATURE_NAMES
    ].copy()
    df_oof["q_dga_global_oof"] = bundle_global.q_te_indep
    df_oof["q_dga_global_train_oof_pred"] = bundle_global.q_tr_oof_pred
    df_oof["q_b1_global_oof"] = preds_global["oof_b1"]
    df_oof["q_pt_epi_global_oof"] = preds_global["oof_pt_epi"]
    df_oof["q_gbr_global_oof"] = preds_global["oof_gbr"]
    df_oof["q_dga_within_natural_oof"] = np.nan
    df_oof.loc[is_ago, "q_dga_within_natural_oof"] = bundle_ago.q_te_indep
    df_oof.loc[is_thc, "q_dga_within_natural_oof"] = bundle_thc.q_te_indep
    df_oof["q_pt_within_natural_oof"] = np.nan
    df_oof.loc[is_ago, "q_pt_within_natural_oof"] = preds_ago["oof_pt_epi"]
    df_oof.loc[is_thc, "q_pt_within_natural_oof"] = preds_thc["oof_pt_epi"]
    df_oof.to_parquet(PROJECT_ROOT / "data" / "processed" / "ws1" / "ws1_oof_predictions.parquet", index=False)

    elapsed = time.time() - t0
    metrics_payload = {
        "workstream": "WS1_Tier1C_THC_vs_FullAgonist_Global_TM6_Activation",
        "provenance": {
            "repository": "ShuklaGroup/THC_binding",
            "citation": "Dutta S, Selvam B, Das A & Shukla D (2022), J. Biol. Chem. 298(4):101764 (PMID: 35227761, DOI: 10.1016/j.jbc.2022.101764)",
        },
        "cv_protocol": {
            "splitter": "LineageGroupSplitter(n_splits=5, stratified=True, random_state=42)",
            "fold_isolation": "Strict two-stage isolation: fold-j predictors train strictly on dga_tr_j (fitted on tr_j alone) and evaluate strictly against independent dga_te_j (fitted on te_j alone) + model-free Test PBR.",
        },
        "runtime_seconds": float(elapsed),
        "data_audit": data_audit,
        "regime_I_global_full_activation_11p8_to_15p2": {
            "description": (
                "Pure-TM6 global activation endpoints A_TM6 (R_TM6 <= 11.8 A) -> B_TM6 (R_TM6 >= 15.2 A) "
                "on pooled 3-CV receptor coordinates (zero ligand-identity input, stratified macro-block CV); "
                "matched intermediate evaluation slice [12.5, 14.5] A (>= 0.7 A buffered from A and B)."
            ),
            "dga_lag_sweep": bundle_global.lag_sweep,
            "h1_benchmark": h1_global,
        },
        "regime_II_within_ligand_natural_transitions": {
            "full_agonist_13p2_to_15p2": {
                "dga_lag_sweep": bundle_ago.lag_sweep,
                "h1_benchmark": h1_ago,
            },
            "delta9_thc_11p8_to_13p5": {
                "dga_lag_sweep": bundle_thc.lag_sweep,
                "h1_benchmark": h1_thc,
            },
        },
        "h2_mori_zwanzig_memory_step_A": {
            "global_pooled_slice_12p5_14p5": h2_global,
            "within_full_agonist_slice_13p7_14p7": h2_ago,
            "within_delta9_thc_slice_12p2_13p1": h2_thc,
        },
        "h3_ligand_bottleneck_rotation": {
            "natural_within_ligand_transitions": h3_natural,
            "shared_boundary_window_12p8_to_13p8": h3_shared_window,
        },
        "sensitivity_analyses": sensitivity,
    }

    metrics_path = RESULTS_WS1_DIR / "ws1_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics_payload, fh, indent=2)

    print(f"[WS1] Completed all analyses in {elapsed:.1f}s -> {metrics_path}")
    return metrics_payload


if __name__ == "__main__":
    main()
