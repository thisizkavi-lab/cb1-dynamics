"""Step 3 of WS2: Leakage-Free Lineage-Grouped Hypothesis Tests (H1 & Two-Step H2 Memory Collapse).

Executes strict lineage-grouped StratifiedGroupKFold (n_splits=5) cross-validation on
`data/processed/ws2/box_aea_cohort_features.parquet`, fitting every scaler, PCA projection,
spline, DGA operator, and committor model exclusively inside training folds. Evaluates
H1 (matched-R_portal and matched-R_TM6 sidechain gating & cooperative epistasis) and
H2 (two-step Mori-Zwanzig memory detection & collapse across Tier A -> Tier B -> Tier C
over delta_t in {0.5, 1.0, 2.0, 5.0, 10.0} ns) against both exact 5,000-run 800-state
`q_+^{MSM}` and fold-isolated continuous stopped-DGA `q_+^{DGA}` + target-free `E_PBR`.
"""

from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
import pickle
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.core import (
    LineageGroupSplitter,
    OracleHistGBRModel,
    PetersTroutCommittorModel,
    PriorBaselineB0,
    ProgressVelocityBaselineB2,
    SplineProgressBaselineB1,
    StoppedDGACommittor,
    assert_contiguous_segments,
    assert_disjoint_group_split,
    bootstrap_bss_and_delta_ell_ci,
    brier_score,
    brier_skill_score,
    build_forward_stopped_pairs,
    committor_log_likelihood,
    compute_lagged_displacements,
    compute_memory_collapse_ratio,
    delta_log_likelihood,
    estimate_reversible_msm_stationary_weights,
    evaluate_committor_pbr,
    evaluate_mori_zwanzig_memory,
    matched_slice_metrics,
    projected_bellman_residual,
    run_group_permutation_null_N1,
    run_time_reversal_diagnostic_N2b,
    separatrix_purity,
)
from src.ws2_box_msm.msm_tpt import estimate_reversible_transition_matrix, solve_discrete_tpt
from src.ws2_box_msm.stream_featurize import TIER_A_COLS, TIER_B_ADDED_COLS, TIER_C_ADDED_COLS

CORE_SWITCH_DISTANCE_COLS = [
    "sc_f155_n393ca_A",
    "sc_f155_f237_ring_A",
    "sc_f155_f237_min_A",
    "sc_f200_w356_min_A",
    "sc_f200_w356_ring_A",
    "sc_y294_y397_oh_A",
    "sc_d163_n393_polar_A",
    "sc_r214_d338_ionic_A",
    "sc_vloop_pocket_min_A",
    "sc_portal_f177_f379_min_A",
]

LIPID_ONLY_COLS = [
    "env_lip_tm17_portal_min_A",
    "env_cer_osm_tm17_min_A",
    "env_plpi_dpps_tm17_min_A",
    "env_glyco_nt_ecl_min_A",
    "env_lip_tm17_contacts_45A",
    "env_lip_tm24_cleft_min_A",
    "env_lip_tm24_contacts_45A",
    "env_aea_lip_shell_min_A",
]

AEA_POSE_COLS = [
    "env_aea_head_pocket_A",
    "env_aea_tail_pocket_A",
    "env_aea_head_minus_tail_depth_A",
    "env_aea_head_s383_polar_A",
    "env_aea_c5_c10_c18_cos",
    "env_aea_end_to_end_A",
    "env_aea_sec_pocket_A",
]


def _fit_fold_dga_pca_pipeline(
    X_raw: np.ndarray,
    in_A: np.ndarray,
    in_B: np.ndarray,
    seg_id: np.ndarray,
    lag_steps: int,
    n_pca: int = 10,
    n_rbf: int = 75,
    reg: float = 1e-4,
    random_state: int = 42,
) -> tuple[StandardScaler, PCA, StoppedDGACommittor, np.ndarray]:
    """Fit StandardScaler + PCA + reversible weights + StoppedDGACommittor strictly on supplied subset."""
    interior = ~(in_A | in_B)
    scaler = StandardScaler().fit(X_raw[interior])
    X_sc = scaler.transform(X_raw)
    pca = PCA(n_components=min(n_pca, X_sc.shape[1]), random_state=random_state).fit(X_sc[interior])
    X_pca = pca.transform(X_sc)

    w_frame, _, _ = estimate_reversible_msm_stationary_weights(
        X_pca,
        seg_id=seg_id,
        lag_steps=lag_steps,
        n_clusters=60,
        random_state=random_state,
    )
    dga = StoppedDGACommittor(
        lag_steps=lag_steps,
        n_rbf=n_rbf,
        include_poly2=True,
        reg=reg,
        random_state=random_state,
    )
    dga.fit(X_pca, in_A=in_A, in_B=in_B, seg_id=seg_id, sample_weights_t0=w_frame)
    return scaler, pca, dga, w_frame


def evaluate_continuous_dga_across_lags(
    X_tier_c: np.ndarray,
    in_A_dga: np.ndarray,
    in_B_dga: np.ndarray,
    seg_id: np.ndarray,
    q_msm_ref: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    lag_steps_list: list[int],
    primary_lag_steps: int = 200,
) -> tuple[dict[str, Any], list[np.ndarray], np.ndarray, list[np.ndarray]]:
    """Validate stopped-DGA across multiple lags (PBR, N2b reversibility, correlation with global MSM).

    Returns:
      - dga_diagnostics dict for all lags
      - primary_train_targets: list of per-fold `q_dga_train` arrays (length = len(tr)) fit ONLY on `tr`
      - primary_test_oof: array (N,) of independent test-fold `q_dga_test` solved ONLY on `te`
      - primary_test_witness_pca: list of per-fold 10D train-PCA test coordinates `X_te_in_tr_pca`
        defining the shared target-free PBR witness subspace at `primary_lag_steps`
    """
    n_total = X_tier_c.shape[0]
    diagnostics: dict[str, Any] = {}
    primary_train_targets: list[np.ndarray] = []
    primary_test_oof = np.zeros(n_total, dtype=np.float64)
    primary_test_witness_pca: list[np.ndarray] = []

    for lag in lag_steps_list:
        lag_ns = lag * 0.1
        fold_pbrs = []
        fold_n2b_l2 = []
        fold_n2b_rho = []
        oof_train_pred_on_te = np.zeros(n_total, dtype=np.float64)
        oof_indep_te = np.zeros(n_total, dtype=np.float64)
        train_targets_this_lag: list[np.ndarray] = []
        test_witness_this_lag: list[np.ndarray] = []

        for f_idx, (tr, te) in enumerate(folds):
            sc_tr, pca_tr, dga_tr, w_tr = _fit_fold_dga_pca_pipeline(
                X_tier_c[tr],
                in_A_dga[tr],
                in_B_dga[tr],
                seg_id[tr],
                lag_steps=lag,
                random_state=42 + f_idx,
            )
            X_tr_pca = pca_tr.transform(sc_tr.transform(X_tier_c[tr]))
            q_tr_hat = dga_tr.predict(X_tr_pca, in_A=in_A_dga[tr], in_B=in_B_dga[tr])
            train_targets_this_lag.append(q_tr_hat)

            # N2b time-reversal diagnostic strictly on train fold with forward-initial weights w(X_{t_0})
            n2b_res = run_time_reversal_diagnostic_N2b(
                dga_tr,
                X_tr_pca,
                in_A_dga[tr],
                in_B_dga[tr],
                seg_id=seg_id[tr],
                weights_forward_initial=w_tr,
            )
            fold_n2b_l2.append(float(n2b_res["l2_reversibility_error"]))
            fold_n2b_rho.append(float(n2b_res["spearman_qplus_vs_1minus_qminus"]))

            # Predict train-fitted DGA onto held-out test fold & compute out-of-sample PBR
            X_te_in_tr_pca = pca_tr.transform(sc_tr.transform(X_tier_c[te]))
            test_witness_this_lag.append(X_te_in_tr_pca)
            oof_train_pred_on_te[te] = dga_tr.predict(X_te_in_tr_pca, in_A=in_A_dga[te], in_B=in_B_dga[te])

            pbr_te = evaluate_committor_pbr(
                predict_interior_fn=lambda X_sub, _d=dga_tr: _d.predict(X_sub),
                X=X_te_in_tr_pca,
                in_A=in_A_dga[te],
                in_B=in_B_dga[te],
                seg_id=seg_id[te],
                lag_steps=lag,
                n_witness=30,
                random_state=100 + f_idx,
            )
            fold_pbrs.append(float(pbr_te))

            # Independent test-fold DGA solved strictly on D_test (zero train leakage into test target!)
            sc_te, pca_te, dga_te, _ = _fit_fold_dga_pca_pipeline(
                X_tier_c[te],
                in_A_dga[te],
                in_B_dga[te],
                seg_id[te],
                lag_steps=lag,
                n_pca=8,
                n_rbf=45,
                reg=5e-4,
                random_state=200 + f_idx,
            )
            X_te_pca = pca_te.transform(sc_te.transform(X_tier_c[te]))
            oof_indep_te[te] = dga_te.predict(X_te_pca, in_A=in_A_dga[te], in_B=in_B_dga[te])

        interior = ~(in_A_dga | in_B_dga)
        sp_tr_vs_msm = float(stats.spearmanr(oof_train_pred_on_te[interior], q_msm_ref[interior]).statistic)
        pr_tr_vs_msm = float(np.corrcoef(oof_train_pred_on_te[interior], q_msm_ref[interior])[0, 1])
        sp_indep_vs_msm = float(stats.spearmanr(oof_indep_te[interior], q_msm_ref[interior]).statistic)
        sp_tr_vs_indep = float(stats.spearmanr(oof_train_pred_on_te[interior], oof_indep_te[interior]).statistic)

        diagnostics[f"{lag_ns:.1f}ns"] = {
            "lag_steps": lag,
            "lag_ns": lag_ns,
            "oof_pbr_mean": float(np.mean(fold_pbrs)),
            "oof_pbr_std": float(np.std(fold_pbrs)),
            "oof_pbr_per_fold": fold_pbrs,
            "n2b_l2_error_mean": float(np.mean(fold_n2b_l2)),
            "n2b_l2_error_max": float(np.max(fold_n2b_l2)),
            "n2b_spearman_mean": float(np.mean(fold_n2b_rho)),
            "n2b_passes_010_threshold": bool(np.mean(fold_n2b_l2) <= 0.10),
            "spearman_oof_dga_vs_global_msm_interior": sp_tr_vs_msm,
            "pearson_oof_dga_vs_global_msm_interior": pr_tr_vs_msm,
            "spearman_indep_test_dga_vs_global_msm_interior": sp_indep_vs_msm,
            "spearman_train_dga_vs_indep_test_dga_interior": sp_tr_vs_indep,
        }

        if lag == primary_lag_steps:
            primary_train_targets = train_targets_this_lag
            primary_test_oof = oof_indep_te.copy()
            primary_test_witness_pca = test_witness_this_lag

    return diagnostics, primary_train_targets, primary_test_oof, primary_test_witness_pca


def evaluate_h1_suite(
    df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    groups: np.ndarray,
    seg_id: np.ndarray,
    in_A_dga: np.ndarray,
    in_B_dga: np.ndarray,
    q_msm: np.ndarray,
    dga_train_targets: list[np.ndarray],
    q_dga_test_oof: np.ndarray,
    progress_col_name: str,
    slice_range: tuple[float, float],
    pbr_lag_steps: int = 200,
    run_pbr: bool = True,
    shared_test_witness_pca: list[np.ndarray] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate all baselines & H1 models on matched slices against both q_msm and fold-isolated q_dga."""
    n = len(df)
    r_prog = df[progress_col_name].to_numpy(dtype=np.float64)
    w_msm = df["msm_weight"].to_numpy(dtype=np.float64)
    w_msm_norm = w_msm / max(float(np.mean(w_msm)), 1e-30)
    zero_B = np.zeros_like(in_A_dga, dtype=bool)

    tier_a_cols = [progress_col_name] + [c for c in TIER_A_COLS if c != progress_col_name]
    core_sw_cols = [progress_col_name] + [c for c in CORE_SWITCH_DISTANCE_COLS if c != progress_col_name]
    tier_b_cols = [progress_col_name] + [c for c in (TIER_A_COLS + TIER_B_ADDED_COLS) if c != progress_col_name]
    tier_c_cols = [progress_col_name] + [c for c in (TIER_A_COLS + TIER_B_ADDED_COLS + TIER_C_ADDED_COLS) if c != progress_col_name]

    X_prog_1d = r_prog[:, None]
    dR_2ns, valid_2ns = compute_lagged_displacements(X_prog_1d, seg_id=seg_id, lag_steps=20)
    X_tier_a = df[tier_a_cols].to_numpy(dtype=np.float64)
    X_core_sw = df[core_sw_cols].to_numpy(dtype=np.float64)
    X_tier_b = df[tier_b_cols].to_numpy(dtype=np.float64)
    X_tier_c = df[tier_c_cols].to_numpy(dtype=np.float64)

    bin_edges = np.arange(slice_range[0], slice_range[1] + 0.5001, 0.5)

    model_specs = [
        ("B0_bin_marginal", "b0", X_prog_1d),
        ("B1_spline_1d", "b1", X_prog_1d),
        ("B2_spline_plus_vel2ns", "b2", X_prog_1d),
        ("TierA_CA_linear", "pt_lin", X_tier_a),
        ("H1_Switches_additive", "pt_lin", X_core_sw),
        ("H1_Switches_sparse_k8_epistasis", "pt_epi_k8", X_core_sw),
        ("H1_Switches_full_epistasis", "pt_epi", X_core_sw),
        ("TierB_Full_sidechains", "pt_lin", X_tier_b),
        ("TierC_Full_lipid_AEA", "pt_lin", X_tier_c),
        ("Tier3_Oracle_HistGBR", "oracle", X_tier_c),
    ]

    results_by_target: dict[str, Any] = {}
    saved_oof: dict[str, np.ndarray] = {}

    for target_label in ("q_msm_phys", "q_dga_oof"):
        y_eval_full = q_msm if target_label == "q_msm_phys" else q_dga_test_oof
        # Never apply pure-distance DGA B-clamping (r_portal <= 8.2 A) to q_msm_phys,
        # because vestibule I2_AB overlaps r_portal <= 8.2 A with q_msm_phys ~ 0.38!
        mask_A_t = in_A_dga
        mask_B_t = zero_B if target_label == "q_msm_phys" else in_B_dga
        oof_preds: dict[str, np.ndarray] = {name: np.zeros(n, dtype=np.float64) for name, _, _ in model_specs}
        pbr_by_model: dict[str, list[float]] = {name: [] for name, _, _ in model_specs}
        top_epistasis_terms_folds: list[list[tuple[str, float]]] = []

        for f_idx, (tr, te) in enumerate(folds):
            y_tr_all = q_msm[tr] if target_label == "q_msm_phys" else dga_train_targets[f_idx]
            int_tr = ~(mask_A_t[tr] | mask_B_t[tr])
            tr_fit = tr[int_tr]
            y_tr_fit = y_tr_all[int_tr]
            A_te = mask_A_t[te]
            B_te = mask_B_t[te]

            for name, kind, X_mat in model_specs:
                if kind == "b0":
                    est = PriorBaselineB0(progress_col=0, bin_edges=bin_edges).fit(X_mat[tr_fit], y_tr_fit)
                    oof_preds[name][te] = est.predict(X_mat[te], in_A=A_te, in_B=B_te)
                elif kind == "b1":
                    est = SplineProgressBaselineB1(progress_col=0, n_knots=5, l2_reg=1e-3).fit(X_mat[tr_fit], y_tr_fit)
                    oof_preds[name][te] = est.predict(X_mat[te], in_A=A_te, in_B=B_te)
                elif kind == "b2":
                    tr_fit_b2 = tr[int_tr & valid_2ns[tr]]
                    y_tr_b2 = y_tr_all[int_tr & valid_2ns[tr]]
                    est = ProgressVelocityBaselineB2(progress_col=0, n_knots=5, l2_reg=1e-3).fit(
                        X_mat[tr_fit_b2], dR_2ns[tr_fit_b2], y_tr_b2
                    )
                    oof_preds[name][te] = est.predict(X_mat[te], dR_2ns[te], in_A=A_te, in_B=B_te)
                elif kind == "pt_lin":
                    est = PetersTroutCommittorModel(include_interactions=False, l2_reg=1e-3).fit(
                        X_mat[tr_fit], y_tr_fit
                    )
                    oof_preds[name][te] = est.predict(X_mat[te], in_A=A_te, in_B=B_te)
                elif kind == "pt_epi_k8":
                    est = PetersTroutCommittorModel(
                        include_interactions=True,
                        l1_reg=2e-4,
                        l2_reg=1e-3,
                        max_features=8,
                        feature_names=core_sw_cols,
                    ).fit(X_mat[tr_fit], y_tr_fit)
                    oof_preds[name][te] = est.predict(X_mat[te], in_A=A_te, in_B=B_te)
                    if target_label == "q_msm_phys" and est.coef_ is not None:
                        order = np.argsort(np.abs(est.coef_))[::-1][:8]
                        top_epistasis_terms_folds.append(
                            [(est.expanded_names_[int(k)], float(est.coef_[int(k)])) for k in order if abs(est.coef_[int(k)]) > 0]
                        )
                elif kind == "pt_epi":
                    est = PetersTroutCommittorModel(
                        include_interactions=True,
                        l2_reg=2e-3,
                        feature_names=core_sw_cols,
                    ).fit(X_mat[tr_fit], y_tr_fit)
                    oof_preds[name][te] = est.predict(X_mat[te], in_A=A_te, in_B=B_te)
                elif kind == "oracle":
                    est = OracleHistGBRModel(
                        max_iter=150, max_leaf_nodes=31, min_samples_leaf=80, l2_regularization=2.0, random_state=42 + f_idx
                    ).fit(X_mat[tr_fit], y_tr_fit)
                    oof_preds[name][te] = est.predict(X_mat[te], in_A=A_te, in_B=B_te)
                else:
                    raise ValueError(f"Unknown model kind {kind}")

            # Evaluate target-free Koopman Bellman residual strictly on models solving the SAME
            # boundary problem (in_A_dga, in_B_dga) over the SINGLE shared fold-isolated witness
            # subspace X_te_in_tr_pca (n_witness=30, random_state=100 + f_idx) used by Stopped-DGA.
            if run_pbr and target_label == "q_dga_oof" and shared_test_witness_pca is not None:
                idx_s, idx_e = build_forward_stopped_pairs(seg_id[te], A_te, B_te, pbr_lag_steps)
                X_wit_start = shared_test_witness_pca[f_idx][idx_s]
                for name, kind, _ in model_specs:
                    if kind == "b2":
                        continue
                    q_te_pred = oof_preds[name][te].copy()
                    q_te_pred[A_te] = 0.0
                    q_te_pred[B_te] = 1.0
                    pbr_val = projected_bellman_residual(
                        q_pred_start=q_te_pred[idx_s],
                        q_pred_stop=q_te_pred[idx_e],
                        X_start_interior=X_wit_start,
                        n_witness=30,
                        random_state=100 + f_idx,
                    )
                    pbr_by_model[name].append(float(pbr_val))

        in_slice = (~(mask_A_t | mask_B_t)) & (r_prog >= slice_range[0]) & (r_prog <= slice_range[1])
        b1_oof = oof_preds["B1_spline_1d"]
        int_eval = ~(mask_A_t | mask_B_t)

        models_summary: dict[str, Any] = {}
        for name, _, _ in model_specs:
            eval_mask = in_slice & valid_2ns if name == "B2_spline_plus_vel2ns" else in_slice
            p_oof = oof_preds[name]
            ms = matched_slice_metrics(
                progress_coord=r_prog[eval_mask],
                q_ref=y_eval_full[eval_mask],
                q_pred=p_oof[eval_mask],
                q_baseline=b1_oof[eval_mask],
                slice_range=slice_range,
                bin_width=0.5,
            )
            ci_unw = bootstrap_bss_and_delta_ell_ci(
                group_ids=groups[eval_mask],
                q_ref=y_eval_full[eval_mask],
                q_pred=p_oof[eval_mask],
                q_baseline=b1_oof[eval_mask],
                n_boot=1000,
                random_state=42,
            )
            ci_w = bootstrap_bss_and_delta_ell_ci(
                group_ids=groups[eval_mask],
                q_ref=y_eval_full[eval_mask],
                q_pred=p_oof[eval_mask],
                q_baseline=b1_oof[eval_mask],
                weights=w_msm_norm[eval_mask],
                n_boot=1000,
                random_state=43,
            )
            sep = separatrix_purity(y_eval_full[int_eval], p_oof[int_eval])

            models_summary[name] = {
                "n_slice_frames": int(eval_mask.sum()),
                "brier_score": brier_score(y_eval_full[eval_mask], p_oof[eval_mask]),
                "log_likelihood": committor_log_likelihood(y_eval_full[eval_mask], p_oof[eval_mask]),
                "bss_vs_b1": ms["bss_int"],
                "bss_ci95": [ci_unw["bss"]["ci_low"], ci_unw["bss"]["ci_high"]],
                "delta_ell_vs_b1": ms["delta_ell_int"],
                "delta_ell_ci95": [ci_unw["delta_ell"]["ci_low"], ci_unw["delta_ell"]["ci_high"]],
                "msm_weighted_bss_vs_b1": ci_w["bss"]["point"],
                "msm_weighted_bss_ci95": [ci_w["bss"]["ci_low"], ci_w["bss"]["ci_high"]],
                "msm_weighted_delta_ell_vs_b1": ci_w["delta_ell"]["point"],
                "spearman_rho_slice": ms["spearman_rho_model"],
                "mean_within_bin_var_q_ref": ms["mean_within_bin_var_q_ref"],
                "ece_slice": ms["calibration"]["ece"],
                "calibration_curve": ms["calibration"],
                "separatrix_purity": sep,
                "per_bin": ms["per_bin"] if name in ("B1_spline_1d", "H1_Switches_full_epistasis", "TierC_Full_lipid_AEA") else [],
                "oof_pbr_mean": float(np.mean(pbr_by_model[name])) if pbr_by_model[name] else None,
                "oof_pbr_std": float(np.std(pbr_by_model[name])) if pbr_by_model[name] else None,
            }

        # Epistasis delta: full epistasis vs additive switches
        epi_mask = in_slice
        epi_vs_add = bootstrap_bss_and_delta_ell_ci(
            group_ids=groups[epi_mask],
            q_ref=y_eval_full[epi_mask],
            q_pred=oof_preds["H1_Switches_full_epistasis"][epi_mask],
            q_baseline=oof_preds["H1_Switches_additive"][epi_mask],
            n_boot=1000,
            random_state=99,
        )

        # N1 lineage-block permutation null on Core_Switches (with epistasis, strict fold isolation)
        n1_res = run_group_permutation_null_N1(
            X=X_core_sw,
            q_target=y_eval_full,
            groups=groups,
            progress_col=0,
            slice_range=slice_range,
            in_A=mask_A_t,
            in_B=mask_B_t,
            fit_interior_only=True,
            include_interactions=True,
            l2_reg=2e-3,
            n_splits=5,
            n_perms=99,
            random_state=42,
            folds=folds,
            fold_q_train=(None if target_label == "q_msm_phys" else dga_train_targets),
        )

        results_by_target[target_label] = {
            "progress_col": progress_col_name,
            "slice_range": list(slice_range),
            "models": models_summary,
            "epistasis_gain_over_additive": {
                "delta_bss": epi_vs_add["bss"]["point"],
                "delta_bss_ci95": [epi_vs_add["bss"]["ci_low"], epi_vs_add["bss"]["ci_high"]],
                "delta_ell": epi_vs_add["delta_ell"]["point"],
                "delta_ell_ci95": [epi_vs_add["delta_ell"]["ci_low"], epi_vs_add["delta_ell"]["ci_high"]],
            },
            "n1_permutation_null": {
                k: v for k, v in n1_res.items() if k not in ("oof_b1", "oof_h1")
            },
            "top_sparse_k8_terms_fold0": top_epistasis_terms_folds[0] if top_epistasis_terms_folds else [],
        }

        if progress_col_name == "r_portal_A":
            for m_key in ("B1_spline_1d", "TierA_CA_linear", "H1_Switches_additive", "H1_Switches_full_epistasis", "TierB_Full_sidechains", "TierC_Full_lipid_AEA", "Tier3_Oracle_HistGBR"):
                saved_oof[f"{target_label}__{m_key}"] = oof_preds[m_key].astype(np.float32)

    return results_by_target, saved_oof


def evaluate_switch_family_ablations(
    df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    groups: np.ndarray,
    in_A_dga: np.ndarray,
    q_msm: np.ndarray,
    slice_range: tuple[float, float] = (9.5, 18.5),
) -> dict[str, Any]:
    """Quantify individual add-one & leave-one-out contributions of each CB1 microswitch family."""
    r_prog = df["r_portal_A"].to_numpy(dtype=np.float64)
    zero_B = np.zeros_like(in_A_dga, dtype=bool)
    in_slice = (~in_A_dga) & (r_prog >= slice_range[0]) & (r_prog <= slice_range[1])

    families = {
        "Portal_Vloop_F177_F379_M103": ["sc_portal_f177_f379_min_A", "sc_vloop_pocket_min_A"],
        "TwinToggle_F200_W356": ["sc_f200_w356_min_A", "sc_f200_w356_ring_A", "sc_sin_chi1_U200", "sc_cos_chi1_U200", "sc_sin_chi1_U356", "sc_cos_chi1_U356", "sc_sin_chi2_U356", "sc_cos_chi2_U356"],
        "Polar_NPxxY_DRY_D163_N393_Y294_Y397": ["sc_d163_n393_polar_A", "sc_y294_y397_oh_A", "sc_r214_d338_ionic_A", "sc_sin_chi1_U294", "sc_cos_chi1_U294", "sc_sin_chi1_U397", "sc_cos_chi1_U397"],
        "TM2_TM4_F155_F237": ["sc_f155_n393ca_A", "sc_f155_f237_ring_A", "sc_f155_f237_min_A", "sc_sin_chi1_U155", "sc_cos_chi1_U155", "sc_sin_chi1_U237", "sc_cos_chi1_U237"],
        "AEA_Internal_Orientation": AEA_POSE_COLS,
        "Neuronal_Lipid_Ceramide_Shell": LIPID_ONLY_COLS,
    }

    def oof_for_cols(cols: list[str], use_b1: bool = False) -> np.ndarray:
        X = df[cols].to_numpy(dtype=np.float64)
        oof = np.zeros(len(df), dtype=np.float64)
        for tr, te in folds:
            tr_fit = tr[~in_A_dga[tr]]
            if use_b1:
                est = SplineProgressBaselineB1(progress_col=0, l2_reg=1e-3).fit(X[tr_fit], q_msm[tr_fit])
            else:
                est = PetersTroutCommittorModel(include_interactions=True, l2_reg=2e-3).fit(X[tr_fit], q_msm[tr_fit])
            oof[te] = est.predict(X[te], in_A=in_A_dga[te], in_B=zero_B[te])
        return oof

    b1_oof = oof_for_cols(["r_portal_A"], use_b1=True)
    out: dict[str, Any] = {}
    for fam_name, fam_cols in families.items():
        oof_f = oof_for_cols(["r_portal_A"] + fam_cols, use_b1=False)
        ci = bootstrap_bss_and_delta_ell_ci(
            group_ids=groups[in_slice],
            q_ref=q_msm[in_slice],
            q_pred=oof_f[in_slice],
            q_baseline=b1_oof[in_slice],
            n_boot=1000,
            random_state=77,
        )
        out[fam_name] = {
            "n_added_features": len(fam_cols),
            "add_one_bss_vs_b1": ci["bss"]["point"],
            "add_one_bss_ci95": [ci["bss"]["ci_low"], ci["bss"]["ci_high"]],
            "add_one_delta_ell_vs_b1": ci["delta_ell"]["point"],
            "add_one_delta_ell_ci95": [ci["delta_ell"]["ci_low"], ci["delta_ell"]["ci_high"]],
        }
    return out


def _run_single_mz_worker(task: dict[str, Any]) -> tuple[str, str, str, dict[str, Any]]:
    """Worker for parallel multi-lag Mori-Zwanzig memory evaluation with strict fold isolation."""
    mem_res = evaluate_mori_zwanzig_memory(
        X_static=task["X_static"],
        seg_id=task["seg_id"],
        q_target=task["q_target"],
        groups=task["groups"],
        lag_steps=task["lag_steps_grid"],
        eval_mask=task["eval_mask"],
        in_A=task["in_A"],
        in_B=task["in_B"],
        fit_interior_only=True,
        include_interactions=False,
        l2_reg=2e-3,
        n_splits=5,
        n_shuffles=task["n_shuffles"],
        n_cells_shuffle=20,
        random_state=task["random_state"],
        folds=task["folds"],
        fold_q_train=task["fold_q_train"],
    )
    mem_res["lag_steps_01ns"] = int(task["lag_steps_01ns"])
    mem_res["lag_ns"] = float(task["lag_ns"])
    return task["target_name"], task["tier_name"], task["lag_key"], mem_res


def evaluate_h2_mori_zwanzig_hierarchy(
    df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    seg_id: np.ndarray,
    groups: np.ndarray,
    in_A_dga: np.ndarray,
    in_B_dga: np.ndarray,
    q_msm: np.ndarray,
    dga_train_targets: list[np.ndarray],
    q_dga_oof: np.ndarray,
    lag_steps_list: list[int],
    max_workers: int = 6,
) -> dict[str, Any]:
    """Run H2 two-step Mori-Zwanzig memory detection & collapse across all lags and feature tiers.

    Static progression is evaluated on all 157,204 (0.1 ns) frames; finite-lag memory
    gains and N2a within-cell history-shuffle nulls (n_shuffles=39, p_min=0.025, plus
    1,000-replicate lineage-cluster bootstrap CIs) are evaluated at exact physical lags
    [0.5, 1.0, 2.0, 5.0, 10.0] ns (k in {1, 2, 4, 10, 20} steps on the uniform 0.5 ns
    trajectory grid, 31,441 frames across all 160 runs), strictly preserving fold-isolated
    training committor targets (zero cross-fold leakage).
    """
    r_portal = df["r_portal_A"].to_numpy(dtype=np.float64)
    zero_B = np.zeros_like(in_A_dga, dtype=bool)
    eval_mask_full = (~in_A_dga) & (r_portal >= 9.0) & (r_portal <= 20.0)

    # Exact uniform 0.5 ns grid (frame_idx % 5 == 0) across all 160 continuous trajectories
    is_s5 = (df["frame_idx"].to_numpy() % 5) == 0
    n_s5 = int(np.sum(is_s5))
    full_to_s5 = np.full(len(df), -1, dtype=np.int64)
    full_to_s5[is_s5] = np.arange(n_s5, dtype=np.int64)

    seg_s5 = seg_id[is_s5]
    assert_contiguous_segments(seg_s5)
    groups_s5 = groups[is_s5]
    in_A_s5 = in_A_dga[is_s5]
    in_B_s5 = in_B_dga[is_s5]
    zero_B_s5 = zero_B[is_s5]
    eval_mask_s5 = eval_mask_full[is_s5]
    q_msm_s5 = q_msm[is_s5]
    q_dga_oof_s5 = q_dga_oof[is_s5]

    folds_s5: list[tuple[np.ndarray, np.ndarray]] = []
    dga_train_s5: list[np.ndarray] = []
    for f_idx, (tr, te) in enumerate(folds):
        keep_tr = is_s5[tr]
        keep_te = is_s5[te]
        tr_s5 = full_to_s5[tr[keep_tr]]
        te_s5 = full_to_s5[te[keep_te]]
        assert_disjoint_group_split(groups_s5, tr_s5, te_s5)
        folds_s5.append((tr_s5, te_s5))
        dga_train_s5.append(dga_train_targets[f_idx][keep_tr])

    tier_defs = [
        ("Step0_ReceptorCA_only", [c for c in TIER_A_COLS if c != "r_portal_A"]),
        ("TierA_CA_backbone", TIER_A_COLS),
        ("TierB_Sidechains", TIER_A_COLS + TIER_B_ADDED_COLS),
        ("TierB_plus_LipidsOnly", TIER_A_COLS + TIER_B_ADDED_COLS + LIPID_ONLY_COLS),
        ("TierC_Full_Lipid_AEA", TIER_A_COLS + TIER_B_ADDED_COLS + TIER_C_ADDED_COLS),
    ]

    static_by_target: dict[str, dict[str, Any]] = {"q_msm_phys": {}, "q_dga_oof": {}}
    tasks: list[dict[str, Any]] = []

    for target_name, y_eval_full, y_eval_s5 in (
        ("q_msm_phys", q_msm, q_msm_s5),
        ("q_dga_oof", q_dga_oof, q_dga_oof_s5),
    ):
        mask_B_full = zero_B if target_name == "q_msm_phys" else in_B_dga
        mask_B_grid = zero_B_s5 if target_name == "q_msm_phys" else in_B_s5
        for tier_name, cols in tier_defs:
            X_tier_full = df[cols].to_numpy(dtype=np.float64)
            oof_stat = np.zeros(len(df), dtype=np.float64)
            for f_idx, (tr, te) in enumerate(folds):
                y_tr_all = q_msm[tr] if target_name == "q_msm_phys" else dga_train_targets[f_idx]
                int_tr = ~(in_A_dga[tr] | mask_B_full[tr])
                tr_fit = tr[int_tr]
                m_st = PetersTroutCommittorModel(include_interactions=False, l2_reg=2e-3).fit(
                    X_tier_full[tr_fit], y_tr_all[int_tr]
                )
                oof_stat[te] = m_st.predict(X_tier_full[te], in_A=in_A_dga[te], in_B=mask_B_full[te])

            static_by_target[target_name][tier_name] = {
                "n_features": len(cols),
                "static_log_likelihood": committor_log_likelihood(
                    y_eval_full[eval_mask_full], oof_stat[eval_mask_full]
                ),
                "static_brier_score": brier_score(y_eval_full[eval_mask_full], oof_stat[eval_mask_full]),
                "static_spearman_rho": float(
                    stats.spearmanr(y_eval_full[eval_mask_full], oof_stat[eval_mask_full]).statistic
                ),
            }

            X_tier_s5 = X_tier_full[is_s5]
            for lag_01ns in lag_steps_list:
                if lag_01ns % 5 != 0:
                    raise ValueError(f"Expected lag_01ns divisible by 5 (0.5 ns), got {lag_01ns}")
                lag_s5 = lag_01ns // 5
                lag_ns = lag_01ns * 0.1
                lag_key = f"{lag_ns:.1f}ns"
                tasks.append(
                    {
                        "target_name": target_name,
                        "tier_name": tier_name,
                        "lag_key": lag_key,
                        "lag_steps_01ns": lag_01ns,
                        "lag_steps_grid": lag_s5,
                        "lag_ns": lag_ns,
                        "X_static": X_tier_s5,
                        "seg_id": seg_s5,
                        "q_target": y_eval_s5,
                        "groups": groups_s5,
                        "eval_mask": eval_mask_s5,
                        "in_A": in_A_s5,
                        "in_B": mask_B_grid,
                        "n_shuffles": 39,
                        "random_state": 42 + lag_01ns,
                        "folds": folds_s5,
                        "fold_q_train": None if target_name == "q_msm_phys" else dga_train_s5,
                    }
                )

    tier_lag_mem: dict[str, dict[str, dict[str, Any]]] = {
        "q_msm_phys": {t_name: {} for t_name, _ in tier_defs},
        "q_dga_oof": {t_name: {} for t_name, _ in tier_defs},
    }
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_run_single_mz_worker, t) for t in tasks]
        done_mz = 0
        for fut in concurrent.futures.as_completed(futs):
            tgt, t_name, lk, m_res = fut.result()
            tier_lag_mem[tgt][t_name][lk] = m_res
            done_mz += 1
            if done_mz % 10 == 0 or done_mz == len(tasks):
                print(f"[WS2 Analysis]   H2 MZ progress: {done_mz}/{len(tasks)} (tier, lag) cells complete", flush=True)

    h2_results: dict[str, Any] = {}
    for target_name in ("q_msm_phys", "q_dga_oof"):
        collapse_by_lag: dict[str, Any] = {}
        for lag_01ns in lag_steps_list:
            lk = f"{lag_01ns * 0.1:.1f}ns"
            d_s0 = tier_lag_mem[target_name]["Step0_ReceptorCA_only"][lk]["delta_ell_mem"]
            d_a = tier_lag_mem[target_name]["TierA_CA_backbone"][lk]["delta_ell_mem"]
            d_b = tier_lag_mem[target_name]["TierB_Sidechains"][lk]["delta_ell_mem"]
            d_bl = tier_lag_mem[target_name]["TierB_plus_LipidsOnly"][lk]["delta_ell_mem"]
            d_c = tier_lag_mem[target_name]["TierC_Full_Lipid_AEA"][lk]["delta_ell_mem"]
            collapse_by_lag[lk] = {
                "delta_ell_mem_Step0_ReceptorCA": d_s0,
                "delta_ell_mem_TierA": d_a,
                "delta_ell_mem_TierB": d_b,
                "delta_ell_mem_TierB_plus_Lipids": d_bl,
                "delta_ell_mem_TierC": d_c,
                "rho_mem_TierA_to_TierB": compute_memory_collapse_ratio(d_a, d_b),
                "rho_mem_TierB_to_TierC": compute_memory_collapse_ratio(d_b, d_c),
                "rho_mem_TierA_to_TierC": compute_memory_collapse_ratio(d_a, d_c),
                "rho_mem_Step0_to_TierC": compute_memory_collapse_ratio(d_s0, d_c),
                "rho_mem_TierB_to_TierB_Lipids": compute_memory_collapse_ratio(d_b, d_bl),
            }

        h2_results[target_name] = {
            "eval_slice_r_portal_A": [9.0, 20.0],
            "n_eval_frames_static_01ns": int(eval_mask_full.sum()),
            "n_eval_frames_grid_05ns": int(eval_mask_s5.sum()),
            "grid_spacing_ns": 0.5,
            "n_n2a_shuffles": 39,
            "static_progression": static_by_target[target_name],
            "memory_by_tier_and_lag": tier_lag_mem[target_name],
            "collapse_summary_by_lag": collapse_by_lag,
        }

    return h2_results


def compute_biophysical_mechanism_profiles(df: pd.DataFrame) -> dict[str, Any]:
    """Summarize physical conformations of key CB1 switches, AEA pose, and lipid wedging across binding states."""
    macro_labels = {
        2: "U_Sol (Ex)",
        3: "U1_Mem (Bulk2)",
        1: "U2_Mem (Bulk1)",
        0: "I1_TM17 (Portal)",
        4: "I2_TM17 (AB Vestibule)",
        5: "B_Pro (B1 Orthosteric)",
    }
    probe_cols = [
        "r_portal_A",
        "ca_r214_d338_A",
        "ca_i119_f379_A",
        "sc_portal_f177_f379_min_A",
        "sc_vloop_pocket_min_A",
        "sc_f200_w356_min_A",
        "sc_f155_f237_min_A",
        "sc_f155_n393ca_A",
        "sc_y294_y397_oh_A",
        "sc_d163_n393_polar_A",
        "env_aea_head_minus_tail_depth_A",
        "env_aea_head_s383_polar_A",
        "env_aea_end_to_end_A",
        "env_lip_tm17_portal_min_A",
        "env_lip_tm17_contacts_45A",
        "env_lip_tm24_cleft_min_A",
    ]
    profiles: dict[str, Any] = {}
    for m_id, m_name in macro_labels.items():
        sub = df[df["macrostate_id"] == m_id]
        if len(sub) == 0:
            continue
        profiles[m_name] = {
            "n_frames_cohort": int(len(sub)),
            "q_msm_median": float(sub["q_msm_phys"].median()),
            "medians": {c: float(sub[c].median()) for c in probe_cols},
            "iqr25_75": {
                c: [float(np.percentile(sub[c], 25)), float(np.percentile(sub[c], 75))] for c in probe_cols
            },
        }
    return profiles


def evaluate_leave_test_lineage_out_msm_audit(
    root: Path,
    df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    groups: np.ndarray,
    in_A_dga: np.ndarray,
    slice_range: tuple[float, float] = (9.5, 18.5),
) -> tuple[dict[str, Any], np.ndarray]:
    """Refit the 5,000-run reversible 800-state MSM & discrete TPT per fold without test lineages.

    Excludes from `CB1_frame_cluster_indices.pkl` every trajectory whose `msm_traj_idx` OR
    `pr_round` in `box_cb1_dcd_catalog.json` belongs to any held-out test trajectory/lineage
    component in fold `j`, verifying that global `q_msm_phys` and downstream matched-slice
    `BSS_int` are invariant to holding out test trajectories/rounds from the 5,000-run MSM.
    """
    box_dir = root / "data/inventory/box_2025"
    proc_dir = root / "data/processed/ws2"

    catalog = json.loads((box_dir / "box_cb1_dcd_catalog.json").read_text())
    cat_items = sorted(catalog["items"], key=lambda x: (int(x["pr_round"]), int(x["frame_seed"])))
    if len(cat_items) != 5000:
        raise ValueError(f"Expected 5,000 catalog items, got {len(cat_items)}")
    cat_pr_rounds = np.array([int(x["pr_round"]) for x in cat_items], dtype=np.int64)

    with open(box_dir / "CB1_frame_cluster_indices.pkl", "rb") as fh:
        dtrajs: list[np.ndarray] = pickle.load(fh)
    if len(dtrajs) != 5000:
        raise ValueError(f"Expected 5,000 dtrajs, got {len(dtrajs)}")

    df_micro = pd.read_parquet(proc_dir / "msm_800_microstates.parquet")
    df_act = df_micro[df_micro["is_active"]].sort_values("active_idx")
    active_set = df_act["microstate_id"].to_numpy(dtype=int)
    full2active = df_micro["active_idx"].to_numpy(dtype=int)
    A_full = df_micro.loc[df_micro["in_A_phys"], "microstate_id"].to_numpy(dtype=int)
    B_full = df_micro.loc[df_micro["in_B_phys"], "microstate_id"].to_numpy(dtype=int)

    micro_ids = df["microstate_id"].to_numpy(dtype=int)
    q_msm_global = df["q_msm_phys"].to_numpy(dtype=np.float64)
    r_prog = df["r_portal_A"].to_numpy(dtype=np.float64)
    zero_B = np.zeros_like(in_A_dga, dtype=bool)

    oof_q_round_excl = np.zeros(len(df), dtype=np.float64)
    oof_q_traj_only = np.zeros(len(df), dtype=np.float64)
    fold_train_q_round_excl: list[np.ndarray] = []
    per_fold_records: list[dict[str, Any]] = []

    for f_idx, (tr, te) in enumerate(folds):
        te_gids = np.unique(groups[te])
        te_cohort_mask = np.isin(groups, te_gids)
        excl_msm = set(int(x) for x in np.unique(df.loc[te_cohort_mask, "msm_traj_idx"]))
        excl_pr = set(int(x) for x in np.unique(df.loc[te_cohort_mask, "pr_round"]))

        dtrajs_tr_round = [
            dt for m, dt in enumerate(dtrajs) if (m not in excl_msm) and (int(cat_pr_rounds[m]) not in excl_pr)
        ]
        P_tr_r, pi_tr_r = estimate_reversible_transition_matrix(
            dtrajs_tr_round, active_set, full2active, lag_frames=350, max_iter=2500, tol=1e-13
        )
        tpt_tr_r = solve_discrete_tpt(P_tr_r, pi_tr_r, active_set, A_full, B_full, lag_ns=35.0)
        q_map_r = tpt_tr_r["q_full"]
        fold_train_q_round_excl.append(q_map_r[micro_ids[tr]])
        oof_q_round_excl[te] = q_map_r[micro_ids[te]]

        dtrajs_tr_traj = [dt for m, dt in enumerate(dtrajs) if m not in excl_msm]
        P_tr_t, pi_tr_t = estimate_reversible_transition_matrix(
            dtrajs_tr_traj, active_set, full2active, lag_frames=350, max_iter=2500, tol=1e-13
        )
        tpt_tr_t = solve_discrete_tpt(P_tr_t, pi_tr_t, active_set, A_full, B_full, lag_ns=35.0)
        oof_q_traj_only[te] = tpt_tr_t["q_full"][micro_ids[te]]

        per_fold_records.append(
            {
                "fold": f_idx,
                "n_excluded_cohort_trajs": len(excl_msm),
                "n_excluded_pr_rounds": len(excl_pr),
                "n_excluded_5000run_dtrajs_round_level": 5000 - len(dtrajs_tr_round),
                "n_train_5000run_dtrajs_round_level": len(dtrajs_tr_round),
                "mfpt_A_to_B_us_round_level": float(tpt_tr_r["mfpt_A_to_B_us"]),
                "harmonic_max_residual_round_level": float(tpt_tr_r["harmonic_max_residual"]),
                "mfpt_A_to_B_us_traj_only": float(tpt_tr_t["mfpt_A_to_B_us"]),
            }
        )

    if np.any(np.isnan(oof_q_round_excl)) or np.any(np.isnan(oof_q_traj_only)):
        raise ValueError("Unexpected NaN in fold-isolated 5,000-run MSM committor refit.")

    # Train downstream H1/TierA/TierC models strictly on fold-isolated q_msm_tr_fold and evaluate on oof_q_round_excl
    core_sw_cols = ["r_portal_A"] + [c for c in CORE_SWITCH_DISTANCE_COLS if c != "r_portal_A"]
    tier_a_cols = ["r_portal_A"] + [c for c in TIER_A_COLS if c != "r_portal_A"]
    tier_c_cols = ["r_portal_A"] + [c for c in (TIER_A_COLS + TIER_B_ADDED_COLS + TIER_C_ADDED_COLS) if c != "r_portal_A"]

    X_b1 = r_prog[:, None]
    X_sw = df[core_sw_cols].to_numpy(dtype=np.float64)
    X_ta = df[tier_a_cols].to_numpy(dtype=np.float64)
    X_tc = df[tier_c_cols].to_numpy(dtype=np.float64)

    oof_b1 = np.zeros(len(df), dtype=np.float64)
    oof_sw = np.zeros(len(df), dtype=np.float64)
    oof_ta = np.zeros(len(df), dtype=np.float64)
    oof_tc = np.zeros(len(df), dtype=np.float64)

    for f_idx, (tr, te) in enumerate(folds):
        int_tr = ~in_A_dga[tr]
        tr_fit = tr[int_tr]
        y_tr_fit = fold_train_q_round_excl[f_idx][int_tr]
        A_te = in_A_dga[te]
        B_te = zero_B[te]

        m_b1 = SplineProgressBaselineB1(progress_col=0, n_knots=5, l2_reg=1e-3).fit(X_b1[tr_fit], y_tr_fit)
        oof_b1[te] = m_b1.predict(X_b1[te], in_A=A_te, in_B=B_te)

        m_sw = PetersTroutCommittorModel(include_interactions=True, l2_reg=2e-3, feature_names=core_sw_cols).fit(
            X_sw[tr_fit], y_tr_fit
        )
        oof_sw[te] = m_sw.predict(X_sw[te], in_A=A_te, in_B=B_te)

        m_ta = PetersTroutCommittorModel(include_interactions=False, l2_reg=1e-3).fit(X_ta[tr_fit], y_tr_fit)
        oof_ta[te] = m_ta.predict(X_ta[te], in_A=A_te, in_B=B_te)

        m_tc = PetersTroutCommittorModel(include_interactions=False, l2_reg=1e-3).fit(X_tc[tr_fit], y_tr_fit)
        oof_tc[te] = m_tc.predict(X_tc[te], in_A=A_te, in_B=B_te)

    in_slice = (~in_A_dga) & (r_prog >= slice_range[0]) & (r_prog <= slice_range[1])
    downstream_models: dict[str, Any] = {}
    for m_name, p_oof in (
        ("H1_Switches_full_epistasis", oof_sw),
        ("TierA_CA_linear", oof_ta),
        ("TierC_Full_lipid_AEA", oof_tc),
    ):
        ms = matched_slice_metrics(
            progress_coord=r_prog[in_slice],
            q_ref=oof_q_round_excl[in_slice],
            q_pred=p_oof[in_slice],
            q_baseline=oof_b1[in_slice],
            slice_range=slice_range,
            bin_width=0.5,
        )
        ci = bootstrap_bss_and_delta_ell_ci(
            group_ids=groups[in_slice],
            q_ref=oof_q_round_excl[in_slice],
            q_pred=p_oof[in_slice],
            q_baseline=oof_b1[in_slice],
            n_boot=1000,
            random_state=42,
        )
        downstream_models[m_name] = {
            "bss_vs_b1": ms["bss_int"],
            "bss_ci95": [ci["bss"]["ci_low"], ci["bss"]["ci_high"]],
            "delta_ell_vs_b1": ms["delta_ell_int"],
            "delta_ell_ci95": [ci["delta_ell"]["ci_low"], ci["delta_ell"]["ci_high"]],
            "spearman_rho_slice": ms["spearman_rho_model"],
        }

    audit_summary = {
        "exclusion_protocol": (
            "Per fold j, exclude from the 5,000-run dtrajs every trajectory whose msm_traj_idx OR "
            "entire adaptive seeding round pr_round belongs to any held-out test lineage component; "
            "refit reversible 800-state MLE transition matrix (lag=35.0 ns) and solve exact TPT."
        ),
        "round_and_lineage_exclusion": {
            "pearson_r_vs_global_msm": float(np.corrcoef(q_msm_global, oof_q_round_excl)[0, 1]),
            "spearman_rho_vs_global_msm": float(stats.spearmanr(q_msm_global, oof_q_round_excl).statistic),
            "mae_vs_global_msm": float(np.mean(np.abs(q_msm_global - oof_q_round_excl))),
            "max_abs_diff_vs_global_msm": float(np.max(np.abs(q_msm_global - oof_q_round_excl))),
            "slice_mae_vs_global_msm": float(np.mean(np.abs(q_msm_global[in_slice] - oof_q_round_excl[in_slice]))),
        },
        "trajectory_lineage_only_exclusion": {
            "pearson_r_vs_global_msm": float(np.corrcoef(q_msm_global, oof_q_traj_only)[0, 1]),
            "spearman_rho_vs_global_msm": float(stats.spearmanr(q_msm_global, oof_q_traj_only).statistic),
            "mae_vs_global_msm": float(np.mean(np.abs(q_msm_global - oof_q_traj_only))),
            "max_abs_diff_vs_global_msm": float(np.max(np.abs(q_msm_global - oof_q_traj_only))),
        },
        "per_fold": per_fold_records,
        "downstream_matched_portal_skill_on_leave_out_msm": downstream_models,
    }
    return audit_summary, oof_q_round_excl


def run_ws2_full_analysis(root: Path) -> dict[str, Any]:
    t0 = time.time()
    proc_dir = root / "data/processed/ws2"
    res_dir = root / "results/ws2"
    res_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(proc_dir / "box_aea_cohort_features.parquet")
    msm_summary = json.loads((proc_dir / "msm_tpt_summary.json").read_text())
    cohort_manifest = json.loads((proc_dir / "cohort_featurization_manifest.json").read_text())

    seg_id = df["cohort_idx"].to_numpy(dtype=np.int64)
    assert_contiguous_segments(seg_id)
    groups = df["lineage_group_id"].to_numpy(dtype=np.int64)
    pr_rounds = df["pr_round"].to_numpy(dtype=np.int64)
    y_strata = np.asarray(pd.Categorical(df["stratum"]).codes, dtype=np.int64)

    # Stratified lineage-grouped 5-fold split (zero shared trajectories, pr_rounds, or <1 A initial TM RMSD)
    splitter = LineageGroupSplitter(n_splits=5, stratified=True, random_state=42)
    folds = list(splitter.split(df[["r_portal_A"]].to_numpy(), y_strata, groups))
    fold_audit = []
    for f_i, (tr, te) in enumerate(folds):
        assert_disjoint_group_split(groups, tr, te)
        assert_disjoint_group_split(pr_rounds, tr, te)
        assert_disjoint_group_split(seg_id, tr, te)
        fold_audit.append(
            {
                "fold": f_i,
                "n_train_frames": int(len(tr)),
                "n_test_frames": int(len(te)),
                "n_train_trajs": int(len(np.unique(seg_id[tr]))),
                "n_test_trajs": int(len(np.unique(seg_id[te]))),
                "n_train_lineage_groups": int(len(np.unique(groups[tr]))),
                "n_test_lineage_groups": int(len(np.unique(groups[te]))),
            }
        )

    # Pure-distance continuous stopped-DGA endpoints (strictly non-circular wrt sidechains & lipids)
    r_portal = df["r_portal_A"].to_numpy(dtype=np.float64)
    in_A_dga = r_portal >= 22.0
    in_B_dga = r_portal <= 8.2
    zero_B = np.zeros_like(in_A_dga, dtype=bool)
    q_msm = df["q_msm_phys"].to_numpy(dtype=np.float64)
    q_msm_pureR = df["q_msm_pureR"].to_numpy(dtype=np.float64)

    all_cols = TIER_A_COLS + TIER_B_ADDED_COLS + TIER_C_ADDED_COLS
    X_all = df[all_cols].to_numpy(dtype=np.float64)

    print("[WS2 Analysis] 1/5 Validating fold-isolated continuous Stopped-DGA across lags [5, 10, 20, 35] ns...", flush=True)
    dga_diag, dga_train_targets, q_dga_test_oof, dga_test_witness_pca = evaluate_continuous_dga_across_lags(
        X_tier_c=X_all,
        in_A_dga=in_A_dga,
        in_B_dga=in_B_dga,
        seg_id=seg_id,
        q_msm_ref=q_msm,
        folds=folds,
        lag_steps_list=[50, 100, 200, 350],
        primary_lag_steps=200,
    )

    print("[WS2 Analysis] 2/5 Running leave-test-lineage-out 5,000-run MSM refit audit...", flush=True)
    leave_out_msm_audit, oof_q_msm_leave_out = evaluate_leave_test_lineage_out_msm_audit(
        root=root,
        df=df,
        folds=folds,
        groups=groups,
        in_A_dga=in_A_dga,
        slice_range=(9.5, 18.5),
    )

    print("[WS2 Analysis] 3/5 Running H1 matched-R_portal & matched-R_TM6 hypothesis tests + N1 nulls...", flush=True)
    h1_portal, oof_portal = evaluate_h1_suite(
        df=df,
        folds=folds,
        groups=groups,
        seg_id=seg_id,
        in_A_dga=in_A_dga,
        in_B_dga=in_B_dga,
        q_msm=q_msm,
        dga_train_targets=dga_train_targets,
        q_dga_test_oof=q_dga_test_oof,
        progress_col_name="r_portal_A",
        slice_range=(9.5, 18.5),
        pbr_lag_steps=200,
        run_pbr=True,
        shared_test_witness_pca=dga_test_witness_pca,
    )

    # Secondary H1 matched-R_TM6 evaluation across interior central 10th-90th percentile of ca_r214_d338_A
    r_tm6 = df["ca_r214_d338_A"].to_numpy(dtype=np.float64)
    tm6_lo, tm6_hi = np.round(np.percentile(r_tm6[~(in_A_dga | in_B_dga)], [10, 90]), 1)
    h1_tm6, _ = evaluate_h1_suite(
        df=df,
        folds=folds,
        groups=groups,
        seg_id=seg_id,
        in_A_dga=in_A_dga,
        in_B_dga=in_B_dga,
        q_msm=q_msm,
        dga_train_targets=dga_train_targets,
        q_dga_test_oof=q_dga_test_oof,
        progress_col_name="ca_r214_d338_A",
        slice_range=(float(tm6_lo), float(tm6_hi)),
        pbr_lag_steps=200,
        run_pbr=False,
    )

    # Pure receptor-only matched-R_TM6 test (excluding r_portal_A so receptor switches alone are tested at fixed TM6!)
    rec_only_cols = ["ca_r214_d338_A"] + [c for c in CORE_SWITCH_DISTANCE_COLS if c != "ca_r214_d338_A"]
    n1_rec_only_tm6 = run_group_permutation_null_N1(
        X=df[rec_only_cols].to_numpy(dtype=np.float64),
        q_target=q_msm,
        groups=groups,
        progress_col=0,
        slice_range=(float(tm6_lo), float(tm6_hi)),
        in_A=in_A_dga,
        in_B=zero_B,
        fit_interior_only=True,
        include_interactions=True,
        l2_reg=2e-3,
        n_splits=5,
        n_perms=99,
        random_state=42,
        folds=folds,
    )

    # Sensitivity check on pure-R_portal MSM TPT target (zero PCCA+ macrostate input)
    n1_pureR_portal = run_group_permutation_null_N1(
        X=df[["r_portal_A"] + CORE_SWITCH_DISTANCE_COLS].to_numpy(dtype=np.float64),
        q_target=q_msm_pureR,
        groups=groups,
        progress_col=0,
        slice_range=(9.5, 18.5),
        in_A=in_A_dga,
        in_B=zero_B,
        fit_interior_only=True,
        include_interactions=True,
        l2_reg=2e-3,
        n_splits=5,
        n_perms=99,
        random_state=42,
        folds=folds,
    )

    switch_ablations = evaluate_switch_family_ablations(
        df=df,
        folds=folds,
        groups=groups,
        in_A_dga=in_A_dga,
        q_msm=q_msm,
        slice_range=(9.5, 18.5),
    )

    print("[WS2 Analysis] 4/5 Running H2 Mori-Zwanzig multi-lag memory collapse tests [0.5, 1, 2, 5, 10] ns...", flush=True)
    h2_mem = evaluate_h2_mori_zwanzig_hierarchy(
        df=df,
        folds=folds,
        seg_id=seg_id,
        groups=groups,
        in_A_dga=in_A_dga,
        in_B_dga=in_B_dga,
        q_msm=q_msm,
        dga_train_targets=dga_train_targets,
        q_dga_oof=q_dga_test_oof,
        lag_steps_list=[5, 10, 20, 50, 100],
    )

    print("[WS2 Analysis] 5/5 Computing biophysical mechanism profiles & saving artifacts...", flush=True)
    bio_profiles = compute_biophysical_mechanism_profiles(df)

    # Save compact OOF predictions table for plotting & verification
    oof_df = pd.DataFrame(
        {
            "cohort_idx": df["cohort_idx"],
            "frame_idx": df["frame_idx"],
            "lineage_group_id": df["lineage_group_id"],
            "macrostate_id": df["macrostate_id"],
            "r_portal_A": df["r_portal_A"],
            "ca_r214_d338_A": df["ca_r214_d338_A"],
            "in_A_dga": in_A_dga,
            "in_B_dga": in_B_dga,
            "q_msm_phys": df["q_msm_phys"],
            "q_msm_leave_out_oof": oof_q_msm_leave_out.astype(np.float32),
            "q_dga_oof": q_dga_test_oof.astype(np.float32),
            **oof_portal,
        }
    )
    oof_df.to_parquet(proc_dir / "ws2_oof_predictions.parquet", compression="zstd", index=False)

    metrics_payload = {
        "workstream": "WS2_Box_5000Run_MSM_Streaming_DCD_MoriZwanzig",
        "runtime_seconds": round(time.time() - t0, 2),
        "global_5000run_msm_tpt": msm_summary,
        "leave_test_lineage_out_msm_audit": leave_out_msm_audit,
        "streamed_cohort_summary": {
            "n_trajectories": cohort_manifest["n_trajectories"],
            "total_frames": cohort_manifest["total_frames"],
            "aggregate_time_us": cohort_manifest["aggregate_time_us"],
            "parquet_size_mb": cohort_manifest["parquet_size_mb"],
            "total_feature_nan_count": cohort_manifest["total_feature_nan_count"],
            "stratum_trajectory_counts": cohort_manifest["stratum_trajectory_counts"],
            "lineage_summary": cohort_manifest["lineage_summary"],
            "cv_fold_audit": fold_audit,
        },
        "continuous_stopped_dga_validation": {
            "dga_endpoint_definition": "in_A = (r_portal_A >= 22.0 A), in_B = (r_portal_A <= 8.2 A), zero sidechain/lipid circularity",
            "n_A_frames": int(in_A_dga.sum()),
            "n_B_frames": int(in_B_dga.sum()),
            "n_interior_frames": int((~(in_A_dga | in_B_dga)).sum()),
            "primary_lag_ns": 20.0,
            "lags": dga_diag,
        },
        "H1_matched_portal_slice": h1_portal,
        "H1_matched_tm6_slice": {
            **h1_tm6,
            "receptor_only_no_portal_N1_check": {
                k: v for k, v in n1_rec_only_tm6.items() if k not in ("oof_b1", "oof_h1")
            },
        },
        "H1_pure_R_portal_target_sensitivity": {
            k: v for k, v in n1_pureR_portal.items() if k not in ("oof_b1", "oof_h1")
        },
        "H1_switch_family_ablations": switch_ablations,
        "H2_mori_zwanzig_memory_collapse": h2_mem,
        "biophysical_macrostate_profiles": bio_profiles,
    }

    out_json = res_dir / "ws2_metrics.json"
    out_json.write_text(json.dumps(metrics_payload, indent=2))
    print(f"[WS2 Analysis] Saved complete metrics to {out_json} in {metrics_payload['runtime_seconds']}s!")
    return metrics_payload


if __name__ == "__main__":
    run_ws2_full_analysis(Path("/Users/kavi/Documents/artico/cb1-dynamics"))
