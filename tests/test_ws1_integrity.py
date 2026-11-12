"""End-to-end data, anti-leakage, boundary, figure, and report-consistency tests for WS1."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core import (
    LineageGroupSplitter,
    assert_contiguous_segments,
    assert_disjoint_group_split,
    assert_no_cross_seam_pairs,
    build_backward_stopped_pairs,
    build_forward_stopped_pairs,
    compute_lagged_displacements,
    evaluate_memory_gain_with_null,
)
from src.ws1_thc_activation.run_ws1_analysis import (
    FEATURE_NAMES,
    FoldIsolatedDGABundle,
    _resegment_subset_indices,
    evaluate_h1_benchmark_suite,
    solve_fold_isolated_dga_bundle,
)


PROJECT_ROOT = Path("/Users/kavi/Documents/artico/cb1-dynamics")
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "ws1"
RESULTS_DIR = PROJECT_ROOT / "results" / "ws1"
FIG_DIR = RESULTS_DIR / "figures"


def test_ws1_processed_parquet_and_unit_audit():
    pq_path = PROCESSED_DIR / "ws1_featurized_segments.parquet"
    audit_path = PROCESSED_DIR / "ws1_data_audit.json"
    assert pq_path.exists(), f"Missing {pq_path}"
    assert audit_path.exists(), f"Missing {audit_path}"

    df = pd.read_parquet(pq_path)
    with open(audit_path, "r", encoding="utf-8") as fh:
        audit = json.load(fh)

    assert len(df) == 278_196
    assert int((df["ligand"] == "full_agonist").sum()) == 173_860
    assert int((df["ligand"] == "delta9_thc").sum()) == 104_336
    assert not df.isna().any().any(), "NaNs detected in processed WS1 dataframe!"

    # Verify exact nm -> Angstrom conversion
    for nm_col, ang_col in [
        ("r_tm6_nm", "r_tm6_angstrom"),
        ("d_toggle_nm", "d_toggle_angstrom"),
        ("d_npxxy_nm", "d_npxxy_angstrom"),
    ]:
        arr_nm = df[nm_col].to_numpy(dtype=np.float64)
        arr_ang = df[ang_col].to_numpy(dtype=np.float64)
        assert np.all(np.isfinite(arr_nm)) and np.all(np.isfinite(arr_ang))
        assert np.allclose(arr_ang, arr_nm * 10.0, rtol=0.0, atol=1e-12)

    assert audit["ensembles"]["full_agonist"]["shared_col0_max_abs_diff_nm"] == 0.0
    assert audit["ensembles"]["delta9_thc"]["shared_col0_max_abs_diff_nm"] == 0.0
    # Verify empirical finding: zero Delta9-THC frames reach R_TM6 >= 15.2 A
    assert audit["ensembles"]["delta9_thc"]["basin_populations"]["n_B_global_ge_15p2"] == 0


def test_ws1_seam_safety_and_buffered_macro_block_disjointness():
    df = pd.read_parquet(PROCESSED_DIR / "ws1_featurized_segments.parquet")
    X_nm = df[["r_tm6_nm", "d_toggle_nm", "d_npxxy_nm"]].to_numpy(dtype=np.float64)
    in_A = df["in_A_global"].to_numpy(dtype=bool)
    in_B = df["in_B_global"].to_numpy(dtype=bool)

    # 1. Verify exact step-jump bounds on every raw threshold segmentation
    for col, thresh in [
        ("seg_id_3cv_0p10nm", 0.10),
        ("seg_id_3cv_0p12nm", 0.12),
        ("seg_id_3cv_0p15nm", 0.15),
        ("seg_id_3cv_0p20nm", 0.20),
    ]:
        seg = df[col].to_numpy(dtype=np.int64)
        assert_contiguous_segments(seg)
        same_seg = seg[1:] == seg[:-1]
        max_within_step = float(np.max(np.abs(X_nm[1:][same_seg] - X_nm[:-1][same_seg])))
        assert max_within_step <= thresh + 1e-12

    # 2. Verify zero cross-seam forward, backward, and history pairs on primary segmentation
    seg_z5 = df["seg_id_z5p0"].to_numpy(dtype=np.int64)
    assert_contiguous_segments(seg_z5)
    s_idx, e_idx = build_forward_stopped_pairs(seg_z5, in_A, in_B, lag_steps=15)
    assert s_idx.size > 50_000
    assert_no_cross_seam_pairs(seg_z5, s_idx, e_idx)

    t0_idx, tend_idx, bstop_idx = build_backward_stopped_pairs(seg_z5, in_A, in_B, lag_steps=15)
    assert_no_cross_seam_pairs(seg_z5, t0_idx, tend_idx)
    assert_no_cross_seam_pairs(seg_z5, bstop_idx, tend_idx)

    _, v_hist = compute_lagged_displacements(X_nm, seg_z5, lag_steps=20)
    idx_h = np.flatnonzero(v_hist)
    assert_no_cross_seam_pairs(seg_z5, idx_h - 20, idx_h)

    # 3. Verify zero segment straddles two valid macro-blocks and buffered gaps hold
    macro_gid = df["macro_group_id"].to_numpy(dtype=np.int64)
    valid_mask = df["macro_valid_mask"].to_numpy(dtype=bool)

    # Every continuous z5 segment must map to AT MOST ONE valid macro_group_id >= 0
    # (3 ultra-long >2,700-frame segments have trailing frames trimmed into -1 buffer,
    #  never into another valid block).
    same_z5 = seg_z5[1:] == seg_z5[:-1]
    changed_mg_in_z5 = same_z5 & (macro_gid[1:] != macro_gid[:-1])
    assert np.all((macro_gid[:-1][changed_mg_in_z5] == -1) | (macro_gid[1:][changed_mg_in_z5] == -1)), (
        "A continuous z5 segment transitioned directly between two valid macro-blocks!"
    )
    valid_seg_groups = (
        pd.DataFrame({"seg": seg_z5[valid_mask], "mg": macro_gid[valid_mask]})
        .groupby("seg")["mg"]
        .nunique()
    )
    assert int(valid_seg_groups.max()) == 1, (
        "A continuous z5 segment was split across multiple valid macro-blocks!"
    )

    # Verify adjacent distinct valid macro-blocks inside each ensemble have >=250 buffer frames between them
    for lig in ("full_agonist", "delta9_thc"):
        sub = df[df["ligand"] == lig]
        mg = sub["macro_group_id"].to_numpy(dtype=np.int64)
        u_valid = sorted(g for g in np.unique(mg) if g >= 0)
        assert len(u_valid) >= 25
        for g1, g2 in zip(u_valid[:-1], u_valid[1:]):
            end_g1 = int(np.max(np.flatnonzero(mg == g1)))
            start_g2 = int(np.min(np.flatnonzero(mg == g2)))
            gap = start_g2 - end_g1 - 1
            assert gap >= 250, f"Buffer gap between {g1} and {g2} in {lig} too narrow: {gap}"

    # 4. Verify LineageGroupSplitter fold disjointness
    g_valid = macro_gid[valid_mask]
    X_valid = X_nm[valid_mask]
    splitter = LineageGroupSplitter(n_splits=5, stratified=False, random_state=42)
    for tr, te in splitter.split(X_valid, None, g_valid):
        assert_disjoint_group_split(g_valid, tr, te)


def test_ws1_oof_predictions_metrics_figures_and_report_match():
    oof_path = PROCESSED_DIR / "ws1_oof_predictions.parquet"
    metrics_path = RESULTS_DIR / "ws1_metrics.json"
    report_path = RESULTS_DIR / "ws1_report.md"

    assert oof_path.exists(), f"Missing {oof_path}"
    assert metrics_path.exists(), f"Missing {metrics_path}"
    assert report_path.exists(), f"Missing {report_path}"

    df_oof = pd.read_parquet(oof_path)
    with open(metrics_path, "r", encoding="utf-8") as fh:
        m = json.load(fh)
    report_text = report_path.read_text(encoding="utf-8")

    # OOF committor bounds & exact Dirichlet boundary conditions
    for col in [
        "q_dga_global_oof",
        "q_b1_global_oof",
        "q_pt_epi_global_oof",
        "q_gbr_global_oof",
        "q_dga_within_natural_oof",
        "q_pt_within_natural_oof",
    ]:
        vals = df_oof[col].to_numpy(dtype=np.float64)
        assert np.all(np.isfinite(vals)), f"Non-finite values in {col}"
        assert np.all((vals >= 0.0) & (vals <= 1.0)), f"Out-of-bounds [0,1] values in {col}"

    r_tm6 = df_oof["r_tm6_angstrom"].to_numpy()
    assert np.all(df_oof.loc[r_tm6 <= 11.8, "q_dga_global_oof"] == 0.0)
    assert np.all(df_oof.loc[r_tm6 >= 15.2, "q_dga_global_oof"] == 1.0)

    # Check N2b reversibility and bootstrap CI ordering in metrics JSON
    lag15 = m["regime_I_global_full_activation_11p8_to_15p2"]["dga_lag_sweep"]["lags"]["15"]
    assert lag15["passes_n2b_threshold_0p10"] is True
    assert lag15["mean_test_pbr_dga_3cv"] < lag15["mean_test_pbr_b1_1cv"]

    h1_glob = m["regime_I_global_full_activation_11p8_to_15p2"]["h1_benchmark"]["pt_epistasis_3cv"]
    bss_ci = h1_glob["bss_vs_b1"]
    assert bss_ci["ci_low"] <= bss_ci["point"] <= bss_ci["ci_high"]

    # Check all 4 publication figures exist and are non-trivial PNGs (>50 KB)
    for fname in [
        "fig1_ws1_committor_landscape_and_matched_slice.png",
        "fig2_ws1_operator_validation_and_h1_benchmark.png",
        "fig3_ws1_mori_zwanzig_memory_and_segmentation_forensics.png",
        "fig4_ws1_h3_ligand_bottleneck_rotation.png",
    ]:
        fpath = FIG_DIR / fname
        assert fpath.exists(), f"Missing figure {fpath}"
        assert fpath.stat().st_size > 50_000, f"Figure {fpath} unexpectedly small ({fpath.stat().st_size} B)"

    # Verify exact numerical and provenance consistency between ws1_metrics.json and ws1_report.md
    expected_strings = [
        "278,196",
        "173,860",
        "104,336",
        "ShuklaGroup/THC_binding",
        "10.1016/j.jbc.2022.101764",
        f"{bss_ci['point']:.4f}",
        f"{lag15['mean_test_pbr_dga_3cv']:.4f}",
        f"{lag15['mean_test_pbr_b1_1cv']:.4f}",
        f"{lag15['mean_n2b_reversibility_l2_err']:.4f}",
    ]
    for s in expected_strings:
        assert s in report_text, f"Expected exact string '{s}' from ws1_metrics.json to appear verbatim in ws1_report.md!"

    for forbidden in ["7689950", "Figure 1D", "Figure 1A–D", "Figure 2B–D", "Figure 3D", "Figure 3C,D"]:
        assert forbidden not in report_text, f"Stale/erroneous reference '{forbidden}' found in ws1_report.md!"


def test_ws1_two_stage_fold_isolation_and_stratification():
    """Verify StratifiedGroupKFold ligand balance and strict bitwise fold isolation."""
    df = pd.read_parquet(PROCESSED_DIR / "ws1_featurized_segments.parquet")
    valid_mask = df["macro_valid_mask"].to_numpy(dtype=bool)
    df_val = df.loc[valid_mask].reset_index(drop=True)
    orig_idx = np.flatnonzero(valid_mask)

    # Verify full 58-block StratifiedGroupKFold balances both ligands in every fold
    is_ago_full = (df_val["ligand"].to_numpy() == "full_agonist").astype(int)
    g_full = df_val["macro_group_id"].to_numpy(dtype=np.int64)
    X_full = df_val[FEATURE_NAMES].to_numpy(dtype=np.float64)
    sp_full = LineageGroupSplitter(n_splits=5, stratified=True, random_state=42)
    for tr_f, te_f in sp_full.split(X_full, is_ago_full, g_full):
        assert np.any(is_ago_full[te_f] == 1) and np.any(is_ago_full[te_f] == 0)
        assert np.any(is_ago_full[tr_f] == 1) and np.any(is_ago_full[tr_f] == 0)

    # Select 5 Full Agonist + 5 Delta9-THC macro-blocks (first 800 frames per block for fast CI)
    ago_blocks = sorted(df_val.loc[df_val["ligand"] == "full_agonist", "macro_group_id"].unique())[:5]
    thc_blocks = sorted(df_val.loc[df_val["ligand"] == "delta9_thc", "macro_group_id"].unique())[:5]
    picked_indices = []
    for blk in ago_blocks + thc_blocks:
        blk_idx = np.flatnonzero(g_full == blk)[:800]
        picked_indices.append(blk_idx)
    sub_pos = np.sort(np.concatenate(picked_indices))

    df_sub = df_val.iloc[sub_pos].reset_index(drop=True)
    sub_orig = orig_idx[sub_pos]
    X = df_sub[FEATURE_NAMES].to_numpy(dtype=np.float64)
    in_A = df_sub["in_A_global"].to_numpy(dtype=bool)
    in_B = df_sub["in_B_global"].to_numpy(dtype=bool)
    seg = _resegment_subset_indices(sub_orig, df_sub["seg_id_z5p0"].to_numpy(dtype=np.int64))
    groups = df_sub["macro_group_id"].to_numpy(dtype=np.int64)
    is_ago = (df_sub["ligand"].to_numpy() == "full_agonist").astype(int)

    bundle_clean = solve_fold_isolated_dga_bundle(
        X,
        in_A,
        in_B,
        seg,
        groups,
        stratify_labels=is_ago,
        candidate_lags=(5,),
        primary_lag=5,
        n_rbf=15,
        n_splits=5,
        random_state=42,
    )

    # 1. Verify every fold contains BOTH Full Agonist and Delta9-THC macro-blocks
    for tr, te in bundle_clean.folds:
        assert np.any(is_ago[te] == 1) and np.any(is_ago[te] == 0), "Unstratified test fold!"
        assert np.any(is_ago[tr] == 1) and np.any(is_ago[tr] == 0), "Unstratified train fold!"

    tr0, te0 = bundle_clean.folds[0]

    # 2. Corrupt interior coordinates of te0 -> fold-0 dga_tr_0 must remain bitwise identical!
    X_pert_te = X.copy()
    te0_int = te0[~(in_A[te0] | in_B[te0])]
    X_pert_te[te0_int] = X_pert_te[te0_int][:, ::-1] * 1.35 + 1.75
    bundle_pert_te = solve_fold_isolated_dga_bundle(
        X_pert_te,
        in_A,
        in_B,
        seg,
        groups,
        stratify_labels=is_ago,
        candidate_lags=(5,),
        primary_lag=5,
        n_rbf=15,
        n_splits=5,
        random_state=42,
    )
    assert np.array_equal(bundle_pert_te.fold_q_train[0], bundle_clean.fold_q_train[0]), (
        "Test-fold coordinates leaked into training-fold DGA targets!"
    )

    # 3. Corrupt other folds' training targets & test reference q_te_indep -> fold-0 OOF predictions
    #    from H1 (B0, B1, B2, PT, GBR) and H2 must depend ONLY on fold_q_train[0] and X[tr0]!
    _, preds_clean = evaluate_h1_benchmark_suite(
        X, bundle_clean, in_A, in_B, seg, groups, slice_range=(12.5, 14.5), n_perms=4, n_boot=30, random_state=42
    )
    poisoned_bundle = FoldIsolatedDGABundle(
        folds=bundle_clean.folds,
        fold_q_train=[
            bundle_clean.fold_q_train[0].copy(),
            *[1.0 - q.copy() for q in bundle_clean.fold_q_train[1:]],
        ],
        q_te_indep=1.0 - bundle_clean.q_te_indep.copy(),
        q_tr_oof_pred=bundle_clean.q_tr_oof_pred.copy(),
        lag_sweep=bundle_clean.lag_sweep,
    )
    _, preds_poison = evaluate_h1_benchmark_suite(
        X, poisoned_bundle, in_A, in_B, seg, groups, slice_range=(12.5, 14.5), n_perms=4, n_boot=30, random_state=42
    )
    for k in ("oof_b0", "oof_b1", "oof_b2", "oof_pt_lin", "oof_pt_epi", "oof_gbr"):
        assert np.array_equal(preds_poison[k][te0], preds_clean[k][te0]), (
            f"Two-stage cross-fold target leakage detected in {k} on fold 0!"
        )

    # Verify evaluate_memory_gain_with_null accepts fold_q_train without leaking q_target into training
    mem_res = evaluate_memory_gain_with_null(
        X_static=X,
        seg_id=seg,
        q_target=bundle_clean.q_te_indep,
        groups=groups,
        lag_steps=5,
        eval_mask=(X[:, 0] >= 12.5) & (X[:, 0] <= 14.5),
        in_A=in_A,
        in_B=in_B,
        include_interactions=True,
        n_shuffles=4,
        folds=bundle_clean.folds,
        fold_q_train=bundle_clean.fold_q_train,
    )
    assert np.isfinite(mem_res["delta_ell_mem"])

    # 4. Corrupt interior coordinates of tr0 -> independent test-fold dga_te_0 on te0 must be bitwise identical!
    X_pert_tr = X.copy()
    tr0_int = tr0[~(in_A[tr0] | in_B[tr0])]
    X_pert_tr[tr0_int] = X_pert_tr[tr0_int][:, ::-1] * 0.75 - 2.10
    bundle_pert_tr = solve_fold_isolated_dga_bundle(
        X_pert_tr,
        in_A,
        in_B,
        seg,
        groups,
        stratify_labels=is_ago,
        candidate_lags=(5,),
        primary_lag=5,
        n_rbf=15,
        n_splits=5,
        random_state=42,
    )
    assert np.array_equal(bundle_pert_tr.q_te_indep[te0], bundle_clean.q_te_indep[te0]), (
        "Training-fold coordinates leaked into independent test-fold dga_te_0!"
    )
