"""Comprehensive integrity, anti-leakage, and mathematical invariant test suite for WS2."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile

import mdtraj as md
import numpy as np
import pandas as pd
import pytest

from src.core import (
    LineageGroupSplitter,
    PetersTroutCommittorModel,
    SplineProgressBaselineB1,
    StoppedDGACommittor,
    assert_contiguous_segments,
    assert_disjoint_group_split,
    build_forward_stopped_pairs,
    compute_lagged_displacements,
)
from src.ws2_box_msm.stream_featurize import (
    CB1_BINDING_FOLDER_ID,
    DCD_FILENAME_REGEX,
    TIER_A_COLS,
    TIER_B_ADDED_COLS,
    TIER_C_ADDED_COLS,
    verify_cb1_strip_topology,
    verify_dcd_binary_header,
)

ROOT = Path("/Users/kavi/Documents/artico/cb1-dynamics")
BOX_DIR = ROOT / "data/inventory/box_2025"
PROC_DIR = ROOT / "data/processed/ws2"
RES_DIR = ROOT / "results/ws2"


def test_residue_identity_guard_passes_real_and_rejects_corrupted() -> None:
    ref = md.load(str(BOX_DIR / "CB1_assym-strip.pdb"), top=str(BOX_DIR / "CB1_assym-strip.prmtop"))
    # Must pass on authentic CB1_assym-strip topology (UniProt = resSeq + 88)
    verify_cb1_strip_topology(ref.topology)

    # Mutating F155^2.42 (resid 66, resSeq 67) or W356^6.48 (resid 267, resSeq 268) must immediately fail
    bad_top = copy.deepcopy(ref.topology)
    bad_top.residue(155 - 89).name = "ALA"
    with pytest.raises(ValueError, match="Residue identity guard FAILED at UniProt 155"):
        verify_cb1_strip_topology(bad_top)

    bad_lig_top = copy.deepcopy(ref.topology)
    bad_lig_top.residue(323).name = "POPC"
    with pytest.raises(ValueError, match="Expected LIG at resid 323"):
        verify_cb1_strip_topology(bad_lig_top)


def test_dcd_binary_header_guard() -> None:
    with tempfile.NamedTemporaryFile(suffix=".dcd", delete=False) as fh:
        valid_bytes = b"\x54\x00\x00\x00CORD" + b"\x00" * 64
        fh.write(valid_bytes)
        p_valid = Path(fh.name)
    try:
        verify_dcd_binary_header(p_valid, expected_bytes=len(valid_bytes))
        with pytest.raises(ValueError, match="size mismatch"):
            verify_dcd_binary_header(p_valid, expected_bytes=len(valid_bytes) + 4)
    finally:
        p_valid.unlink(missing_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".dcd", delete=False) as fh:
        fh.write(b"<html>Box error page</html>")
        p_bad = Path(fh.name)
    try:
        with pytest.raises(ValueError, match="Invalid DCD magic header"):
            verify_dcd_binary_header(p_bad)
    finally:
        p_bad.unlink(missing_ok=True)


def test_global_5000run_msm_and_tpt_invariants() -> None:
    df_micro = pd.read_parquet(PROC_DIR / "msm_800_microstates.parquet")
    summary = json.loads((PROC_DIR / "msm_tpt_summary.json").read_text())

    assert len(df_micro) == 800
    assert int(df_micro["is_active"].sum()) == 799
    assert int(df_micro["n_frames"].sum()) == 4_829_682

    inv = summary["markov_invariants"]
    assert inv["row_stochasticity_max_abs_err"] < 1e-12
    assert inv["detailed_balance_max_abs_err"] < 1e-10
    assert inv["stationary_pi_sum_abs_err"] < 1e-12

    tpt = summary["tpt_primary_phys"]
    assert tpt["harmonic_max_residual"] < 1e-10
    assert tpt["flux_conservation_rel_err"] < 1e-8
    assert tpt["mfpt_A_to_B_us"] > 0.0

    act = df_micro[df_micro["is_active"]]
    assert np.all((act["q_msm_phys"] >= 0.0) & (act["q_msm_phys"] <= 1.0))
    assert np.all(act.loc[act["in_A_phys"], "q_msm_phys"] == 0.0)
    assert np.all(act.loc[act["in_B_phys"], "q_msm_phys"] == 1.0)


def test_streamed_cohort_parquet_budget_cleanliness_and_no_leftover_dcd() -> None:
    pq_path = PROC_DIR / "box_aea_cohort_features.parquet"
    manifest = json.loads((PROC_DIR / "cohort_featurization_manifest.json").read_text())
    catalog = json.loads((BOX_DIR / "box_cb1_dcd_catalog.json").read_text())

    assert str(catalog["folder_id"]) == CB1_BINDING_FOLDER_ID
    assert pq_path.exists()
    pq_mb = pq_path.stat().st_size / (1024 * 1024)
    assert pq_mb < 20.0, f"Parquet size {pq_mb:.2f} MB exceeds <20 MB limit"

    # Zero leftover temporary .dcd files in box_2025/
    leftover_dcds = list(BOX_DIR.rglob("*.dcd"))
    assert len(leftover_dcds) == 0, f"Leftover temp .dcd files found: {leftover_dcds}"

    df = pd.read_parquet(pq_path)
    n_trajs = int(df["cohort_idx"].nunique())
    n_prs = int(df["pr_round"].nunique())
    assert 150 <= n_trajs <= 200, f"Expected 150-200 streamed trajectories, got {n_trajs}"
    assert 120_000 <= len(df) <= 190_000, f"Expected 120k-190k frames, got {len(df)}"
    assert n_prs >= 15, f"Expected >=15 distinct pr_<round> lineages, got {n_prs}"

    # Verify zero macrostate_rep_dcd filenames
    for name in df["dcd_name"].unique():
        assert DCD_FILENAME_REGEX.match(str(name)), f"Invalid continuous DCD filename: {name}"
        assert "macrostate_rep" not in str(name)

    all_feat_cols = TIER_A_COLS + TIER_B_ADDED_COLS + TIER_C_ADDED_COLS
    assert len(all_feat_cols) == 62
    for c in all_feat_cols:
        assert c in df.columns, f"Missing feature column {c}"
        assert int(df[c].isna().sum()) == 0, f"NaNs found in {c}"
        assert np.all(np.isfinite(df[c].to_numpy())), f"Non-finite values found in {c}"
    assert manifest["total_feature_nan_count"] == 0


def test_zero_leakage_lineage_splits_and_no_cross_seam_pairs() -> None:
    df = pd.read_parquet(PROC_DIR / "box_aea_cohort_features.parquet")
    seg_id = df["cohort_idx"].to_numpy(dtype=np.int64)
    groups = df["lineage_group_id"].to_numpy(dtype=np.int64)
    pr_rounds = df["pr_round"].to_numpy(dtype=np.int64)
    y_strata = np.asarray(pd.Categorical(df["stratum"]).codes, dtype=np.int64)

    assert_contiguous_segments(seg_id)

    splitter = LineageGroupSplitter(n_splits=5, stratified=True, random_state=42)
    for tr, te in splitter.split(df[["r_portal_A"]].to_numpy(), y_strata, groups):
        assert_disjoint_group_split(groups, tr, te)
        assert_disjoint_group_split(pr_rounds, tr, te)
        assert_disjoint_group_split(seg_id, tr, te)

        # Verify finite-lag displacements and stopped pairs on train & test never cross trajectory boundaries
        dX_te, valid_te = compute_lagged_displacements(df.iloc[te][["r_portal_A"]].to_numpy(), seg_id[te], lag_steps=50)
        assert dX_te.shape[0] == len(te)
        assert np.any(valid_te)

        r_te = df.iloc[te]["r_portal_A"].to_numpy()
        s0, s1 = build_forward_stopped_pairs(seg_id[te], r_te >= 22.0, r_te <= 8.2, lag_steps=100)
        assert len(s0) > 0
        assert np.all(seg_id[te][s0] == seg_id[te][s1])


def test_strict_train_only_scaler_and_dga_isolation() -> None:
    rng = np.random.default_rng(123)
    X_tr = rng.normal(12.0, 2.5, size=(400, 4))
    seg_tr = np.repeat(np.arange(4, dtype=np.int64), 100)
    in_A_tr = X_tr[:, 0] >= 16.0
    in_B_tr = X_tr[:, 0] <= 8.0
    q_tr = np.clip((16.0 - X_tr[:, 0]) / 8.0, 0.0, 1.0)

    X_eval = rng.normal(12.0, 1.5, size=(50, 4))
    X_te_corrupted = X_eval + 5000.0  # Massive distribution shift on unseen test data

    b1 = SplineProgressBaselineB1(progress_col=0).fit(X_tr, q_tr)
    pt = PetersTroutCommittorModel(include_interactions=True).fit(X_tr, q_tr)
    dga = StoppedDGACommittor(lag_steps=5, n_rbf=15, random_state=42).fit(X_tr, in_A_tr, in_B_tr, seg_id=seg_tr)

    pred_b1_before = b1.predict(X_eval)
    pred_pt_before = pt.predict(X_eval)
    pred_dga_before = dga.predict(X_eval)

    # Predicting on wildly shifted X_te_corrupted must never mutate internal train-fitted state
    _ = b1.predict(X_te_corrupted)
    _ = pt.predict(X_te_corrupted)
    _ = dga.predict(X_te_corrupted)

    np.testing.assert_allclose(pred_b1_before, b1.predict(X_eval), rtol=0, atol=0)
    np.testing.assert_allclose(pred_pt_before, pt.predict(X_eval), rtol=0, atol=0)
    np.testing.assert_allclose(pred_dga_before, dga.predict(X_eval), rtol=0, atol=0)


def test_two_stage_dga_zero_cross_fold_leakage() -> None:
    """Verify two-stage DGA -> H1/H2 training targets for fold j have zero dependency on fold j test data."""
    from src.ws2_box_msm.run_ws2_analysis import _fit_fold_dga_pca_pipeline

    rng = np.random.default_rng(7)
    n_segs = 10
    T_seg = 120
    X_all = rng.normal(13.0, 3.2, size=(n_segs * T_seg, 8))
    seg_id = np.repeat(np.arange(n_segs, dtype=np.int64), T_seg)
    groups = seg_id // 2
    in_A = X_all[:, 0] >= 17.5
    in_B = X_all[:, 0] <= 8.5

    splitter = LineageGroupSplitter(n_splits=5, stratified=False, random_state=42)
    tr0, te0 = next(iter(splitter.split(X_all, None, groups)))

    sc_clean, pca_clean, dga_clean, _ = _fit_fold_dga_pca_pipeline(
        X_all[tr0], in_A[tr0], in_B[tr0], seg_id[tr0], lag_steps=5, n_pca=4, n_rbf=20, random_state=42
    )
    q_tr_clean = dga_clean.predict(pca_clean.transform(sc_clean.transform(X_all[tr0])), in_A=in_A[tr0], in_B=in_B[tr0])
    pt_clean = PetersTroutCommittorModel(include_interactions=False).fit(
        X_all[tr0][~(in_A[tr0] | in_B[tr0])], q_tr_clean[~(in_A[tr0] | in_B[tr0])]
    )

    # Massively corrupt held-out fold 0 test coordinates X_all[te0] and re-run fold-0 training pipeline
    X_pert = X_all.copy()
    X_pert[te0] = -9999.0 + rng.normal(0.0, 500.0, size=X_pert[te0].shape)
    in_A_pert = X_pert[:, 0] >= 17.5
    in_B_pert = X_pert[:, 0] <= 8.5

    sc_pert, pca_pert, dga_pert, _ = _fit_fold_dga_pca_pipeline(
        X_pert[tr0], in_A_pert[tr0], in_B_pert[tr0], seg_id[tr0], lag_steps=5, n_pca=4, n_rbf=20, random_state=42
    )
    q_tr_pert = dga_pert.predict(
        pca_pert.transform(sc_pert.transform(X_pert[tr0])), in_A=in_A_pert[tr0], in_B=in_B_pert[tr0]
    )
    pt_pert = PetersTroutCommittorModel(include_interactions=False).fit(
        X_pert[tr0][~(in_A_pert[tr0] | in_B_pert[tr0])], q_tr_pert[~(in_A_pert[tr0] | in_B_pert[tr0])]
    )

    np.testing.assert_allclose(q_tr_clean, q_tr_pert, rtol=0, atol=0)
    np.testing.assert_allclose(pt_clean.coef_, pt_pert.coef_, rtol=0, atol=0)
    assert pt_clean.intercept_ == pt_pert.intercept_


def test_pbr_shared_witness_subspace_and_monotonic_dga_boundary() -> None:
    """Verify target-free E_PBR is evaluated strictly on a single shared witness subspace for DGA-boundary models."""
    metrics = json.loads((RES_DIR / "ws2_metrics.json").read_text())
    h1_msm = metrics["H1_matched_portal_slice"]["q_msm_phys"]["models"]
    h1_dga = metrics["H1_matched_portal_slice"]["q_dga_oof"]["models"]
    dga_pbr_20 = metrics["continuous_stopped_dga_validation"]["lags"]["20.0ns"]["oof_pbr_mean"]

    # Discrete global MSM does not satisfy pure-distance r_portal <= 8.2 A = 1.0, so PBR is omitted there
    for m_name, m_val in h1_msm.items():
        assert m_val["oof_pbr_mean"] is None, f"Unexpected PBR on mismatched boundary for q_msm_phys/{m_name}"

    pbr_b1 = h1_dga["B1_spline_1d"]["oof_pbr_mean"]
    pbr_sw = h1_dga["H1_Switches_full_epistasis"]["oof_pbr_mean"]
    pbr_ta = h1_dga["TierA_CA_linear"]["oof_pbr_mean"]
    pbr_tc = h1_dga["TierC_Full_lipid_AEA"]["oof_pbr_mean"]
    pbr_or = h1_dga["Tier3_Oracle_HistGBR"]["oof_pbr_mean"]

    for v in (pbr_b1, pbr_sw, pbr_ta, pbr_tc, dga_pbr_20, pbr_or):
        assert v is not None and 0.0 < float(v) < 0.05

    # Monotonic reduction across hierarchy on the shared 10D witness subspace
    assert pbr_or < dga_pbr_20 < pbr_tc < pbr_ta < pbr_sw < pbr_b1, (
        f"Non-monotonic shared-witness PBR chain: Oracle={pbr_or}, DGA={dga_pbr_20}, "
        f"TierC={pbr_tc}, TierA={pbr_ta}, Switches={pbr_sw}, B1={pbr_b1}"
    )


def test_no_spurious_in_B_dga_clamping_on_q_msm_phys() -> None:
    """Ensure held-out q_msm_phys predictions are NEVER clamped to 1.0 on r_portal <= 8.2 A vestibule frames."""
    df_oof = pd.read_parquet(PROC_DIR / "ws2_oof_predictions.parquet")
    vest_lowR = (df_oof["r_portal_A"] <= 8.2) & (df_oof["macrostate_id"] != 5)
    assert int(vest_lowR.sum()) >= 1000, f"Expected >=1000 vestibule frames at r_portal <= 8.2 A, got {vest_lowR.sum()}"

    for col in (
        "q_msm_phys__B1_spline_1d",
        "q_msm_phys__H1_Switches_full_epistasis",
        "q_msm_phys__TierC_Full_lipid_AEA",
    ):
        vals = df_oof.loc[vest_lowR, col].to_numpy(dtype=np.float64)
        assert int(np.sum(vals == 1.0)) == 0, f"Found {np.sum(vals == 1.0)} spuriously clamped 1.0 frames in {col}"

    true_vest_mean = float(df_oof.loc[vest_lowR, "q_msm_phys"].mean())
    pred_tc_vest_mean = float(df_oof.loc[vest_lowR, "q_msm_phys__TierC_Full_lipid_AEA"].mean())
    pred_b1_vest_mean = float(df_oof.loc[vest_lowR, "q_msm_phys__B1_spline_1d"].mean())
    assert abs(pred_tc_vest_mean - true_vest_mean) < 0.20, (
        f"Vestibule r<=8.2 TierC mean {pred_tc_vest_mean:.4f} too far from true {true_vest_mean:.4f}"
    )
    assert pred_tc_vest_mean < pred_b1_vest_mean - 0.25, (
        f"Expected TierC ({pred_tc_vest_mean:.4f}) to substantially correct 1D B1 overprediction ({pred_b1_vest_mean:.4f})"
    )

    # Conversely, DGA-boundary models MUST clamp in_B_dga to 1.0
    assert np.all(df_oof.loc[df_oof["in_B_dga"], "q_dga_oof__TierC_Full_lipid_AEA"].to_numpy() == 1.0)


def test_leave_test_lineage_out_5000run_msm_invariance() -> None:
    """Verify fold-isolated 5,000-run MSM refit (excluding test lineages & pr_rounds) matches global q_msm_phys."""
    metrics = json.loads((RES_DIR / "ws2_metrics.json").read_text())
    audit = metrics["leave_test_lineage_out_msm_audit"]

    assert len(audit["per_fold"]) == 5
    for f in audit["per_fold"]:
        assert f["n_excluded_cohort_trajs"] >= 20
        assert f["n_excluded_5000run_dtrajs_round_level"] >= 100
        assert f["harmonic_max_residual_round_level"] < 1e-10

    assert audit["round_and_lineage_exclusion"]["pearson_r_vs_global_msm"] > 0.998
    assert audit["round_and_lineage_exclusion"]["mae_vs_global_msm"] < 0.01
    assert audit["trajectory_lineage_only_exclusion"]["pearson_r_vs_global_msm"] > 0.9995
    assert audit["trajectory_lineage_only_exclusion"]["mae_vs_global_msm"] < 0.005

    dm = audit["downstream_matched_portal_skill_on_leave_out_msm"]
    glob_tc_bss = metrics["H1_matched_portal_slice"]["q_msm_phys"]["models"]["TierC_Full_lipid_AEA"]["bss_vs_b1"]
    assert dm["TierC_Full_lipid_AEA"]["bss_vs_b1"] > 0.75
    assert abs(dm["TierC_Full_lipid_AEA"]["bss_vs_b1"] - glob_tc_bss) < 0.03


def test_ws2_metrics_figures_and_report_consistency() -> None:
    metrics_path = RES_DIR / "ws2_metrics.json"
    report_path = RES_DIR / "ws2_report.md"
    fig_dir = RES_DIR / "figures"

    assert metrics_path.exists()
    assert report_path.exists()
    metrics = json.loads(metrics_path.read_text())
    report_text = report_path.read_text()

    expected_figs = [
        "fig1_global_msm_tpt_landscape.png",
        "fig2_h1_matched_slice_gating.png",
        "fig3_h2_mori_zwanzig_memory_collapse.png",
        "fig4_lipid_portal_biophysical_mechanism.png",
    ]
    for fn in expected_figs:
        fp = fig_dir / fn
        assert fp.exists() and fp.stat().st_size > 20_000, f"Missing or empty figure {fp}"

    # Verify all 5 Mori-Zwanzig lags are reported without cherry-picking
    for lk in ("0.5ns", "1.0ns", "2.0ns", "5.0ns", "10.0ns"):
        assert lk in metrics["H2_mori_zwanzig_memory_collapse"]["q_msm_phys"]["collapse_summary_by_lag"]
        assert lk in metrics["H2_mori_zwanzig_memory_collapse"]["q_dga_oof"]["collapse_summary_by_lag"]

    # Verify key numeric anchors and verified DOIs appear identically in ws2_report.md
    mfpt_str = f"{metrics['global_5000run_msm_tpt']['tpt_primary_phys']['mfpt_A_to_B_us']:.2f}"
    assert mfpt_str in report_text, f"Expected exact MFPT {mfpt_str} in ws2_report.md"
    assert "4,829,682" in report_text
    assert "10.1016/j.jbc.2026.111434" in report_text
    assert "10.1101/2024.10.25.620304" in report_text
    assert "acschemneuro" not in report_text
    assert "638909" not in report_text
