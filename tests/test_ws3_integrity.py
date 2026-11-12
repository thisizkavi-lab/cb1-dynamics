"""Comprehensive Integrity, Zero-Leakage, and Provenance Tests for Workstream 3 (WS3).

Tests:
  1. 16-PDB quantitative benchmark completeness, +122 flavodoxin offset on 5XR8/7V3Z, Chain D on 8GHV,
     stubbed F237 handling on 8WU1, and all three structural dichotomy assertions.
  2. All 6 GPCRmd topologies pass canonical P21554 CB1 residue guards (isolating CB1 from Gs in 2067),
     and temporary raw .xtc directory `data/inventory/gpcrmd_temp/` is completely clean.
  3. `gpcrmd_open6_features.parquet` (<10 MB, 20,501 frames, 8.90 us actual / 8.97 us nominal) has zero
     NaNs/Infs, valid PBC distances across ICL3 TER splits, exact water-wire conductance encoding, and
     distinct Na+ vs K+ ion sentinel handling.
  4. Strict zero-leakage buffered macro-block splitting (>= 2*tau_max temporal gap), adversarial fold
     poisoning test on X[te] and y[te] (verifying bitwise-identical train scalers, Peters-Trout weights,
     Stopped-DGA operators, and shared-witness interior PCA), shared-witness PBR schema, and <5%
     right-censoring enforcement across all evaluated local committor cohorts.
  5. Presence of all WS3 deliverables (`ws3_metrics.json`, `ws3_report.md`, 3 publication PNGs),
     verified literature citations, and exact numerical consistency between `ws3_metrics.json` and
     `ws3_report.md`.
"""

from __future__ import annotations

import json
from pathlib import Path

import mdtraj as md
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.core import (
    LineageGroupSplitter,
    PetersTroutCommittorModel,
    StoppedDGACommittor,
    compute_lagged_displacements,
    evaluate_mori_zwanzig_memory,
)
from src.ws3_gpcrmd_water.gpcrmd_featurize import GPCRMD_SPECS, build_topology_index
from src.ws3_gpcrmd_water.ws3_analysis import (
    audit_and_select_local_targets,
    build_buffered_macro_segments,
    evaluate_static_and_memory_suite,
    prepare_uniform_dt_cohort,
)


PROJECT_ROOT = Path("/Users/kavi/Documents/artico/cb1-dynamics")
PDB16_PARQUET = PROJECT_ROOT / "data" / "processed" / "ws3" / "pdb16_structural_benchmark.parquet"
GPCRMD_PARQUET = PROJECT_ROOT / "data" / "processed" / "ws3" / "gpcrmd_open6_features.parquet"
PROVENANCE_JSON = PROJECT_ROOT / "data" / "processed" / "ws3" / "gpcrmd_open6_provenance.json"
TOPO_DIR = PROJECT_ROOT / "data" / "processed" / "ws3" / "topologies"
TEMP_XTC_DIR = PROJECT_ROOT / "data" / "inventory" / "gpcrmd_temp"
RESULTS_DIR = PROJECT_ROOT / "results" / "ws3"
FIG_DIR = RESULTS_DIR / "figures"
METRICS_JSON = RESULTS_DIR / "ws3_metrics.json"
REPORT_MD = RESULTS_DIR / "ws3_report.md"


def test_pdb16_benchmark_integrity() -> None:
    """Verify 16-PDB structural benchmark accuracy, +122 offset, Chain D 8GHV, and dichotomy claims."""
    assert PDB16_PARQUET.exists(), f"Missing {PDB16_PARQUET}"
    df = pd.read_parquet(PDB16_PARQUET)
    expected_ids = {
        "5TGZ", "5U09", "9BA0", "6KQI", "7FEE",
        "5XRA", "5XR8", "7V3Z",
        "6N4B", "6KPG", "7WV9", "8GHV", "8IKG", "9B54",
        "8WU1", "8WRZ",
    }
    assert len(df) == 16
    assert set(df["pdb_id"].tolist()) == expected_ids

    by_id = df.set_index("pdb_id")

    # +122 flavodoxin offset checks on 5XR8 and 7V3Z
    assert by_id.loc["5XR8", "tm6_h8_auth_offset"] == 122
    assert by_id.loc["7V3Z", "tm6_h8_auth_offset"] == 122
    assert abs(float(by_id.loc["5XR8", "S173_S383_CA_A"]) - 11.99) < 0.05
    assert abs(float(by_id.loc["7V3Z", "S173_S383_CA_A"]) - 12.32) < 0.05

    # Chain D on 8GHV check
    assert by_id.loc["8GHV", "cb1_chain"] == "D"
    assert abs(float(by_id.loc["8GHV", "R214_D338_CA_A"]) - 15.53) < 0.05

    # Only 8WU1 has stubbed F237 sidechain; zero unexpected NaNs elsewhere
    assert bool(by_id.loc["8WU1", "F237_sidechain_resolved"]) is False
    assert np.isnan(by_id.loc["8WU1", "F237_chi1_deg"])
    non_clr_cols = [
        c for c in df.columns
        if c not in ("F237_chi1_deg", "F237_chi2_deg", "F237_F155_ring_min_A", "F155_F237_ring_min_A", "min_clr_tm2_tm4_A")
    ]
    assert not df[non_clr_cols].isna().any().any()

    # Dichotomy 1: Open-TM6 transducer-free agonist crystals vs transducer-coupled complexes
    no_trans = df[df["state_class"] == "Active_like_no_transducer"]
    coupled = df[df["state_class"].isin(["Active_Gi", "Active_BetaArrestin1"])]
    assert (no_trans["R214_D338_CA_A"] > 14.0).all()
    assert (no_trans["F155sc_N393CA_A"] < 13.0).all()
    assert (no_trans["F155_chi1_deg"] > 50.0).all()
    assert (no_trans["Y294_Y397_OH_A"] > 5.0).all()

    assert (coupled["R214_D338_CA_A"] > 14.3).all()
    assert (coupled["F155sc_N393CA_A"] > 14.5).all()
    assert (coupled["F155_chi1_deg"] < -50.0).all()
    assert (coupled["Y294_Y397_OH_A"] < 4.3).all()

    # Dichotomy 2: Matched-ligand FUB Gi1 (6N4B) vs beta-arrestin-1 (8WU1)
    assert abs(float(by_id.loc["6N4B", "W356_chi1_deg"]) - (-58.95)) < 0.5
    assert abs(float(by_id.loc["8WU1", "W356_chi1_deg"]) - (-177.42)) < 0.5
    assert float(by_id.loc["8WU1", "D163_W356_CA_A"]) < float(by_id.loc["6N4B", "D163_W356_CA_A"]) - 1.5

    # Dichotomy 3: NAM ORG27569 (6KQI) contracts orthosteric lid while locking TM6
    assert float(by_id.loc["6KQI", "S173_F268_CA_A"]) < 13.5
    assert float(by_id.loc["6KQI", "R214_D338_CA_A"]) < 7.5


def test_gpcrmd_topologies_and_temp_cleanup() -> None:
    """Verify all 6 GPCRmd topologies pass CB1 residue guards and temp .xtc directory is empty."""
    for gid, spec in GPCRMD_SPECS.items():
        pdb_path = TOPO_DIR / f"gpcrmd_{gid}.pdb"
        assert pdb_path.exists(), f"Missing topology {pdb_path}"
        top = md.load_pdb(str(pdb_path)).topology
        t_idx = build_topology_index(top, spec["cb1_chains"], gid)
        assert t_idx["rmap_pos210"] == "THR"
        assert t_idx["wat_O"].size > 10_000
        assert t_idx["popc_heavy"].size > 5_000
        if gid == 1201:
            assert t_idx["na_ions"].size == 0 and t_idx["k_ions"].size > 50
        else:
            assert t_idx["na_ions"].size > 20 and t_idx["k_ions"].size == 0

    if TEMP_XTC_DIR.exists():
        leftover = [p.name for p in TEMP_XTC_DIR.iterdir() if p.is_file()]
        assert leftover == [], f"Leftover temporary files in {TEMP_XTC_DIR}: {leftover}"


def test_gpcrmd_open6_features_integrity() -> None:
    """Verify gpcrmd_open6_features.parquet size (<10 MB), frame count (20,501), zero NaNs, and physics."""
    assert GPCRMD_PARQUET.exists(), f"Missing {GPCRMD_PARQUET}"
    size_mb = GPCRMD_PARQUET.stat().st_size / 1e6
    assert size_mb < 10.0, f"Parquet exceeds 10 MB limit: {size_mb:.2f} MB"

    df = pd.read_parquet(GPCRMD_PARQUET)
    assert len(df) == 20_501, f"Expected 20,501 total frames, got {len(df)}"
    assert set(df["gpcrmd_id"].unique().tolist()) == {1201, 2067, 22, 164, 162, 163}
    assert not df.isna().any().any(), f"Found NaNs in columns: {df.columns[df.isna().any()].tolist()}"

    num_cols = df.select_dtypes(include=[np.number]).columns
    assert np.all(np.isfinite(df[num_cols].to_numpy())), "Found non-finite values in numeric columns"

    # Monotonicity and PBC distance sanity per trajectory
    expected_counts = {1201: 5001, 2067: 6000, 22: 2500, 164: 2500, 162: 2000, 163: 2500}
    for gid, exp_n in expected_counts.items():
        sub = df[df["gpcrmd_id"] == gid]
        assert len(sub) == exp_n, f"GPCRmd {gid}: expected {exp_n} frames, got {len(sub)}"
        assert np.all(np.diff(sub["frame_idx"].to_numpy()) == 1)
        assert np.all(np.diff(sub["time_ns"].to_numpy()) > 0.0)
        assert sub["R214_D338_CA_A"].min() > 4.5 and sub["R214_D338_CA_A"].max() < 24.0

    # Water-wire conductance encoding invariants
    for prefix in ("wire_D163_W356NE1", "wire_D163_W356", "wire_D163_N393Y397", "wire_Y294_Y397"):
        hops = df[f"{prefix}_hops"].to_numpy()
        cond = df[f"{prefix}_cond"].to_numpy()
        assert np.all((cond >= 0.0) & (cond <= 1.0))
        assert np.array_equal(hops == -1, cond == 0.0)

    # Distinct Na+ vs K+ channel encoding
    df_1201 = df[df["gpcrmd_id"] == 1201]
    df_nacl = df[df["gpcrmd_id"] != 1201]
    assert (~df_1201["has_na_ions"]).all() and df_1201["has_k_ions"].all()
    assert (df_1201["min_Na_D163_A"] == 99.0).all() and (df_1201["min_K_D163_A"] < 90.0).all()
    assert df_nacl["has_na_ions"].all() and (~df_nacl["has_k_ions"]).all()
    assert (df_nacl["min_Na_D163_A"] < 90.0).all() and (df_nacl["min_K_D163_A"] == 99.0).all()


def test_zero_leakage_splits_and_censoring_guards() -> None:
    """Verify buffered macro-block gaps, adversarial X/y fold-poisoning isolation, shared PBR, and <5% censoring."""
    df = pd.read_parquet(GPCRMD_PARQUET)

    # 1. Verify >= 32.0 ns (> 2*15 ns) physical gap on 0.2-ns cohort and >= 52.0 ns (> 2*25 ns) on 1.0-ns cohort
    c_raw = prepare_uniform_dt_cohort(df, [1201, 22, 164, 162, 163], target_dt_ns=0.2)
    c_buf, seg_id, grp_id = build_buffered_macro_segments(
        c_raw, block_duration_ns=80.0, buffer_duration_ns=32.0
    )
    for gid, sub in c_buf.groupby("gpcrmd_id"):
        blocks = [(g, gdf["time_ns"].min(), gdf["time_ns"].max()) for g, gdf in sub.groupby("macro_group_id")]
        blocks.sort(key=lambda t: t[1])
        for i in range(len(blocks) - 1):
            gap_ns = blocks[i + 1][1] - blocks[i][2]
            assert gap_ns >= 32.0 - 1e-6, f"Trajectory {gid} block gap {gap_ns:.2f} ns < 32.0 ns!"

    c_nacl_raw = prepare_uniform_dt_cohort(df, [2067, 22, 164, 162, 163], target_dt_ns=1.0)
    c_nacl_buf, seg_n, grp_n = build_buffered_macro_segments(
        c_nacl_raw, block_duration_ns=140.0, buffer_duration_ns=52.0
    )
    for gid, sub in c_nacl_buf.groupby("gpcrmd_id"):
        blocks = [(g, gdf["time_ns"].min(), gdf["time_ns"].max()) for g, gdf in sub.groupby("macro_group_id")]
        blocks.sort(key=lambda t: t[1])
        for i in range(len(blocks) - 1):
            gap_ns = blocks[i + 1][1] - blocks[i][2]
            assert gap_ns >= 52.0 - 1e-6, f"Trajectory {gid} NaCl block gap {gap_ns:.2f} ns < 52.0 ns!"

    # 2. Verify zero cross-seam displacement pairs
    feat_cols = ["F155_F237_ring_min_A", "R214_D338_CA_A", "Y294_Y397_OH_A", "n_wat_core_pore"]
    X_sub = c_buf[feat_cols].to_numpy(dtype=np.float64)
    _, valid_lag = compute_lagged_displacements(X_sub, seg_id, lag_steps=25)
    valid_idx = np.flatnonzero(valid_lag)
    assert np.array_equal(seg_id[valid_idx], seg_id[valid_idx - 25])

    # 3. Adversarial fold-poisoning test (corrupting BOTH X[te] and y[te] causes ZERO change in tr fits)
    splitter = LineageGroupSplitter(n_splits=5, random_state=42)
    tr, te = next(splitter.split(X_sub, None, grp_id))
    assert set(grp_id[tr].tolist()).isdisjoint(set(grp_id[te].tolist()))
    in_A = X_sub[:, 0] <= 4.2
    in_B = X_sub[:, 0] >= 5.8
    y_clean = (X_sub[:, 0] > 5.0).astype(float)

    X_poison = X_sub.copy()
    y_poison = y_clean.copy()
    X_poison[te] = X_poison[te] * -999.0 + 1e6
    y_poison[te] = 1.0 - y_poison[te]

    m_clean = PetersTroutCommittorModel(include_interactions=False, l2_reg=5e-3).fit(X_sub[tr], y_clean[tr])
    m_poison = PetersTroutCommittorModel(include_interactions=False, l2_reg=5e-3).fit(X_poison[tr], y_poison[tr])
    assert np.array_equal(m_clean.scaler_.mean_, m_poison.scaler_.mean_)
    assert np.array_equal(m_clean.coef_, m_poison.coef_)

    dga_clean = StoppedDGACommittor(lag_steps=10, n_rbf=25, include_poly2=False, reg=1e-3).fit(
        X_sub[tr], in_A[tr], in_B[tr], seg_id=seg_id[tr]
    )
    dga_poison = StoppedDGACommittor(lag_steps=10, n_rbf=25, include_poly2=False, reg=1e-3).fit(
        X_poison[tr], in_A[tr], in_B[tr], seg_id=seg_id[tr]
    )
    assert np.array_equal(dga_clean.coef_plus_, dga_poison.coef_plus_)

    int_tr = ~(in_A[tr] | in_B[tr])
    sc1 = StandardScaler().fit(X_sub[tr][int_tr])
    pca1 = PCA(n_components=3, random_state=42).fit(sc1.transform(X_sub[tr][int_tr]))
    sc2 = StandardScaler().fit(X_poison[tr][int_tr])
    pca2 = PCA(n_components=3, random_state=42).fit(sc2.transform(X_poison[tr][int_tr]))
    assert np.array_equal(pca1.components_, pca2.components_)

    # 4. Adversarial non-unc_mask target poisoning test for Mori-Zwanzig memory evaluation:
    # Poisoning y_full[~unc_mask] with -999.0, +999.0, or NaN must leave evaluate_mori_zwanzig_memory
    # (and evaluate_static_and_memory_suite's memory profile) 100% bitwise identical!
    _, best_nacl, _ = audit_and_select_local_targets(c_nacl_buf, seg_n, grp_n, tau_max_ns=25.0)
    assert best_nacl is not None
    raw_nacl = best_nacl["_raw"]
    in_A_n = raw_nacl["in_A"]
    in_B_n = raw_nacl["in_B"]
    unc_n = raw_nacl["uncens_mask"]
    X_n = c_nacl_buf[["F200_W356_ring_min_A", "R214_D338_CA_A", "n_wat_core_pore", "wire_D163_W356_cond"]].to_numpy(dtype=np.float64)

    in_A_mem_fit = in_A_n | ((~in_B_n) & (~unc_n))
    assert np.array_equal(~(in_A_mem_fit | in_B_n), unc_n)
    unc_n_idx = np.flatnonzero(unc_n)
    g_u_n = grp_n[unc_n_idx]
    mem_folds_n = []
    for _, te_u in LineageGroupSplitter(n_splits=5, stratified=False, random_state=42).split(X_n[unc_n_idx], None, g_u_n):
        te_mask = np.isin(grp_n, np.unique(g_u_n[te_u]))
        mem_folds_n.append((np.flatnonzero(~te_mask), np.flatnonzero(te_mask)))

    y_base_n = np.where(in_B_n, 1.0, np.where(in_A_n, 0.0, 0.5))
    y_base_n[unc_n] = raw_nacl["y_hit"][unc_n]
    y_neg999 = y_base_n.copy()
    y_neg999[~unc_n] = -999.0
    y_pos999 = y_base_n.copy()
    y_pos999[~unc_n] = +999.0
    y_nan_guard = y_base_n.copy()
    y_nan_guard[~unc_n] = np.nan

    ref_mem = evaluate_mori_zwanzig_memory(
        X_static=X_n, seg_id=seg_n, q_target=y_base_n, groups=grp_n, lag_steps=2,
        eval_mask=unc_n, in_A=in_A_mem_fit, in_B=in_B_n, folds=mem_folds_n,
        fit_interior_only=True, include_interactions=False, l2_reg=5e-3, n_splits=5, n_shuffles=10, random_state=42,
    )
    for y_adv in (y_neg999, y_pos999, y_nan_guard):
        adv_mem = evaluate_mori_zwanzig_memory(
            X_static=X_n, seg_id=seg_n, q_target=y_adv, groups=grp_n, lag_steps=2,
            eval_mask=unc_n, in_A=in_A_mem_fit, in_B=in_B_n, folds=mem_folds_n,
            fit_interior_only=True, include_interactions=False, l2_reg=5e-3, n_splits=5, n_shuffles=10, random_state=42,
        )
        assert ref_mem["delta_ell_mem"] == adv_mem["delta_ell_mem"]
        assert ref_mem["bss_mem"] == adv_mem["bss_mem"]
        assert ref_mem["n2a_null_q95"] == adv_mem["n2a_null_q95"]
        assert ref_mem["p_value_vs_n2a"] == adv_mem["p_value_vs_n2a"]

    # End-to-end suite check with poisoned raw["y_hit"][~unc_n] = -999.0 vs +999.0
    bundle_neg = {**best_nacl, "_raw": {**raw_nacl, "y_hit": np.where(unc_n, raw_nacl["y_hit"], -999.0)}}
    bundle_pos = {**best_nacl, "_raw": {**raw_nacl, "y_hit": np.where(unc_n, raw_nacl["y_hit"], +999.0)}}
    suite_neg = evaluate_static_and_memory_suite(c_nacl_buf, seg_n, grp_n, bundle_neg, True, [1, 2], n_splits=5, n_shuffles=5, n_perms=5)
    suite_pos = evaluate_static_and_memory_suite(c_nacl_buf, seg_n, grp_n, bundle_pos, True, [1, 2], n_splits=5, n_shuffles=5, n_perms=5)
    assert suite_neg["mori_zwanzig_memory_profile"] == suite_pos["mori_zwanzig_memory_profile"]

    # 5. Verify <5% right-censoring & shared-witness PBR across ALL evaluated Pillar B cohorts
    metrics = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    cohorts = metrics["pillar_b_explicit_solvent_memory_closure"]["cohorts"]
    assert len(cohorts) >= 4
    for cname, cres in cohorts.items():
        cens = float(cres["target_metadata"]["censoring_rate"])
        assert cens < 0.05, f"Cohort {cname} violated <5% right-censoring requirement: {cens:.4f}"
        dga_diag = cres["stopped_dga_diagnostics"]
        assert dga_diag["shared_witness_pca_dim"] == 8
        pbr_dict = dga_diag["mean_oof_pbr_shared_witness"]
        for req_k in ("B1", "PT_prot", "PT_prot_env", "StoppedDGA_prot", "StoppedDGA_prot_env"):
            assert req_k in pbr_dict and pbr_dict[req_k] is not None


def test_ws3_outputs_and_report_consistency() -> None:
    """Verify existence of figures/JSON/report, verified citations, and exact JSON <-> Markdown sync."""
    for fig_name in ("fig_pdb16_dichotomy.png", "fig_gi1_vs_gs_coupling.png", "fig_water_na_memory_closure.png"):
        fpath = FIG_DIR / fig_name
        assert fpath.exists() and fpath.stat().st_size > 30_000, f"Missing or empty figure {fpath}"

    assert METRICS_JSON.exists() and REPORT_MD.exists()
    metrics = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    report_text = REPORT_MD.read_text(encoding="utf-8")

    # Provenance, formula, residue-guard, & full-author literature checks
    assert "20,501" in report_text
    assert "31659318" in report_text  # Shao et al. 2019 PMID (6KQI)
    assert "35637350" in report_text  # Yang et al. 2022 PMID (7FEE/7WV9)
    assert "37160876" in report_text  # Krishna Kumar et al. 2023 PMID (8GHV)
    assert "40866700" in report_text  # Hua et al. 2025 Nature replacement PMID (5XRA/5XR8)
    assert "10.1038/s41586-025-09454-5" in report_text  # Hua et al. 2025 Nature DOI
    assert "Dutta S, Zhao L, Shukla D" in report_text  # Full author list for PMID: 41962866
    assert "Lorpaiboon C, Guo SC, Strahan J, Weare J, Dinner AR" in report_text  # Full author list for PMID: 38391020
    assert "exp(-d/5.0)" in report_text and "exp(-d/4.0)" not in report_text

    # Pillar A exact numerical checks
    gi_y_mean = metrics["pillar_a_gi1_vs_gs_coupling"]["cohort_summary"]["Gi1_1201"]["metrics"]["Y294_Y397_OH_A"]["mean"]
    gs_y_mean = metrics["pillar_a_gi1_vs_gs_coupling"]["cohort_summary"]["Gs_2067"]["metrics"]["Y294_Y397_OH_A"]["mean"]
    gi_wat = metrics["pillar_a_gi1_vs_gs_coupling"]["cohort_summary"]["Gi1_1201"]["metrics"]["n_wat_core_pore"]["mean"]
    gs_wat = metrics["pillar_a_gi1_vs_gs_coupling"]["cohort_summary"]["Gs_2067"]["metrics"]["n_wat_core_pore"]["mean"]
    assert f"{gi_y_mean:.2f}" in report_text
    assert f"{gs_y_mean:.2f}" in report_text
    assert f"{gi_wat:.2f}" in report_text
    assert f"{gs_wat:.2f}" in report_text

    # Pillar B exact numerical checks across Primary (Fine_0p2ns_5traj), NaCl5_1p0ns_ExplicitNa, and Open6_1p0ns_AllTraj
    cohorts = metrics["pillar_b_explicit_solvent_memory_closure"]["cohorts"]
    for ckey in ("Fine_0p2ns_5traj", "NaCl5_1p0ns_ExplicitNa", "Open6_1p0ns_AllTraj"):
        c_obj = cohorts[ckey]
        cens_pct = f"{100.0 * c_obj['target_metadata']['censoring_rate']:.2f}%"
        dell_env = c_obj["static_committor_evaluation"]["prot_env_vs_prot"]["delta_ell"]["point"]
        n2b_env = c_obj["stopped_dga_diagnostics"]["n2b_reversibility_prot_env"]["l2_reversibility_error"]
        rho_lag0 = c_obj["mori_zwanzig_memory_profile"][0]["rho_mem_collapse_prot_to_env"]
        assert cens_pct in report_text, f"{ckey} censoring {cens_pct} missing from ws3_report.md"
        assert f"{dell_env:+.4f}" in report_text or f"{dell_env:.4f}" in report_text, (
            f"{ckey} static gain {dell_env:.4f} missing from ws3_report.md"
        )
        assert f"{n2b_env:.4f}" in report_text, f"{ckey} N2b {n2b_env:.4f} missing from ws3_report.md"
        assert f"{rho_lag0:.4f}" in report_text, f"{ckey} lag-0 rho_mem {rho_lag0:.4f} missing from ws3_report.md"
