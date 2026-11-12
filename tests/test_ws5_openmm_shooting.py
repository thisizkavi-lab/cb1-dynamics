"""Fast (<30 s) local physics, Hamiltonian-identity, queue/checkpoint, and notebook tests for WS5."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import time

import nbformat
import numpy as np
import openmm as mm
import openmm.unit as unit
import pandas as pd
import pytest

from src.ws5_openmm_shooting import (
    ARM_SPECS,
    CampaignConfig,
    ShootingCampaignQueue,
    build_interleaved_manifest,
    compute_clopper_pearson_ci,
    compute_kaplan_meier_curve,
    compute_logrank_test,
    extract_ws5_frame_observables,
    load_seed_bundle,
    plot_prospective_campaign_readout,
    summarize_prospective_campaign,
)
from src.ws5_openmm_shooting.seed_builder import (
    SEEDS_DIR,
    create_unrestrained_production_system,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
COLAB_NB_PATH = REPO_ROOT / "notebooks" / "03_Prospective_OpenMM_Shooting_Colab.ipynb"


def test_ws5_seed_budget_and_zero_raw_binaries() -> None:
    manifest_path = SEEDS_DIR / "ws5_seed_manifest.json"
    assert manifest_path.is_file(), f"Missing seed manifest: {manifest_path}"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["hamiltonian_identity_verified"] is True

    gz_files = sorted(SEEDS_DIR.glob("*.gz"))
    assert len(gz_files) == 5, f"Expected 5 compressed seed files (.pdb.gz / .xml.gz), found {[p.name for p in gz_files]}"

    total_bytes = sum(p.stat().st_size for p in gz_files)
    assert total_bytes < 15 * 1024 * 1024, (
        f"WS5 seed directory ({total_bytes / 1e6:.2f} MB) exceeds 15 MB GitHub budget!"
    )

    for key, finfo in manifest["files"].items():
        fpath = REPO_ROOT / finfo["path"]
        assert fpath.is_file(), f"Missing seed file for {key}: {fpath}"
        actual_sha = hashlib.sha256(fpath.read_bytes()).hexdigest()
        assert actual_sha == finfo["sha256_gzip"], f"SHA-256 mismatch on {fpath.name}"

    leaked = [p for p in SEEDS_DIR.rglob("*") if p.suffix.lower() in {".dcd", ".xtc", ".trr", ".tmp", ".part"}]
    assert not leaked, f"Found raw/temp binary files in {SEEDS_DIR}: {leaked}"


def test_bitwise_identical_hamiltonian_arm_a_vs_arm_b_and_f237l_arm_c() -> None:
    pdb_a, sys_a, idx_a = load_seed_bundle("Arm_A_Fake_Open_Trapped", SEEDS_DIR)
    pdb_b, sys_b, idx_b = load_seed_bundle("Arm_B_Committed_Open", SEEDS_DIR)
    pdb_c, sys_c, idx_c = load_seed_bundle("Arm_C_Mutant_F237L", SEEDS_DIR)

    # 1. Saved production System XMLs for Arm A and Arm B must be bitwise identical
    xml_a = mm.XmlSerializer.serialize(sys_a)
    xml_b = mm.XmlSerializer.serialize(sys_b)
    assert xml_a == xml_b, "Arm A and Arm B production Systems are NOT bitwise identical!"

    # 2. Independently rebuilding OpenMM Systems directly from reloaded PDB topologies must also match bitwise!
    rebuilt_a = mm.XmlSerializer.serialize(create_unrestrained_production_system(pdb_a.topology))
    rebuilt_b = mm.XmlSerializer.serialize(create_unrestrained_production_system(pdb_b.topology))
    assert rebuilt_a == rebuilt_b == xml_a, (
        "Rebuilt OpenMM Systems from Arm A and Arm B PDB topologies differ!"
    )

    # 3. Exact atom-for-atom and residue-for-residue identity between Arm A and Arm B
    atoms_a = [(a.index, a.name, a.residue.name, a.residue.index) for a in pdb_a.topology.atoms()]
    atoms_b = [(a.index, a.name, a.residue.name, a.residue.index) for a in pdb_b.topology.atoms()]
    assert atoms_a == atoms_b, "Atom/residue ordering differs between Arm A and Arm B!"

    # 4. Zero residual steering/restraint forces in production Hamiltonians (Arms A, B, and C)
    for label, s_obj in (("Arm_A", sys_a), ("Arm_B", sys_b), ("Arm_C", sys_c)):
        force_names = [type(s_obj.getForce(i)).__name__ for i in range(s_obj.getNumForces())]
        for forbidden in ("CustomBondForce", "CustomTorsionForce", "CustomExternalForce", "CustomCompoundBondForce"):
            assert forbidden not in force_names, f"{label} retains steering restraint {forbidden}!"
        assert "MonteCarloMembraneBarostat" in force_names, f"{label} missing MonteCarloMembraneBarostat!"

    # 5. Verify 3.0 amu HMR on non-water hydrogens
    rec_h_masses = [
        sys_a.getParticleMass(a.index).value_in_unit(unit.amu)
        for a in list(pdb_a.topology.chains())[0].atoms()
        if a.element is not None and a.element.symbol == "H"
    ]
    assert len(rec_h_masses) > 2000
    assert np.allclose(rec_h_masses, 3.0, atol=0.05), f"Expected 3.0 amu HMR on protein H, got {rec_h_masses[:5]}"

    # 6. Verify Arm C carries ONLY the F237L swap (PHE=20 atoms -> LEU=19 atoms => N_wt - 1 atoms)
    assert idx_a["res237_name"] == "PHE"
    assert idx_b["res237_name"] == "PHE"
    assert idx_c["res237_name"] == "LEU"
    assert pdb_a.topology.getNumAtoms() - pdb_c.topology.getNumAtoms() == 1
    assert idx_a["wat_O"].size == idx_c["wat_O"].size
    assert idx_a["na_ions"].size == idx_c["na_ions"].size
    assert idx_a["chl_heavy"].size == idx_c["chl_heavy"].size == 56  # 2 crystallographic CHL1 x 28 heavy atoms


def test_matched_open_tm6_and_orthogonal_stage3_separation_at_t0() -> None:
    obs_by_arm = {}
    for arm_name in ARM_SPECS:
        pdb, _, t_idx = load_seed_bundle(arm_name, SEEDS_DIR)
        bv = pdb.topology.getPeriodicBoxVectors()
        box_nm = np.array(
            [
                bv[0][0].value_in_unit(unit.nanometer),
                bv[1][1].value_in_unit(unit.nanometer),
                bv[2][2].value_in_unit(unit.nanometer),
            ]
        )
        obs_by_arm[arm_name] = extract_ws5_frame_observables(
            np.array(pdb.positions.value_in_unit(unit.nanometer)),
            box_nm,
            t_idx,
        )

    oa = obs_by_arm["Arm_A_Fake_Open_Trapped"]
    ob = obs_by_arm["Arm_B_Committed_Open"]
    oc = obs_by_arm["Arm_C_Mutant_F237L"]

    # All 3 arms must start at matched open cytoplasmic TM6 (14.3 +/- 0.25 A)
    for arm_name, obs in obs_by_arm.items():
        assert 14.05 <= obs["R214_D338_CA_A"] <= 14.55, (
            f"{arm_name} R214-D338={obs['R214_D338_CA_A']:.2f} A outside matched 14.3 +/- 0.25 A window!"
        )
    assert abs(oa["R214_D338_CA_A"] - ob["R214_D338_CA_A"]) <= 0.20

    # Arm A & Arm C: inward F155 (+65 deg), inward F/L237 (-80 deg), unlatched Y294-Y397 (~5.2 A)
    for trapped_name, ot in (("Arm_A", oa), ("Arm_C", oc)):
        assert 30.0 <= ot["F155_chi1_deg"] <= 100.0, f"{trapped_name} F155 chi1 not inward g+: {ot['F155_chi1_deg']}"
        assert -115.0 <= ot["X237_chi1_deg"] <= -45.0, f"{trapped_name} X237 chi1 not inward g-: {ot['X237_chi1_deg']}"
        assert ot["F155sc_N393CA_A"] < 13.0, f"{trapped_name} F155sc-N393CA too large: {ot['F155sc_N393CA_A']}"
        assert 4.65 <= ot["Y294_Y397_OH_A"] <= 6.00, f"{trapped_name} Y294-Y397 not unlatched: {ot['Y294_Y397_OH_A']}"

    # Arm B: outward F155 (-65 deg), trans F237 (~ +/-180 deg), latched Y294-Y397 (~3.1-3.6 A)
    assert -100.0 <= ob["F155_chi1_deg"] <= -30.0, f"Arm_B F155 chi1 not outward g-: {ob['F155_chi1_deg']}"
    assert abs(ob["X237_chi1_deg"]) >= 145.0, f"Arm_B F237 chi1 not trans: {ob['X237_chi1_deg']}"
    assert ob["F155sc_N393CA_A"] > 14.0, f"Arm_B F155sc-N393CA not flipped outward: {ob['F155sc_N393CA_A']}"
    assert ob["Y294_Y397_OH_A"] <= 3.85, f"Arm_B Y294-Y397 not latched: {ob['Y294_Y397_OH_A']}"
    assert ob["n_wat_pore"] >= 6, f"Expected pre-hydrated internal polar cavity, got {ob['n_wat_pore']} waters"
    assert ob["min_CHL1_cleft_A"] < 12.0


def test_unbiased_4fs_hmr_md_checkpoint_resume_continuity_and_stale_lock(tmp_path: Path) -> None:
    # 1. Verify default 900 ns interleaved campaign schedule
    default_cfg = CampaignConfig()
    default_jobs = build_interleaved_manifest(default_cfg)
    assert len(default_jobs) == 45
    assert [j["arm_code"] for j in default_jobs[:6]] == ["A", "B", "C", "A", "B", "C"]
    assert [j["replicate"] for j in default_jobs[:6]] == [1, 1, 1, 2, 2, 2]

    # 2. Run 50 unbiased 4fs HMR steps split across 2 x 25-step chunks with mid-run interruption & stale-lock recovery!
    micro_cfg = CampaignConfig(
        n_replicates_per_arm=1,
        traj_duration_ns=0.0002,   # 50 steps x 0.004 ps = 0.20 ps = 0.0002 ns
        chunk_duration_ns=0.0001,  # 25 steps per chunk (2 chunks total)
        dt_ps=0.004,
        report_stride_ps=0.020,    # every 5 steps -> 11 frames total (step 0, 5, ..., 50)
        stale_lock_timeout_s=1500.0,
    )
    q_worker1 = ShootingCampaignQueue(
        work_dir=tmp_path / "drive_queue",
        seeds_dir=SEEDS_DIR,
        cfg=micro_cfg,
        worker_id="tab1_worker",
        prefer_gpu=True,
    )
    job_a = q_worker1.jobs[0]
    assert q_worker1.try_claim_job(job_a, now_epoch=1000.0) is True

    # Competing Tab 2 must NOT steal an active lock (< 25 min old)
    q_worker2 = ShootingCampaignQueue(
        work_dir=tmp_path / "drive_queue",
        seeds_dir=SEEDS_DIR,
        cfg=micro_cfg,
        worker_id="tab2_worker",
        prefer_gpu=False,  # Also tests cross-platform State XML portability (GPU/OpenCL -> CPU)!
    )
    assert q_worker2.try_claim_job(job_a, now_epoch=1600.0) is False  # 600 s <= 1500 s

    # Worker 1 runs ONLY chunk 1 (25 steps) and then "disconnects"
    r1 = q_worker1.run_job(job_a, max_chunks_this_call=1, verbose=False)
    assert r1["completed"] is False
    assert r1["current_step"] == 25

    # Simulate 30 minutes passing (> 1500 s stale threshold): Tab 2 reclaims stale lock & resumes chunk 2!
    q_worker1.refresh_heartbeat(job_a["job_id"], chunk_idx=1, now_epoch=1000.0)
    assert q_worker2.try_claim_job(job_a, now_epoch=2800.0) is True  # 1800 s > 1500 s stale lock reclaimed!
    r2 = q_worker2.run_job(job_a, max_chunks_this_call=1, verbose=False)
    assert r2["completed"] is True
    assert r2["current_step"] == 50

    df_pq = pd.read_parquet(r2["parquet_path"])
    assert list(df_pq["step"]) == list(range(0, 55, 5))
    assert np.all(np.isfinite(df_pq["potential_energy_kJ_mol"]))
    assert np.all(np.isfinite(df_pq["kinetic_energy_kJ_mol"]))
    assert np.all((df_pq["R214_D338_CA_A"] > 12.0) & (df_pq["R214_D338_CA_A"] < 17.0))


def test_prospective_statistical_readout_and_colab_notebook_validity(tmp_path: Path) -> None:
    p_hat, lo, hi = compute_clopper_pearson_ci(8, 10)
    assert np.isclose(p_hat, 0.8) and 0.44 < lo < 0.8 < hi < 0.98

    km = compute_kaplan_meier_curve(np.array([2.0, 5.0, 20.0, 20.0]), np.array([True, True, False, False]))
    assert np.isclose(km["survival"].iloc[-1], 0.5)

    lr = compute_logrank_test(
        np.array([2.0, 3.0, 4.0, 5.0, 6.0]),
        np.array([True, True, True, True, True]),
        np.array([20.0, 20.0, 20.0, 20.0, 20.0]),
        np.array([False, False, False, False, False]),
    )
    assert lr["p_value"] < 0.01

    # Synthetic 12-trajectory readout smoke test (4 reps x 3 arms, dt = 0.25 ns non-default spacing!)
    # verifying pure R214_D338 commitment, WT-only pre-hitting memory_closure, dynamic lag_steps, and non-None N2a p-values
    tdir = tmp_path / "synth_trajs"
    tdir.mkdir(parents=True)
    rng = np.random.default_rng(2026)
    times_ns = np.arange(0, 81) * 0.25  # 0.0 .. 20.0 ns at 0.25 ns spacing => lag_steps = 2 for 0.5 ns!
    steps = (times_ns * 250000).astype(np.int64)
    for rep in range(1, 5):
        for arm_name, collapse_flag in (
            ("Arm_A_Fake_Open_Trapped", True),
            ("Arm_B_Committed_Open", False),
            ("Arm_C_Mutant_F237L", False),
        ):
            jid = f"rep{rep:02d}_{arm_name[:5]}"
            tm6 = (14.25 - 0.16 * times_ns if collapse_flag else 14.28 + 0.04 * times_ns) + rng.normal(0, 0.05, size=times_ns.size)
            tm6[0] = 14.26  # Strictly inside (12.80, 15.30) A at t=0
            f155 = (11.4 if collapse_flag else 15.4) + rng.normal(0, 0.15, size=times_ns.size)
            latch = (5.2 if collapse_flag else 3.2) + rng.normal(0, 0.12, size=times_ns.size)
            df = pd.DataFrame(
                {
                    "job_id": jid,
                    "arm_name": arm_name,
                    "arm_code": arm_name[4],
                    "replicate": rep,
                    "step": steps,
                    "time_ns": times_ns,
                    "R214_D338_CA_A": tm6,
                    "F155sc_N393CA_A": f155,
                    "Y294_Y397_OH_A": latch,
                    "wire_conductance_d163_ny": np.where(collapse_flag, 0.25, 0.85) + rng.normal(0, 0.03, size=times_ns.size),
                    "min_CHL1_cleft_A": np.where(collapse_flag, 3.8, 7.8) + rng.normal(0, 0.1, size=times_ns.size),
                }
            )
            df.to_parquet(tdir / f"{jid}_observables.parquet", index=False)

    summary = summarize_prospective_campaign(tdir, n_memory_shuffles=20)
    assert summary["n_trajectories"] == 12
    assert summary["comparisons"]["Arm_A_vs_Arm_B_Single_Hamiltonian"]["logrank"]["p_value"] < 0.05
    mc = summary["memory_closure"]
    assert mc["available"] is True
    assert mc["hamiltonian_scope"] == "WT_only_Arm_A_Arm_B"
    assert mc["n_wt_resolved_trajectories"] == 8  # Strictly 4 Arm_A + 4 Arm_B (Arm_C F237L excluded!)
    assert np.isclose(mc["dt_ns"], 0.25) and mc["lag_steps"] == 2 and np.isclose(mc["lag_ns"], 0.5)
    assert mc["n2a_p_1d"] is not None and 0.0 < mc["n2a_p_1d"] <= 1.0
    assert mc["n2a_p_stage3"] is not None and 0.0 < mc["n2a_p_stage3"] <= 1.0
    out_png = tmp_path / "readout.png"
    fig = plot_prospective_campaign_readout(summary, out_png=out_png)
    assert out_png.is_file() and out_png.stat().st_size > 10_000
    fig.clf()

    # Regression guard (#56 Defect 1): 4/4 Arm_B runs lingering near seed at t=2.0 ns (14.25 A, latched 3.12 A,
    # outward F155 15.65 A) and THEN collapsing to 11.5 A at t=4.0 ns MUST get q_outcome == 0 (never B_committed!)
    cdir = tmp_path / "arm_b_collapse_check"
    cdir.mkdir(parents=True)
    t_grid = np.arange(0.0, 10.5, 0.5)
    for rep in range(1, 5):
        tm6_c = np.where(t_grid < 4.0, 14.25, 11.50)
        df_c = pd.DataFrame(
            {
                "job_id": f"rep{rep:02d}_Arm_B",
                "arm_name": "Arm_B_Committed_Open",
                "arm_code": "B",
                "replicate": rep,
                "step": (t_grid * 250000).astype(np.int64),
                "time_ns": t_grid,
                "R214_D338_CA_A": tm6_c,
                "F155sc_N393CA_A": np.full_like(t_grid, 15.65),
                "Y294_Y397_OH_A": np.full_like(t_grid, 3.12),
                "wire_conductance_d163_ny": np.full_like(t_grid, 1.0),
                "min_CHL1_cleft_A": np.full_like(t_grid, 8.0),
            }
        )
        df_c.to_parquet(cdir / f"rep{rep:02d}_Arm_B_observables.parquet", index=False)
    sum_c = summarize_prospective_campaign(cdir)
    st_b = sum_c["arm_stats"]["Arm_B_Committed_Open"]
    assert list(sum_c["traj_summary_df"]["q_outcome"]) == [0, 0, 0, 0]
    assert st_b["n_collapsed"] == 4
    assert st_b["n_committed_B"] == 0
    assert st_b["q_hat_resolved"] == 0.0
    assert st_b["open_persistence_rate"] == 0.0

    # Validate Colab notebook structure & required tokens
    assert COLAB_NB_PATH.is_file()
    nb = nbformat.read(COLAB_NB_PATH, as_version=4)
    nbformat.validate(nb)
    assert len(nb.cells) >= 8
    full_src = "\n".join(c.source for c in nb.cells)
    for tok in (
        "google.colab",
        "/content/drive/MyDrive/cb1-dynamics-phase4/ws5_shooting",
        "Arm_A_Fake_Open_Trapped",
        "Arm_B_Committed_Open",
        "Arm_C_Mutant_F237L",
        "ShootingCampaignQueue",
        "summarize_prospective_campaign",
        "388.31",
    ):
        assert tok in full_src, f"Colab notebook missing required token: {tok!r}"
