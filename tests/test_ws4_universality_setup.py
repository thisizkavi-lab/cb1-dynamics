"""Fast Local Unit & Provenance Tests for Workstream 4 (`WS4` Multi-GPCR Universality).

Target runtime: < 15 seconds on local CPU (zero `.xtc` downloads locally).
Verifies:
  1. `data/metadata/gpcr_family_manifest.json` schema, all 6 Class A GPCRs (`CB1, CB2, S1P1, B2AR, MOR, A2AR`),
     conserved core BW residues (`D2.50, R3.50, Y5.58, W6.48, N7.49, Y7.53`), and evolutionary divergence
     between `CB1` (`F155^2.42 - F237^4.46`) and `CB2` (`F72^2.42 - L154^4.46`).
  2. Live `HTTP Range: bytes=0-127` binary verification (`HTTP 206` + `XTC magic == 1995` + `natoms` match)
     across all 6 GPCRs in the manifest.
  3. Synthetic multi-GPCR (`CB1` TER-split two-chain `PHE4.46` + periodic box jump vs `CB2` `LEU4.46`)
     `mdtraj.Trajectory` topology indexing, PBC whole-receptor reassembly, water-wire BFS conductance,
     ion sentinel encoding, and zero-NaN feature extraction.
  4. End-to-end `U1` (Mori-Zwanzig water/Na+ memory collapse & StoppedDGA PBR) + `U2` (`2.42-4.46`
     evolutionary specificity) statistical suite and 4-panel figure generation.
  5. `nbformat` schema, strict alternating markdown/code structure, Colab badge, Drive checkpoint path
     in `notebooks/02_Multi_GPCR_Universality_Colab.ipynb`, and strict local workspace hygiene (`0` `.xtc`/`.dcd`
     binaries and `0` files `> 25 MB`).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import mdtraj as md
import nbformat
import numpy as np
import pandas as pd
import pytest

from src.ws4_gpcr_universality import (
    CANONICAL_BW_POSITIONS,
    build_bw_topology_index,
    featurize_receptor_chunk,
    load_gpcr_family_manifest,
    pbc_wrap_receptor_angstrom,
    run_superfamily_universality_suite,
    stream_and_featurize_receptor,
    verify_manifest_live_probes,
)
from src.ws4_gpcr_universality.universality_analysis import build_buffered_macro_segments


PROJECT_ROOT = Path("/Users/kavi/Documents/artico/cb1-dynamics")
MANIFEST_PATH = PROJECT_ROOT / "data" / "metadata" / "gpcr_family_manifest.json"
COLAB_NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "02_Multi_GPCR_Universality_Colab.ipynb"
EXPECTED_RECEPTORS = {"CB1", "CB2", "S1P1", "B2AR", "MOR", "A2AR"}


def _build_mock_gpcr_trajectory(
    bw446_resname: str = "PHE",
    split_chains: bool = True,
    apply_pbc_jump_nm: float = 8.0,
    bridge_water: bool = True,
    n_frames: int = 6,
) -> tuple[md.Trajectory, dict[str, Any]]:
    """Construct a compact explicit-solvent + Na+ + POPC mock GPCR `mdtraj.Trajectory`."""
    top = md.Topology()
    ch0 = top.add_chain()
    ch1 = top.add_chain() if split_chains else ch0
    ch_env = top.add_chain()

    # 12 canonical BW positions: TM1-TM5 on ch0, TM6-TM7 on ch1
    bw_defs = [
        ("1.35", ch0, 119, "ILE", np.array([-8.0, 6.0, 10.0])),
        ("2.42", ch0, 155, "PHE", np.array([-6.0, -5.0, -4.0])),
        ("2.50", ch0, 163, "ASP", np.array([-2.0, 0.0, -1.0])),
        ("3.36", ch0, 200, "PHE", np.array([2.5, 2.0, 2.0])),
        ("3.50", ch0, 214, "ARG", np.array([3.0, -2.0, -10.0])),
        ("4.46", ch0, 237, bw446_resname, np.array([-6.0, -8.3, -4.0])),
        ("5.58", ch0, 294, "TYR", np.array([4.0, 3.0, -8.0])),
        ("6.30", ch1, 338, "ASP", np.array([14.0, -2.0, -10.0])),
        ("6.48", ch1, 356, "TRP", np.array([2.5, -1.5, 2.5])),
        ("7.35", ch1, 379, "PHE", np.array([-4.0, 6.5, 10.5])),
        ("7.49", ch1, 393, "ASN", np.array([5.8, 0.0, -1.0])),
        ("7.53", ch1, 397, "TYR", np.array([3.5, 0.5, -7.8])),
    ]

    sidechain_templates: dict[str, list[tuple[str, str, np.ndarray]]] = {
        "ILE": [("CB", "C", [0.8, 0.0, 0.0]), ("CG1", "C", [1.5, 0.6, 0.0]), ("CD1", "C", [2.3, 0.6, 0.3])],
        "LEU": [("CB", "C", [0.8, 0.0, 0.0]), ("CG", "C", [1.5, 0.5, 0.0]), ("CD1", "C", [2.2, 1.0, 0.2]), ("CD2", "C", [2.2, -0.3, -0.2])],
        "PHE": [
            ("CB", "C", [0.0, -0.8, 0.0]),
            ("CG", "C", [0.0, -1.5, 0.0]),
            ("CD1", "C", [0.6, -2.0, 0.0]),
            ("CD2", "C", [-0.6, -2.0, 0.0]),
            ("CE1", "C", [0.6, -2.8, 0.0]),
            ("CE2", "C", [-0.6, -2.8, 0.0]),
            ("CZ", "C", [0.0, -3.2, 0.0]),
        ],
        "ASP": [("CB", "C", [0.8, 0.0, 0.0]), ("CG", "C", [1.4, 0.0, 0.0]), ("OD1", "O", [1.9, 0.4, 0.0]), ("OD2", "O", [1.9, -0.4, 0.0])],
        "ASN": [("CB", "C", [-0.8, 0.0, 0.0]), ("CG", "C", [-1.4, 0.0, 0.0]), ("OD1", "O", [-1.9, 0.4, 0.0]), ("ND2", "N", [-1.9, -0.4, 0.0])],
        "ARG": [("CB", "C", [0.8, 0.0, 0.0]), ("CG", "C", [1.5, 0.3, 0.0]), ("NE", "N", [2.2, 0.3, 0.0]), ("NH1", "N", [2.9, 0.7, 0.0]), ("NH2", "N", [2.9, -0.2, 0.0])],
        "TYR": [("CB", "C", [0.0, -0.8, 0.0]), ("CG", "C", [0.0, -1.5, 0.0]), ("CZ", "C", [0.0, -2.8, 0.0]), ("OH", "O", [0.0, -3.6, 0.0])],
        "TRP": [("CB", "C", [0.0, 0.8, 0.0]), ("CG", "C", [0.0, 1.5, 0.0]), ("CD1", "C", [0.6, 2.0, 0.0]), ("NE1", "N", [0.4, 2.8, 0.0]), ("CZ2", "C", [-0.6, 3.2, 0.0])],
    }

    atom_coords_A: list[np.ndarray] = []
    ch1_atom_indices: list[int] = []
    bw_residue_map: dict[str, dict[str, Any]] = {}

    for bw, ch_obj, rseq, rname, base_ca_A in bw_defs:
        res = top.add_residue(rname, ch_obj, resSeq=rseq)
        bw_residue_map[bw] = {
            "chain_idx": ch_obj.index,
            "resSeq": rseq,
            "resName": rname,
            "is_wt_identity": True,
        }
        bb_offsets = [
            ("N", "N", np.array([-0.9, 0.4, 0.0])),
            ("CA", "C", np.array([0.0, 0.0, 0.0])),
            ("C", "C", np.array([0.9, 0.4, 0.0])),
            ("O", "O", np.array([1.2, 1.2, 0.0])),
        ]
        for aname, elem, off in bb_offsets:
            a = top.add_atom(aname, md.element.get_by_symbol(elem), res)
            atom_coords_A.append(base_ca_A + off)
            if ch_obj.index == 1:
                ch1_atom_indices.append(a.index)
        for aname, elem, off_list in sidechain_templates[rname]:
            a = top.add_atom(aname, md.element.get_by_symbol(elem), res)
            atom_coords_A.append(base_ca_A + np.array(off_list, dtype=np.float64))
            if ch_obj.index == 1:
                ch1_atom_indices.append(a.index)

    # Add explicit water oxygens: 1 bridging D2.50 OD1 (-0.1, 0.4, -1.0) <-> N7.49 OD1 (3.9, 0.4, -1.0) when bridge_water=True
    wat_positions = [
        np.array([1.9, 0.4, -1.0]) if bridge_water else np.array([25.0, 25.0, 25.0]),
        np.array([-2.0, 2.2, 1.5]),
    ]
    for wi, wpos in enumerate(wat_positions, start=1):
        wres = top.add_residue("HOH", ch_env, resSeq=1000 + wi)
        top.add_atom("O", md.element.oxygen, wres)
        atom_coords_A.append(wpos)

    # Add explicit Na+ ion within 2.4 A of D2.50 OD2 (-0.1, -0.4, -1.0)
    na_res = top.add_residue("SOD", ch_env, resSeq=2001)
    top.add_atom("SOD", md.element.sodium, na_res)
    atom_coords_A.append(np.array([-0.1, -2.5, -1.0]))

    # Add explicit POPC heavy atoms near TM2-TM4 cleft (-6.0, -6.5, -4.0)
    popc_res = top.add_residue("POPC", ch_env, resSeq=3001)
    top.add_atom("C21", md.element.carbon, popc_res)
    atom_coords_A.append(np.array([-8.2, -6.5, -4.0]))

    base_nm = np.stack(atom_coords_A, axis=0) / 10.0  # (N_atoms, 3) in nm
    xyz = np.repeat(base_nm[None, :, :], n_frames, axis=0)
    # Apply slight thermal jitter + artificial +1 periodic box jump on chain 1 to test PBC reassembly!
    rng = np.random.default_rng(7)
    xyz += rng.normal(0.0, 0.003, size=xyz.shape)
    if split_chains and apply_pbc_jump_nm != 0.0:
        xyz[:, ch1_atom_indices, 0] += apply_pbc_jump_nm

    box_nm = np.tile(np.array([[8.0, 8.0, 9.5]], dtype=np.float32), (n_frames, 1))
    traj = md.Trajectory(xyz=xyz.astype(np.float32), topology=top, unitcell_lengths=box_nm, unitcell_angles=np.full((n_frames, 3), 90.0, dtype=np.float32))
    traj_entry = {
        "traj_id": f"mock_{bw446_resname}",
        "gpcrmd_dyn_id": 9999,
        "pdb_id": "MOCK",
        "state_class": "Inactive",
        "dt_ns": 0.2,
        "receptor_chains": [0, 1] if split_chains else [0],
        "bw_residue_map": bw_residue_map,
    }
    return traj, traj_entry


def test_gpcr_family_manifest_schema_and_bw_conservation() -> None:
    """Verify `gpcr_family_manifest.json` covers all 6 GPCRs with exact BW and XTC 1995 provenance."""
    assert MANIFEST_PATH.exists(), f"Missing manifest: {MANIFEST_PATH}"
    manifest = load_gpcr_family_manifest(MANIFEST_PATH)

    assert manifest["colab_checkpoint_dir"] == "/content/drive/MyDrive/cb1-dynamics-phase4/ws4_universality"
    assert tuple(manifest["canonical_bw_positions"]) == CANONICAL_BW_POSITIONS
    assert set(manifest["receptors"].keys()) == EXPECTED_RECEPTORS

    # Evolutionary control check: CB1 has F237^4.46, CB2 has L154^4.46, and CB1 is uniquely Phe at 4.46
    cb1_wt = manifest["receptors"]["CB1"]["bw_canonical_wt"]
    cb2_wt = manifest["receptors"]["CB2"]["bw_canonical_wt"]
    assert cb1_wt["2.42"]["wt_resname"] == "PHE" and cb1_wt["2.42"]["canonical_resseq"] == 155
    assert cb1_wt["4.46"]["wt_resname"] == "PHE" and cb1_wt["4.46"]["canonical_resseq"] == 237
    assert cb2_wt["2.42"]["wt_resname"] == "PHE" and cb2_wt["2.42"]["canonical_resseq"] == 72
    assert cb2_wt["4.46"]["wt_resname"] == "LEU" and cb2_wt["4.46"]["canonical_resseq"] == 154
    assert all(
        t["dt_ns"] == 1.0 and t["approx_duration_ns"] == 398.0
        for t in manifest["receptors"]["CB2"]["trajectories"]
    )
    assert manifest["receptors"]["CB2"]["approx_aggregate_us"] == 1.59
    assert manifest["receptors"]["MOR"]["approx_aggregate_us"] == 1.25
    assert manifest["receptors"]["A2AR"]["approx_aggregate_us"] == 1.7

    for rec_key, rinfo in manifest["receptors"].items():
        trajs = rinfo["trajectories"]
        assert len(trajs) >= 2, f"{rec_key}: expected >=2 open trajectories, got {len(trajs)}"
        wt_map = rinfo["bw_canonical_wt"]
        assert wt_map["2.50"]["wt_resname"] == "ASP"
        assert wt_map["3.50"]["wt_resname"] == "ARG"
        assert wt_map["5.58"]["wt_resname"] == "TYR"
        assert wt_map["6.48"]["wt_resname"] == "TRP"
        assert wt_map["7.49"]["wt_resname"] == "ASN"
        assert wt_map["7.53"]["wt_resname"] == "TYR"
        if rec_key != "CB1":
            assert wt_map["4.46"]["wt_resname"] not in ("PHE", "TYR", "TRP")

        for t in trajs:
            assert t["xtc_probe"]["http_status"] == 206
            assert t["xtc_probe"]["magic"] == 1995
            assert t["xtc_probe"]["natoms"] == t["topology_summary"]["pdb_natoms"]
            assert t["topology_summary"]["n_water_O"] >= 1000
            assert t["topology_summary"]["n_lipid_heavy"] >= 500
            assert set(t["bw_residue_map"].keys()) == set(CANONICAL_BW_POSITIONS)
            for bw_k, bw_v in t["bw_residue_map"].items():
                assert bw_v["is_wt_identity"] is True, f"{t['traj_id']} mutated at {bw_k}: {bw_v}"


def test_live_http206_xtc_magic_1995_across_all_6_gpcrs() -> None:
    """Live-probe one `.xtc` URL per GPCR (`Range: bytes=0-127`) to confirm HTTP 206 + magic 1995."""
    manifest = load_gpcr_family_manifest(MANIFEST_PATH)
    live_res = verify_manifest_live_probes(manifest, one_per_receptor=True, max_workers=6)
    assert len(live_res) == 6
    seen_recs = set()
    for item in live_res:
        seen_recs.add(item["receptor"])
        assert item["http_status"] == 206
        assert item["magic"] == 1995
        assert item["natoms"] > 20000
        assert item["total_bytes"] > 50_000_000
    assert seen_recs == EXPECTED_RECEPTORS


def test_synthetic_multi_gpcr_pbc_featurizer_and_guards() -> None:
    """Verify PBC TER-split healing, aromatic vs aliphatic 4.46 indexing, water-wire BFS, and 0 NaNs."""
    traj_cb1, entry_cb1 = _build_mock_gpcr_trajectory(
        bw446_resname="PHE", split_chains=True, apply_pbc_jump_nm=8.0, bridge_water=True
    )
    t_idx_cb1 = build_bw_topology_index(traj_cb1.topology, entry_cb1)
    assert t_idx_cb1["is_aromatic_pair_242_446"] == 1
    xyz_cb1 = pbc_wrap_receptor_angstrom(traj_cb1, t_idx_cb1)
    rows_cb1 = featurize_receptor_chunk(xyz_cb1, t_idx_cb1, "CB1", entry_cb1)
    df_cb1 = pd.DataFrame(rows_cb1)

    assert int(df_cb1.isna().sum().sum()) == 0
    # Unwrapped R3.50 (x=3) to D6.30 (x=14) CA distance is ~11.0 A despite +80 A box jump on chain 1!
    assert np.allclose(df_cb1["d_350_630_CA_A"], 11.0, atol=0.25)
    assert np.all(df_cb1["wire_D250_NPxxY_cond"] >= 0.49)  # 1 bridging water -> hops=1 -> cond=0.5
    assert np.all(df_cb1["coord_Na_D250_3p2"] >= 1)
    assert np.all(df_cb1["min_K_D250_A"] == 99.0)  # K+ absent sentinel

    # CB2-like mock (LEU at 4.46, bridging water removed)
    traj_cb2, entry_cb2 = _build_mock_gpcr_trajectory(
        bw446_resname="LEU", split_chains=False, apply_pbc_jump_nm=0.0, bridge_water=False
    )
    t_idx_cb2 = build_bw_topology_index(traj_cb2.topology, entry_cb2)
    assert t_idx_cb2["is_aromatic_pair_242_446"] == 0
    xyz_cb2 = pbc_wrap_receptor_angstrom(traj_cb2, t_idx_cb2)
    df_cb2 = pd.DataFrame(featurize_receptor_chunk(xyz_cb2, t_idx_cb2, "CB2", entry_cb2))
    assert int(df_cb2.isna().sum().sum()) == 0
    assert np.all(df_cb2["wire_D250_NPxxY_cond"] == 0.0)

    # Conserved core mutation guard must reject D2.50 -> ALA
    bad_entry = copy.deepcopy(entry_cb2)
    bad_entry["bw_residue_map"]["2.50"]["resName"] = "ALA"
    with pytest.raises(AssertionError):
        build_bw_topology_index(traj_cb2.topology, bad_entry, enforce_conserved_core=True)


def test_drive_checkpoint_resume_without_redownload(tmp_path: Path) -> None:
    """Verify `stream_and_featurize_receptor` resumes cleanly from Drive `.parquet` checkpoints."""
    traj_cb2, entry_cb2 = _build_mock_gpcr_trajectory(bw446_resname="LEU", split_chains=False)
    t_idx = build_bw_topology_index(traj_cb2.topology, entry_cb2)
    xyz = pbc_wrap_receptor_angstrom(traj_cb2, t_idx)
    df_part = pd.DataFrame(featurize_receptor_chunk(xyz, t_idx, "CB2", entry_cb2))

    # Simulate an interrupted Colab session where traj_parts/mock_LEU.parquet was already saved to Drive
    parts_dir = tmp_path / "traj_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    df_part.to_parquet(parts_dir / "mock_LEU.parquet", index=False)

    mock_manifest = {
        "receptors": {
            "CB2": {
                "trajectories": [
                    {
                        **entry_cb2,
                        "pdb_url": "https://invalid.example.invalid/never_called.pdb",
                        "pdb_size_bytes": 123,
                        "xtc_url": "https://invalid.example.invalid/never_called.xtc",
                        "xtc_size_bytes": 456,
                    }
                ]
            }
        }
    }
    # Resume from traj_parts checkpoint and promote to CB2_features.parquet without touching network
    df_resumed = stream_and_featurize_receptor("CB2", mock_manifest, out_dir=tmp_path)
    assert len(df_resumed) == len(df_part)
    assert (tmp_path / "CB2_features.parquet").exists()

    # Second call resumes directly from receptor-level CB2_features.parquet
    df_cached = stream_and_featurize_receptor("CB2", mock_manifest, out_dir=tmp_path)
    assert len(df_cached) == len(df_part)
    assert int(df_cached.isna().sum().sum()) == 0


def test_cross_receptor_u1_u2_statistical_suite_and_figure(tmp_path: Path) -> None:
    """Smoke-test U1 Mori-Zwanzig closure + U2 2.42-4.46 evolutionary specificity + 4-panel figure."""
    rng = np.random.default_rng(123)
    rec_pairs = {
        "CB1": ("PHE", "PHE", 1),
        "CB2": ("PHE", "LEU", 0),
        "S1P1": ("PHE", "ILE", 0),
        "B2AR": ("PHE", "ILE", 0),
        "MOR": ("TYR", "ASN", 0),
        "A2AR": ("PHE", "ILE", 0),
    }

    df_by_rec: dict[str, pd.DataFrame] = {}
    for r_i, (rkey, (r242, r446, is_arom)) in enumerate(rec_pairs.items()):
        rows: list[dict[str, Any]] = []
        dt_i = 1.0 if rkey == "CB2" else 0.2
        n_f = 400 if rkey == "CB2" else 250
        relax_rate = 0.48 if rkey == "CB2" else 0.30
        sw_prob = 0.14 if rkey == "CB2" else 0.08
        for t_j in range(2):
            # Slow hidden telegraph water-wire/Na+ mode z_solv(t) in {-1, +1} driving D2.50-N7.49 transitions
            z_solv = np.zeros(n_f, dtype=np.float64)
            aperture = np.zeros(n_f, dtype=np.float64)
            z_solv[0] = 1.0 if t_j == 1 else -1.0
            aperture[0] = 4.6
            for k in range(1, n_f):
                z_solv[k] = -z_solv[k - 1] if rng.uniform() < sw_prob else z_solv[k - 1]
                target_eq = 4.6 - 1.25 * z_solv[k - 1]
                aperture[k] = aperture[k - 1] - relax_rate * (aperture[k - 1] - target_eq) + rng.normal(0.0, 0.13)

            aperture = np.clip(aperture, 2.9, 6.6)
            wire_cond = np.clip(0.5 + 0.42 * z_solv + rng.normal(0.0, 0.05, n_f), 0.0, 1.0)
            pore_wat = np.clip(np.round(8.0 + 4.5 * z_solv + rng.normal(0.0, 0.8, n_f)), 0, 25).astype(int)
            na_coord = (z_solv < 0.0).astype(int)
            prox_na = np.where(na_coord > 0, 0.65, 0.12) + rng.uniform(0.0, 0.04, n_f)

            if is_arom == 1:
                # CB1 bimodal aromatic cleft lock coupled to lipid entry
                gate = (rng.uniform(0.0, 1.0, n_f) > 0.45).astype(float)
                d_242_446 = np.where(gate > 0.5, rng.normal(3.7, 0.25, n_f), rng.normal(5.4, 0.35, n_f))
                chi_242 = np.where(gate > 0.5, rng.normal(-65.0, 12.0, n_f), rng.normal(170.0, 14.0, n_f))
                chi_446 = np.where(gate > 0.5, rng.normal(-60.0, 12.0, n_f), rng.normal(65.0, 15.0, n_f))
                lip_cleft = np.where(gate > 0.5, rng.normal(5.2, 0.4, n_f), rng.normal(3.6, 0.3, n_f))
            else:
                d_242_446 = rng.normal(5.1 + 0.1 * r_i, 0.4, n_f)
                chi_242 = rng.normal(-65.0, 10.0, n_f)
                chi_446 = rng.normal(-62.0, 10.0, n_f)
                lip_cleft = rng.normal(4.8, 0.5, n_f)

            for f in range(n_f):
                rows.append({
                    "receptor": rkey,
                    "traj_id": f"{rkey}_syn_{t_j}",
                    "gpcrmd_dyn_id": 100 + 10 * r_i + t_j,
                    "pdb_id": "SYNP",
                    "state_class": "Active" if t_j == 1 else "Inactive",
                    "frame_idx": f,
                    "dt_ns": dt_i,
                    "time_ns": round(f * dt_i, 4),
                    "bw242_resname": r242,
                    "bw446_resname": r446,
                    "is_aromatic_pair_242_446": is_arom,
                    "has_bw446_chi1": 1,
                    "d_350_630_CA_A": float(11.5 + 2.0 * t_j + rng.normal(0.0, 0.3)),
                    "d_350_630_sc_min_A": float(6.0 + 2.0 * t_j + rng.normal(0.0, 0.3)),
                    "d_250_749_CA_A": float(aperture[f] + 3.2),
                    "d_250_749_polar_A": float(aperture[f]),
                    "d_250_648_CA_A": float(9.5 + rng.normal(0.0, 0.2)),
                    "d_336_648_CA_A": float(8.5 + rng.normal(0.0, 0.2)),
                    "d_336_648_sc_min_A": float(4.2 + rng.normal(0.0, 0.25)),
                    "d_336_648_sc_com_A": float(5.5 + rng.normal(0.0, 0.25)),
                    "d_558_753_CA_A": float(9.0 + rng.normal(0.0, 0.25)),
                    "d_558_753_OH_A": float(4.6 + rng.normal(0.0, 0.3)),
                    "d_135_735_CA_A": float(12.0 + rng.normal(0.0, 0.3)),
                    "d_242_446_CA_A": float(8.2 + rng.normal(0.0, 0.2)),
                    "d_242_446_sc_min_A": float(max(2.8, d_242_446[f])),
                    "d_242_446_sc_com_A": float(max(3.5, d_242_446[f] + 1.2)),
                    "d_242sc_749CA_A": float(12.5 + rng.normal(0.0, 0.3)),
                    "bw242_chi1_deg": float(((chi_242[f] + 180.0) % 360.0) - 180.0),
                    "bw446_chi1_deg": float(((chi_446[f] + 180.0) % 360.0) - 180.0),
                    "bw336_chi1_deg": float(rng.normal(-65.0, 12.0)),
                    "bw648_chi1_deg": float(rng.normal(-70.0, 15.0)),
                    "bw753_chi1_deg": float(rng.normal(-60.0, 12.0)),
                    "n_wat_core_pore": int(pore_wat[f]),
                    "n_wat_D250": int(max(0, pore_wat[f] // 3)),
                    "n_wat_W648": 2,
                    "n_wat_N749": 2,
                    "n_wat_Y558": 1,
                    "n_wat_Y753": 1,
                    "hops_D250_W648": 1,
                    "wire_D250_W648_cond": float(wire_cond[f]),
                    "hops_D250_NPxxY": 1,
                    "wire_D250_NPxxY_cond": float(wire_cond[f]),
                    "hops_558_753": 2,
                    "wire_558_753_cond": 0.333,
                    "has_na_ions": 1,
                    "has_k_ions": 0,
                    "min_Na_D250_A": float(2.6 if na_coord[f] else 7.5),
                    "coord_Na_D250_3p2": int(na_coord[f]),
                    "prox_Na_D250": float(prox_na[f]),
                    "min_K_D250_A": 99.0,
                    "coord_K_D250_3p2": 0,
                    "prox_K_D250": 0.0,
                    "min_cation_D250_A": float(2.6 if na_coord[f] else 7.5),
                    "coord_cation_D250_3p2": int(na_coord[f]),
                    "prox_cation_D250": float(prox_na[f]),
                    "min_lipid_TM2_TM4_cleft_A": float(max(2.8, lip_cleft[f])),
                    "min_lipid_TM1_TM7_portal_A": 4.5,
                })
        df_by_rec[rkey] = pd.DataFrame(rows)

    out_json = tmp_path / "ws4_summary.json"
    out_fig = tmp_path / "fig_ws4.png"
    summary = run_superfamily_universality_suite(
        df_by_rec,
        out_json_path=out_json,
        out_fig_path=out_fig,
        tau_max_ns=2.0,
        n_splits=3,
        n_shuffles=6,
        n_perms=6,
        n_boot=60,
    )

    assert out_json.exists() and out_json.stat().st_size > 1000
    assert out_fig.exists() and out_fig.stat().st_size > 30_000
    assert summary["U1_water_na_closure"]["per_receptor"]["CB2"]["eff_dt_ns"] == 1.0
    assert summary["U1_water_na_closure"]["n_receptors_positive_static_env_gain"] >= 5
    assert summary["U1_water_na_closure"]["n_receptors_positive_memory_collapse"] >= 5

    # Explicit seam-buffer regression test: >= 2*tau_max gap preserved without clipping even for large buffer_ns
    df_cb1_u = df_by_rec["CB1"].copy()
    df_cb1_u["eff_dt_ns"] = 0.2
    for test_buf in (4.0, 8.5):
        df_b, seg_b, _ = build_buffered_macro_segments(df_cb1_u, macro_block_ns=25.0, buffer_ns=test_buf)
        for _, sub_t in df_b.groupby("traj_id", sort=False):
            s_loc = seg_b[sub_t.index.to_numpy()]
            uniq_s = list(dict.fromkeys(s_loc.tolist()))
            for idx_s in range(len(uniq_s) - 1):
                t_prev_end = float(sub_t.loc[s_loc == uniq_s[idx_s], "time_ns"].iloc[-1])
                t_next_start = float(sub_t.loc[s_loc == uniq_s[idx_s + 1], "time_ns"].iloc[0])
                assert t_next_start - t_prev_end >= test_buf - 1e-9

    u2 = summary["U2_tm2_tm4_specificity"]
    assert u2["evolutionary_contrasts"]["CB1_vs_non_aromatic_446_controls"]["cb1_is_unique_aromatic_446"] is True
    assert u2["per_receptor"]["CB1"]["aromatic_lock_occupancy_4p2A"]["point"] > 0.30
    assert u2["per_receptor"]["CB2"]["aromatic_lock_occupancy_4p2A"]["point"] == 0.0
    assert u2["evolutionary_contrasts"]["CB1_vs_CB2"]["delta_joint_rotamer_entropy_bits"] > 0.5


def test_colab_notebook_structure_and_local_workspace_hygiene() -> None:
    """Validate `02_Multi_GPCR_Universality_Colab.ipynb` schema & strict local 0-xtc / <25MB hygiene."""
    assert COLAB_NOTEBOOK_PATH.exists(), f"Missing Colab notebook: {COLAB_NOTEBOOK_PATH}"
    nb = nbformat.read(str(COLAB_NOTEBOOK_PATH), as_version=4)
    nbformat.validate(nb)
    assert len(nb.cells) >= 11

    # Strict alternating Markdown <-> Code cells
    for idx, cell in enumerate(nb.cells):
        exp = "markdown" if idx % 2 == 0 else "code"
        assert cell.cell_type == exp, f"Cell {idx} ({cell.get('id')}) is {cell.cell_type}, expected {exp}"

    full_text = "\n".join(c.source for c in nb.cells)
    assert (
        "https://colab.research.google.com/github/thisizkavi-lab/cb1-dynamics/blob/main/notebooks/02_Multi_GPCR_Universality_Colab.ipynb"
        in full_text
    )
    assert "/content/drive/MyDrive/cb1-dynamics-phase4/ws4_universality" in full_text
    assert "stream_and_featurize_receptor" in full_text

    # Strict local hygiene check: 0 .xtc/.dcd trajectory binaries anywhere in repo, and 0 tracked/Phase-4 files > 25 MB
    traj_binaries = [
        p for p in PROJECT_ROOT.rglob("*")
        if p.is_file() and ".git" not in p.parts and p.suffix.lower() in (".xtc", ".dcd", ".trr")
    ]
    assert not traj_binaries, f"Found forbidden local trajectory binaries: {traj_binaries}"

    oversized_files = [
        (p, p.stat().st_size)
        for p in PROJECT_ROOT.rglob("*")
        if p.is_file()
        and ".git" not in p.parts
        and "box_2025" not in p.parts  # pre-existing gitignored Phase 1 Shukla Box MSM pickles used by WS2
        and p.stat().st_size > 25 * 1024 * 1024
    ]
    assert not oversized_files, f"Found local files exceeding 25 MB limit: {oversized_files}"

    ws4_owned_roots = [
        PROJECT_ROOT / "src" / "ws4_gpcr_universality",
        MANIFEST_PATH,
        COLAB_NOTEBOOK_PATH,
        PROJECT_ROOT / "tests" / "test_ws4_universality_setup.py",
    ]
    for root in ws4_owned_roots:
        files = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]
        for fpath in files:
            assert fpath.stat().st_size < 5 * 1024 * 1024, f"WS4 file unexpectedly large: {fpath}"
