"""Step 2 of WS2: Lineage-Stratified Streaming .dcd Featurizer for Shukla 2025/2026 CB1 Box.

Selects a deterministic, lineage-diverse stratified cohort of 160 continuous .dcd
trajectories (from folder 236736306087 `CB1_anandamide_binding`, NEVER `macrostate_rep_dcd`),
enforces binary `DCD CORD` and `UniProt = resSeq + 88` biological residue-identity guards
on every file, computes 3 nested translation/rotation-invariant physical feature tiers
(Tier A: CA_backbone, Tier B: +Sidechain_switches, Tier C: +Env_lipid_sterol_AEA),
deletes each temporary 67-84 MB .dcd immediately after featurization (<200 MB peak disk),
clusters connected lineage components (`pr_<round>` U initial TM heavy-atom RMSD < 1.0 A),
and writes `<20 MB` zstd-compressed `data/processed/ws2/box_aea_cohort_features.parquet`.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import pickle
import re
import tempfile
import time
from typing import Any
import urllib.request

import mdtraj as md
import numpy as np
import pandas as pd

from src.ws2_box_msm.msm_tpt import ORTHOSTERIC_POCKET_UNIPROT, UNIPROT_STRIP_OFFSET_ZERO_IDX

SHARE_TOKEN = "q9w1f99mybxo6hrgo8sfr3sbm3l8xrmu"
CB1_BINDING_FOLDER_ID = "236736306087"
DCD_FILENAME_REGEX = re.compile(r"^CB1_assym_pr_(\d+)_frame_(\d+)-strip\.dcd$")

# Mandatory UniProt = resSeq + 88 (0-based resid = UniProt - 89) residue guards (#17)
EXPECTED_RESIDUES_UNIPROT: dict[int, str] = {
    102: "PHE",
    103: "MET",
    108: "PHE",
    119: "ILE",
    123: "SER",
    155: "PHE",  # resSeq 67  (2.42)
    163: "ASP",  # resSeq 75  (2.50)
    177: "PHE",  # resSeq 89  (2.64)
    193: "LEU",  # resSeq 105 (3.29)
    200: "PHE",  # resSeq 112 (3.36)
    214: "ARG",  # resSeq 126 (3.50)
    237: "PHE",  # resSeq 149 (4.46)
    241: "TRP",  # resSeq 153 (4.50)
    290: "ILE",  # resSeq 202 (5.54)
    294: "TYR",  # resSeq 206 (5.58)
    338: "ASP",  # resSeq 250 (6.30)
    343: "LYS",  # resSeq 255 (6.35)
    356: "TRP",  # resSeq 268 (6.48)
    379: "PHE",  # resSeq 291 (7.35)
    383: "SER",  # resSeq 295 (7.39)
    384: "MET",  # resSeq 296 (7.40)
    393: "ASN",  # resSeq 305 (7.49)
    397: "TYR",  # resSeq 309 (7.53)
}

TIER_A_COLS = [
    "r_portal_A",
    "ca_r214_d338_A",
    "ca_r214_k343_A",
    "ca_f200_w356_A",
    "ca_d163_w356_A",
    "ca_d163_n393_A",
    "ca_f155_f237_A",
    "ca_f155_n393_A",
    "ca_y294_y397_A",
    "ca_i290_y397_A",
    "ca_m103_d163_A",
    "ca_i119_f379_A",
    "ca_s123_m384_A",
]

AROMATIC_SWITCH_UNIPROT = [155, 200, 237, 294, 356, 397]
TIER_B_ADDED_COLS = [
    *(
        f"sc_{fn}_{kind}_U{u}"
        for u in AROMATIC_SWITCH_UNIPROT
        for kind in ("chi1", "chi2")
        for fn in ("sin", "cos")
    ),
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

TIER_C_ADDED_COLS = [
    "env_aea_head_pocket_A",
    "env_aea_tail_pocket_A",
    "env_aea_head_minus_tail_depth_A",
    "env_aea_head_s383_polar_A",
    "env_aea_c5_c10_c18_cos",
    "env_aea_end_to_end_A",
    "env_aea_sec_pocket_A",
    "env_lip_tm17_portal_min_A",
    "env_cer_osm_tm17_min_A",
    "env_plpi_dpps_tm17_min_A",
    "env_glyco_nt_ecl_min_A",
    "env_lip_tm17_contacts_45A",
    "env_lip_tm24_cleft_min_A",
    "env_lip_tm24_contacts_45A",
    "env_aea_lip_shell_min_A",
]


def verify_dcd_binary_header(dcd_path: str | Path, expected_bytes: int | None = None) -> None:
    """Assert valid CHARMM/NAMD 32-bit DCD `CORD` magic header and optional exact byte count."""
    p = Path(dcd_path)
    if not p.exists():
        raise FileNotFoundError(f"DCD file missing: {p}")
    actual_bytes = p.stat().st_size
    if expected_bytes is not None and actual_bytes != expected_bytes:
        raise ValueError(f"DCD size mismatch for {p.name}: got {actual_bytes}, expected {expected_bytes}")
    with open(p, "rb") as fh:
        hdr = fh.read(12)
    if len(hdr) < 8 or hdr[4:8] != b"CORD":
        raise ValueError(f"Invalid DCD magic header in {p.name}: bytes[4:8]={hdr[4:8]!r} != b'CORD'")


def verify_cb1_strip_topology(topology: md.Topology) -> None:
    """Assert exact UniProt = resSeq + 88 amino-acid identities and ligand/lipid composition."""
    if topology.n_residues < 351:
        raise ValueError(f"Expected >=351 residues in CB1_assym-strip topology, got {topology.n_residues}")
    for u_num, expected_name in EXPECTED_RESIDUES_UNIPROT.items():
        r_idx = u_num - UNIPROT_STRIP_OFFSET_ZERO_IDX
        res = topology.residue(r_idx)
        if res.name != expected_name:
            raise ValueError(
                f"Residue identity guard FAILED at UniProt {u_num} (resid={r_idx}, resSeq={r_idx + 1}): "
                f"found {res.name!r}, expected {expected_name!r}"
            )
    # Verify two anandamide ligands at resid 323 and 324
    for r_lig in (323, 324):
        if topology.residue(r_lig).name != "LIG":
            raise ValueError(f"Expected LIG at resid {r_lig}, got {topology.residue(r_lig).name!r}")


def _min_heavy_dist_A(xyz_nm: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray) -> np.ndarray:
    """Vectorized minimum pairwise distance (in Angstroms) between two atom sets across all frames."""
    diff = xyz_nm[:, idx_a, None, :] - xyz_nm[:, None, idx_b, :]
    d2 = np.einsum("fijk,fijk->fij", diff, diff)
    return np.sqrt(np.min(d2, axis=(1, 2))) * 10.0


def _min_dist_and_contacts_A(
    xyz_nm: np.ndarray,
    idx_env: np.ndarray,
    idx_site: np.ndarray,
    cutoff_nm: float = 0.45,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (min_dist_A, n_env_atoms_within_cutoff) in one memory-clean pass over frame chunks."""
    n_f = xyz_nm.shape[0]
    min_d = np.empty(n_f, dtype=np.float32)
    n_cont = np.empty(n_f, dtype=np.float32)
    cut2 = cutoff_nm * cutoff_nm
    chunk = 250
    for s in range(0, n_f, chunk):
        e = min(n_f, s + chunk)
        diff = xyz_nm[s:e, idx_env, None, :] - xyz_nm[s:e, None, idx_site, :]
        d2 = np.einsum("fijk,fijk->fij", diff, diff)
        min_per_env = np.min(d2, axis=2)  # (chunk, n_env)
        min_d[s:e] = np.sqrt(np.min(min_per_env, axis=1)) * 10.0
        n_cont[s:e] = np.sum(min_per_env <= cut2, axis=1, dtype=np.float32)
    return min_d, n_cont


class CB1FeatureIndexCache:
    """Precomputes all atom index arrays once from validated CB1_assym-strip topology."""

    def __init__(self, ref_pdb: Path, prmtop: Path, resid_csv: Path) -> None:
        self.prmtop_str = str(prmtop)
        self.ref_traj = md.load(str(ref_pdb), top=str(prmtop))
        self.top = self.ref_traj.topology
        verify_cb1_strip_topology(self.top)

        self.ca_all = self.top.select("resid 0 to 322 and name CA")
        # Transmembrane heavy atoms from CB1_resid.csv (where Position contains '.') for lineage RMSD
        df_res = pd.read_csv(resid_csv)
        tm_resids = [int(row["OriNo"]) - UNIPROT_STRIP_OFFSET_ZERO_IDX for _, row in df_res.iterrows() if "." in str(row["Position"])]
        tm_sel_str = "resid " + " ".join(str(r) for r in tm_resids) + " and element != H"
        self.tm_heavy_idx = self.top.select(tm_sel_str)

        # Orthosteric pocket CA indices
        self.pocket_ca_idx = np.array(
            [self.top.select(f"resid {u - UNIPROT_STRIP_OFFSET_ZERO_IDX} and name CA")[0] for u in ORTHOSTERIC_POCKET_UNIPROT],
            dtype=int,
        )

        # Tier A CA pairs
        ca_pair_uniprot = [
            (214, 338),  # R_TM6 (3.50-6.30)
            (214, 343),  # 3.50-6.35
            (200, 356),  # 3.36-6.48
            (163, 356),  # 2.50-6.48
            (163, 393),  # 2.50-7.49
            (155, 237),  # 2.42-4.46
            (155, 393),  # 2.42-7.49
            (294, 397),  # 5.58-7.53
            (290, 397),  # 5.54-7.53
            (103, 163),  # N-term M103 - D163
            (119, 379),  # TM1-TM7 outer portal
            (123, 384),  # TM1-TM7 mid portal
        ]
        self.ca_pairs = np.array(
            [
                [
                    self.top.select(f"resid {u1 - UNIPROT_STRIP_OFFSET_ZERO_IDX} and name CA")[0],
                    self.top.select(f"resid {u2 - UNIPROT_STRIP_OFFSET_ZERO_IDX} and name CA")[0],
                ]
                for u1, u2 in ca_pair_uniprot
            ],
            dtype=int,
        )

        # Tier B chi1 / chi2 torsion atom quadruplets
        chi1_quads, _ = md.compute_chi1(self.ref_traj)
        chi2_quads, _ = md.compute_chi2(self.ref_traj)
        c1_list, c2_list = [], []
        for u in AROMATIC_SWITCH_UNIPROT:
            r_idx = u - UNIPROT_STRIP_OFFSET_ZERO_IDX
            m1 = [row for row in chi1_quads if self.top.atom(int(row[1])).residue.index == r_idx]
            m2 = [row for row in chi2_quads if self.top.atom(int(row[1])).residue.index == r_idx]
            if len(m1) != 1 or len(m2) != 1:
                raise ValueError(f"Missing chi1/chi2 for UniProt {u}: {len(m1)}, {len(m2)}")
            c1_list.append(m1[0])
            c2_list.append(m2[0])
        self.chi1_quads = np.array(c1_list, dtype=int)
        self.chi2_quads = np.array(c2_list, dtype=int)

        # Tier B sidechain heavy-atom groups
        def sc_heavy(u: int) -> np.ndarray:
            r = u - UNIPROT_STRIP_OFFSET_ZERO_IDX
            return self.top.select(f"resid {r} and element != H and not (name N or name CA or name C or name O)")

        self.f155_ring = self.top.select(f"resid {155 - 89} and name CG CD1 CD2 CE1 CE2 CZ")
        self.f237_ring = self.top.select(f"resid {237 - 89} and name CG CD1 CD2 CE1 CE2 CZ")
        self.f200_ring = self.top.select(f"resid {200 - 89} and name CG CD1 CD2 CE1 CE2 CZ")
        self.w356_ring = self.top.select(f"resid {356 - 89} and name CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2")
        self.n393_ca = self.top.select(f"resid {393 - 89} and name CA")[0]

        self.f155_sc = sc_heavy(155)
        self.f237_sc = sc_heavy(237)
        self.f200_sc = sc_heavy(200)
        self.w356_sc = sc_heavy(356)
        self.f177_sc = sc_heavy(177)
        self.f379_sc = sc_heavy(379)

        self.y294_oh = self.top.select(f"resid {294 - 89} and name OH")[0]
        self.y397_oh = self.top.select(f"resid {397 - 89} and name OH")[0]
        self.d163_cg = self.top.select(f"resid {163 - 89} and name CG")[0]
        self.n393_cg = self.top.select(f"resid {393 - 89} and name CG")[0]
        self.r214_cz = self.top.select(f"resid {214 - 89} and name CZ")[0]
        self.d338_cg = self.top.select(f"resid {338 - 89} and name CG")[0]

        self.vloop_sc = np.concatenate([sc_heavy(102), sc_heavy(103), sc_heavy(108)])
        self.orth_core_sc = np.concatenate([sc_heavy(177), sc_heavy(356), sc_heavy(379)])

        # Tier C: Two LIG copies (resid 323 and 324)
        self.lig_info = []
        for r_lig in (323, 324):
            all_idx = self.top.select(f"resid {r_lig}")
            masses = np.array([self.top.atom(int(a)).element.mass for a in all_idx], dtype=np.float64)
            masses /= masses.sum()
            heavy_idx = self.top.select(f"resid {r_lig} and element != H")
            head_idx = self.top.select(f"resid {r_lig} and name C20 O1 N C21 C22 O2")
            tail_idx = self.top.select(f"resid {r_lig} and name C1 C2 C3 C4 C5")
            polar_idx = self.top.select(f"resid {r_lig} and name O1 N O2")
            c5 = self.top.select(f"resid {r_lig} and name C5")[0]
            c10 = self.top.select(f"resid {r_lig} and name C10")[0]
            c18 = self.top.select(f"resid {r_lig} and name C18")[0]
            c1 = self.top.select(f"resid {r_lig} and name C1")[0]
            o2 = self.top.select(f"resid {r_lig} and name O2")[0]
            self.lig_info.append(
                {
                    "all_idx": all_idx,
                    "masses": masses,
                    "heavy_idx": heavy_idx,
                    "head_idx": head_idx,
                    "tail_idx": tail_idx,
                    "polar_idx": polar_idx,
                    "c5": c5,
                    "c10": c10,
                    "c18": c18,
                    "c1": c1,
                    "o2": o2,
                }
            )

        self.s383_h178_polar = self.top.select(
            f"(resid {383 - 89} and name OG) or (resid {178 - 89} and name ND1 NE2)"
        )

        # Tier C: Lipid / ceramide / ganglioside shell heavy atoms (resid 325..350)
        self.lip_all_heavy = self.top.select("resid 325 to 350 and element != H")
        self.cer_osm_heavy = self.top.select("(resname CER1 or resname OSM) and element != H")
        self.plpi_dpps_heavy = self.top.select("(resname PLPI or resname DPPS) and element != H")
        self.glyco_heavy = self.top.select("(resname BGLC or resname BGAL or resname ANE5) and element != H")

        # Receptor portal & TM2-TM4 cleft heavy atoms
        portal_u = [119, 123, 177, 193, 379, 383, 384]
        self.tm17_portal_heavy = self.top.select(
            "resid " + " ".join(str(u - 89) for u in portal_u) + " and element != H"
        )
        nt_ecl_u = [102, 103, 108, 116, 177, 366, 373, 379]
        self.nt_ecl_heavy = self.top.select(
            "resid " + " ".join(str(u - 89) for u in nt_ecl_u) + " and element != H"
        )
        tm24_u = [155, 237, 241]
        self.tm24_cleft_heavy = self.top.select(
            "resid " + " ".join(str(u - 89) for u in tm24_u) + " and element != H"
        )


def featurize_single_dcd(
    dcd_path: Path,
    expected_bytes: int,
    expected_frames: int,
    cache: CB1FeatureIndexCache,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Verify binary/biological invariants and compute all Tier A/B/C features + frame-0 TM coords."""
    verify_dcd_binary_header(dcd_path, expected_bytes=expected_bytes)
    t = md.load(str(dcd_path), top=cache.prmtop_str)
    verify_cb1_strip_topology(t.topology)
    if t.n_frames != expected_frames:
        raise ValueError(f"Frame count mismatch in {dcd_path.name}: got {t.n_frames}, expected {expected_frames}")

    # Superpose frame 0 onto reference TM heavy atoms solely to record canonical initial TM coords for lineage RMSD
    t_f0 = t[0]
    t_f0.superpose(cache.ref_traj, 0, cache.tm_heavy_idx, cache.tm_heavy_idx)
    x0_tm_heavy_nm = t_f0.xyz[0, cache.tm_heavy_idx, :].astype(np.float32).copy()

    # All internal features are strictly translation/rotation invariant directly on raw coordinates!
    xyz = t.xyz  # (n_frames, n_atoms, 3) in nm
    n_f = t.n_frames
    pocket_center = xyz[:, cache.pocket_ca_idx, :].mean(axis=1)  # (n_f, 3)

    # Compute pocket COM distance for both LIG 323 and LIG 324 -> permutation-invariant primary/secondary assignment
    lig_coms = []
    lig_dists_A = []
    for li in cache.lig_info:
        com = np.einsum("fia,i->fa", xyz[:, li["all_idx"], :], li["masses"])
        d_A = np.linalg.norm(com - pocket_center, axis=1) * 10.0
        lig_coms.append(com)
        lig_dists_A.append(d_A)

    prim_is_1 = lig_dists_A[1] < lig_dists_A[0]  # True where LIG 324 is closer to pocket than LIG 323
    r_portal_A = np.where(prim_is_1, lig_dists_A[1], lig_dists_A[0]).astype(np.float32)
    r_sec_A = np.where(prim_is_1, lig_dists_A[0], lig_dists_A[1]).astype(np.float32)

    feats: dict[str, np.ndarray] = {"r_portal_A": r_portal_A}

    # Tier A: 12 CA-CA distances (A)
    ca_dists_A = md.compute_distances(t, cache.ca_pairs, periodic=False) * 10.0
    for col_idx, col_name in enumerate(TIER_A_COLS[1:]):
        feats[col_name] = ca_dists_A[:, col_idx].astype(np.float32)

    # Tier B: 6 aromatic switches x (sin/cos chi1, sin/cos chi2)
    chi1_rad = md.compute_dihedrals(t, cache.chi1_quads, periodic=False)
    chi2_rad = md.compute_dihedrals(t, cache.chi2_quads, periodic=False)
    for i_u, u in enumerate(AROMATIC_SWITCH_UNIPROT):
        feats[f"sc_sin_chi1_U{u}"] = np.sin(chi1_rad[:, i_u]).astype(np.float32)
        feats[f"sc_cos_chi1_U{u}"] = np.cos(chi1_rad[:, i_u]).astype(np.float32)
        feats[f"sc_sin_chi2_U{u}"] = np.sin(chi2_rad[:, i_u]).astype(np.float32)
        feats[f"sc_cos_chi2_U{u}"] = np.cos(chi2_rad[:, i_u]).astype(np.float32)

    # Tier B: Sidechain centroids and heavy-atom distances
    f155_c = xyz[:, cache.f155_ring, :].mean(axis=1)
    f237_c = xyz[:, cache.f237_ring, :].mean(axis=1)
    f200_c = xyz[:, cache.f200_ring, :].mean(axis=1)
    w356_c = xyz[:, cache.w356_ring, :].mean(axis=1)

    feats["sc_f155_n393ca_A"] = (np.linalg.norm(f155_c - xyz[:, cache.n393_ca, :], axis=1) * 10.0).astype(np.float32)
    feats["sc_f155_f237_ring_A"] = (np.linalg.norm(f155_c - f237_c, axis=1) * 10.0).astype(np.float32)
    feats["sc_f155_f237_min_A"] = _min_heavy_dist_A(xyz, cache.f155_sc, cache.f237_sc).astype(np.float32)
    feats["sc_f200_w356_min_A"] = _min_heavy_dist_A(xyz, cache.f200_sc, cache.w356_sc).astype(np.float32)
    feats["sc_f200_w356_ring_A"] = (np.linalg.norm(f200_c - w356_c, axis=1) * 10.0).astype(np.float32)

    atom_pairs_B = np.array(
        [
            [cache.y294_oh, cache.y397_oh],
            [cache.d163_cg, cache.n393_cg],
            [cache.r214_cz, cache.d338_cg],
        ],
        dtype=int,
    )
    b_pair_A = md.compute_distances(t, atom_pairs_B, periodic=False) * 10.0
    feats["sc_y294_y397_oh_A"] = b_pair_A[:, 0].astype(np.float32)
    feats["sc_d163_n393_polar_A"] = b_pair_A[:, 1].astype(np.float32)
    feats["sc_r214_d338_ionic_A"] = b_pair_A[:, 2].astype(np.float32)
    feats["sc_vloop_pocket_min_A"] = _min_heavy_dist_A(xyz, cache.vloop_sc, cache.orth_core_sc).astype(np.float32)
    feats["sc_portal_f177_f379_min_A"] = _min_heavy_dist_A(xyz, cache.f177_sc, cache.f379_sc).astype(np.float32)

    # Tier C: Primary & secondary AEA pose/internal features (computed for both LIG copies, selected by prim_is_1)
    head_d, tail_d, polar_s383, cos_ang, e2e_d, aea_lip_d = [], [], [], [], [], []
    for li in cache.lig_info:
        hc = xyz[:, li["head_idx"], :].mean(axis=1)
        tc = xyz[:, li["tail_idx"], :].mean(axis=1)
        head_d.append(np.linalg.norm(hc - pocket_center, axis=1) * 10.0)
        tail_d.append(np.linalg.norm(tc - pocket_center, axis=1) * 10.0)
        polar_s383.append(_min_heavy_dist_A(xyz, li["polar_idx"], cache.s383_h178_polar))
        ang = md.compute_angles(t, np.array([[li["c5"], li["c10"], li["c18"]]], dtype=int), periodic=False)[:, 0]
        cos_ang.append(np.cos(ang))
        e2e_d.append(np.linalg.norm(xyz[:, li["o2"], :] - xyz[:, li["c1"], :], axis=1) * 10.0)
        d_al, _ = _min_dist_and_contacts_A(xyz, cache.lip_all_heavy, li["heavy_idx"], cutoff_nm=0.45)
        aea_lip_d.append(d_al)

    hp = np.where(prim_is_1, head_d[1], head_d[0]).astype(np.float32)
    tp = np.where(prim_is_1, tail_d[1], tail_d[0]).astype(np.float32)
    feats["env_aea_head_pocket_A"] = hp
    feats["env_aea_tail_pocket_A"] = tp
    feats["env_aea_head_minus_tail_depth_A"] = (hp - tp).astype(np.float32)
    feats["env_aea_head_s383_polar_A"] = np.where(prim_is_1, polar_s383[1], polar_s383[0]).astype(np.float32)
    feats["env_aea_c5_c10_c18_cos"] = np.where(prim_is_1, cos_ang[1], cos_ang[0]).astype(np.float32)
    feats["env_aea_end_to_end_A"] = np.where(prim_is_1, e2e_d[1], e2e_d[0]).astype(np.float32)
    feats["env_aea_sec_pocket_A"] = r_sec_A

    # Tier C: Explicit neuronal lipid / ceramide / ganglioside shell wedging
    lip_tm17_min, lip_tm17_cnt = _min_dist_and_contacts_A(xyz, cache.lip_all_heavy, cache.tm17_portal_heavy, 0.45)
    cer_tm17_min, _ = _min_dist_and_contacts_A(xyz, cache.cer_osm_heavy, cache.tm17_portal_heavy, 0.45)
    plpi_tm17_min, _ = _min_dist_and_contacts_A(xyz, cache.plpi_dpps_heavy, cache.tm17_portal_heavy, 0.45)
    glyco_nt_min, _ = _min_dist_and_contacts_A(xyz, cache.glyco_heavy, cache.nt_ecl_heavy, 0.45)
    lip_tm24_min, lip_tm24_cnt = _min_dist_and_contacts_A(xyz, cache.lip_all_heavy, cache.tm24_cleft_heavy, 0.45)

    feats["env_lip_tm17_portal_min_A"] = lip_tm17_min
    feats["env_cer_osm_tm17_min_A"] = cer_tm17_min
    feats["env_plpi_dpps_tm17_min_A"] = plpi_tm17_min
    feats["env_glyco_nt_ecl_min_A"] = glyco_nt_min
    feats["env_lip_tm17_contacts_45A"] = lip_tm17_cnt
    feats["env_lip_tm24_cleft_min_A"] = lip_tm24_min
    feats["env_lip_tm24_contacts_45A"] = lip_tm24_cnt
    feats["env_aea_lip_shell_min_A"] = np.where(prim_is_1, aea_lip_d[1], aea_lip_d[0]).astype(np.float32)

    # Verify zero NaNs or Infs across all 62 extracted physical features and quantize float32 noise
    # below 2^-9 (~0.00195 A / rad-sin, >50x finer than MD thermal precision) so zstd Parquet stays ~12 MB (<20 MB).
    for k, arr in feats.items():
        if arr.shape != (n_f,) or not np.all(np.isfinite(arr)):
            raise ValueError(f"Non-finite or malformed feature {k} in {dcd_path.name}")
        feats[k] = np.ldexp(np.round(np.ldexp(arr.astype(np.float64), 9)), -9).astype(np.float32)

    return feats, x0_tm_heavy_nm


def select_stratified_cohort(
    catalog_items: list[dict[str, Any]],
    dtrajs: list[np.ndarray],
    lig_com_list: list[np.ndarray],
    q_msm_phys_map: np.ndarray,
    pocket_center_nm: np.ndarray,
    seed: int = 20261112,
) -> list[dict[str, Any]]:
    """Select 160 lineage-diverse continuous .dcd runs oversampling transition portal/vestibule dynamics."""
    rng = np.random.default_rng(seed)
    sorted_cat = sorted(catalog_items, key=lambda x: (x["pr_round"], x["frame_seed"]))
    if len(sorted_cat) != 5000:
        raise ValueError(f"Expected 5000 catalog items, got {len(sorted_cat)}")

    pools: dict[str, list[dict[str, Any]]] = {
        "I_to_B1": [],
        "Portal_to_BulkA": [],
        "Portal_AB_Explorer": [],
        "Deep_B1": [],
        "Deep_BulkA": [],
    }

    for msm_idx, (item, dt, lc) in enumerate(zip(sorted_cat, dtrajs, lig_com_list)):
        name = item["name"]
        if not DCD_FILENAME_REGEX.match(name) or "macrostate_rep" in name:
            raise ValueError(f"Forbidden DCD filename in CB1_anandamide_binding catalog: {name}")
        if len(dt) < 800:
            continue
        q_tr = q_msm_phys_map[dt]
        if np.any(np.isnan(q_tr)):
            continue
        r_tr = np.linalg.norm(lc - pocket_center_nm[None, :], axis=1) * 10.0

        rec = {
            **item,
            "msm_traj_idx": msm_idx,
            "n_frames": int(len(dt)),
            "q_min": float(q_tr.min()),
            "q_max": float(q_tr.max()),
            "q_mean": float(q_tr.mean()),
            "r_min_A": float(r_tr.min()),
            "r_max_A": float(r_tr.max()),
        }
        in_I = (q_tr > 0.05) & (q_tr < 0.95)
        in_B = (q_tr >= 0.95) | (r_tr <= 8.5)
        if in_I.sum() >= 30 and in_B.sum() >= 20:
            pools["I_to_B1"].append(rec)
        elif (r_tr < 18.0).sum() >= 30 and (r_tr >= 22.0).sum() >= 30:
            pools["Portal_to_BulkA"].append(rec)
        elif ((q_tr > 0.02) & (q_tr < 0.95)).sum() >= 100:
            pools["Portal_AB_Explorer"].append(rec)
        elif in_B.mean() >= 0.80:
            pools["Deep_B1"].append(rec)
        elif (r_tr >= 22.0).mean() >= 0.95:
            pools["Deep_BulkA"].append(rec)

    quotas = {
        "I_to_B1": 52,
        "Portal_to_BulkA": 44,
        "Portal_AB_Explorer": 44,
        "Deep_B1": 10,
        "Deep_BulkA": 10,
    }

    selected: list[dict[str, Any]] = []
    used_msm_idx: set[int] = set()
    pr_round_counts: dict[int, int] = {}

    for stratum, target_n in quotas.items():
        candidates = list(pools[stratum])
        rng.shuffle(candidates)
        # Prioritize unseen pr_rounds first to maximize lineage diversity, then allow <= 2 per pr_round
        for max_per_pr in (1, 2, 4):
            for rec in candidates:
                if len([x for x in selected if x["stratum"] == stratum]) >= target_n:
                    break
                mid = rec["msm_traj_idx"]
                pr = int(rec["pr_round"])
                if mid in used_msm_idx:
                    continue
                if pr_round_counts.get(pr, 0) < max_per_pr:
                    rec_copy = dict(rec)
                    rec_copy["stratum"] = stratum
                    selected.append(rec_copy)
                    used_msm_idx.add(mid)
                    pr_round_counts[pr] = pr_round_counts.get(pr, 0) + 1

    selected.sort(key=lambda x: x["msm_traj_idx"])
    for c_idx, rec in enumerate(selected):
        rec["cohort_idx"] = c_idx
    return selected


def compute_connected_lineage_groups(
    pr_rounds: list[int],
    x0_tm_heavy_nm: np.ndarray,
    rmsd_cutoff_A: float = 1.0,
) -> tuple[list[int], dict[str, Any]]:
    """Build connected lineage components merging shared `pr_<round>` OR initial TM heavy-atom RMSD < 1.0 A."""
    n = len(pr_rounds)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # 1. Merge by shared pr_<round>
    pr_edges = 0
    for i in range(n):
        for j in range(i + 1, n):
            if pr_rounds[i] == pr_rounds[j]:
                union(i, j)
                pr_edges += 1

    # 2. Pairwise Kabsch-aligned heavy-atom TM RMSD on frame 0
    # Load into an mdtraj Trajectory using first frame topology slice is unnecessary: Kabsch via centered SVD!
    coords = x0_tm_heavy_nm.astype(np.float64) - x0_tm_heavy_nm.mean(axis=1, keepdims=True)
    n_atoms = coords.shape[1]
    rmsd_mat_A = np.zeros((n, n), dtype=np.float64)
    rmsd_edges = 0
    norms = np.sum(coords * coords, axis=(1, 2))
    for i in range(n):
        xi = coords[i]
        for j in range(i + 1, n):
            xj = coords[j]
            H = xi.T @ xj
            U, S, Vt = np.linalg.svd(H)
            d = np.sign(np.linalg.det(U @ Vt))
            S_signed = S[0] + S[1] + d * S[2]
            msd = max(0.0, (norms[i] + norms[j] - 2.0 * S_signed) / n_atoms)
            r_A = float(np.sqrt(msd) * 10.0)
            rmsd_mat_A[i, j] = rmsd_mat_A[j, i] = r_A
            if r_A < rmsd_cutoff_A:
                union(i, j)
                rmsd_edges += 1

    roots = [find(i) for i in range(n)]
    unique_roots = sorted(set(roots))
    root_to_gid = {r: gid for gid, r in enumerate(unique_roots)}
    group_ids = [root_to_gid[r] for r in roots]

    off_diag = rmsd_mat_A[np.triu_indices(n, k=1)]
    stats = {
        "n_trajectories": n,
        "n_distinct_pr_rounds": len(set(pr_rounds)),
        "n_connected_lineage_groups": len(unique_roots),
        "n_shared_pr_pairs": pr_edges,
        "n_rmsd_lt_1A_pairs": rmsd_edges,
        "initial_tm_rmsd_min_A": float(off_diag.min()) if len(off_diag) else 0.0,
        "initial_tm_rmsd_median_A": float(np.median(off_diag)) if len(off_diag) else 0.0,
        "initial_tm_rmsd_p05_A": float(np.percentile(off_diag, 5)) if len(off_diag) else 0.0,
    }
    return group_ids, stats


def _download_and_featurize_worker(
    rec: dict[str, Any],
    temp_dir: Path,
    cache: CB1FeatureIndexCache,
) -> tuple[int, dict[str, np.ndarray], np.ndarray]:
    """Download one .dcd to temp_dir, run guards & featurization, and IMMEDIATELY unlink temp .dcd."""
    dl_url = (
        f"https://uofi.app.box.com/index.php?rm=box_download_shared_file"
        f"&shared_name={SHARE_TOKEN}&file_id={rec['typed_id']}"
    )
    fd, tmp_str = tempfile.mkstemp(prefix=f"ws2_{rec['cohort_idx']:03d}_", suffix=".dcd", dir=str(temp_dir))
    os.close(fd)
    tmp_path = Path(tmp_str)
    try:
        for attempt in range(4):
            try:
                req = urllib.request.Request(dl_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=90) as resp, open(tmp_path, "wb") as out_f:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        out_f.write(chunk)
                if tmp_path.stat().st_size == rec["size"]:
                    break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(1.5 * (attempt + 1))

        feats, x0_tm = featurize_single_dcd(
            dcd_path=tmp_path,
            expected_bytes=int(rec["size"]),
            expected_frames=int(rec["n_frames"]),
            cache=cache,
        )
        return int(rec["cohort_idx"]), feats, x0_tm
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def run_streaming_featurization(
    box_dir: Path,
    out_dir: Path,
    max_workers: int = 2,
    limit_test: int | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog_data = json.loads((box_dir / "box_cb1_dcd_catalog.json").read_text())
    if str(catalog_data["folder_id"]) != CB1_BINDING_FOLDER_ID:
        raise ValueError(f"Catalog folder_id mismatch: {catalog_data['folder_id']}")

    with open(box_dir / "CB1_frame_cluster_indices.pkl", "rb") as fh:
        dtrajs: list[np.ndarray] = pickle.load(fh)
    with open(box_dir / "CB1_lig_com_cor.pkl", "rb") as fh:
        lig_com_list: list[np.ndarray] = pickle.load(fh)
    with open(box_dir / "CB1_msm_weight.pkl", "rb") as fh:
        msm_weights_list: list[np.ndarray] = pickle.load(fh)

    df_micro = pd.read_parquet(out_dir / "msm_800_microstates.parquet")
    q_phys_map = df_micro["q_msm_phys"].to_numpy()
    q_pureR_map = df_micro["q_msm_pureR"].to_numpy()
    q_macro_map = df_micro["q_msm_macro"].to_numpy()
    mtM_map = df_micro["macrostate_id"].to_numpy()

    cache = CB1FeatureIndexCache(
        ref_pdb=box_dir / "CB1_assym-strip.pdb",
        prmtop=box_dir / "CB1_assym-strip.prmtop",
        resid_csv=box_dir / "CB1_resid.csv",
    )
    pocket_center_ref_nm = cache.ref_traj.xyz[0, cache.pocket_ca_idx, :].mean(axis=0)

    cohort = select_stratified_cohort(
        catalog_items=catalog_data["items"],
        dtrajs=dtrajs,
        lig_com_list=lig_com_list,
        q_msm_phys_map=q_phys_map,
        pocket_center_nm=pocket_center_ref_nm,
    )
    if limit_test is not None:
        cohort = cohort[:limit_test]

    temp_dir = box_dir / "_tmp_stream"
    temp_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    results_by_cidx: dict[int, tuple[dict[str, np.ndarray], np.ndarray]] = {}
    # max_workers=2 guarantees <= 2 * 84 MB = 168 MB (< 200 MB) peak temporary .dcd usage!
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_download_and_featurize_worker, rec, temp_dir, cache): rec for rec in cohort}
        done_cnt = 0
        for fut in concurrent.futures.as_completed(futs):
            c_idx, feats, x0_tm = fut.result()
            results_by_cidx[c_idx] = (feats, x0_tm)
            done_cnt += 1
            if done_cnt % 10 == 0 or done_cnt == len(cohort):
                elapsed = time.time() - t0
                print(
                    f"[WS2 Stream] {done_cnt:3d}/{len(cohort)} .dcd trajectories featurized & unlinked "
                    f"({elapsed:.1f}s elapsed, {elapsed / done_cnt:.2f}s/traj)",
                    flush=True,
                )

    # Remove empty temp_dir
    for leftover in temp_dir.glob("*.dcd"):
        leftover.unlink()
    if temp_dir.exists() and not any(temp_dir.iterdir()):
        temp_dir.rmdir()

    # Compute connected lineage groups across streamed cohort
    x0_stack = np.stack([results_by_cidx[rec["cohort_idx"]][1] for rec in cohort], axis=0)
    pr_list = [int(rec["pr_round"]) for rec in cohort]
    lineage_gids, lineage_stats = compute_connected_lineage_groups(pr_list, x0_stack, rmsd_cutoff_A=1.0)

    # Assemble DataFrame in deterministic cohort order
    dfs: list[pd.DataFrame] = []
    com_r_corr_list: list[float] = []
    for rec, gid in zip(cohort, lineage_gids):
        c_idx = int(rec["cohort_idx"])
        m_idx = int(rec["msm_traj_idx"])
        feats, _ = results_by_cidx[c_idx]
        dt = dtrajs[m_idx]
        n_f = len(dt)
        lc = lig_com_list[m_idx]
        r_pkl_raw = np.linalg.norm(lc - pocket_center_ref_nm[None, :], axis=1) * 10.0
        r_pkl_A = np.ldexp(np.round(np.ldexp(r_pkl_raw, 9)), -9).astype(np.float32)

        # Correlation check between superposed pickle R_portal and exact internal primary-AEA R_portal
        if np.std(r_pkl_A) > 1e-3 and np.std(feats["r_portal_A"]) > 1e-3:
            com_r_corr_list.append(float(np.corrcoef(r_pkl_A, feats["r_portal_A"])[0, 1]))

        meta_cols = {
            "cohort_idx": np.full(n_f, c_idx, dtype=np.int32),
            "msm_traj_idx": np.full(n_f, m_idx, dtype=np.int32),
            "pr_round": np.full(n_f, int(rec["pr_round"]), dtype=np.int32),
            "frame_seed": np.full(n_f, int(rec["frame_seed"]), dtype=np.int32),
            "lineage_group_id": np.full(n_f, int(gid), dtype=np.int32),
            "stratum": [rec["stratum"]] * n_f,
            "dcd_name": [rec["name"]] * n_f,
            "frame_idx": np.arange(n_f, dtype=np.int32),
            "time_ns": (np.arange(n_f, dtype=np.float32) * np.float32(0.1)),
            "n_traj_frames": np.full(n_f, n_f, dtype=np.int32),
            "microstate_id": dt.astype(np.int32),
            "macrostate_id": mtM_map[dt].astype(np.int32),
            "msm_weight": msm_weights_list[m_idx].astype(np.float32),
            "q_msm_phys": q_phys_map[dt].astype(np.float32),
            "q_msm_pureR": q_pureR_map[dt].astype(np.float32),
            "q_msm_macro": q_macro_map[dt].astype(np.float32),
            "r_portal_pkl_A": r_pkl_A,
        }
        dfs.append(pd.DataFrame({**meta_cols, **feats}))

    df_all = pd.concat(dfs, ignore_index=True)
    # Convert repeated strings to dictionary categoricals for ultra-compact Parquet storage
    df_all["stratum"] = df_all["stratum"].astype("category")
    df_all["dcd_name"] = df_all["dcd_name"].astype("category")

    all_feature_cols = TIER_A_COLS + TIER_B_ADDED_COLS + TIER_C_ADDED_COLS
    nan_counts = {c: int(df_all[c].isna().sum()) for c in all_feature_cols}
    total_nans = sum(nan_counts.values())
    if total_nans != 0:
        raise ValueError(f"Found {total_nans} NaNs in extracted features!")

    out_name = "box_aea_cohort_features.parquet" if limit_test is None else "_smoke_test_features.parquet"
    parquet_path = out_dir / out_name
    df_all.to_parquet(parquet_path, compression="zstd", compression_level=15, index=False)
    parquet_mb = parquet_path.stat().st_size / (1024 * 1024)

    manifest = {
        "n_trajectories": len(cohort),
        "total_frames": int(len(df_all)),
        "aggregate_time_us": round(len(df_all) * 0.1 / 1000.0, 3),
        "parquet_path": str(parquet_path),
        "parquet_size_bytes": int(parquet_path.stat().st_size),
        "parquet_size_mb": round(parquet_mb, 3),
        "total_feature_nan_count": total_nans,
        "tier_a_cols": TIER_A_COLS,
        "tier_b_added_cols": TIER_B_ADDED_COLS,
        "tier_c_added_cols": TIER_C_ADDED_COLS,
        "stratum_trajectory_counts": pd.Series([r["stratum"] for r in cohort]).value_counts().to_dict(),
        "stratum_frame_counts": {str(k): int(v) for k, v in df_all["stratum"].value_counts().items()},
        "lineage_summary": lineage_stats,
        "median_r_portal_pkl_vs_dcd_corr": float(np.median(com_r_corr_list)) if com_r_corr_list else None,
        "cohort_records": [
            {
                **r,
                "lineage_group_id": int(gid),
            }
            for r, gid in zip(cohort, lineage_gids)
        ],
    }

    if limit_test is None:
        with open(out_dir / "cohort_featurization_manifest.json", "w") as fh:
            json.dump(manifest, fh, indent=2)
    else:
        parquet_path.unlink(missing_ok=True)

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--limit-test", type=int, default=None)
    args = parser.parse_args()

    root = Path("/Users/kavi/Documents/artico/cb1-dynamics")
    res = run_streaming_featurization(
        box_dir=root / "data/inventory/box_2025",
        out_dir=root / "data/processed/ws2",
        max_workers=args.max_workers,
        limit_test=args.limit_test,
    )
    print(
        f"Streaming featurization finished: {res['n_trajectories']} runs, "
        f"{res['total_frames']} frames, parquet={res['parquet_size_mb']} MB, "
        f"NaNs={res['total_feature_nan_count']}, lineage_groups={res['lineage_summary']['n_connected_lineage_groups']}"
    )
