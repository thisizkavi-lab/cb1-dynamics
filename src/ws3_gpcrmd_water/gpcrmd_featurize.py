"""Streaming downloader, binary/biological validator, and explicit-solvent featurizer for GPCRmd Open 6.

Streams the 6 verified open all-atom CB1 bilayer trajectories from GPCRmd (8.97 us, 20,501 frames):
  - ID 1201: 6N4B + MDMB-FUBINACA + cognate Gi1 (1.0 us, 5,001 frames, KCl buffer)
  - ID 2067: 6N4B apo + non-cognate Gs (6.0 us, 6,000 frames, NaCl buffer)
  - ID 22  : 5U09 + taranabant rep A (0.5 us, 2,500 frames, NaCl buffer)
  - ID 164 : 5U09 + taranabant rep B (0.5 us, 2,500 frames, NaCl buffer)
  - ID 162 : 5TGZ apo (0.467 us, 2,000 frames, NaCl buffer)
  - ID 163 : 5U09 apo (0.5 us, 2,500 frames, NaCl buffer)

Mandatory Guardrails Enforced on Every Trajectory:
  1. HTTP download byte-size check + XTC binary header assertion (magic == 1995 & natoms match).
  2. Strict CB1 chain isolation (preventing Galpha_s resSeq collision in 2067 and handling TER-split
     TM1-5 / TM6-H8 chains in 22, 164, 162, 163) + exact P21554 residue identity assertions across
     all 12 core microswitches.
  3. Exact periodic boundary condition (PBC) minimum-image wrapping around the CB1 7TM core centroid
     before computing inter-chain protein, water-wire, Na+/K+, and POPC wedging coordinates.
  4. Zero-NaN encoding (explicitly documenting disconnected water-wire conductance = 0.0, hops = -1,
     and separate Na+ vs K+ channels with sentinel 99.0 A when ion species is absent).
  5. Immediate deletion of multi-GB raw .xtc files in data/inventory/gpcrmd_temp/ after verification.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
import time
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import mdtraj as md
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


PROJECT_ROOT = Path("/Users/kavi/Documents/artico/cb1-dynamics")
MANIFEST_PATH = PROJECT_ROOT / "data" / "inventory" / "datasets_manifest.json"
TEMP_XTC_DIR = PROJECT_ROOT / "data" / "inventory" / "gpcrmd_temp"
TOPO_DIR = PROJECT_ROOT / "data" / "processed" / "ws3" / "topologies"
OUT_PARQUET = PROJECT_ROOT / "data" / "processed" / "ws3" / "gpcrmd_open6_features.parquet"
PROVENANCE_JSON = PROJECT_ROOT / "data" / "processed" / "ws3" / "gpcrmd_open6_provenance.json"

# Explicit CB1 chain indices in mdtraj topology (verified against P21554 sequence)
GPCRMD_SPECS: dict[int, dict[str, Any]] = {
    1201: {
        "cb1_chains": [3],
        "state_label": "Active_Gi1_FUB",
        "transducer_type": "Gi1",
        "ligand_label": "MDMB-FUBINACA",
        "dt_ns": 0.2,
    },
    2067: {
        "cb1_chains": [0],
        "state_label": "Active_Gs_Apo",
        "transducer_type": "Gs",
        "ligand_label": "Apo",
        "dt_ns": 1.0,
    },
    22: {
        "cb1_chains": [0, 1],
        "state_label": "Inactive_5U09_Taranabant_A",
        "transducer_type": "None",
        "ligand_label": "Taranabant",
        "dt_ns": 0.2,
    },
    164: {
        "cb1_chains": [0, 1],
        "state_label": "Inactive_5U09_Taranabant_B",
        "transducer_type": "None",
        "ligand_label": "Taranabant",
        "dt_ns": 0.2,
    },
    162: {
        "cb1_chains": [0, 1],
        "state_label": "Inactive_5TGZ_Apo",
        "transducer_type": "None",
        "ligand_label": "Apo",
        "dt_ns": 0.2,
    },
    163: {
        "cb1_chains": [0, 1],
        "state_label": "Inactive_5U09_Apo",
        "transducer_type": "None",
        "ligand_label": "Apo",
        "dt_ns": 0.2,
    },
}

REQUIRED_CB1_RESIDUES = {
    155: "PHE",
    163: "ASP",
    173: "SER",
    200: "PHE",
    204: "VAL",
    214: "ARG",
    237: "PHE",
    268: "PHE",
    294: "TYR",
    300: "LYS",
    338: "ASP",
    341: "LEU",
    343: "LYS",
    356: "TRP",
    383: "SER",
    393: "ASN",
    397: "TYR",
}

RING_ATOM_NAMES = {
    "PHE": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "TRP": ("CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    "TYR": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
}

CHI_ATOM_NAMES = {
    "PHE": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
    "TRP": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
}


def download_with_retry(
    url: str,
    dest: Path,
    expected_bytes: int | None = None,
    retries: int = 5,
    n_workers: int = 20,
    slice_bytes: int = 16 * 1024 * 1024,
) -> dict[str, Any]:
    """Download file via parallel HTTP 206 Range streams (using os.pwrite) to overcome high-RTT single-TCP throttling."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if expected_bytes is not None and dest.exists() and dest.stat().st_size == expected_bytes:
        h = hashlib.sha256()
        with open(dest, "rb") as fh:
            h.update(fh.read(1024 * 1024))
        return {"size_bytes": expected_bytes, "head_1mb_sha256": h.hexdigest()}

    # Small files (< 32 MiB) or unknown size: single-stream download
    if expected_bytes is None or expected_bytes <= 2 * slice_bytes:
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (CB1-WS3-Audit/1.0)"})
                with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as fh:
                    shutil.copyfileobj(resp, fh, length=4 * 1024 * 1024)
                if expected_bytes is not None and dest.stat().st_size != expected_bytes:
                    raise IOError(f"Size mismatch for {dest.name}: {dest.stat().st_size} != {expected_bytes}")
                break
            except Exception as exc:
                if attempt == retries - 1:
                    raise RuntimeError(f"Failed downloading {url}: {exc}") from exc
                time.sleep(1.5 * (attempt + 1))
    else:
        # Parallel Range download into pre-allocated temporary file, renamed atomically on completion
        part_path = dest.with_suffix(dest.suffix + ".part")
        if dest.exists():
            dest.unlink()
        fd = os.open(str(part_path), os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o644)
        try:
            os.ftruncate(fd, expected_bytes)
            ranges: list[tuple[int, int]] = []
            for start in range(0, expected_bytes, slice_bytes):
                end = min(start + slice_bytes - 1, expected_bytes - 1)
                ranges.append((start, end))

            def fetch_slice(start_end: tuple[int, int]) -> int:
                s_byte, e_byte = start_end
                want = e_byte - s_byte + 1
                for att in range(retries):
                    try:
                        headers = {
                            "User-Agent": "Mozilla/5.0 (CB1-WS3-Audit/1.0)",
                            "Range": f"bytes={s_byte}-{e_byte}",
                        }
                        req = urllib.request.Request(url, headers=headers)
                        with urllib.request.urlopen(req, timeout=120) as resp:
                            if resp.getcode() != 206:
                                raise IOError(f"Expected HTTP 206 for Range {s_byte}-{e_byte}, got {resp.getcode()}")
                            offset = s_byte
                            rem = want
                            while rem > 0:
                                buf = resp.read(min(1024 * 1024, rem))
                                if not buf:
                                    raise IOError(f"Premature EOF at offset {offset} (rem={rem})")
                                written = os.pwrite(fd, buf, offset)
                                if written != len(buf):
                                    raise IOError(f"Short pwrite: {written} != {len(buf)}")
                                offset += written
                                rem -= written
                        return want
                    except Exception as exc:
                        if att == retries - 1:
                            raise RuntimeError(f"Range {s_byte}-{e_byte} failed on {dest.name}: {exc}") from exc
                        time.sleep(1.5 * (att + 1))
                return 0

            done_bytes = 0
            next_report = 0.25
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futs = [pool.submit(fetch_slice, r) for r in ranges]
                for fut in as_completed(futs):
                    done_bytes += fut.result()
                    frac = done_bytes / expected_bytes
                    if frac >= next_report:
                        print(f"    [{dest.name}] downloaded {done_bytes / 1e6:.1f}/{expected_bytes / 1e6:.1f} MB ({100*frac:.0f}%)", flush=True)
                        next_report += 0.25
        finally:
            os.close(fd)

        if part_path.stat().st_size != expected_bytes:
            raise IOError(f"Incomplete parallel download for {dest.name}")
        part_path.rename(dest)

    h = hashlib.sha256()
    with open(dest, "rb") as fh:
        h.update(fh.read(1024 * 1024))
    return {"size_bytes": dest.stat().st_size, "head_1mb_sha256": h.hexdigest()}


def verify_xtc_magic(xtc_path: Path, expected_natoms: int) -> dict[str, int]:
    """Verify GROMACS XTC magic number (1995) and atom count on binary header."""
    with open(xtc_path, "rb") as fh:
        head = fh.read(16)
    if len(head) < 16:
        raise ValueError(f"XTC file too short: {xtc_path}")
    magic, natoms, step0 = struct.unpack(">iii", head[:12])
    t0_ps = struct.unpack(">f", head[12:16])[0]
    if magic != 1995:
        raise AssertionError(
            f"XTC magic guard failed on {xtc_path.name}: magic={magic} (expected 1995; likely HTML login redirect!)"
        )
    if natoms != expected_natoms:
        raise AssertionError(
            f"XTC natoms mismatch on {xtc_path.name}: header={natoms} vs topology={expected_natoms}"
        )
    return {"magic": int(magic), "natoms": int(natoms), "step0": int(step0), "t0_ps": float(t0_ps)}


def calc_dihedrals_batch_deg(coords: np.ndarray, idx4: list[int]) -> np.ndarray:
    """Vectorized dihedral angle computation in degrees [-180, 180] over (N_frames, N_atoms, 3)."""
    p0 = coords[:, idx4[0], :]
    p1 = coords[:, idx4[1], :]
    p2 = coords[:, idx4[2], :]
    p3 = coords[:, idx4[3], :]
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1_norm = np.linalg.norm(b1, axis=1, keepdims=True)
    b1_u = b1 / np.clip(b1_norm, 1e-8, None)
    v = b0 - np.sum(b0 * b1_u, axis=1, keepdims=True) * b1_u
    w = b2 - np.sum(b2 * b1_u, axis=1, keepdims=True) * b1_u
    x = np.sum(v * w, axis=1)
    y = np.sum(np.cross(b1_u, v) * w, axis=1)
    return np.degrees(np.arctan2(y, x))


def bfs_shortest_water_hops(
    wat_pos: np.ndarray,
    wat_adj: list[list[int]],
    src_atoms: np.ndarray,
    dst_atoms: np.ndarray,
    cutoff_A: float = 3.5,
) -> int:
    """Compute minimum number of bridging water molecules between src_atoms and dst_atoms.

    Returns:
      - 0 if min distance(src_atoms, dst_atoms) <= cutoff_A (direct polar H-bond),
      - k >= 1 if shortest H-bonded water chain uses k water oxygens,
      - -1 if no connected H-bond path exists.
    """
    direct_d = float(np.min(np.linalg.norm(src_atoms[:, None, :] - dst_atoms[None, :, :], axis=-1)))
    if direct_d <= cutoff_A:
        return 0
    if wat_pos.shape[0] == 0:
        return -1

    d_src = np.min(np.linalg.norm(wat_pos[:, None, :] - src_atoms[None, :, :], axis=-1), axis=1)
    d_dst = np.min(np.linalg.norm(wat_pos[:, None, :] - dst_atoms[None, :, :], axis=-1), axis=1)
    src_nodes = np.flatnonzero(d_src <= cutoff_A)
    dst_set = set(np.flatnonzero(d_dst <= cutoff_A).tolist())
    if src_nodes.size == 0 or not dst_set:
        return -1

    visited: set[int] = set(src_nodes.tolist())
    queue: deque[tuple[int, int]] = deque((int(u), 1) for u in src_nodes)
    while queue:
        u, hops = queue.popleft()
        if u in dst_set:
            return hops
        for v in wat_adj[u]:
            if v not in visited:
                visited.add(v)
                queue.append((v, hops + 1))
    return -1


def hops_to_conductance(hops: int) -> float:
    """Map shortest H-bond hop count to bounded conductance C_wire = 1/(1 + L_hops), 0 if disconnected."""
    if hops < 0:
        return 0.0
    return 1.0 / (1.0 + float(hops))


def build_topology_index(top: md.Topology, cb1_chains: list[int], gid: int) -> dict[str, Any]:
    """Validate CB1 residue identities and precompute atom-index lookup arrays for fast featurization."""
    rmap: dict[int, md.core.topology.Residue] = {}
    chain_ca_indices: list[np.ndarray] = []
    cb1_chain_blocks: list[dict[str, Any]] = []
    cb1_all_heavy: list[int] = []

    for ci in cb1_chains:
        ch = top.chain(ci)
        ch_cas = []
        ch_all_atoms = []
        ch_res_blocks: list[tuple[int, np.ndarray]] = []
        ch_core_cas = []
        for res in ch.residues:
            if not res.is_protein:
                continue
            rmap[res.resSeq] = res
            r_atoms = []
            r_ca = None
            for a in res.atoms:
                r_atoms.append(a.index)
                ch_all_atoms.append(a.index)
                if a.element.symbol != "H":
                    cb1_all_heavy.append(a.index)
                if a.name == "CA":
                    ch_cas.append(a.index)
                    r_ca = a.index
                    if res.resSeq in (163, 200, 214, 237, 294, 356, 393):
                        ch_core_cas.append(a.index)
            if r_ca is not None and r_atoms:
                ch_res_blocks.append((r_ca, np.array(r_atoms, dtype=np.int32)))
        if ch_cas:
            chain_ca_indices.append(np.array(ch_cas, dtype=np.int32))
            cb1_chain_blocks.append({
                "res_blocks": ch_res_blocks,
                "all_atoms": np.array(ch_all_atoms, dtype=np.int32),
                "core_cas": np.array(ch_core_cas if ch_core_cas else ch_cas, dtype=np.int32),
            })

    # Mandatory residue-identity guard across all 17 core CB1 positions
    for pos, exp_name in REQUIRED_CB1_RESIDUES.items():
        if pos not in rmap:
            raise AssertionError(f"GPCRmd {gid}: missing required CB1 residue {exp_name}{pos}")
        if rmap[pos].name != exp_name:
            raise AssertionError(
                f"GPCRmd {gid}: residue identity guard failed at {pos}: got {rmap[pos].name}, expected {exp_name}"
            )
    if 210 in rmap and rmap[210].name not in ("THR", "ALA", "ILE"):
        raise AssertionError(f"GPCRmd {gid}: unexpected residue at 210: {rmap[210].name}")

    def atom_idx(pos: int, name: str) -> int:
        for a in rmap[pos].atoms:
            if a.name == name:
                return a.index
        raise KeyError(f"GPCRmd {gid}: missing atom {name} in {rmap[pos]}")

    def heavy_indices(positions: list[int], only_names: tuple[str, ...] | None = None, exclude_bb: bool = False) -> np.ndarray:
        out = []
        for p in positions:
            if p not in rmap:
                continue
            for a in rmap[p].atoms:
                if a.element.symbol == "H":
                    continue
                if exclude_bb and a.name in ("N", "CA", "C", "O"):
                    continue
                if only_names is not None and a.name not in only_names:
                    continue
                out.append(a.index)
        if not out:
            raise AssertionError(f"GPCRmd {gid}: empty atom selection for {positions} ({only_names})")
        return np.array(out, dtype=np.int32)

    # Core centroid CAs for PBC centering
    core_center_cas = np.array([atom_idx(p, "CA") for p in (163, 200, 214, 237, 294, 356, 393)], dtype=np.int32)
    pore_center_cas = np.array([atom_idx(p, "CA") for p in (163, 356, 393, 397)], dtype=np.int32)
    ic_axis_cas = np.array([atom_idx(p, "CA") for p in (154, 214, 237, 294, 338, 397) if p in rmap], dtype=np.int32)
    ec_axis_cas = np.array([atom_idx(p, "CA") for p in (134, 173, 196, 241, 279, 358, 383) if p in rmap], dtype=np.int32)

    # Dihedral atom quadruplets
    chi_quads = {}
    for pos in (155, 237, 200, 356):
        rname = rmap[pos].name
        q1, q2 = CHI_ATOM_NAMES[rname]
        chi_quads[f"{pos}_chi1"] = [atom_idx(pos, a) for a in q1]
        chi_quads[f"{pos}_chi2"] = [atom_idx(pos, a) for a in q2]

    # Solvent, ions, and POPC
    wat_O = np.array(
        [a.index for a in top.atoms if a.residue.name in ("HOH", "TIP3", "SOL", "WAT") and a.element.symbol == "O"],
        dtype=np.int32,
    )
    na_ions = np.array(
        [a.index for a in top.atoms if a.residue.name in ("SOD", "NA", "NA+")],
        dtype=np.int32,
    )
    k_ions = np.array(
        [a.index for a in top.atoms if a.residue.name in ("POT", "K", "K+")],
        dtype=np.int32,
    )
    # Group POPC heavy atoms by residue so we can PBC-wrap whole lipid molecules cleanly
    popc_res_atoms: list[np.ndarray] = []
    for res in top.residues:
        if res.name in ("POPC", "POP"):
            h_idx = [a.index for a in res.atoms if a.element.symbol != "H"]
            if h_idx:
                popc_res_atoms.append(np.array(h_idx, dtype=np.int32))
    popc_heavy = np.concatenate(popc_res_atoms) if popc_res_atoms else np.empty(0, dtype=np.int32)
    env_atoms = np.concatenate([arr for arr in (wat_O, na_ions, k_ions, popc_heavy) if arr.size > 0])

    return {
        "rmap_pos210": rmap[210].name if 210 in rmap else "UNK",
        "chain_ca_indices": chain_ca_indices,
        "cb1_chain_blocks": cb1_chain_blocks,
        "cb1_all_heavy": np.array(cb1_all_heavy, dtype=np.int32),
        "env_atoms": env_atoms,
        "core_center_cas": core_center_cas,
        "pore_center_cas": pore_center_cas,
        "ic_axis_cas": ic_axis_cas,
        "ec_axis_cas": ec_axis_cas,
        "ca": {p: atom_idx(p, "CA") for p in (155, 163, 173, 200, 204, 214, 237, 268, 294, 300, 338, 341, 343, 356, 383, 393, 397)},
        "chi_quads": chi_quads,
        "f155_sc": heavy_indices([155], exclude_bb=True),
        "f237_sc": heavy_indices([237], exclude_bb=True),
        "f155_ring": heavy_indices([155], only_names=RING_ATOM_NAMES["PHE"]),
        "f237_ring": heavy_indices([237], only_names=RING_ATOM_NAMES["PHE"]),
        "f200_ring": heavy_indices([200], only_names=RING_ATOM_NAMES["PHE"]),
        "w356_ring": heavy_indices([356], only_names=RING_ATOM_NAMES["TRP"]),
        "w356_ne1": heavy_indices([356], only_names=("NE1",)),
        "y294_oh": atom_idx(294, "OH"),
        "y397_oh": atom_idx(397, "OH"),
        "d163_carboxyl": heavy_indices([163], only_names=("OD1", "OD2")),
        "n393_polar": heavy_indices([393], only_names=("OD1", "ND2")),
        "n393_y397_polar": np.concatenate([
            heavy_indices([393], only_names=("OD1", "ND2")),
            heavy_indices([397], only_names=("OH",)),
        ]),
        "r214_guan": heavy_indices([214], only_names=("NE", "NH1", "NH2")),
        "d338_carboxyl": heavy_indices([338], only_names=("OD1", "OD2")),
        "d163_heavy": heavy_indices([163], exclude_bb=True),
        "w356_heavy": heavy_indices([356], exclude_bb=True),
        "n393_heavy": heavy_indices([393], exclude_bb=True),
        "y294_heavy": heavy_indices([294], exclude_bb=True),
        "y397_heavy": heavy_indices([397], exclude_bb=True),
        "tm2_tm4_cleft": heavy_indices([154, 155, 158, 237, 238, 241], exclude_bb=True),
        "tm1_tm7_portal": heavy_indices([126, 130, 174, 177, 379, 383], exclude_bb=True),
        "wat_O": wat_O,
        "na_ions": na_ions,
        "k_ions": k_ions,
        "popc_heavy": popc_heavy,
    }


def pbc_wrap_chunk_angstrom(chunk: md.Trajectory, t_idx: dict[str, Any]) -> np.ndarray:
    """Convert chunk xyz to Angstroms, unwrap CB1 covalent/inter-chain continuity, and wrap solvent/lipids."""
    if chunk.unitcell_lengths is None:
        raise ValueError("Missing unitcell_lengths in XTC chunk; cannot apply exact PBC wrapping.")
    xyz = np.array(chunk.xyz, dtype=np.float64) * 10.0  # (F, N_atoms, 3) in Angstroms
    box = np.array(chunk.unitcell_lengths, dtype=np.float64) * 10.0  # (F, 3) in Angstroms
    if np.min(box) < 35.0:
        raise ValueError(f"Non-physical unitcell_lengths (min={np.min(box):.2f} A); expected >35 A bilayer box.")

    # Step 1a: Unwrap covalent peptide continuity strictly within each CB1 chain independently
    chain_blocks = t_idx["cb1_chain_blocks"]
    for cb in chain_blocks:
        rblocks = cb["res_blocks"]
        first_ca, first_atoms = rblocks[0]
        prev_ca = xyz[:, first_ca, :].copy()  # (F, 3)
        xyz[:, first_atoms, :] -= box[:, None, :] * np.round(
            (xyz[:, first_atoms, :] - prev_ca[:, None, :]) / box[:, None, :]
        )
        for r_ca, r_atoms in rblocks[1:]:
            ca_pos = xyz[:, r_ca, :]
            ca_unwrapped = ca_pos - box * np.round((ca_pos - prev_ca) / box)
            xyz[:, r_atoms, :] -= box[:, None, :] * np.round(
                (xyz[:, r_atoms, :] - ca_unwrapped[:, None, :]) / box[:, None, :]
            )
            prev_ca = ca_unwrapped

    # Step 1b: Align any TER-split CB1 chain (TM6-H8) rigidly to TM1-TM5 via transmembrane core C-alphas
    # (||c1_core - c0_core|| ~ 9.5 A << box/2 ~ 41 A, immune to solvent-exposed ICL3 stub flailing!)
    if len(chain_blocks) > 1:
        c0_core = np.mean(xyz[:, chain_blocks[0]["core_cas"], :], axis=1, keepdims=True)  # (F, 1, 3)
        for cb in chain_blocks[1:]:
            ck_core = np.mean(xyz[:, cb["core_cas"], :], axis=1, keepdims=True)
            shift_chain = box[:, None, :] * np.round((ck_core - c0_core) / box[:, None, :])
            xyz[:, cb["all_atoms"], :] -= shift_chain

    # Step 2: Translate intact CB1 7TM core centroid to origin, then wrap solvent/ions/POPC into [-box/2, +box/2)
    core_com = np.mean(xyz[:, t_idx["core_center_cas"], :], axis=1, keepdims=True)
    xyz -= core_com
    env_atoms = t_idx["env_atoms"]
    xyz[:, env_atoms, :] -= box[:, None, :] * np.round(xyz[:, env_atoms, :] / box[:, None, :])
    return xyz


def featurize_chunk(
    xyz: np.ndarray,
    t_idx: dict[str, Any],
    gid: int,
    pdb_id: str,
    spec: dict[str, Any],
    frame_offset: int,
) -> list[dict[str, Any]]:
    """Extract all protein, water-wire, ion, and POPC wedging features on PBC-wrapped chunk (in A)."""
    n_f = xyz.shape[0]
    ca = t_idx["ca"]

    def pair_dist_vec(i1: int, i2: int) -> np.ndarray:
        return np.linalg.norm(xyz[:, i1, :] - xyz[:, i2, :], axis=1)

    def set_min_dist_vec(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
        diff = xyz[:, s1, :][:, :, None, :] - xyz[:, s2, :][:, None, :, :]
        return np.min(np.linalg.norm(diff, axis=-1), axis=(1, 2))

    def set_com_dist_vec(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
        c1 = np.mean(xyz[:, s1, :], axis=1)
        c2 = np.mean(xyz[:, s2, :], axis=1)
        return np.linalg.norm(c1 - c2, axis=1)

    # Vectorized protein distances across chunk + live physical sanity guard
    r214_d338 = pair_dist_vec(ca[214], ca[338])
    if np.min(r214_d338) < 4.0 or np.max(r214_d338) > 30.0:
        raise AssertionError(
            f"GPCRmd {gid}: non-physical R214-D338 CA distance [{np.min(r214_d338):.2f}, {np.max(r214_d338):.2f}] A; check PBC!"
        )
    r214_l341 = pair_dist_vec(ca[214], ca[341])
    r214_k343 = pair_dist_vec(ca[214], ca[343])
    d163_n393 = pair_dist_vec(ca[163], ca[393])
    d163_w356 = pair_dist_vec(ca[163], ca[356])
    f200_w356_ca = pair_dist_vec(ca[200], ca[356])
    f155_f237_ca = pair_dist_vec(ca[155], ca[237])
    y294_y397_ca = pair_dist_vec(ca[294], ca[397])
    r214_k300 = pair_dist_vec(ca[214], ca[300])
    s173_s383 = pair_dist_vec(ca[173], ca[383])
    s173_f268 = pair_dist_vec(ca[173], ca[268])

    f155sc_n393ca = np.linalg.norm(np.mean(xyz[:, t_idx["f155_sc"], :], axis=1) - xyz[:, ca[393], :], axis=1)
    f237sc_v204ca = np.linalg.norm(np.mean(xyz[:, t_idx["f237_sc"], :], axis=1) - xyz[:, ca[204], :], axis=1)
    f155_f237_ring_min = set_min_dist_vec(t_idx["f155_ring"], t_idx["f237_ring"])
    f155_f237_ring_com = set_com_dist_vec(t_idx["f155_ring"], t_idx["f237_ring"])
    f200_w356_ring_min = set_min_dist_vec(t_idx["f200_ring"], t_idx["w356_ring"])
    f200_w356_ring_com = set_com_dist_vec(t_idx["f200_ring"], t_idx["w356_ring"])
    y294_y397_oh = pair_dist_vec(t_idx["y294_oh"], t_idx["y397_oh"])
    d163_n393_polar = set_min_dist_vec(t_idx["d163_carboxyl"], t_idx["n393_polar"])
    r214_d338_ionic = set_min_dist_vec(t_idx["r214_guan"], t_idx["d338_carboxyl"])

    # Vectorized dihedrals
    chis = {k: calc_dihedrals_batch_deg(xyz, q) for k, q in t_idx["chi_quads"].items()}

    # Pore cylinder axis per frame
    pore_mid = np.mean(xyz[:, t_idx["pore_center_cas"], :], axis=1)  # (F, 3)
    tm_vec = np.mean(xyz[:, t_idx["ec_axis_cas"], :], axis=1) - np.mean(xyz[:, t_idx["ic_axis_cas"], :], axis=1)
    tm_hat = tm_vec / np.clip(np.linalg.norm(tm_vec, axis=1, keepdims=True), 1e-8, None)

    wat_O = t_idx["wat_O"]
    na_ions = t_idx["na_ions"]
    k_ions = t_idx["k_ions"]
    popc_heavy = t_idx["popc_heavy"]
    has_na = na_ions.size > 0
    has_k = k_ions.size > 0

    rows: list[dict[str, Any]] = []
    dt_ns = float(spec["dt_ns"])

    for f in range(n_f):
        g_frame = frame_offset + f
        pos_f = xyz[f]
        wat_all = pos_f[wat_O]

        # Pre-filter waters within 16.5 A of pore midpoint using fast L_inf box filter then exact Euclidean radius
        rel_wat = wat_all - pore_mid[f]
        box_mask = (np.abs(rel_wat[:, 0]) <= 16.5) & (np.abs(rel_wat[:, 1]) <= 16.5) & (np.abs(rel_wat[:, 2]) <= 16.5)
        rel_box = rel_wat[box_mask]
        wat_box = wat_all[box_mask]
        d2_pore = np.sum(rel_box * rel_box, axis=1)
        near_mask = d2_pore <= (16.5 * 16.5)
        wat_near = wat_box[near_mask]
        rel_near = rel_box[near_mask]

        # Cylinder count: |axial| <= 12 A and radial <= 8 A around D163-W356-N393-Y397 pore axis
        z_ax = rel_near @ tm_hat[f]
        r_perp2 = np.sum(rel_near * rel_near, axis=1) - z_ax * z_ax
        n_wat_pore = int(np.sum((np.abs(z_ax) <= 12.0) & (r_perp2 <= 64.0)))

        # Local 3.5 A hydration counts around microswitch sidechains
        def count_hyd(sel_idx: np.ndarray) -> int:
            if wat_near.shape[0] == 0:
                return 0
            pts = pos_f[sel_idx]
            dmin = np.min(np.linalg.norm(wat_near[:, None, :] - pts[None, :, :], axis=-1), axis=1)
            return int(np.sum(dmin <= 3.5))

        n_wat_d163 = count_hyd(t_idx["d163_heavy"])
        n_wat_w356 = count_hyd(t_idx["w356_heavy"])
        n_wat_n393 = count_hyd(t_idx["n393_heavy"])
        n_wat_y294 = count_hyd(t_idx["y294_heavy"])
        n_wat_y397 = count_hyd(t_idx["y397_heavy"])

        # Build H-bond adjacency on core waters (within 13.5 A of pore_mid) via cKDTree
        wire_mask = np.sum(rel_near * rel_near, axis=1) <= (13.5 * 13.5)
        wat_wire = wat_near[wire_mask]
        n_w = wat_wire.shape[0]
        wat_adj: list[list[int]] = [[] for _ in range(n_w)]
        if n_w > 1:
            tree = cKDTree(wat_wire)
            for i_w, j_w in tree.query_pairs(r=3.5):
                wat_adj[i_w].append(j_w)
                wat_adj[j_w].append(i_w)

        d163_pts = pos_f[t_idx["d163_carboxyl"]]
        w356_ne1_pts = pos_f[t_idx["w356_ne1"]]
        w356_sc_pts = pos_f[t_idx["w356_heavy"]]
        ny_pts = pos_f[t_idx["n393_y397_polar"]]
        y294_oh_pt = pos_f[[t_idx["y294_oh"]]]
        y397_oh_pt = pos_f[[t_idx["y397_oh"]]]

        hops_d163_w356ne1 = bfs_shortest_water_hops(wat_wire, wat_adj, d163_pts, w356_ne1_pts, 3.5)
        hops_d163_w356 = bfs_shortest_water_hops(wat_wire, wat_adj, d163_pts, w356_sc_pts, 3.5)
        hops_d163_ny = bfs_shortest_water_hops(wat_wire, wat_adj, d163_pts, ny_pts, 3.5)
        hops_y294_y397 = bfs_shortest_water_hops(wat_wire, wat_adj, y294_oh_pt, y397_oh_pt, 3.5)

        # Explicit Na+ and K+ coordination to D163^2.50 carboxylates
        if has_na:
            na_pts = pos_f[na_ions]
            d_na = np.min(np.linalg.norm(na_pts[:, None, :] - d163_pts[None, :, :], axis=-1), axis=1)
            min_na = float(np.min(d_na))
            coord_na = int(np.sum(d_na < 3.2))
            prox_na = float(math.exp(-min_na / 5.0))
        else:
            min_na = 99.0
            coord_na = 0
            prox_na = 0.0

        if has_k:
            k_pts = pos_f[k_ions]
            d_k = np.min(np.linalg.norm(k_pts[:, None, :] - d163_pts[None, :, :], axis=-1), axis=1)
            min_k = float(np.min(d_k))
            coord_k = int(np.sum(d_k < 3.2))
            prox_k = float(math.exp(-min_k / 5.0))
        else:
            min_k = 99.0
            coord_k = 0
            prox_k = 0.0

        min_cation = min(min_na, min_k)
        coord_cation = coord_na + coord_k

        # Explicit POPC heavy-atom wedging distances (fast L_inf box pre-filter within 28 A of origin)
        popc_pts = pos_f[popc_heavy]
        pbox_mask = (np.abs(popc_pts[:, 0]) <= 28.0) & (np.abs(popc_pts[:, 1]) <= 28.0) & (np.abs(popc_pts[:, 2]) <= 28.0)
        popc_near = popc_pts[pbox_mask]
        if popc_near.shape[0] == 0:
            popc_near = popc_pts

        cleft_pts = pos_f[t_idx["tm2_tm4_cleft"]]
        portal_pts = pos_f[t_idx["tm1_tm7_portal"]]
        popc_tree = cKDTree(popc_near)
        d_cleft, _ = popc_tree.query(cleft_pts, k=1)
        d_portal, _ = popc_tree.query(portal_pts, k=1)
        min_popc_cleft = float(np.min(d_cleft))
        min_popc_portal = float(np.min(d_portal))

        rows.append({
            "gpcrmd_id": gid,
            "pdb_id": pdb_id,
            "state_label": spec["state_label"],
            "transducer_type": spec["transducer_type"],
            "ligand_label": spec["ligand_label"],
            "pos210_resname": t_idx["rmap_pos210"],
            "has_na_ions": has_na,
            "has_k_ions": has_k,
            "frame_idx": g_frame,
            "dt_ns": dt_ns,
            "time_ns": round(g_frame * dt_ns, 4),
            # Tier P1: Backbone C-alpha gates (A)
            "R214_D338_CA_A": round(float(r214_d338[f]), 3),
            "R214_L341_CA_A": round(float(r214_l341[f]), 3),
            "R214_K343_CA_A": round(float(r214_k343[f]), 3),
            "D163_N393_CA_A": round(float(d163_n393[f]), 3),
            "D163_W356_CA_A": round(float(d163_w356[f]), 3),
            "F200_W356_CA_A": round(float(f200_w356_ca[f]), 3),
            "F155_F237_CA_A": round(float(f155_f237_ca[f]), 3),
            "Y294_Y397_CA_A": round(float(y294_y397_ca[f]), 3),
            "R214_K300_CA_A": round(float(r214_k300[f]), 3),
            "S173_S383_CA_A": round(float(s173_s383[f]), 3),
            "S173_F268_CA_A": round(float(s173_f268[f]), 3),
            # Tier P2: Sidechain torsions (deg) & contacts (A)
            "F155_chi1_deg": round(float(chis["155_chi1"][f]), 2),
            "F155_chi2_deg": round(float(chis["155_chi2"][f]), 2),
            "F237_chi1_deg": round(float(chis["237_chi1"][f]), 2),
            "F237_chi2_deg": round(float(chis["237_chi2"][f]), 2),
            "F200_chi1_deg": round(float(chis["200_chi1"][f]), 2),
            "F200_chi2_deg": round(float(chis["200_chi2"][f]), 2),
            "W356_chi1_deg": round(float(chis["356_chi1"][f]), 2),
            "W356_chi2_deg": round(float(chis["356_chi2"][f]), 2),
            "F155sc_N393CA_A": round(float(f155sc_n393ca[f]), 3),
            "F237sc_V204CA_A": round(float(f237sc_v204ca[f]), 3),
            "F155_F237_ring_min_A": round(float(f155_f237_ring_min[f]), 3),
            "F155_F237_ring_com_A": round(float(f155_f237_ring_com[f]), 3),
            "F200_W356_ring_min_A": round(float(f200_w356_ring_min[f]), 3),
            "F200_W356_ring_com_A": round(float(f200_w356_ring_com[f]), 3),
            "Y294_Y397_OH_A": round(float(y294_y397_oh[f]), 3),
            "D163_N393_polar_min_A": round(float(d163_n393_polar[f]), 3),
            "R214_D338_ionic_min_A": round(float(r214_d338_ionic[f]), 3),
            # Tier E1: Explicit hydration & H-bonded water-wire percolation
            "n_wat_core_pore": n_wat_pore,
            "n_wat_D163_3p5": n_wat_d163,
            "n_wat_W356_3p5": n_wat_w356,
            "n_wat_N393_3p5": n_wat_n393,
            "n_wat_Y294_3p5": n_wat_y294,
            "n_wat_Y397_3p5": n_wat_y397,
            "wire_D163_W356NE1_hops": hops_d163_w356ne1,
            "wire_D163_W356NE1_cond": round(hops_to_conductance(hops_d163_w356ne1), 4),
            "wire_D163_W356_hops": hops_d163_w356,
            "wire_D163_W356_cond": round(hops_to_conductance(hops_d163_w356), 4),
            "wire_D163_N393Y397_hops": hops_d163_ny,
            "wire_D163_N393Y397_cond": round(hops_to_conductance(hops_d163_ny), 4),
            "wire_Y294_Y397_hops": hops_y294_y397,
            "wire_Y294_Y397_cond": round(hops_to_conductance(hops_y294_y397), 4),
            # Tier E2: Explicit Na+ / K+ ions & POPC membrane wedging
            "min_Na_D163_A": round(min_na, 3),
            "coord_Na_D163_3p2": coord_na,
            "prox_Na_D163": round(prox_na, 5),
            "min_K_D163_A": round(min_k, 3),
            "coord_K_D163_3p2": coord_k,
            "prox_K_D163": round(prox_k, 5),
            "min_cation_D163_A": round(min_cation, 3),
            "coord_cation_D163_3p2": coord_cation,
            "min_POPC_TM2_TM4_cleft_A": round(min_popc_cleft, 3),
            "min_POPC_TM1_TM7_portal_A": round(min_popc_portal, 3),
        })
    return rows


def stream_and_featurize_all(
    out_parquet: Path = OUT_PARQUET,
    provenance_json: Path = PROVENANCE_JSON,
    chunk_size: int = 500,
) -> pd.DataFrame:
    """Download, verify, featurize, and clean up all 6 open GPCRmd trajectories."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    ds_by_gid: dict[int, dict[str, Any]] = {}
    for ds in manifest["immediately_downloadable_open_md_datasets"]:
        did = ds["dataset_id"]
        if did.startswith("gpcrmd_"):
            gid = int(did.split("_")[1])
            ds_by_gid[gid] = ds

    TOPO_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_XTC_DIR.mkdir(parents=True, exist_ok=True)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    # Order smaller trajectories first to validate end-to-end quickly, then 2067 (2.1 GB) and 1201 (3.8 GB)
    ordered_gids = [22, 164, 162, 163, 2067, 1201]
    all_dfs: list[pd.DataFrame] = []
    provenance: dict[str, Any] = {
        "encoding_documentation": {
            "water_wire_conductance": "C_wire = 1.0 / (1.0 + L_hops) where L_hops=0 for direct polar contact <=3.5 A, L_hops>=1 for bridging waters (O-O <=3.5 A), and C_wire=0.0 (hops=-1) when disconnected. Zero NaNs.",
            "ion_sentinel": "min_Na_D163_A = 99.0 (prox_Na_D163 = 0.0, coord_Na_D163_3p2 = 0) when has_na_ions == False (ID 1201 uses KCl buffer); min_K_D163_A = 99.0 when has_k_ions == False. min_cation_D163_A = min(Na+, K+) is finite across all 6 trajectories.",
            "pbc_handling": "Exact minimum-image convention around CB1 7TM core C-alpha centroid (after aligning TER-split TM6-H8 chain to TM1-TM5 chain) applied on every frame prior to distance/KDTree calculations.",
        },
        "trajectories": {},
    }

    for gid in ordered_gids:
        ds = ds_by_gid[gid]
        spec = GPCRMD_SPECS[gid]
        pdb_id = ds["pdb_id"]
        pdb_info = ds["files"]["pdb"]
        xtc_info = ds["files"]["xtc_rep1"]

        pdb_path = TOPO_DIR / f"gpcrmd_{gid}.pdb"
        download_with_retry(pdb_info["url"], pdb_path, expected_bytes=pdb_info["file_size_bytes"])

        top_traj = md.load_pdb(str(pdb_path))
        t_idx = build_topology_index(top_traj.topology, spec["cb1_chains"], gid)

        ckpt_pq = TEMP_XTC_DIR / f"ckpt_{gid}.parquet"
        ckpt_meta = TEMP_XTC_DIR / f"ckpt_{gid}.json"
        if ckpt_pq.exists() and ckpt_meta.exists():
            df_g = pd.read_parquet(ckpt_pq)
            provenance["trajectories"][str(gid)] = json.loads(ckpt_meta.read_text(encoding="utf-8"))
            all_dfs.append(df_g)
            print(f"[WS3] Loaded cached checkpoint for GPCRmd ID {gid} ({len(df_g)} frames).", flush=True)
            continue

        xtc_path = TEMP_XTC_DIR / f"gpcrmd_{gid}.xtc"
        t_start = time.time()
        print(f"[WS3] Streaming GPCRmd ID {gid} ({spec['state_label']}, {xtc_info['file_size_bytes']/1e6:.1f} MB) ...", flush=True)
        dl_meta = download_with_retry(xtc_info["url"], xtc_path, expected_bytes=xtc_info["file_size_bytes"])
        hdr_meta = verify_xtc_magic(xtc_path, expected_natoms=top_traj.n_atoms)

        traj_rows: list[dict[str, Any]] = []
        frame_offset = 0
        for chunk in md.iterload(str(xtc_path), top=str(pdb_path), chunk=chunk_size):
            xyz_wrapped = pbc_wrap_chunk_angstrom(chunk, t_idx)
            chunk_rows = featurize_chunk(xyz_wrapped, t_idx, gid, pdb_id, spec, frame_offset)
            traj_rows.extend(chunk_rows)
            frame_offset += chunk.n_frames
            print(f"  [ID {gid}] featurized {frame_offset} frames ...", flush=True)

        df_g = pd.DataFrame(traj_rows)
        if df_g.isna().any().any():
            nan_cols = df_g.columns[df_g.isna().any()].tolist()
            raise AssertionError(f"NaN guard failed on GPCRmd {gid}: {nan_cols}")

        # Delete bulky temporary .xtc immediately after verification
        xtc_path.unlink()
        elapsed = time.time() - t_start
        print(
            f"[WS3] Completed ID {gid}: {len(df_g)} frames ({df_g['time_ns'].max():.1f} ns) in {elapsed:.1f}s; "
            f"deleted temp {xtc_path.name}.",
            flush=True,
        )

        t_prov = {
            "dataset_id": ds["dataset_id"],
            "pdb_id": pdb_id,
            "state_label": spec["state_label"],
            "frames_extracted": int(len(df_g)),
            "dt_ns": spec["dt_ns"],
            "duration_ns": round(float(len(df_g) * spec["dt_ns"]), 2),
            "xtc_download": dl_meta,
            "xtc_header": hdr_meta,
            "pos210_resname": t_idx["rmap_pos210"],
            "n_water_O": int(t_idx["wat_O"].size),
            "n_na_ions": int(t_idx["na_ions"].size),
            "n_k_ions": int(t_idx["k_ions"].size),
            "n_popc_heavy": int(t_idx["popc_heavy"].size),
        }
        df_g.to_parquet(ckpt_pq, index=False)
        ckpt_meta.write_text(json.dumps(t_prov, indent=2), encoding="utf-8")
        provenance["trajectories"][str(gid)] = t_prov
        all_dfs.append(df_g)

    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df.to_parquet(out_parquet, index=False)
    provenance["total_frames"] = int(len(full_df))
    provenance["parquet_size_bytes"] = int(out_parquet.stat().st_size)
    provenance_json.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    # Clean up temporary checkpoints in TEMP_XTC_DIR so directory stays completely clean
    for tmp_f in TEMP_XTC_DIR.iterdir():
        if tmp_f.is_file():
            tmp_f.unlink()
    print(
        f"[WS3] Saved {len(full_df)} total frames to {out_parquet} "
        f"({out_parquet.stat().st_size / 1e6:.2f} MB).",
        flush=True,
    )
    return full_df


if __name__ == "__main__":
    stream_and_featurize_all()
