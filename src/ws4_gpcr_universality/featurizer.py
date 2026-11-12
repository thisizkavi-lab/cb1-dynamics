"""Multi-GPCR Streaming HTTP 206 Downloader, PBC Whole-Receptor Reassembler, and BW Featurizer.

Designed for turnkey execution on Google Colab CPU / High-RAM CPU runtimes with automatic
per-trajectory and per-receptor Google Drive Parquet checkpointing and immediate raw .xtc cleanup.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import struct
import tempfile
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


CANONICAL_BW_POSITIONS: tuple[str, ...] = (
    "1.35",
    "2.42",
    "2.50",
    "3.36",
    "3.50",
    "4.46",
    "5.58",
    "6.30",
    "6.48",
    "7.35",
    "7.49",
    "7.53",
)

CONSERVED_CORE_BW_GUARDS: dict[str, tuple[str, ...]] = {
    "2.50": ("ASP",),
    "3.50": ("ARG",),
    "5.58": ("TYR",),
    "6.48": ("TRP",),
    "7.49": ("ASN",),
    "7.53": ("TYR",),
}

AROMATIC_RESNAMES: frozenset[str] = frozenset({
    "PHE", "TYR", "TRP", "HIS", "HSD", "HSE", "HSP", "HID", "HIE", "HIP",
})

WATER_RESNAMES: frozenset[str] = frozenset({"HOH", "TIP3", "TIP3P", "SOL", "WAT", "SPC"})
NA_ION_RESNAMES: frozenset[str] = frozenset({"SOD", "NA", "NA+"})
K_ION_RESNAMES: frozenset[str] = frozenset({"POT", "K", "K+"})
MEMBRANE_LIPID_RESNAMES: frozenset[str] = frozenset({
    "POPC", "POP", "POPE", "POPS", "POPG", "DOPC", "DPPC", "DSPC", "SOPC", "SDPC",
    "PLPC", "PAPC", "PAPS", "OAPE", "PDOP", "PLES", "PSM", "NSM", "LSM", "SM18",
    "CHL1", "CLR", "CHOL", "OL", "PA", "PC", "PE",
})

CHI1_FOURTH_ATOM: dict[str, str | None] = {
    "PHE": "CG",
    "TYR": "CG",
    "TRP": "CG",
    "HIS": "CG",
    "HSD": "CG",
    "HSE": "CG",
    "HSP": "CG",
    "HID": "CG",
    "HIE": "CG",
    "HIP": "CG",
    "LEU": "CG",
    "MET": "CG",
    "ASP": "CG",
    "ASN": "CG",
    "GLU": "CG",
    "GLN": "CG",
    "LYS": "CG",
    "ARG": "CG",
    "PRO": "CG",
    "ILE": "CG1",
    "VAL": "CG1",
    "SER": "OG",
    "THR": "OG1",
    "CYS": "SG",
    "CYX": "SG",
    "ALA": None,
    "GLY": None,
}

USER_AGENT = "Mozilla/5.0 (CB1-WS4-Universality/1.0)"


def load_gpcr_family_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Load and sanity-check `gpcr_family_manifest.json`."""
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing GPCR family manifest: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if "receptors" not in data or not isinstance(data["receptors"], dict):
        raise ValueError(f"Invalid manifest schema in {path}: missing 'receptors' mapping.")
    return data


def probe_xtc_http206_magic1995(url: str, timeout: float = 20.0) -> dict[str, Any]:
    """Send `HTTP Range: bytes=0-127` and verify `HTTP 206` + big-endian XTC magic `1995`."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Range": "bytes=0-127"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        status = int(resp.getcode())
        content_range = resp.headers.get("Content-Range", "")
        content_type = resp.headers.get("Content-Type", "")
        buf = resp.read(128)

    if status != 206:
        raise AssertionError(
            f"Expected HTTP 206 Partial Content for {url}, got HTTP {status} (Content-Type={content_type})"
        )
    if len(buf) < 16:
        raise AssertionError(f"Short header ({len(buf)} bytes) returned from {url}")

    magic, natoms, step0 = struct.unpack(">iii", buf[:12])
    t0_ps = float(struct.unpack(">f", buf[12:16])[0])
    if magic != 1995:
        raise AssertionError(
            f"XTC binary magic check failed for {url}: magic={magic} != 1995 (likely HTML login redirect!)"
        )

    m_total = re.search(r"/(\d+)\s*$", content_range)
    total_bytes = int(m_total.group(1)) if m_total else -1
    return {
        "http_status": status,
        "magic": int(magic),
        "natoms": int(natoms),
        "step0": int(step0),
        "t0_ps": round(t0_ps, 3),
        "total_bytes": total_bytes,
        "content_range": content_range,
    }


def verify_manifest_live_probes(
    manifest: dict[str, Any],
    one_per_receptor: bool = True,
    max_workers: int = 6,
) -> list[dict[str, Any]]:
    """Live-probe manifest `.xtc` URLs (`Range: bytes=0-127`) across all 6 GPCRs."""
    targets: list[tuple[str, dict[str, Any]]] = []
    for rec_key, rec_info in manifest["receptors"].items():
        trajs = rec_info.get("trajectories", [])
        if not trajs:
            raise AssertionError(f"Receptor {rec_key} has zero trajectories in manifest!")
        selected = trajs[:1] if one_per_receptor else trajs
        for t in selected:
            targets.append((rec_key, t))

    def _check(item: tuple[str, dict[str, Any]]) -> dict[str, Any]:
        rec_key, t = item
        live = probe_xtc_http206_magic1995(t["xtc_url"])
        exp_natoms = int(t["topology_summary"]["pdb_natoms"])
        if live["natoms"] != exp_natoms:
            raise AssertionError(
                f"{t['traj_id']}: live XTC natoms={live['natoms']} != manifest pdb_natoms={exp_natoms}"
            )
        return {"receptor": rec_key, "traj_id": t["traj_id"], **live}

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for res in pool.map(_check, targets):
            results.append(res)
    return results


def download_parallel_http206(
    url: str,
    dest: Path,
    expected_bytes: int | None = None,
    n_workers: int = 16,
    slice_bytes: int = 16 * 1024 * 1024,
    retries: int = 5,
) -> dict[str, Any]:
    """Stream file via 16-way parallel `HTTP 206` Range requests using `os.pwrite`."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if expected_bytes is not None and expected_bytes > 0 and dest.exists() and dest.stat().st_size == expected_bytes:
        h = hashlib.sha256()
        with open(dest, "rb") as fh:
            h.update(fh.read(1024 * 1024))
        return {"size_bytes": expected_bytes, "head_1mb_sha256": h.hexdigest()}

    if expected_bytes is None or expected_bytes <= 0 or expected_bytes <= 2 * slice_bytes:
        tmp_path = dest.with_suffix(dest.suffix + ".part")
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=120) as resp, open(tmp_path, "wb") as fh:
                    shutil.copyfileobj(resp, fh, length=4 * 1024 * 1024)
                if expected_bytes is not None and expected_bytes > 0 and tmp_path.stat().st_size != expected_bytes:
                    raise IOError(f"Size mismatch for {dest.name}: {tmp_path.stat().st_size} != {expected_bytes}")
                tmp_path.replace(dest)
                break
            except Exception as exc:
                if attempt == retries - 1:
                    raise RuntimeError(f"Failed downloading {url}: {exc}") from exc
                time.sleep(1.5 * (attempt + 1))
    else:
        part_path = dest.with_suffix(dest.suffix + ".part")
        fd = os.open(part_path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o644)
        try:
            os.ftruncate(fd, expected_bytes)
            ranges = [
                (off, min(off + slice_bytes - 1, expected_bytes - 1))
                for off in range(0, expected_bytes, slice_bytes)
            ]

            def fetch_slice(byte_range: tuple[int, int]) -> int:
                b_start, b_end = byte_range
                want = b_end - b_start + 1
                for att in range(retries):
                    try:
                        req = urllib.request.Request(
                            url,
                            headers={"User-Agent": USER_AGENT, "Range": f"bytes={b_start}-{b_end}"},
                        )
                        with urllib.request.urlopen(req, timeout=120) as resp:
                            if resp.getcode() != 206:
                                raise IOError(f"Expected HTTP 206 for Range request, got {resp.getcode()}")
                            pos = b_start
                            rem = want
                            while rem > 0:
                                chunk = resp.read(min(1024 * 1024, rem))
                                if not chunk:
                                    break
                                written = os.pwrite(fd, chunk, pos)
                                pos += written
                                rem -= written
                            if rem != 0:
                                raise IOError(f"Short Range read ({rem} bytes remaining) on {dest.name}")
                        return want
                    except Exception as exc:
                        if att == retries - 1:
                            raise RuntimeError(f"Slice {b_start}-{b_end} failed for {url}: {exc}") from exc
                        time.sleep(1.5 * (att + 1))
                return 0

            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futs = [pool.submit(fetch_slice, r) for r in ranges]
                for fut in as_completed(futs):
                    fut.result()
        finally:
            os.close(fd)

        if part_path.stat().st_size != expected_bytes:
            raise IOError(f"Incomplete parallel download for {dest.name}")
        part_path.replace(dest)

    h = hashlib.sha256()
    with open(dest, "rb") as fh:
        h.update(fh.read(1024 * 1024))
    return {"size_bytes": dest.stat().st_size, "head_1mb_sha256": h.hexdigest()}


def verify_xtc_header_1995(xtc_path: str | Path, expected_natoms: int) -> dict[str, Any]:
    """Assert local `.xtc` file starts with GROMACS magic `1995` and matches `expected_natoms`."""
    p = Path(xtc_path)
    with open(p, "rb") as fh:
        head = fh.read(16)
    if len(head) < 16:
        raise ValueError(f"XTC file too short: {p}")
    magic, natoms, step0 = struct.unpack(">iii", head[:12])
    t0_ps = float(struct.unpack(">f", head[12:16])[0])
    if magic != 1995:
        raise AssertionError(f"XTC magic guard failed on {p.name}: magic={magic} != 1995")
    if natoms != expected_natoms:
        raise AssertionError(f"XTC natoms mismatch on {p.name}: {natoms} != {expected_natoms}")
    return {"magic": int(magic), "natoms": int(natoms), "step0": int(step0), "t0_ps": t0_ps}


def calc_dihedrals_batch_deg(coords: np.ndarray, idx4: list[int] | None) -> np.ndarray:
    """Vectorized dihedral angle in degrees `[-180, 180]` over `(F, N_atoms, 3)`; returns `0.0` if `idx4 is None`."""
    if idx4 is None:
        return np.zeros(coords.shape[0], dtype=np.float64)
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
    """Minimum number of bridging water oxygens (`0` direct H-bond, `k>=1` waters, `-1` disconnected)."""
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
    """Bounded water-wire conductance `C_wire = 1 / (1 + L_hops)`, `0.0` when disconnected (`hops < 0`)."""
    if hops < 0:
        return 0.0
    return 1.0 / (1.0 + float(hops))


def build_bw_topology_index(
    top: md.Topology,
    traj_entry: dict[str, Any],
    enforce_conserved_core: bool = True,
) -> dict[str, Any]:
    """Validate 12 canonical Ballesteros-Weinstein residues and precompute atom-index lookup arrays."""
    traj_id = str(traj_entry.get("traj_id", "unknown_traj"))
    receptor_chains: list[int] = [int(c) for c in traj_entry["receptor_chains"]]
    bw_map_spec: dict[str, dict[str, Any]] = traj_entry["bw_residue_map"]

    for bw in CANONICAL_BW_POSITIONS:
        if bw not in bw_map_spec:
            raise AssertionError(f"{traj_id}: missing canonical BW position {bw} in bw_residue_map")

    # Resolve exact mdtraj Residue object per BW position
    bw_res_objs: dict[str, md.core.topology.Residue] = {}
    core_bw_set = {"2.50", "3.36", "3.50", "4.46", "5.58", "6.48", "7.49"}
    core_res_indices: set[int] = set()

    for bw in CANONICAL_BW_POSITIONS:
        spec = bw_map_spec[bw]
        ci = int(spec["chain_idx"])
        if ci not in receptor_chains:
            raise AssertionError(f"{traj_id}: BW {bw} chain_idx={ci} not in receptor_chains={receptor_chains}")
        target_seq = int(spec["resSeq"])
        expected_name = str(spec["resName"]).upper()
        matches = [
            r for r in top.chain(ci).residues
            if r.is_protein and int(r.resSeq) == target_seq and r.name.upper() == expected_name
        ]
        if len(matches) != 1:
            got = [(r.resSeq, r.name) for r in top.chain(ci).residues if r.is_protein and int(r.resSeq) == target_seq]
            raise AssertionError(
                f"{traj_id}: BW {bw} expected unique ({ci}, {target_seq}, {expected_name}), found {got}"
            )
        res_obj = matches[0]
        if enforce_conserved_core and bw in CONSERVED_CORE_BW_GUARDS:
            allowed = CONSERVED_CORE_BW_GUARDS[bw]
            if res_obj.name.upper() not in allowed:
                raise AssertionError(
                    f"{traj_id}: conserved Class A BW {bw} guard failed: got {res_obj.name}, expected {allowed}"
                )
        bw_res_objs[bw] = res_obj
        if bw in core_bw_set:
            core_res_indices.add(res_obj.index)

    # Build receptor chain blocks for covalent unwrapping and TER-split core alignment
    receptor_chain_blocks: list[dict[str, Any]] = []
    for ci in receptor_chains:
        ch = top.chain(ci)
        ch_cas: list[int] = []
        ch_all_atoms: list[int] = []
        ch_res_blocks: list[tuple[int, np.ndarray]] = []
        ch_core_cas: list[int] = []
        for res in ch.residues:
            if not res.is_protein:
                continue
            r_atoms: list[int] = []
            r_ca: int | None = None
            for a in res.atoms:
                r_atoms.append(a.index)
                ch_all_atoms.append(a.index)
                if a.name == "CA":
                    ch_cas.append(a.index)
                    r_ca = a.index
                    if res.index in core_res_indices:
                        ch_core_cas.append(a.index)
            if r_ca is not None and r_atoms:
                ch_res_blocks.append((r_ca, np.array(r_atoms, dtype=np.int32)))
        if ch_cas:
            receptor_chain_blocks.append({
                "chain_idx": ci,
                "res_blocks": ch_res_blocks,
                "all_atoms": np.array(ch_all_atoms, dtype=np.int32),
                "core_cas": np.array(ch_core_cas if ch_core_cas else ch_cas, dtype=np.int32),
            })

    if not receptor_chain_blocks:
        raise AssertionError(f"{traj_id}: zero protein atoms found in receptor_chains={receptor_chains}")

    def atom_idx(bw: str, atom_name: str) -> int:
        res = bw_res_objs[bw]
        for a in res.atoms:
            if a.name == atom_name:
                return a.index
        raise KeyError(f"{traj_id}: missing atom '{atom_name}' in BW {bw} ({res.name}{res.resSeq})")

    def sidechain_heavy(bws: list[str], only_names: tuple[str, ...] | None = None) -> np.ndarray:
        out: list[int] = []
        for bw in bws:
            res = bw_res_objs[bw]
            r_sc = [
                a.index for a in res.atoms
                if a.element.symbol != "H"
                and a.name not in ("N", "CA", "C", "O", "OXT")
                and (only_names is None or a.name in only_names)
            ]
            if not r_sc and only_names is None:
                # GLY fallback to CA so sidechain distance arrays are never empty
                r_sc = [atom_idx(bw, "CA")]
            out.extend(r_sc)
        if not out:
            raise AssertionError(f"{traj_id}: empty heavy-atom selection for {bws} (only_names={only_names})")
        return np.array(out, dtype=np.int32)

    def chi1_quad(bw: str) -> list[int] | None:
        res = bw_res_objs[bw]
        rname = res.name.upper()
        fourth = CHI1_FOURTH_ATOM.get(rname)
        if fourth is None:
            return None
        names = ("N", "CA", "CB", fourth)
        idx_map = {a.name: a.index for a in res.atoms}
        if all(nm in idx_map for nm in names):
            return [idx_map[nm] for nm in names]
        return None

    ca_map = {bw: atom_idx(bw, "CA") for bw in CANONICAL_BW_POSITIONS}
    core_center_cas = np.array(
        [ca_map[bw] for bw in ("2.50", "3.36", "3.50", "4.46", "5.58", "6.48", "7.49")],
        dtype=np.int32,
    )
    pore_center_cas = np.array([ca_map[bw] for bw in ("2.50", "6.48", "7.49", "7.53")], dtype=np.int32)
    ic_axis_cas = np.array([ca_map[bw] for bw in ("3.50", "5.58", "6.30", "7.53")], dtype=np.int32)
    ec_axis_cas = np.array([ca_map[bw] for bw in ("1.35", "3.36", "6.48", "7.35")], dtype=np.int32)

    wat_O = np.array(
        [a.index for a in top.atoms if a.residue.name.upper() in WATER_RESNAMES and a.element.symbol == "O"],
        dtype=np.int32,
    )
    na_ions = np.array(
        [a.index for a in top.atoms if a.residue.name.upper() in NA_ION_RESNAMES],
        dtype=np.int32,
    )
    k_ions = np.array(
        [a.index for a in top.atoms if a.residue.name.upper() in K_ION_RESNAMES],
        dtype=np.int32,
    )
    lipid_heavy = np.array(
        [
            a.index for a in top.atoms
            if (not a.residue.is_protein)
            and a.residue.name.upper() in MEMBRANE_LIPID_RESNAMES
            and a.element.symbol != "H"
        ],
        dtype=np.int32,
    )
    env_atoms = (
        np.concatenate([arr for arr in (wat_O, na_ions, k_ions, lipid_heavy) if arr.size > 0])
        if any(arr.size > 0 for arr in (wat_O, na_ions, k_ions, lipid_heavy))
        else np.empty(0, dtype=np.int32)
    )

    bw242_name = bw_res_objs["2.42"].name.upper()
    bw446_name = bw_res_objs["4.46"].name.upper()
    is_aromatic_pair = int((bw242_name in AROMATIC_RESNAMES) and (bw446_name in AROMATIC_RESNAMES))

    return {
        "traj_id": traj_id,
        "bw242_resname": bw242_name,
        "bw446_resname": bw446_name,
        "is_aromatic_pair_242_446": is_aromatic_pair,
        "receptor_chain_blocks": receptor_chain_blocks,
        "core_center_cas": core_center_cas,
        "pore_center_cas": pore_center_cas,
        "ic_axis_cas": ic_axis_cas,
        "ec_axis_cas": ec_axis_cas,
        "ca": ca_map,
        "chi1_quads": {
            "bw242_chi1_deg": chi1_quad("2.42"),
            "bw446_chi1_deg": chi1_quad("4.46"),
            "bw336_chi1_deg": chi1_quad("3.36"),
            "bw648_chi1_deg": chi1_quad("6.48"),
            "bw753_chi1_deg": chi1_quad("7.53"),
        },
        "has_bw446_chi1": int(chi1_quad("4.46") is not None),
        "sc_242": sidechain_heavy(["2.42"]),
        "sc_446": sidechain_heavy(["4.46"]),
        "sc_336": sidechain_heavy(["3.36"]),
        "sc_648": sidechain_heavy(["6.48"]),
        "sc_350": sidechain_heavy(["3.50"]),
        "sc_630": sidechain_heavy(["6.30"]),
        "sc_250": sidechain_heavy(["2.50"]),
        "sc_749": sidechain_heavy(["7.49"]),
        "sc_558": sidechain_heavy(["5.58"]),
        "sc_753": sidechain_heavy(["7.53"]),
        "d250_carboxyl": sidechain_heavy(["2.50"], only_names=("OD1", "OD2")),
        "n749_polar": sidechain_heavy(["7.49"], only_names=("OD1", "ND2")),
        "npxxy_polar": np.concatenate([
            sidechain_heavy(["7.49"], only_names=("OD1", "ND2")),
            sidechain_heavy(["7.53"], only_names=("OH",)),
        ]),
        "y558_oh": atom_idx("5.58", "OH"),
        "y753_oh": atom_idx("7.53", "OH"),
        "tm2_tm4_cleft": sidechain_heavy(["2.42", "4.46"]),
        "tm1_tm7_portal": sidechain_heavy(["1.35", "7.35"]),
        "wat_O": wat_O,
        "na_ions": na_ions,
        "k_ions": k_ions,
        "lipid_heavy": lipid_heavy,
        "env_atoms": env_atoms,
    }


def pbc_wrap_receptor_angstrom(chunk: md.Trajectory, t_idx: dict[str, Any]) -> np.ndarray:
    """Unwrap covalent/TER-split GPCR chains in Angstroms and wrap water/ions/lipids into `[-box/2, +box/2)`."""
    if chunk.unitcell_lengths is None:
        raise ValueError(f"{t_idx['traj_id']}: missing unitcell_lengths in trajectory chunk.")
    xyz = np.array(chunk.xyz, dtype=np.float64) * 10.0
    box = np.array(chunk.unitcell_lengths, dtype=np.float64) * 10.0
    if np.min(box) < 35.0:
        raise ValueError(f"{t_idx['traj_id']}: non-physical box lengths (min={np.min(box):.2f} A < 35 A).")

    chain_blocks = t_idx["receptor_chain_blocks"]
    for cb in chain_blocks:
        rblocks = cb["res_blocks"]
        first_ca, first_atoms = rblocks[0]
        prev_ca = xyz[:, first_ca, :].copy()
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

    if len(chain_blocks) > 1:
        c0_core = np.mean(xyz[:, chain_blocks[0]["core_cas"], :], axis=1, keepdims=True)
        for cb in chain_blocks[1:]:
            ck_core = np.mean(xyz[:, cb["core_cas"], :], axis=1, keepdims=True)
            shift_chain = box[:, None, :] * np.round((ck_core - c0_core) / box[:, None, :])
            xyz[:, cb["all_atoms"], :] -= shift_chain

    core_com = np.mean(xyz[:, t_idx["core_center_cas"], :], axis=1, keepdims=True)
    xyz -= core_com
    env_atoms = t_idx["env_atoms"]
    if env_atoms.size > 0:
        xyz[:, env_atoms, :] -= box[:, None, :] * np.round(xyz[:, env_atoms, :] / box[:, None, :])
    return xyz


def featurize_receptor_chunk(
    xyz: np.ndarray,
    t_idx: dict[str, Any],
    receptor_key: str,
    traj_entry: dict[str, Any],
    frame_offset: int = 0,
) -> list[dict[str, Any]]:
    """Extract standardized BW protein, water-wire, Na+/K+, and lipid cleft features (`0` NaNs)."""
    n_f = xyz.shape[0]
    ca = t_idx["ca"]
    traj_id = str(traj_entry["traj_id"])
    dyn_id = int(traj_entry["gpcrmd_dyn_id"])
    pdb_id = str(traj_entry["pdb_id"])
    state_class = str(traj_entry["state_class"])
    dt_ns = float(traj_entry["dt_ns"])

    def pair_dist(i1: int, i2: int) -> np.ndarray:
        return np.linalg.norm(xyz[:, i1, :] - xyz[:, i2, :], axis=1)

    def set_min_dist(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
        diff = xyz[:, s1, :][:, :, None, :] - xyz[:, s2, :][:, None, :, :]
        return np.min(np.linalg.norm(diff, axis=-1), axis=(1, 2))

    def set_com_dist(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
        c1 = np.mean(xyz[:, s1, :], axis=1)
        c2 = np.mean(xyz[:, s2, :], axis=1)
        return np.linalg.norm(c1 - c2, axis=1)

    d_350_630_ca = pair_dist(ca["3.50"], ca["6.30"])
    if np.min(d_350_630_ca) < 3.5 or np.max(d_350_630_ca) > 34.0:
        raise AssertionError(
            f"{traj_id}: non-physical d_350_630_CA_A [{np.min(d_350_630_ca):.2f}, {np.max(d_350_630_ca):.2f}] A; check PBC!"
        )

    d_350_630_sc = set_min_dist(t_idx["sc_350"], t_idx["sc_630"])
    d_250_749_ca = pair_dist(ca["2.50"], ca["7.49"])
    d_250_749_polar = set_min_dist(t_idx["d250_carboxyl"], t_idx["n749_polar"])
    d_250_648_ca = pair_dist(ca["2.50"], ca["6.48"])
    d_336_648_ca = pair_dist(ca["3.36"], ca["6.48"])
    d_336_648_min = set_min_dist(t_idx["sc_336"], t_idx["sc_648"])
    d_336_648_com = set_com_dist(t_idx["sc_336"], t_idx["sc_648"])
    d_558_753_ca = pair_dist(ca["5.58"], ca["7.53"])
    d_558_753_oh = pair_dist(t_idx["y558_oh"], t_idx["y753_oh"])
    d_135_735_ca = pair_dist(ca["1.35"], ca["7.35"])

    d_242_446_ca = pair_dist(ca["2.42"], ca["4.46"])
    d_242_446_min = set_min_dist(t_idx["sc_242"], t_idx["sc_446"])
    d_242_446_com = set_com_dist(t_idx["sc_242"], t_idx["sc_446"])
    d_242sc_749ca = np.linalg.norm(
        np.mean(xyz[:, t_idx["sc_242"], :], axis=1) - xyz[:, ca["7.49"], :],
        axis=1,
    )

    chis = {k: calc_dihedrals_batch_deg(xyz, q) for k, q in t_idx["chi1_quads"].items()}

    pore_mid = np.mean(xyz[:, t_idx["pore_center_cas"], :], axis=1)
    tm_vec = np.mean(xyz[:, t_idx["ec_axis_cas"], :], axis=1) - np.mean(xyz[:, t_idx["ic_axis_cas"], :], axis=1)
    tm_hat = tm_vec / np.clip(np.linalg.norm(tm_vec, axis=1, keepdims=True), 1e-8, None)

    wat_O = t_idx["wat_O"]
    na_ions = t_idx["na_ions"]
    k_ions = t_idx["k_ions"]
    lipid_heavy = t_idx["lipid_heavy"]
    has_na = int(na_ions.size > 0)
    has_k = int(k_ions.size > 0)
    has_lipid = lipid_heavy.size > 0

    rows: list[dict[str, Any]] = []
    for f in range(n_f):
        g_frame = frame_offset + f
        pos_f = xyz[f]

        if wat_O.size > 0:
            wat_all = pos_f[wat_O]
            rel_wat = wat_all - pore_mid[f]
            box_mask = (
                (np.abs(rel_wat[:, 0]) <= 16.5)
                & (np.abs(rel_wat[:, 1]) <= 16.5)
                & (np.abs(rel_wat[:, 2]) <= 16.5)
            )
            rel_box = rel_wat[box_mask]
            wat_box = wat_all[box_mask]
            d2_pore = np.sum(rel_box * rel_box, axis=1)
            near_mask = d2_pore <= (16.5 * 16.5)
            wat_near = wat_box[near_mask]
            rel_near = rel_box[near_mask]

            z_ax = rel_near @ tm_hat[f]
            r_perp2 = np.sum(rel_near * rel_near, axis=1) - z_ax * z_ax
            n_wat_pore = int(np.sum((np.abs(z_ax) <= 12.0) & (r_perp2 <= 64.0)))
        else:
            wat_near = np.empty((0, 3), dtype=np.float64)
            rel_near = np.empty((0, 3), dtype=np.float64)
            n_wat_pore = 0

        def count_hyd(sel_idx: np.ndarray) -> int:
            if wat_near.shape[0] == 0:
                return 0
            pts = pos_f[sel_idx]
            dmin = np.min(np.linalg.norm(wat_near[:, None, :] - pts[None, :, :], axis=-1), axis=1)
            return int(np.sum(dmin <= 3.5))

        n_wat_d250 = count_hyd(t_idx["sc_250"])
        n_wat_w648 = count_hyd(t_idx["sc_648"])
        n_wat_n749 = count_hyd(t_idx["sc_749"])
        n_wat_y558 = count_hyd(t_idx["sc_558"])
        n_wat_y753 = count_hyd(t_idx["sc_753"])

        wire_mask = np.sum(rel_near * rel_near, axis=1) <= (13.5 * 13.5) if rel_near.shape[0] > 0 else np.zeros(0, dtype=bool)
        wat_wire = wat_near[wire_mask]
        n_w = wat_wire.shape[0]
        wat_adj: list[list[int]] = [[] for _ in range(n_w)]
        if n_w > 1:
            tree = cKDTree(wat_wire)
            for i_w, j_w in tree.query_pairs(r=3.5):
                wat_adj[i_w].append(j_w)
                wat_adj[j_w].append(i_w)

        d250_pts = pos_f[t_idx["d250_carboxyl"]]
        w648_pts = pos_f[t_idx["sc_648"]]
        npxxy_pts = pos_f[t_idx["npxxy_polar"]]
        y558_pt = pos_f[[t_idx["y558_oh"]]]
        y753_pt = pos_f[[t_idx["y753_oh"]]]

        hops_d250_w648 = bfs_shortest_water_hops(wat_wire, wat_adj, d250_pts, w648_pts, 3.5)
        hops_d250_npxxy = bfs_shortest_water_hops(wat_wire, wat_adj, d250_pts, npxxy_pts, 3.5)
        hops_558_753 = bfs_shortest_water_hops(wat_wire, wat_adj, y558_pt, y753_pt, 3.5)

        if has_na:
            na_pts = pos_f[na_ions]
            d_na = np.min(np.linalg.norm(na_pts[:, None, :] - d250_pts[None, :, :], axis=-1), axis=1)
            min_na = float(np.min(d_na))
            coord_na = int(np.sum(d_na < 3.2))
            prox_na = float(math.exp(-min_na / 5.0))
        else:
            min_na = 99.0
            coord_na = 0
            prox_na = 0.0

        if has_k:
            k_pts = pos_f[k_ions]
            d_k = np.min(np.linalg.norm(k_pts[:, None, :] - d250_pts[None, :, :], axis=-1), axis=1)
            min_k = float(np.min(d_k))
            coord_k = int(np.sum(d_k < 3.2))
            prox_k = float(math.exp(-min_k / 5.0))
        else:
            min_k = 99.0
            coord_k = 0
            prox_k = 0.0

        min_cation = min(min_na, min_k)
        coord_cation = coord_na + coord_k
        prox_cation = max(prox_na, prox_k)

        if has_lipid:
            lip_pts = pos_f[lipid_heavy]
            lbox_mask = (
                (np.abs(lip_pts[:, 0]) <= 30.0)
                & (np.abs(lip_pts[:, 1]) <= 30.0)
                & (np.abs(lip_pts[:, 2]) <= 30.0)
            )
            lip_near = lip_pts[lbox_mask] if np.any(lbox_mask) else lip_pts
            cleft_pts = pos_f[t_idx["tm2_tm4_cleft"]]
            portal_pts = pos_f[t_idx["tm1_tm7_portal"]]
            min_lip_cleft = float(np.min(np.linalg.norm(lip_near[:, None, :] - cleft_pts[None, :, :], axis=-1)))
            min_lip_portal = float(np.min(np.linalg.norm(lip_near[:, None, :] - portal_pts[None, :, :], axis=-1)))
        else:
            min_lip_cleft = 99.0
            min_lip_portal = 99.0

        rows.append({
            "receptor": receptor_key,
            "traj_id": traj_id,
            "gpcrmd_dyn_id": dyn_id,
            "pdb_id": pdb_id,
            "state_class": state_class,
            "frame_idx": g_frame,
            "dt_ns": dt_ns,
            "time_ns": round(g_frame * dt_ns, 4),
            "bw242_resname": t_idx["bw242_resname"],
            "bw446_resname": t_idx["bw446_resname"],
            "is_aromatic_pair_242_446": t_idx["is_aromatic_pair_242_446"],
            "has_bw446_chi1": t_idx["has_bw446_chi1"],
            "d_350_630_CA_A": float(d_350_630_ca[f]),
            "d_350_630_sc_min_A": float(d_350_630_sc[f]),
            "d_250_749_CA_A": float(d_250_749_ca[f]),
            "d_250_749_polar_A": float(d_250_749_polar[f]),
            "d_250_648_CA_A": float(d_250_648_ca[f]),
            "d_336_648_CA_A": float(d_336_648_ca[f]),
            "d_336_648_sc_min_A": float(d_336_648_min[f]),
            "d_336_648_sc_com_A": float(d_336_648_com[f]),
            "d_558_753_CA_A": float(d_558_753_ca[f]),
            "d_558_753_OH_A": float(d_558_753_oh[f]),
            "d_135_735_CA_A": float(d_135_735_ca[f]),
            "d_242_446_CA_A": float(d_242_446_ca[f]),
            "d_242_446_sc_min_A": float(d_242_446_min[f]),
            "d_242_446_sc_com_A": float(d_242_446_com[f]),
            "d_242sc_749CA_A": float(d_242sc_749ca[f]),
            "bw242_chi1_deg": float(chis["bw242_chi1_deg"][f]),
            "bw446_chi1_deg": float(chis["bw446_chi1_deg"][f]),
            "bw336_chi1_deg": float(chis["bw336_chi1_deg"][f]),
            "bw648_chi1_deg": float(chis["bw648_chi1_deg"][f]),
            "bw753_chi1_deg": float(chis["bw753_chi1_deg"][f]),
            "n_wat_core_pore": n_wat_pore,
            "n_wat_D250": n_wat_d250,
            "n_wat_W648": n_wat_w648,
            "n_wat_N749": n_wat_n749,
            "n_wat_Y558": n_wat_y558,
            "n_wat_Y753": n_wat_y753,
            "hops_D250_W648": hops_d250_w648,
            "wire_D250_W648_cond": hops_to_conductance(hops_d250_w648),
            "hops_D250_NPxxY": hops_d250_npxxy,
            "wire_D250_NPxxY_cond": hops_to_conductance(hops_d250_npxxy),
            "hops_558_753": hops_558_753,
            "wire_558_753_cond": hops_to_conductance(hops_558_753),
            "has_na_ions": has_na,
            "has_k_ions": has_k,
            "min_Na_D250_A": min_na,
            "coord_Na_D250_3p2": coord_na,
            "prox_Na_D250": prox_na,
            "min_K_D250_A": min_k,
            "coord_K_D250_3p2": coord_k,
            "prox_K_D250": prox_k,
            "min_cation_D250_A": min_cation,
            "coord_cation_D250_3p2": coord_cation,
            "prox_cation_D250": prox_cation,
            "min_lipid_TM2_TM4_cleft_A": min_lip_cleft,
            "min_lipid_TM1_TM7_portal_A": min_lip_portal,
        })

    return rows


def _is_valid_checkpoint_parquet(parquet_path: Path, expected_traj_ids: set[str] | None = None) -> bool:
    """Verify an existing Parquet checkpoint is readable, non-empty, and free of NaNs."""
    if not parquet_path.exists() or parquet_path.stat().st_size == 0:
        return False
    try:
        df = pd.read_parquet(parquet_path)
        if len(df) == 0 or int(df.isna().sum().sum()) > 0:
            return False
        if expected_traj_ids is not None and set(df["traj_id"].unique().tolist()) != expected_traj_ids:
            return False
        return True
    except Exception:
        return False


def stream_and_featurize_receptor(
    receptor_key: str,
    manifest: dict[str, Any],
    out_dir: str | Path,
    temp_dir: str | Path | None = None,
    n_workers: int = 16,
    chunk_frames: int = 500,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """Stream, PBC-wrap, featurize, and checkpoint all open GPCRmd trajectories for `receptor_key`.

    Checkpoints both per-trajectory (`<out_dir>/traj_parts/<traj_id>.parquet`) and per-receptor
    (`<out_dir>/<receptor_key>_features.parquet`) on Google Drive so interrupted Colab sessions
    resume seamlessly without re-downloading completed `.xtc` files. Every `.xtc` and `.pdb`
    temporary file is deleted immediately in `finally:`.
    """
    if receptor_key not in manifest["receptors"]:
        raise KeyError(f"Unknown receptor '{receptor_key}' in manifest.")
    rec_spec = manifest["receptors"][receptor_key]
    trajs = rec_spec["trajectories"]
    expected_ids = {str(t["traj_id"]) for t in trajs}

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    rec_parquet = out_dir_path / f"{receptor_key}_features.parquet"
    parts_dir = out_dir_path / "traj_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    if not force_recompute and _is_valid_checkpoint_parquet(rec_parquet, expected_ids):
        print(f"[{receptor_key}] Loaded Drive checkpoint {rec_parquet.name}", flush=True)
        return pd.read_parquet(rec_parquet)

    owns_temp = temp_dir is None
    work_tmp = Path(tempfile.mkdtemp(prefix=f"ws4_{receptor_key}_")) if owns_temp else Path(temp_dir)
    work_tmp.mkdir(parents=True, exist_ok=True)

    traj_dfs: list[pd.DataFrame] = []
    try:
        for t_entry in trajs:
            traj_id = str(t_entry["traj_id"])
            part_parquet = parts_dir / f"{traj_id}.parquet"
            if not force_recompute and _is_valid_checkpoint_parquet(part_parquet, {traj_id}):
                print(f"  [{receptor_key}] Resuming cached trajectory {traj_id}", flush=True)
                traj_dfs.append(pd.read_parquet(part_parquet))
                continue

            pdb_tmp = work_tmp / f"{traj_id}.pdb"
            xtc_tmp = work_tmp / f"{traj_id}.xtc"
            try:
                t0 = time.time()
                print(
                    f"  [{receptor_key}] Streaming {traj_id} ({t_entry['xtc_size_bytes'] / 1e6:.1f} MB)...",
                    flush=True,
                )
                download_parallel_http206(
                    t_entry["pdb_url"],
                    pdb_tmp,
                    expected_bytes=int(t_entry["pdb_size_bytes"]),
                    n_workers=4,
                )
                top = md.load_pdb(str(pdb_tmp)).topology
                t_idx = build_bw_topology_index(top, t_entry)

                download_parallel_http206(
                    t_entry["xtc_url"],
                    xtc_tmp,
                    expected_bytes=int(t_entry["xtc_size_bytes"]),
                    n_workers=n_workers,
                )
                verify_xtc_header_1995(xtc_tmp, expected_natoms=top.n_atoms)

                rows: list[dict[str, Any]] = []
                frame_offset = 0
                for chunk in md.iterload(str(xtc_tmp), top=top, chunk=chunk_frames):
                    xyz_A = pbc_wrap_receptor_angstrom(chunk, t_idx)
                    rows.extend(featurize_receptor_chunk(xyz_A, t_idx, receptor_key, t_entry, frame_offset))
                    frame_offset += chunk.n_frames

                df_t = pd.DataFrame(rows)
                if len(df_t) == 0 or int(df_t.isna().sum().sum()) > 0:
                    raise AssertionError(f"{traj_id}: invalid featurized DataFrame (rows={len(df_t)}).")
                tmp_part = part_parquet.with_suffix(".parquet.tmp")
                df_t.to_parquet(tmp_part, index=False)
                tmp_part.replace(part_parquet)
                traj_dfs.append(df_t)
                print(
                    f"  [{receptor_key}] Checkpointed {traj_id}: {len(df_t)} frames in {time.time() - t0:.1f}s",
                    flush=True,
                )
            finally:
                for p_clean in (xtc_tmp, pdb_tmp, xtc_tmp.with_suffix(".xtc.part"), pdb_tmp.with_suffix(".pdb.part")):
                    if p_clean.exists():
                        p_clean.unlink()

        df_rec = pd.concat(traj_dfs, ignore_index=True)
        if int(df_rec.isna().sum().sum()) > 0:
            raise AssertionError(f"{receptor_key}: NaNs detected in concatenated DataFrame!")
        tmp_rec = rec_parquet.with_suffix(".parquet.tmp")
        df_rec.to_parquet(tmp_rec, index=False)
        tmp_rec.replace(rec_parquet)
        print(f"[{receptor_key}] Saved {rec_parquet.name} ({len(df_rec)} frames across {len(trajs)} trajectories).", flush=True)
        return df_rec
    finally:
        if owns_temp and work_tmp.exists():
            shutil.rmtree(work_tmp, ignore_errors=True)
