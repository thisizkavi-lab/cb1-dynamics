"""PBC-aware physical observable extractor for WS5 prospective OpenMM shooting trajectories.

Supports both WT human CB1 (`P21554`, `M109-M411`, `F237^4.46`) and the single evolutionary
point mutant `F237L` (`Arm_C_Mutant_F237L`). Enforces strict residue-identity assertions
on receptor Chain 0 (`UniProt = local_res_idx + 109`) before extracting:
  - Cytoplasmic TM6 opening (`R214-D338`, `R214-L341`, `R214-K343` C-alpha & ionic lock)
  - Stage-3 rotamer switches (`F155^2.42`, `F/L237^4.46`, `F200^3.36`, `W356^6.48` chi1/chi2)
  - Extrahelical flip distances (`F155sc-N393CA`, `X237sc-V204CA`, `F155-X237` min contact)
  - Intracellular TM5-TM7 latch (`Y294^5.58-Y397^7.53` OH & C-alpha)
  - Buried polar/toggle coupling (`D163-W356`, `D163-N393`, `F200-W356` ring contact)
  - Explicit-solvent internal pore hydration & shortest H-bond water-wire conductance
    (`D163-NPxxY` and `D163-W356`, `C_wire = 1 / (1 + hops)`, `0.0` if disconnected)
  - Allosteric `Na+` coordination to `D163^2.50` (`min_Na_D163_A`, `coord_Na_D163_3p2`, `prox_Na_D163`)
  - Crystallographic cholesterol (`CHL1`) & phospholipid (`POPC`/`PA`/`PC`/`OL`) wedging
    into the `TM2-TM3-TM4` (`154, 155, 158, 237, 238, 241`) cleft.
"""

from __future__ import annotations

from collections import deque
import math
from typing import Any

import numpy as np
import openmm.app as app
from scipy.spatial import cKDTree


REQUIRED_WT_ANCHORS: dict[int, set[str]] = {
    109: {"MET"},
    155: {"PHE"},
    163: {"ASP"},
    173: {"SER"},
    200: {"PHE"},
    204: {"VAL"},
    210: {"THR"},  # Guard against T210A thermostabilizing mutation
    214: {"ARG"},
    237: {"PHE", "LEU"},  # WT PHE237 (Arms A/B) or mutant LEU237 (Arm C)
    257: {"CYS", "CYX"},
    264: {"CYS", "CYX"},
    273: {"GLU"},  # Guard against E273K mutation
    283: {"THR"},  # Guard against T283V mutation
    294: {"TYR"},
    338: {"ASP"},
    340: {"ARG"},  # Guard against R340E mutation
    341: {"LEU"},
    343: {"LYS"},
    356: {"TRP"},
    383: {"SER"},
    393: {"ASN"},
    397: {"TYR"},
    411: {"MET"},
}

WATER_RESNAMES = {"HOH", "WAT", "TIP3", "SOL", "TP3"}
NA_RESNAMES = {"NA", "NA+", "SOD"}
CHL_RESNAMES = {"CHL1", "CHL", "CLR", "CHOL"}
LIPID_RESNAMES = {"POPC", "POP", "PA", "PC", "OL", "POPE", "POPS"}

CORE_7TM_UNIPROT = (
    tuple(range(120, 145))
    + tuple(range(152, 176))
    + tuple(range(190, 218))
    + tuple(range(232, 252))
    + tuple(range(274, 302))
    + tuple(range(338, 366))
    + tuple(range(376, 400))
)


def _calc_dihedral_deg(pos_A: np.ndarray, q: tuple[int, int, int, int]) -> float:
    """Compute IUPAC dihedral angle in degrees [-180, 180] for 4 atom indices."""
    p0, p1, p2, p3 = pos_A[q[0]], pos_A[q[1]], pos_A[q[2]], pos_A[q[3]]
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1_norm = np.linalg.norm(b1)
    if b1_norm < 1e-8:
        return 0.0
    b1_u = b1 / b1_norm
    v = b0 - np.dot(b0, b1_u) * b1_u
    w = b2 - np.dot(b2, b1_u) * b1_u
    x = float(np.dot(v, w))
    y = float(np.dot(np.cross(b1_u, v), w))
    return float(np.degrees(np.arctan2(y, x)))


def _bfs_shortest_water_hops(
    wat_wire: np.ndarray,
    wat_adj: list[list[int]],
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    cutoff_A: float = 3.5,
) -> int:
    """Return shortest water-mediated H-bond hop count (0=direct <=3.5 A, -1=disconnected)."""
    d_direct = float(np.min(np.linalg.norm(src_pts[:, None, :] - dst_pts[None, :, :], axis=-1)))
    if d_direct <= cutoff_A:
        return 0
    if wat_wire.shape[0] == 0:
        return -1

    d_src = np.min(np.linalg.norm(wat_wire[:, None, :] - src_pts[None, :, :], axis=-1), axis=1)
    d_dst = np.min(np.linalg.norm(wat_wire[:, None, :] - dst_pts[None, :, :], axis=-1), axis=1)
    starts = np.where(d_src <= cutoff_A)[0]
    targets = set(np.where(d_dst <= cutoff_A)[0].tolist())
    if starts.size == 0 or not targets:
        return -1

    visited: dict[int, int] = {}
    q: deque[tuple[int, int]] = deque()
    for s in starts.tolist():
        if s in targets:
            return 1
        visited[s] = 1
        q.append((s, 1))

    while q:
        u, hops = q.popleft()
        nh = hops + 1
        for v in wat_adj[u]:
            if v in targets:
                return nh
            if v not in visited:
                visited[v] = nh
                q.append((v, nh))
    return -1


def build_ws5_topology_indices(topology: app.Topology) -> dict[str, Any]:
    """Validate WT/F237L CB1 Chain 0 (`M109-M411`) and precompute fast NumPy index arrays."""
    chains = list(topology.chains())
    if not chains:
        raise ValueError("Empty OpenMM Topology.")
    rec_chain = chains[0]
    rec_residues = [r for r in rec_chain.residues() if r.name not in WATER_RESNAMES | NA_RESNAMES | CHL_RESNAMES | LIPID_RESNAMES | {"CL", "CL-", "CLA"}]
    if len(rec_residues) != 303:
        raise AssertionError(
            f"Expected 303 contiguous CB1 residues (M109..M411) in receptor chain 0, found {len(rec_residues)}"
        )

    u_to_res: dict[int, app.Residue] = {}
    for local_i, res in enumerate(rec_residues):
        u_num = 109 + local_i
        u_to_res[u_num] = res
        if u_num in REQUIRED_WT_ANCHORS:
            allowed = REQUIRED_WT_ANCHORS[u_num]
            if res.name not in allowed:
                raise AssertionError(
                    f"CB1 P21554 anchor mismatch at UniProt {u_num}: got {res.name!r}, expected {allowed}"
                )

    res237_name = u_to_res[237].name

    def atom_idx(u_num: int, aname: str) -> int:
        for a in u_to_res[u_num].atoms():
            if a.name == aname:
                return a.index
        raise KeyError(f"Missing atom {aname!r} on CB1 residue {u_num} ({u_to_res[u_num].name})")

    def heavy_sc(u_nums: list[int]) -> np.ndarray:
        bb = {"N", "CA", "C", "O", "OXT"}
        out = [
            a.index
            for u in u_nums
            for a in u_to_res[u].atoms()
            if a.element is not None and a.element.symbol != "H" and a.name not in bb
        ]
        return np.array(out, dtype=np.int64)

    def ring_atoms(u_num: int, names: tuple[str, ...]) -> np.ndarray:
        amap = {a.name: a.index for a in u_to_res[u_num].atoms()}
        return np.array([amap[n] for n in names if n in amap], dtype=np.int64)

    ca_map = {u: atom_idx(u, "CA") for u in u_to_res}
    rec_res_blocks = [
        (ca_map[u], np.array([a.index for a in u_to_res[u].atoms()], dtype=np.int64))
        for u in sorted(u_to_res)
    ]
    rec_atom_set = {a.index for r in rec_residues for a in r.atoms()}

    wat_O: list[int] = []
    na_ions: list[int] = []
    chl_heavy: list[int] = []
    lipid_heavy: list[int] = []
    env_atoms: list[int] = []

    for res in topology.residues():
        r_atoms = list(res.atoms())
        if not r_atoms or r_atoms[0].index in rec_atom_set:
            continue
        for a in r_atoms:
            env_atoms.append(a.index)
        rname = res.name.upper()
        if rname in WATER_RESNAMES:
            for a in r_atoms:
                if a.element is not None and a.element.symbol == "O":
                    wat_O.append(a.index)
        elif rname in NA_RESNAMES:
            for a in r_atoms:
                na_ions.append(a.index)
        elif rname in CHL_RESNAMES:
            for a in r_atoms:
                if a.element is not None and a.element.symbol != "H":
                    chl_heavy.append(a.index)
        elif rname in LIPID_RESNAMES:
            for a in r_atoms:
                if a.element is not None and a.element.symbol != "H":
                    lipid_heavy.append(a.index)

    chi_quads = {
        "F155_chi1_deg": (atom_idx(155, "N"), atom_idx(155, "CA"), atom_idx(155, "CB"), atom_idx(155, "CG")),
        "F155_chi2_deg": (atom_idx(155, "CA"), atom_idx(155, "CB"), atom_idx(155, "CG"), atom_idx(155, "CD1")),
        "X237_chi1_deg": (atom_idx(237, "N"), atom_idx(237, "CA"), atom_idx(237, "CB"), atom_idx(237, "CG")),
        "X237_chi2_deg": (atom_idx(237, "CA"), atom_idx(237, "CB"), atom_idx(237, "CG"), atom_idx(237, "CD1")),
        "F200_chi1_deg": (atom_idx(200, "N"), atom_idx(200, "CA"), atom_idx(200, "CB"), atom_idx(200, "CG")),
        "F200_chi2_deg": (atom_idx(200, "CA"), atom_idx(200, "CB"), atom_idx(200, "CG"), atom_idx(200, "CD1")),
        "W356_chi1_deg": (atom_idx(356, "N"), atom_idx(356, "CA"), atom_idx(356, "CB"), atom_idx(356, "CG")),
        "W356_chi2_deg": (atom_idx(356, "CA"), atom_idx(356, "CB"), atom_idx(356, "CG"), atom_idx(356, "CD1")),
    }

    return {
        "res237_name": res237_name,
        "n_atoms": topology.getNumAtoms(),
        "ca": ca_map,
        "rec_res_blocks": rec_res_blocks,
        "core_7tm_cas": np.array([ca_map[u] for u in CORE_7TM_UNIPROT], dtype=np.int64),
        "pore_center_cas": np.array([ca_map[u] for u in (163, 356, 393, 397)], dtype=np.int64),
        "ec_axis_cas": np.array([ca_map[u] for u in (176, 180, 270, 368, 375)], dtype=np.int64),
        "ic_axis_cas": np.array([ca_map[u] for u in (146, 214, 298, 338, 397)], dtype=np.int64),
        "f155_sc": heavy_sc([155]),
        "x237_sc": heavy_sc([237]),
        "f200_ring": ring_atoms(200, ("CG", "CD1", "CD2", "CE1", "CE2", "CZ")),
        "w356_ring": ring_atoms(356, ("CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2")),
        "w356_heavy": heavy_sc([356]),
        "d163_carboxyl": np.array([atom_idx(163, "OD1"), atom_idx(163, "OD2")], dtype=np.int64),
        "n393_polar": np.array([atom_idx(393, "OD1"), atom_idx(393, "ND2")], dtype=np.int64),
        "n393_y397_polar": np.array(
            [atom_idx(393, "OD1"), atom_idx(393, "ND2"), atom_idx(397, "OH")], dtype=np.int64
        ),
        "r214_guan": np.array([atom_idx(214, "NH1"), atom_idx(214, "NH2"), atom_idx(214, "NE")], dtype=np.int64),
        "d338_carboxyl": np.array([atom_idx(338, "OD1"), atom_idx(338, "OD2")], dtype=np.int64),
        "y294_oh": atom_idx(294, "OH"),
        "y397_oh": atom_idx(397, "OH"),
        "tm2_tm4_cleft": heavy_sc([154, 155, 158, 237, 238, 241]),
        "chi_quads": chi_quads,
        "env_atoms": np.array(env_atoms, dtype=np.int64),
        "wat_O": np.array(wat_O, dtype=np.int64),
        "na_ions": np.array(na_ions, dtype=np.int64),
        "chl_heavy": np.array(chl_heavy, dtype=np.int64),
        "lipid_heavy": np.array(lipid_heavy, dtype=np.int64),
    }


def _unwrap_and_center_frame_angstrom(
    positions_nm: np.ndarray,
    box_lengths_nm: np.ndarray | None,
    t_idx: dict[str, Any],
) -> np.ndarray:
    """Convert single-frame positions (N, 3) from nm to A and apply exact PBC wrapping around 7TM core."""
    xyz = np.asarray(positions_nm, dtype=np.float64) * 10.0
    if box_lengths_nm is None:
        core_com = np.mean(xyz[t_idx["core_7tm_cas"]], axis=0, keepdims=True)
        return xyz - core_com

    box = np.asarray(box_lengths_nm, dtype=np.float64) * 10.0
    if np.min(box) < 30.0:
        core_com = np.mean(xyz[t_idx["core_7tm_cas"]], axis=0, keepdims=True)
        return xyz - core_com

    # 1. Unwrap covalent peptide chain 0 sequentially
    rblocks = t_idx["rec_res_blocks"]
    first_ca, first_atoms = rblocks[0]
    prev_ca = xyz[first_ca].copy()
    xyz[first_atoms] -= box * np.round((xyz[first_atoms] - prev_ca) / box)
    for r_ca, r_atoms in rblocks[1:]:
        ca_unw = xyz[r_ca] - box * np.round((xyz[r_ca] - prev_ca) / box)
        xyz[r_atoms] -= box * np.round((xyz[r_atoms] - ca_unw) / box)
        prev_ca = ca_unw

    # 2. Center 7TM C-alpha centroid at origin and wrap solvent/lipids/ions into [-box/2, +box/2)
    core_com = np.mean(xyz[t_idx["core_7tm_cas"]], axis=0, keepdims=True)
    xyz -= core_com
    env = t_idx["env_atoms"]
    if env.size > 0:
        xyz[env] -= box * np.round(xyz[env] / box)
    return xyz


def extract_ws5_frame_observables(
    positions_nm: np.ndarray,
    box_lengths_nm: np.ndarray | None,
    t_idx: dict[str, Any],
    time_ns: float = 0.0,
    step: int = 0,
) -> dict[str, Any]:
    """Compute all Stage-2 and Stage-3 physical observables on a single simulation frame."""
    pos = _unwrap_and_center_frame_angstrom(positions_nm, box_lengths_nm, t_idx)
    ca = t_idx["ca"]

    def pdist(i: int, j: int) -> float:
        return float(np.linalg.norm(pos[i] - pos[j]))

    def min_dist(s1: np.ndarray, s2: np.ndarray) -> float:
        if s1.size == 0 or s2.size == 0:
            return 99.0
        return float(np.min(np.linalg.norm(pos[s1][:, None, :] - pos[s2][None, :, :], axis=-1)))

    r214_d338 = pdist(ca[214], ca[338])
    r214_l341 = pdist(ca[214], ca[341])
    r214_k343 = pdist(ca[214], ca[343])
    r214_d338_ionic = min_dist(t_idx["r214_guan"], t_idx["d338_carboxyl"])

    f155sc_n393ca = float(np.linalg.norm(np.mean(pos[t_idx["f155_sc"]], axis=0) - pos[ca[393]]))
    x237sc_v204ca = float(np.linalg.norm(np.mean(pos[t_idx["x237_sc"]], axis=0) - pos[ca[204]]))
    f155_x237_min = min_dist(t_idx["f155_sc"], t_idx["x237_sc"])

    y294_y397_oh = pdist(t_idx["y294_oh"], t_idx["y397_oh"])
    y294_y397_ca = pdist(ca[294], ca[397])
    d163_w356_ca = pdist(ca[163], ca[356])
    d163_n393_ca = pdist(ca[163], ca[393])
    d163_n393_polar = min_dist(t_idx["d163_carboxyl"], t_idx["n393_polar"])
    f200_w356_ring_min = min_dist(t_idx["f200_ring"], t_idx["w356_ring"])

    chis = {k: _calc_dihedral_deg(pos, q) for k, q in t_idx["chi_quads"].items()}

    # Explicit internal pore hydration & water-wire conductance
    pore_mid = np.mean(pos[t_idx["pore_center_cas"]], axis=0)
    tm_vec = np.mean(pos[t_idx["ec_axis_cas"]], axis=0) - np.mean(pos[t_idx["ic_axis_cas"]], axis=0)
    tm_hat = tm_vec / max(float(np.linalg.norm(tm_vec)), 1e-8)

    wat_O = t_idx["wat_O"]
    if wat_O.size > 0:
        wat_all = pos[wat_O]
        rel = wat_all - pore_mid
        bmask = (np.abs(rel[:, 0]) <= 16.5) & (np.abs(rel[:, 1]) <= 16.5) & (np.abs(rel[:, 2]) <= 16.5)
        rel_b = rel[bmask]
        wat_b = wat_all[bmask]
        d2 = np.sum(rel_b * rel_b, axis=1)
        nmask = d2 <= (16.5 * 16.5)
        wat_near = wat_b[nmask]
        rel_near = rel_b[nmask]
        z_ax = rel_near @ tm_hat
        r_perp2 = np.sum(rel_near * rel_near, axis=1) - z_ax * z_ax
        n_wat_pore = int(np.sum((np.abs(z_ax) <= 12.0) & (r_perp2 <= 64.0)))

        wmask = np.sum(rel_near * rel_near, axis=1) <= (13.5 * 13.5)
        wat_wire = wat_near[wmask]
        n_w = wat_wire.shape[0]
        wat_adj: list[list[int]] = [[] for _ in range(n_w)]
        if n_w > 1:
            tree = cKDTree(wat_wire)
            for iw, jw in tree.query_pairs(r=3.5):
                wat_adj[iw].append(jw)
                wat_adj[jw].append(iw)
        hops_ny = _bfs_shortest_water_hops(wat_wire, wat_adj, pos[t_idx["d163_carboxyl"]], pos[t_idx["n393_y397_polar"]], 3.5)
        hops_w356 = _bfs_shortest_water_hops(wat_wire, wat_adj, pos[t_idx["d163_carboxyl"]], pos[t_idx["w356_heavy"]], 3.5)
    else:
        n_wat_pore = 0
        hops_ny = -1
        hops_w356 = -1

    cond_ny = 0.0 if hops_ny < 0 else float(1.0 / (1.0 + hops_ny))
    cond_w356 = 0.0 if hops_w356 < 0 else float(1.0 / (1.0 + hops_w356))

    # Na+ ion coordination to D163^2.50
    na_ions = t_idx["na_ions"]
    if na_ions.size > 0:
        d_na = np.min(np.linalg.norm(pos[na_ions][:, None, :] - pos[t_idx["d163_carboxyl"]][None, :, :], axis=-1), axis=1)
        min_na = float(np.min(d_na))
        coord_na = int(np.sum(d_na < 3.2))
        prox_na = float(math.exp(-min_na / 5.0))
    else:
        min_na = 99.0
        coord_na = 0
        prox_na = 0.0

    # Cholesterol (CHL1) & POPC heavy-atom wedging into TM2-TM3-TM4 cleft
    cleft_pts = pos[t_idx["tm2_tm4_cleft"]]
    chl_idx = t_idx["chl_heavy"]
    if chl_idx.size > 0:
        min_chl_cleft = float(np.min(np.linalg.norm(pos[chl_idx][:, None, :] - cleft_pts[None, :, :], axis=-1)))
    else:
        min_chl_cleft = 99.0

    lip_idx = t_idx["lipid_heavy"]
    if lip_idx.size > 0:
        lip_pts = pos[lip_idx]
        lmask = (np.abs(lip_pts[:, 0]) <= 28.0) & (np.abs(lip_pts[:, 1]) <= 28.0) & (np.abs(lip_pts[:, 2]) <= 28.0)
        lip_sub = lip_pts[lmask]
        if lip_sub.shape[0] > 0:
            min_popc_cleft = float(np.min(np.linalg.norm(lip_sub[:, None, :] - cleft_pts[None, :, :], axis=-1)))
        else:
            min_popc_cleft = 99.0
    else:
        min_popc_cleft = 99.0

    return {
        "step": int(step),
        "time_ns": float(time_ns),
        "res237_name": t_idx["res237_name"],
        "R214_D338_CA_A": r214_d338,
        "R214_L341_CA_A": r214_l341,
        "R214_K343_CA_A": r214_k343,
        "R214_D338_ionic_min_A": r214_d338_ionic,
        "F155_chi1_deg": chis["F155_chi1_deg"],
        "F155_chi2_deg": chis["F155_chi2_deg"],
        "X237_chi1_deg": chis["X237_chi1_deg"],
        "X237_chi2_deg": chis["X237_chi2_deg"],
        "F200_chi1_deg": chis["F200_chi1_deg"],
        "F200_chi2_deg": chis["F200_chi2_deg"],
        "W356_chi1_deg": chis["W356_chi1_deg"],
        "W356_chi2_deg": chis["W356_chi2_deg"],
        "F155sc_N393CA_A": f155sc_n393ca,
        "X237sc_V204CA_A": x237sc_v204ca,
        "F155_X237_sc_min_A": f155_x237_min,
        "Y294_Y397_OH_A": y294_y397_oh,
        "Y294_Y397_CA_A": y294_y397_ca,
        "D163_W356_CA_A": d163_w356_ca,
        "D163_N393_CA_A": d163_n393_ca,
        "D163_N393_polar_A": d163_n393_polar,
        "F200_W356_ring_min_A": f200_w356_ring_min,
        "n_wat_pore": n_wat_pore,
        "hops_d163_ny": hops_ny,
        "wire_conductance_d163_ny": cond_ny,
        "hops_d163_w356": hops_w356,
        "wire_conductance_d163_w356": cond_w356,
        "min_Na_D163_A": min_na,
        "coord_Na_D163_3p2": coord_na,
        "prox_Na_D163": prox_na,
        "min_CHL1_cleft_A": min_chl_cleft,
        "min_POPC_cleft_A": min_popc_cleft,
        "min_lipid_cleft_A": min(min_chl_cleft, min_popc_cleft),
    }
