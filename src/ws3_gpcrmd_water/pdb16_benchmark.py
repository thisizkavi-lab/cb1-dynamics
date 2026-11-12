"""Quantitative 16-PDB experimental structural dichotomy benchmark for human CB1 (P21554).

Programmatically parses deposited mmCIF atomic coordinates across all 16 representative
human CB1 experimental structures, strictly enforcing:
  - +122 TM6-H8 auth_seq_id offset on ICL3-flavodoxin constructs 5XR8 and 7V3Z,
  - Chain D selection on 8GHV (Chain A/R as appropriate on all other entries),
  - Canonical P21554 amino-acid identity verification on every resolved microswitch.

Outputs:
  - data/processed/ws3/pdb16_structural_benchmark.parquet
  - results/ws3/figures/fig_pdb16_dichotomy.png
"""

from __future__ import annotations

import math
import os
import urllib.request
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/Users/kavi/Documents/artico/cb1-dynamics")
INVENTORY_PDB_DIR = PROJECT_ROOT / "data" / "inventory" / "pdb_structures"
WS3_MMCIF_DIR = PROJECT_ROOT / "data" / "processed" / "ws3" / "pdb_mmcif"
OUT_PARQUET = PROJECT_ROOT / "data" / "processed" / "ws3" / "pdb16_structural_benchmark.parquet"
OUT_FIG = PROJECT_ROOT / "results" / "ws3" / "figures" / "fig_pdb16_dichotomy.png"

# Canonical 1-letter to 3-letter mapping for residue-identity assertions
AA1_TO_AA3 = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}

# Canonical human CB1 (P21554) identities at audited positions, plus known engineered mutations
EXPECTED_RESIDUES = {
    154: {"HIS", "LEU"},  # H154L in 7V3Z
    155: {"PHE"},
    163: {"ASP"},
    173: {"SER"},
    200: {"PHE"},
    203: {"SER", "LYS"},  # S203K in 6KQI, 7FEE, 9BA0
    210: {"THR", "ALA", "ILE"},  # T210A / T210I in engineered constructs
    214: {"ARG"},
    237: {"PHE"},
    268: {"PHE"},
    294: {"TYR"},
    300: {"LYS"},
    338: {"ASP"},
    341: {"LEU"},
    343: {"LYS"},
    356: {"TRP"},
    383: {"SER"},
    393: {"ASN", "ASP"},  # N393D in 7FEE
    397: {"TYR"},
}

ORTHOSTERIC_CAGE_RESIDUES = [173, 174, 177, 193, 196, 268, 279, 359, 363, 379, 383]

CHI_ATOMS = {
    "PHE": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
    "TYR": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
    "TRP": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
}

RING_HEAVY_ATOMS = {
    "PHE": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "TRP": ("CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
}

PDB16_SPECS: list[dict[str, Any]] = [
    {
        "pdb_id": "5TGZ", "cb1_chain": "A", "tm6_h8_offset": 0,
        "method": "X-ray", "resolution_A": 2.80,
        "state_class": "Inactive",
        "ligands": "AM6538 (antagonist)", "transducer": "None (ICL3 Flavodoxin)",
        "is_wt_tm_core": False,
    },
    {
        "pdb_id": "5U09", "cb1_chain": "A", "tm6_h8_offset": 0,
        "method": "X-ray", "resolution_A": 2.60,
        "state_class": "Inactive",
        "ligands": "Taranabant (inv. agonist)", "transducer": "None (ICL3 PGS)",
        "is_wt_tm_core": False,
    },
    {
        "pdb_id": "9BA0", "cb1_chain": "R", "tm6_h8_offset": 0,
        "method": "Cryo-EM", "resolution_A": 3.13,
        "state_class": "Inactive",
        "ligands": "MRI-1891 (biased inv. agonist)", "transducer": "CNb36 + PGS",
        "is_wt_tm_core": False,
    },
    {
        "pdb_id": "6KQI", "cb1_chain": "A", "tm6_h8_offset": 0,
        "method": "X-ray", "resolution_A": 3.25,
        "state_class": "NAM_trapped",
        "ligands": "CP55,940 + ORG27569 (NAM)", "transducer": "None (ICL3 PGS)",
        "is_wt_tm_core": False,
    },
    {
        "pdb_id": "7FEE", "cb1_chain": "A", "tm6_h8_offset": 0,
        "method": "X-ray", "resolution_A": 2.70,
        "state_class": "Mutant_locked_PAM",
        "ligands": "CP55,940 + ZCZ011 (PAM)", "transducer": "None (ICL3 PGS, N393D)",
        "is_wt_tm_core": False,
    },
    {
        "pdb_id": "5XRA", "cb1_chain": "A", "tm6_h8_offset": 0,
        "method": "X-ray", "resolution_A": 2.80,
        "state_class": "Active_like_no_transducer",
        "ligands": "AM11542 + CLR", "transducer": "None (ICL3 Flavodoxin)",
        "is_wt_tm_core": False,
    },
    {
        "pdb_id": "5XR8", "cb1_chain": "A", "tm6_h8_offset": 122,
        "method": "X-ray", "resolution_A": 2.95,
        "state_class": "Active_like_no_transducer",
        "ligands": "AM841 + CLR", "transducer": "None (ICL3 Flavodoxin)",
        "is_wt_tm_core": False,
    },
    {
        "pdb_id": "7V3Z", "cb1_chain": "A", "tm6_h8_offset": 122,
        "method": "X-ray", "resolution_A": 3.29,
        "state_class": "Active_like_no_transducer",
        "ligands": "CP55,940 + CLR", "transducer": "None (ICL3 Flavodoxin)",
        "is_wt_tm_core": False,
    },
    {
        "pdb_id": "6N4B", "cb1_chain": "R", "tm6_h8_offset": 0,
        "method": "Cryo-EM", "resolution_A": 3.00,
        "state_class": "Active_Gi",
        "ligands": "MDMB-FUBINACA + CLR", "transducer": "Gi1 + scFv16",
        "is_wt_tm_core": True,
    },
    {
        "pdb_id": "6KPG", "cb1_chain": "R", "tm6_h8_offset": 0,
        "method": "Cryo-EM", "resolution_A": 3.00,
        "state_class": "Active_Gi",
        "ligands": "AM841", "transducer": "Gi1 + scFv16",
        "is_wt_tm_core": False,
    },
    {
        "pdb_id": "7WV9", "cb1_chain": "R", "tm6_h8_offset": 0,
        "method": "Cryo-EM", "resolution_A": 3.36,
        "state_class": "Active_Gi",
        "ligands": "CP55,940 + ZCZ011 (PAM)", "transducer": "Gi2 + scFv16",
        "is_wt_tm_core": True,
    },
    {
        "pdb_id": "8GHV", "cb1_chain": "D", "tm6_h8_offset": 0,
        "method": "Cryo-EM", "resolution_A": 2.80,
        "state_class": "Active_Gi",
        "ligands": "AMG315 (eCB analog)", "transducer": "Gi1 + scFv16",
        "is_wt_tm_core": True,
    },
    {
        "pdb_id": "8IKG", "cb1_chain": "R", "tm6_h8_offset": 0,
        "method": "Cryo-EM", "resolution_A": 3.40,
        "state_class": "Active_Gi",
        "ligands": "Q2B (biased ago-PAM)", "transducer": "Gi1 + scFv16",
        "is_wt_tm_core": True,
    },
    {
        "pdb_id": "9B54", "cb1_chain": "R", "tm6_h8_offset": 0,
        "method": "Cryo-EM", "resolution_A": 2.86,
        "state_class": "Active_Gi",
        "ligands": "VIP36 (cryptic biased agonist)", "transducer": "Gi1 + scFv16",
        "is_wt_tm_core": True,
    },
    {
        "pdb_id": "8WU1", "cb1_chain": "R", "tm6_h8_offset": 0,
        "method": "Cryo-EM", "resolution_A": 3.20,
        "state_class": "Active_BetaArrestin1",
        "ligands": "MDMB-FUBINACA", "transducer": "beta-arrestin-1 + Fab30",
        "is_wt_tm_core": True,
    },
    {
        "pdb_id": "8WRZ", "cb1_chain": "R", "tm6_h8_offset": 0,
        "method": "Cryo-EM", "resolution_A": 3.60,
        "state_class": "Active_BetaArrestin1",
        "ligands": "AM841", "transducer": "beta-arrestin-1 + scFv30",
        "is_wt_tm_core": False,
    },
]


def ensure_mmcif(pdb_id: str) -> Path:
    """Return local mmCIF path, downloading from RCSB into ws3/pdb_mmcif/ if absent."""
    inv_path = INVENTORY_PDB_DIR / f"{pdb_id}.cif"
    if inv_path.exists() and inv_path.stat().st_size > 10_000:
        return inv_path
    WS3_MMCIF_DIR.mkdir(parents=True, exist_ok=True)
    ws3_path = WS3_MMCIF_DIR / f"{pdb_id}.cif"
    if not ws3_path.exists() or ws3_path.stat().st_size < 10_000:
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        if not data.startswith(b"data_"):
            raise ValueError(f"Invalid mmCIF payload downloaded for {pdb_id}")
        ws3_path.write_bytes(data)
    return ws3_path


def parse_mmcif_atoms(cif_path: Path) -> tuple[dict[tuple[str, int], dict[str, Any]], list[np.ndarray]]:
    """Parse ATOM records (model 1, altloc . or A) and CLR heavy-atom coordinates from mmCIF."""
    lines = cif_path.read_text(encoding="utf-8", errors="replace").splitlines()
    # Locate _atom_site loop headers
    i = 0
    n = len(lines)
    col_names: list[str] = []
    while i < n:
        if lines[i].strip() == "loop_" and i + 1 < n and lines[i + 1].strip().startswith("_atom_site."):
            i += 1
            while i < n and lines[i].strip().startswith("_atom_site."):
                col_names.append(lines[i].strip().split(".")[1])
                i += 1
            break
        i += 1
    if not col_names:
        raise ValueError(f"No _atom_site loop found in {cif_path}")

    idx = {name: k for k, name in enumerate(col_names)}
    residues: dict[tuple[str, int], dict[str, Any]] = {}
    clr_coords: list[np.ndarray] = []

    while i < n:
        raw = lines[i].strip()
        i += 1
        if not raw or raw.startswith("#") or raw.startswith("loop_") or raw.startswith("_"):
            if raw.startswith("#") or raw.startswith("loop_") or raw.startswith("_"):
                break
            continue
        parts = raw.split()
        if len(parts) < len(col_names):
            continue
        group = parts[idx["group_PDB"]]
        model_num = parts[idx["pdbx_PDB_model_num"]] if "pdbx_PDB_model_num" in idx else "1"
        if model_num not in ("1", "?", "."):
            continue
        alt = parts[idx["label_alt_id"]]
        if alt not in (".", "?", "A"):
            continue
        type_sym = parts[idx["type_symbol"]]
        if type_sym in ("H", "D"):
            continue

        comp = parts[idx["auth_comp_id"]] if "auth_comp_id" in idx else parts[idx["label_comp_id"]]
        x = float(parts[idx["Cartn_x"]])
        y = float(parts[idx["Cartn_y"]])
        z = float(parts[idx["Cartn_z"]])
        coord = np.array([x, y, z], dtype=np.float64)

        if comp in ("CLR", "CHL", "Y01"):
            clr_coords.append(coord)

        if group != "ATOM":
            continue
        asym = parts[idx["auth_asym_id"]]
        seq_str = parts[idx["auth_seq_id"]]
        ins_code = parts[idx["pdbx_PDB_ins_code"]] if "pdbx_PDB_ins_code" in idx else "?"
        if ins_code not in ("?", "."):
            continue
        try:
            auth_seq = int(seq_str)
        except ValueError:
            continue

        atom_name = parts[idx["label_atom_id"]].strip('"')
        key = (asym, auth_seq)
        if key not in residues:
            residues[key] = {"resname": comp, "atoms": {}}
        if atom_name not in residues[key]["atoms"]:
            residues[key]["atoms"][atom_name] = coord

    return residues, clr_coords


def calc_dihedral_deg(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Compute IUPAC torsion angle in degrees [-180, 180]."""
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1_norm = np.linalg.norm(b1)
    if b1_norm < 1e-8:
        return float("nan")
    b1_u = b1 / b1_norm
    v = b0 - np.dot(b0, b1_u) * b1_u
    w = b2 - np.dot(b2, b1_u) * b1_u
    x = float(np.dot(v, w))
    y = float(np.dot(np.cross(b1_u, v), w))
    return math.degrees(math.atan2(y, x))


def extract_structure_row(spec: dict[str, Any]) -> dict[str, Any]:
    """Extract verified structural coordinates and enforce amino-acid identity guards."""
    pdb_id = spec["pdb_id"]
    chain = spec["cb1_chain"]
    tm6_offset = spec["tm6_h8_offset"]
    cif_path = ensure_mmcif(pdb_id)
    residues, clr_coords = parse_mmcif_atoms(cif_path)

    def get_res(uniprot_pos: int) -> dict[str, Any]:
        auth_pos = uniprot_pos + (tm6_offset if uniprot_pos >= 320 else 0)
        key = (chain, auth_pos)
        if key not in residues:
            raise KeyError(f"{pdb_id} missing residue UniProt {uniprot_pos} (chain {chain}, auth {auth_pos})")
        res = residues[key]
        if uniprot_pos in EXPECTED_RESIDUES and res["resname"] not in EXPECTED_RESIDUES[uniprot_pos]:
            raise AssertionError(
                f"Residue guard failed on {pdb_id} pos {uniprot_pos} (auth {auth_pos}): "
                f"got {res['resname']}, expected {EXPECTED_RESIDUES[uniprot_pos]}"
            )
        return res

    def ca_dist(p1: int, p2: int) -> float:
        r1, r2 = get_res(p1), get_res(p2)
        return float(np.linalg.norm(r1["atoms"]["CA"] - r2["atoms"]["CA"]))

    def get_chis(pos: int) -> tuple[float, float, bool]:
        res = get_res(pos)
        rname = res["resname"]
        chi1_names, chi2_names = CHI_ATOMS[rname]
        atoms = res["atoms"]
        if not all(a in atoms for a in chi1_names):
            return float("nan"), float("nan"), False
        c1 = calc_dihedral_deg(*(atoms[a] for a in chi1_names))
        c2 = (
            calc_dihedral_deg(*(atoms[a] for a in chi2_names))
            if all(a in atoms for a in chi2_names)
            else float("nan")
        )
        return round(c1, 2), round(c2, 2), True

    f200_chi1, f200_chi2, _ = get_chis(200)
    w356_chi1, w356_chi2, _ = get_chis(356)
    f155_chi1, f155_chi2, _ = get_chis(155)
    f237_chi1, f237_chi2, f237_resolved = get_chis(237)

    # F155 sidechain heavy-atom centroid (beyond CA) to N393 CA
    r155 = get_res(155)
    sc155 = [v for k, v in r155["atoms"].items() if k not in ("N", "CA", "C", "O")]
    f155_com = np.mean(np.stack(sc155, axis=0), axis=0)
    n393_ca = get_res(393)["atoms"]["CA"]
    f155sc_n393ca = float(np.linalg.norm(f155_com - n393_ca))

    # F155-F237 minimum aromatic ring heavy-atom distance (and CB-CB distance)
    r237 = get_res(237)
    ring155 = [r155["atoms"][a] for a in RING_HEAVY_ATOMS["PHE"] if a in r155["atoms"]]
    ring237 = [r237["atoms"][a] for a in RING_HEAVY_ATOMS["PHE"] if a in r237["atoms"]]
    if len(ring155) >= 3 and len(ring237) >= 3:
        dmat = np.linalg.norm(np.stack(ring155)[:, None, :] - np.stack(ring237)[None, :, :], axis=-1)
        f155_f237_ring_min = float(np.min(dmat))
    else:
        f155_f237_ring_min = float("nan")

    f155_f237_cb = (
        float(np.linalg.norm(r155["atoms"]["CB"] - r237["atoms"]["CB"]))
        if "CB" in r155["atoms"] and "CB" in r237["atoms"]
        else float("nan")
    )

    # Y294-Y397 phenolic OH distance
    y294_oh = get_res(294)["atoms"]["OH"]
    y397_oh = get_res(397)["atoms"]["OH"]
    y294_y397_oh = float(np.linalg.norm(y294_oh - y397_oh))

    # Orthosteric pocket cage radius of gyration across conserved 11 pocket residues
    cage_cas = np.stack([get_res(p)["atoms"]["CA"] for p in ORTHOSTERIC_CAGE_RESIDUES], axis=0)
    cage_center = np.mean(cage_cas, axis=0)
    ortho_cage_rg = float(np.sqrt(np.mean(np.sum((cage_cas - cage_center) ** 2, axis=1))))

    # Check crystallographic cholesterol wedging at TM2-TM4 cleft (<4.2 A to H154/S158/F237/C238/W241)
    cleft_atoms = []
    for p in (154, 158, 237, 238, 241):
        key = (chain, p)
        if key in residues:
            cleft_atoms.extend(residues[key]["atoms"].values())
    clr_wedged = False
    min_clr_cleft_A = float("nan")
    if clr_coords and cleft_atoms:
        c_arr = np.stack(clr_coords, axis=0)
        p_arr = np.stack(cleft_atoms, axis=0)
        min_clr_cleft_A = round(float(np.min(np.linalg.norm(c_arr[:, None, :] - p_arr[None, :, :], axis=-1))), 2)
        clr_wedged = min_clr_cleft_A < 4.2

    return {
        "pdb_id": pdb_id,
        "state_class": spec["state_class"],
        "method": spec["method"],
        "resolution_A": spec["resolution_A"],
        "cb1_chain": chain,
        "tm6_h8_auth_offset": tm6_offset,
        "is_wt_tm_core": spec["is_wt_tm_core"],
        "ligands": spec["ligands"],
        "transducer": spec["transducer"],
        "R214_D338_CA_A": round(ca_dist(214, 338), 2),
        "R214_L341_CA_A": round(ca_dist(214, 341), 2),
        "R214_K343_CA_A": round(ca_dist(214, 343), 2),
        "F200_chi1_deg": f200_chi1,
        "F200_chi2_deg": f200_chi2,
        "W356_chi1_deg": w356_chi1,
        "W356_chi2_deg": w356_chi2,
        "F155_chi1_deg": f155_chi1,
        "F155_chi2_deg": f155_chi2,
        "F237_chi1_deg": f237_chi1,
        "F237_chi2_deg": f237_chi2,
        "F237_sidechain_resolved": f237_resolved,
        "F155sc_N393CA_A": round(f155sc_n393ca, 2),
        "F155_F237_CA_A": round(ca_dist(155, 237), 2),
        "F155_F237_CB_A": round(f155_f237_cb, 2),
        "F155_F237_ring_min_A": round(f155_f237_ring_min, 2) if not math.isnan(f155_f237_ring_min) else float("nan"),
        "Y294_Y397_OH_A": round(y294_y397_oh, 2),
        "D163_W356_CA_A": round(ca_dist(163, 356), 2),
        "D163_N393_CA_A": round(ca_dist(163, 393), 2),
        "F200_W356_CA_A": round(ca_dist(200, 356), 2),
        "S173_S383_CA_A": round(ca_dist(173, 383), 2),
        "S173_F268_CA_A": round(ca_dist(173, 268), 2),
        "ortho_cage_rg_A": round(ortho_cage_rg, 2),
        "R214_K300_CA_A": round(ca_dist(214, 300), 2),
        "clr_wedged_tm2_tm4": clr_wedged,
        "min_clr_tm2_tm4_A": min_clr_cleft_A,
    }


def build_pdb16_benchmark(out_parquet: Path = OUT_PARQUET, out_fig: Path = OUT_FIG) -> pd.DataFrame:
    """Build the 16-PDB structural benchmark table and generate fig_pdb16_dichotomy.png."""
    rows = [extract_structure_row(spec) for spec in PDB16_SPECS]
    df = pd.DataFrame(rows)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    plot_pdb16_dichotomy(df, out_fig)
    return df


def plot_pdb16_dichotomy(df: pd.DataFrame, out_fig: Path) -> None:
    """Render publication-grade 4-panel structural dichotomy figure (300 DPI)."""
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
    })

    palette = {
        "Inactive": ("#4C566A", "s", "Inactive (5TGZ, 5U09, 9BA0)"),
        "NAM_trapped": ("#D08770", "D", "NAM ORG27569-trapped (6KQI)"),
        "Mutant_locked_PAM": ("#EBCB8B", "P", "Mutant-locked + PAM (7FEE)"),
        "Active_like_no_transducer": ("#BF616A", "^", "Agonist crystal, NO transducer (5XRA, 5XR8, 7V3Z)"),
        "Active_Gi": ("#2E8B57", "o", "Active Gi-coupled (6N4B, 6KPG, 7WV9, 8GHV, 8IKG, 9B54)"),
        "Active_BetaArrestin1": ("#5E81AC", "v", "Active β-arrestin-1 (8WU1, 8WRZ)"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.6), dpi=300)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # --- Panel A: TM6 Opening vs F155^2.42 Outward Clearance ---
    ax_a.axvspan(13.8, 17.8, color="#E8F4EA", alpha=0.45, zorder=0)
    ax_a.axhspan(8.5, 13.0, xmin=0.55, xmax=1.0, color="#FDE8E8", alpha=0.55, zorder=0)
    ax_a.axhline(13.5, color="#555555", ls="--", lw=1.0, zorder=1)
    ax_a.axvline(13.8, color="#555555", ls=":", lw=1.0, zorder=1)

    offsets_a = {
        "5U09": (-0.62, -0.58),
        "9BA0": (-0.72, 0.42),
        "6KQI": (0.32, -0.48),
        "5TGZ": (-0.32, -0.62),
        "7FEE": (0.28, 0.28),
        "5XRA": (-1.05, 0.20),
        "5XR8": (0.35, -0.25),
        "7V3Z": (0.28, 0.20),
        "8IKG": (-1.08, 0.36),
        "7WV9": (-1.18, -0.02),
        "8WU1": (0.42, 0.30),
        "6KPG": (-1.18, -0.42),
        "6N4B": (-0.90, -0.78),
        "9B54": (0.48, -0.10),
        "8GHV": (0.45, -0.58),
        "8WRZ": (-1.05, 0.16),
    }

    for state, (col, marker, label) in palette.items():
        sub = df[df["state_class"] == state]
        ax_a.scatter(
            sub["R214_D338_CA_A"], sub["F155sc_N393CA_A"],
            c=col, marker=marker, s=90, edgecolors="black", linewidths=0.8, label=label, zorder=4
        )
        for _, r in sub.iterrows():
            dx, dy = offsets_a.get(r["pdb_id"], (0.20, 0.20))
            ax_a.annotate(
                r["pdb_id"],
                xy=(r["R214_D338_CA_A"], r["F155sc_N393CA_A"]),
                xytext=(r["R214_D338_CA_A"] + dx, r["F155sc_N393CA_A"] + dy),
                arrowprops=dict(arrowstyle="-", lw=0.55, color="#777777"),
                fontsize=7.6, fontweight="bold", color="#222222", zorder=5
            )

    ax_a.text(
        14.05, 10.25,
        "Transducer-free agonist trap:\nTM6 open (14.2–14.7 Å),\nYET F155/F237 inward\n(CLR-wedged in 5XRA)",
        fontsize=8.0, color="#8B1E24", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BF616A", alpha=0.92)
    )
    ax_a.set_xlabel(r"Cytoplasmic TM6 Opening $d(\mathrm{R214^{3.50}\!-\!D338^{6.30}}\ \mathrm{C\alpha})$ (Å)")
    ax_a.set_ylabel(r"$\mathrm{F155^{2.42}}$ Outward Clearance $d(\mathrm{F155_{sc}\!-\!N393^{7.49}\ C\alpha})$ (Å)")
    ax_a.set_title("(a) Open TM6 ≠ Active Core: F155–F237 Dichotomy Across 16 Structures")
    ax_a.set_xlim(5.8, 18.2)
    ax_a.set_ylim(8.5, 17.6)
    ax_a.grid(True, alpha=0.25)

    # --- Panel B: Concerted F155-F237 Rotamer Switch + Y294-Y397 Tyr Lock ---
    for state, (col, marker, _) in palette.items():
        sub = df[df["state_class"] == state]
        ax_b.scatter(
            sub["F155_chi1_deg"], sub["Y294_Y397_OH_A"],
            c=col, marker=marker, s=90, edgecolors="black", linewidths=0.8, zorder=4
        )

    offsets_b = {
        "6KQI": (-28.0, 0.35),
        "5U09": (8.0, 0.22),
        "9BA0": (8.0, -0.35),
        "7FEE": (-28.0, 0.35),
        "5TGZ": (-27.0, -0.55),
        "7V3Z": (8.0, 0.22),
        "5XR8": (-28.0, 0.32),
        "5XRA": (-28.0, -0.52),
        "8IKG": (-22.0, 0.52),
        "8WRZ": (-27.0, 0.30),
        "8WU1": (-26.0, -0.52),
        "6N4B": (-28.0, -0.04),
        "7WV9": (9.0, -0.52),
        "6KPG": (9.5, -0.10),
        "8GHV": (9.5, 0.14),
        "9B54": (8.5, 0.42),
    }
    for _, r in df.iterrows():
        dx, dy = offsets_b.get(r["pdb_id"], (6.0, 0.20))
        ax_b.annotate(
            r["pdb_id"],
            xy=(r["F155_chi1_deg"], r["Y294_Y397_OH_A"]),
            xytext=(r["F155_chi1_deg"] + dx, r["Y294_Y397_OH_A"] + dy),
            arrowprops=dict(arrowstyle="-", lw=0.55, color="#777777"),
            fontsize=7.5, fontweight="bold", color="#222222", zorder=5
        )
    ax_b.axhline(4.4, color="#2E8B57", ls="--", lw=1.0, zorder=1)
    ax_b.axvline(0.0, color="#555555", ls=":", lw=1.0, zorder=1)
    ax_b.axvspan(-108, -38, ymin=0.0, ymax=0.22, color="#E8F4EA", alpha=0.5, zorder=0)
    ax_b.text(
        -106, 5.25,
        "Active signaling quadrant:\nF155 gauche- (-57° to -87°)\nF237 trans (154° to 190°)\nY294–Y397 H-bond (2.9–4.2 Å)",
        fontsize=8.0, color="#1D5B38", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2E8B57", alpha=0.92)
    )
    ax_b.text(
        42, 6.45,
        "5XRA / 5XR8 / 7V3Z:\nY294–Y397 unformed (5.1–5.5 Å)\n& F155/F237 still inward!",
        fontsize=8.0, color="#8B1E24", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BF616A", alpha=0.92)
    )
    ax_b.set_xlabel(r"$\mathrm{F155^{2.42}}$ Rotamer $\chi_1$ (°)  [Inward $+59^\circ..+167^\circ$ $\rightarrow$ Outward $-87^\circ..-57^\circ$]")
    ax_b.set_ylabel(r"Intracellular Tyr Lock $d(\mathrm{Y294^{5.58}\!-\!Y397^{7.53}\ OH})$ (Å)")
    ax_b.set_title("(b) Coupled F155 Rotamer Flip & Y294–Y397 Polar Latch Completion")
    ax_b.set_xlim(-115, 190)
    ax_b.set_ylim(2.0, 12.8)
    ax_b.grid(True, alpha=0.25)

    # --- Panel C: Matched-Ligand Gi1 vs Beta-Arrestin-1 Pathway Bifurcation ---
    pairs = df.set_index("pdb_id")
    metrics_c = [
        ("TM6\nR214–D338 (Å)", "R214_D338_CA_A"),
        ("Na+-Toggle\nD163–W356 (Å)", "D163_W356_CA_A"),
        ("TM5 Cleft\nR214–K300 (Å)", "R214_K300_CA_A"),
        ("|W356 χ1|/10\n(° / 10)", "W356_chi1_deg"),
    ]
    x_pos = np.arange(len(metrics_c))
    width = 0.19
    bar_specs = [
        ("6N4B", "6N4B: FUB + Gi1 (WT)", "#2E8B57"),
        ("8WU1", "8WU1: FUB + βarr1 (WT)", "#5E81AC"),
        ("6KPG", "6KPG: AM841 + Gi1", "#66BB6A"),
        ("8WRZ", "8WRZ: AM841 + βarr1", "#81A1C1"),
    ]
    for k, (pid, blabel, bcol) in enumerate(bar_specs):
        vals = []
        for _, colname in metrics_c:
            v = float(pairs.loc[pid, colname])
            if colname == "W356_chi1_deg":
                v = abs(v) / 10.0
            vals.append(v)
        bars = ax_c.bar(
            x_pos + (k - 1.5) * width, vals, width=width,
            color=bcol, edgecolor="black", linewidth=0.7, label=blabel, zorder=3
        )
        for rect, v in zip(bars, vals):
            ax_c.text(
                rect.get_x() + rect.get_width() / 2.0, rect.get_height() + 0.25,
                f"{v:.1f}", ha="center", va="bottom", fontsize=7.0, rotation=0
            )
    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels([m[0] for m in metrics_c], fontsize=8.5)
    ax_c.set_ylabel("Distance (Å) or |W356 χ1| / 10 (°)")
    ax_c.set_ylim(0, 20.5)
    ax_c.set_title("(c) Gi1 vs β-Arrestin-1 Bifurcation at Matched ~15 Å TM6 Opening")
    ax_c.legend(loc="upper left", fontsize=7.6, ncol=2, framealpha=0.92)
    ax_c.grid(True, axis="y", alpha=0.25)

    # --- Panel D: Orthosteric Agonist Compaction vs Intracellular Clamping by NAM ORG27569 (6KQI) ---
    offsets_d = {
        "7FEE": (-0.42, -0.58),
        "6KQI": (0.16, 0.25),
        "5TGZ": (-0.42, 0.35),
        "5U09": (-0.45, 0.32),
        "9BA0": (-0.42, 0.38),
        "8IKG": (0.14, -0.52),
        "8WU1": (-0.48, 0.35),
        "8GHV": (-0.22, 0.48),
        "7V3Z": (-0.52, -0.45),
        "5XR8": (-0.48, -0.95),
        "5XRA": (0.12, -0.92),
        "6KPG": (0.30, -0.35),
        "8WRZ": (0.16, 0.14),
        "9B54": (-0.12, 0.45),
        "6N4B": (0.18, -0.45),
        "7WV9": (0.18, 0.35),
    }
    for state, (col, marker, _) in palette.items():
        sub = df[df["state_class"] == state]
        ax_d.scatter(
            sub["S173_F268_CA_A"], sub["R214_D338_CA_A"],
            c=col, marker=marker, s=90, edgecolors="black", linewidths=0.8, zorder=4
        )
        for _, r in sub.iterrows():
            dx, dy = offsets_d.get(r["pdb_id"], (0.14, 0.22))
            ax_d.annotate(
                r["pdb_id"],
                xy=(r["S173_F268_CA_A"], r["R214_D338_CA_A"]),
                xytext=(r["S173_F268_CA_A"] + dx, r["R214_D338_CA_A"] + dy),
                arrowprops=dict(arrowstyle="-", lw=0.55, color="#777777"),
                fontsize=7.5, fontweight="bold", color="#222222", zorder=5
            )

    r_6kqi = pairs.loc["6KQI"]
    ax_d.annotate(
        "6KQI (CP55,940 + NAM ORG27569):\nECL2/TM2 orthosteric lid contracted\nlike active agonists (high affinity),\nYET TM6 clamped shut (7.20 Å)!",
        xy=(r_6kqi["S173_F268_CA_A"], r_6kqi["R214_D338_CA_A"]),
        xytext=(11.75, 10.65),
        arrowprops=dict(arrowstyle="->", color="#D08770", lw=1.5),
        fontsize=8.0, color="#7A3E1D",
        bbox=dict(boxstyle="round,pad=0.3", fc="#FFF8F3", ec="#D08770", alpha=0.95),
        zorder=6,
    )
    ax_d.set_xlim(11.6, 15.65)
    ax_d.set_ylim(6.0, 18.0)
    ax_d.set_xlabel(r"Orthosteric Lid Closure $d(\mathrm{S173^{2.60}\!-\!F268^{ECL2}}\ \mathrm{C\alpha})$ (Å)  [← Agonist-Contracted]")
    ax_d.set_ylabel(r"Cytoplasmic TM6 Opening $d(\mathrm{R214^{3.50}\!-\!D338^{6.30}}\ \mathrm{C\alpha})$ (Å)")
    ax_d.set_title("(d) Allosteric Decoupling: ORG27569 Contracts Pocket While Locking TM6")
    ax_d.grid(True, alpha=0.25)

    handles = [
        mpatches.Patch(color=col, label=lbl) for _, (col, _, lbl) in palette.items()
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=3, fontsize=8.5,
        frameon=True, bbox_to_anchor=(0.5, 0.005)
    )
    fig.suptitle(
        "Human CB1 (P21554) 16-Structure Experimental Benchmark: Microswitch & Allosteric Dichotomy",
        fontsize=12.5, fontweight="bold", y=0.992
    )
    fig.tight_layout(rect=[0.0, 0.055, 1.0, 0.975])
    fig.savefig(out_fig, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    df_out = build_pdb16_benchmark()
    print(df_out[[
        "pdb_id", "state_class", "R214_D338_CA_A", "F155_chi1_deg",
        "F237_chi1_deg", "F155sc_N393CA_A", "Y294_Y397_OH_A", "D163_W356_CA_A", "S173_F268_CA_A"
    ]].to_string(index=False))
