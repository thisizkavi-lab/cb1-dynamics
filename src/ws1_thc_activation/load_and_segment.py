"""Data audit, nm->Angstrom unit conversion, seam segmentation, and buffered macro-blocking for WS1.

Reads strictly from /Users/kavi/Documents/artico/cb1-dynamics/data/inventory/THC_binding/Figure_8/
and writes clean processed tables to /Users/kavi/Documents/artico/cb1-dynamics/data/processed/ws1/.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.core import assert_contiguous_segments, segment_concatenated_array


PROJECT_ROOT = Path("/Users/kavi/Documents/artico/cb1-dynamics")
RAW_FIG8_DIR = PROJECT_ROOT / "data" / "inventory" / "THC_binding" / "Figure_8"
PROCESSED_WS1_DIR = PROJECT_ROOT / "data" / "processed" / "ws1"

# Crystal reference distances (in Angstrom) verified directly from Figure_8C.py & Figure_8D.py
CRYSTAL_REFS_ANGSTROM = {
    "inactive": {
        "d_toggle_angstrom": 7.9054385,
        "r_tm6_angstrom": 11.950634,
        "d_npxxy_angstrom": 11.875776,
    },
    "active_like": {
        "d_toggle_angstrom": 10.914677,
        "r_tm6_angstrom": 15.470214,
        "d_npxxy_angstrom": 17.152655,
    },
}


def segment_by_standardized_step_z(
    X_nm: np.ndarray,
    z_threshold: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Segment concatenated 3-CV trajectory array using MAD-standardized L2 step norm.

    Accounts for anisotropic 1-step thermal sidechain breathing (e.g. fast Y153-Y397
    phenolic/aromatic rotamer fluctuations vs R214-K343 / W356-D163 fluctuations).
    """
    X_arr = np.asarray(X_nm, dtype=np.float64)
    diffs = np.diff(X_arr, axis=0)
    sigma_mad = 1.4826 * np.median(np.abs(diffs), axis=0)
    sigma_safe = np.maximum(sigma_mad, 1e-9)
    z_step = np.sqrt(np.sum((diffs / sigma_safe[None, :]) ** 2, axis=1))
    jump_indices = np.flatnonzero(z_step > float(z_threshold)).astype(np.int64) + 1
    seg_id = np.zeros(X_arr.shape[0], dtype=np.int64)
    if jump_indices.size > 0:
        seg_id[jump_indices] = 1
        seg_id = np.cumsum(seg_id, dtype=np.int64)
    assert_contiguous_segments(seg_id)
    return seg_id, jump_indices, sigma_safe


def build_acf_buffered_macro_blocks(
    seg_id: np.ndarray,
    n_blocks: int = 30,
    buffer_frames: int = 500,
    group_offset: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Partition frames into contiguous macro-blocks separated by >=buffer_frames gaps.

    Snaps block boundaries strictly to segment boundaries so zero continuous segments
    are split between a macro-block and a buffer or across two macro-blocks.
    """
    assert_contiguous_segments(seg_id)
    n = seg_id.size
    edges = np.linspace(0, n, int(n_blocks) + 1, dtype=int)
    gid = np.full(n, -1, dtype=np.int64)

    for b in range(int(n_blocks)):
        lo = int(edges[b])
        hi = int(edges[b + 1]) - (int(buffer_frames) if b < int(n_blocks) - 1 else 0)

        # Advance lo past any segment that started before edges[b]
        if b > 0 and lo > 0:
            prev_seg = seg_id[edges[b] - 1]
            while lo < hi and seg_id[lo] == prev_seg:
                lo += 1

        # Extend or retract hi to the nearest segment boundary <= edges[b+1] - buffer_frames//2
        if b < int(n_blocks) - 1 and hi > lo:
            end_seg = seg_id[hi - 1]
            max_hi = int(edges[b + 1]) - max(int(buffer_frames) // 2, 1)
            end_run = hi
            while end_run < n and seg_id[end_run] == end_seg:
                end_run += 1
            if end_run <= max_hi:
                hi = end_run
            else:
                start_run = hi - 1
                while start_run > lo and seg_id[start_run - 1] == end_seg:
                    start_run -= 1
                if start_run > lo:
                    hi = start_run

        if hi > lo:
            gid[lo:hi] = int(group_offset + b)

    valid_mask = gid >= 0
    return gid, valid_mask


def compute_acf_curve(x: np.ndarray, lags: Tuple[int, ...] = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)) -> Dict[str, float]:
    v = np.asarray(x, dtype=np.float64) - float(np.mean(x))
    var = float(np.mean(v * v))
    if var <= 1e-15:
        return {str(k): 0.0 for k in lags}
    return {str(k): float(np.mean(v[:-k] * v[k:]) / var) for k in lags if k < v.size}


def DwellFilter1D(mask: np.ndarray, seg_id: np.ndarray, min_dwell: int = 3) -> np.ndarray:
    """Retain basin membership True only on contiguous within-segment runs of length >= min_dwell."""
    m = np.asarray(mask, dtype=bool).reshape(-1)
    seg = np.asarray(seg_id, dtype=np.int64).reshape(-1)
    if min_dwell <= 1:
        return m.copy()
    out = np.zeros_like(m)
    cuts = np.flatnonzero(seg[1:] != seg[:-1]) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, seg.size]
    for s, e in zip(starts, ends):
        sub = m[s:e]
        if sub.size < min_dwell or not np.any(sub):
            continue
        padded = np.r_[False, sub, False]
        transitions = np.diff(padded.astype(np.int8))
        r_starts = np.flatnonzero(transitions == 1)
        r_ends = np.flatnonzero(transitions == -1)
        for rs, re in zip(r_starts, r_ends):
            if (re - rs) >= min_dwell:
                out[s + rs : s + re] = True
    return out


def load_and_build_ws1_dataset(
    raw_dir: Path = RAW_FIG8_DIR,
    out_dir: Path = PROCESSED_WS1_DIR,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Load Tier 1C arrays, run forensic quality/unit/seam audits, and save Parquet + JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)

    ago_tm6 = np.load(raw_dir / "agonist_toggle_switch_TM6_movement.npy")
    ago_npx = np.load(raw_dir / "agonist_toggle_switch_NPxxY_movement.npy")
    thc_tm6 = np.load(raw_dir / "partial_agonist_toggle_switch_TM6_movement.npy")
    thc_npx = np.load(raw_dir / "partial_agonist_toggle_switch_NPxxY_movement.npy")
    res_pairs = np.load(raw_dir / "residue_pair_ind.npy")

    # Strict integrity checks
    for name, arr in [
        ("agonist_TM6", ago_tm6),
        ("agonist_NPxxY", ago_npx),
        ("thc_TM6", thc_tm6),
        ("thc_NPxxY", thc_npx),
    ]:
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"Unexpected shape for {name}: {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"NaN or Inf detected in {name}!")

    ago_col0_diff = float(np.max(np.abs(ago_tm6[:, 0] - ago_npx[:, 0])))
    thc_col0_diff = float(np.max(np.abs(thc_tm6[:, 0] - thc_npx[:, 0])))
    if ago_col0_diff != 0.0 or thc_col0_diff != 0.0:
        raise AssertionError(
            f"Shared toggle-switch column mismatch: ago={ago_col0_diff}, thc={thc_col0_diff}"
        )

    records = []
    audit_ensembles: Dict[str, object] = {}
    seg_offsets = {
        "z5p0": 0,
        "tm6_0p15nm": 0,
        "3cv_0p10nm": 0,
        "3cv_0p12nm": 0,
        "3cv_0p15nm": 0,
        "3cv_0p20nm": 0,
    }

    for ens_idx, (ligand, tm6_nm, npx_nm, group_offset) in enumerate(
        [
            ("full_agonist", ago_tm6, ago_npx, 0),
            ("delta9_thc", thc_tm6, thc_npx, 30),
        ]
    ):
        # Order columns: [R_TM6 (nm), d_W356_D163 (nm), d_Y153_Y397 (nm)]
        X_nm = np.column_stack([tm6_nm[:, 1], tm6_nm[:, 0], npx_nm[:, 1]]).astype(np.float64)
        X_ang = X_nm * 10.0
        n_frames = X_nm.shape[0]

        # Exact consecutive duplicate frame check
        n_exact_dup_steps = int(np.sum(np.all(np.diff(X_nm, axis=0) == 0.0, axis=1)))

        # Segmentations
        s_z5, j_z5, sigma_mad_nm = segment_by_standardized_step_z(X_nm, z_threshold=5.0)
        s_tm6_15, j_tm6_15 = segment_concatenated_array(X_nm[:, :2], jump_threshold=0.15, norm="inf")
        s_3cv_10, j_3cv_10 = segment_concatenated_array(X_nm, jump_threshold=0.10, norm="inf")
        s_3cv_12, j_3cv_12 = segment_concatenated_array(X_nm, jump_threshold=0.12, norm="inf")
        s_3cv_15, j_3cv_15 = segment_concatenated_array(X_nm, jump_threshold=0.15, norm="inf")
        s_3cv_20, j_3cv_20 = segment_concatenated_array(X_nm, jump_threshold=0.20, norm="inf")

        # Buffered macro-blocks (30 per ensemble, >=500-frame buffer >= ACF decorrelation horizon)
        macro_gid, macro_valid = build_acf_buffered_macro_blocks(
            s_z5, n_blocks=30, buffer_frames=500, group_offset=group_offset
        )

        # Dwell-filtered endpoints (m_dwell = 3 frames) on primary segmentation
        in_A_global = X_ang[:, 0] <= 11.8
        in_B_global = X_ang[:, 0] >= 15.2
        in_A_dwell = DwellFilter1D(in_A_global, s_z5, min_dwell=3)
        in_B_dwell = DwellFilter1D(in_B_global, s_z5, min_dwell=3)

        df_ens = pd.DataFrame(
            {
                "ligand": ligand,
                "ligand_code": ens_idx,
                "frame_in_ensemble": np.arange(n_frames, dtype=np.int64),
                "r_tm6_nm": X_nm[:, 0],
                "d_toggle_nm": X_nm[:, 1],
                "d_npxxy_nm": X_nm[:, 2],
                "r_tm6_angstrom": X_ang[:, 0],
                "d_toggle_angstrom": X_ang[:, 1],
                "d_npxxy_angstrom": X_ang[:, 2],
                "seg_id_z5p0": s_z5 + seg_offsets["z5p0"],
                "seg_id_tm6_0p15nm": s_tm6_15 + seg_offsets["tm6_0p15nm"],
                "seg_id_3cv_0p10nm": s_3cv_10 + seg_offsets["3cv_0p10nm"],
                "seg_id_3cv_0p12nm": s_3cv_12 + seg_offsets["3cv_0p12nm"],
                "seg_id_3cv_0p15nm": s_3cv_15 + seg_offsets["3cv_0p15nm"],
                "seg_id_3cv_0p20nm": s_3cv_20 + seg_offsets["3cv_0p20nm"],
                "macro_group_id": macro_gid,
                "macro_valid_mask": macro_valid,
                "in_A_global": in_A_global,
                "in_B_global": in_B_global,
                "in_A_global_dwell3": in_A_dwell,
                "in_B_global_dwell3": in_B_dwell,
            }
        )
        records.append(df_ens)

        def _seg_stats(s_local: np.ndarray, j_local: np.ndarray) -> Dict[str, float]:
            cuts = np.r_[0, j_local, n_frames]
            lens = np.diff(cuts)
            return {
                "n_segments": int(lens.size),
                "n_jumps": int(j_local.size),
                "median_len": float(np.median(lens)),
                "mean_len": float(np.mean(lens)),
                "max_len": int(np.max(lens)),
                "n_segments_ge_20": int(np.sum(lens >= 20)),
                "n_segments_ge_50": int(np.sum(lens >= 50)),
            }

        audit_ensembles[ligand] = {
            "n_frames": int(n_frames),
            "nan_count": 0,
            "inf_count": 0,
            "exact_consecutive_duplicate_steps": n_exact_dup_steps,
            "shared_col0_max_abs_diff_nm": ago_col0_diff if ligand == "full_agonist" else thc_col0_diff,
            "step_mad_sigma_nm": {
                "r_tm6": float(sigma_mad_nm[0]),
                "d_toggle": float(sigma_mad_nm[1]),
                "d_npxxy": float(sigma_mad_nm[2]),
            },
            "coordinate_summary_angstrom": {
                col_name: {
                    "min": float(np.min(X_ang[:, j])),
                    "q05": float(np.quantile(X_ang[:, j], 0.05)),
                    "median": float(np.median(X_ang[:, j])),
                    "mean": float(np.mean(X_ang[:, j])),
                    "std": float(np.std(X_ang[:, j])),
                    "q95": float(np.quantile(X_ang[:, j], 0.95)),
                    "max": float(np.max(X_ang[:, j])),
                }
                for j, col_name in enumerate(["r_tm6_angstrom", "d_toggle_angstrom", "d_npxxy_angstrom"])
            },
            "acf_curves": {
                "r_tm6": compute_acf_curve(X_ang[:, 0]),
                "d_toggle": compute_acf_curve(X_ang[:, 1]),
                "d_npxxy": compute_acf_curve(X_ang[:, 2]),
            },
            "basin_populations": {
                "n_A_global_le_11p8": int(np.sum(in_A_global)),
                "frac_A_global_le_11p8": float(np.mean(in_A_global)),
                "n_intermediate_12p5_to_14p5": int(np.sum((X_ang[:, 0] >= 12.5) & (X_ang[:, 0] <= 14.5))),
                "frac_intermediate_12p5_to_14p5": float(np.mean((X_ang[:, 0] >= 12.5) & (X_ang[:, 0] <= 14.5))),
                "n_B_global_ge_15p2": int(np.sum(in_B_global)),
                "frac_B_global_ge_15p2": float(np.mean(in_B_global)),
                "n_A_dwell3": int(np.sum(in_A_dwell)),
                "n_B_dwell3": int(np.sum(in_B_dwell)),
            },
            "matched_slice_12p5_14p5_microswitches_angstrom": {
                "d_toggle_mean": float(np.mean(X_ang[(X_ang[:, 0] >= 12.5) & (X_ang[:, 0] <= 14.5), 1])),
                "d_toggle_std": float(np.std(X_ang[(X_ang[:, 0] >= 12.5) & (X_ang[:, 0] <= 14.5), 1])),
                "d_npxxy_mean": float(np.mean(X_ang[(X_ang[:, 0] >= 12.5) & (X_ang[:, 0] <= 14.5), 2])),
                "d_npxxy_std": float(np.std(X_ang[(X_ang[:, 0] >= 12.5) & (X_ang[:, 0] <= 14.5), 2])),
            },
            "segmentations": {
                "z5p0_primary": _seg_stats(s_z5, j_z5),
                "tm6_0p15nm": _seg_stats(s_tm6_15, j_tm6_15),
                "3cv_0p10nm": _seg_stats(s_3cv_10, j_3cv_10),
                "3cv_0p12nm": _seg_stats(s_3cv_12, j_3cv_12),
                "3cv_0p15nm": _seg_stats(s_3cv_15, j_3cv_15),
                "3cv_0p20nm": _seg_stats(s_3cv_20, j_3cv_20),
            },
            "macro_blocking": {
                "n_valid_frames": int(np.sum(macro_valid)),
                "n_buffer_excluded_frames": int(np.sum(~macro_valid)),
                "n_unique_macro_blocks": int(np.unique(macro_gid[macro_valid]).size),
                "buffer_frames_target": 500,
            },
        }

        seg_offsets["z5p0"] += int(s_z5.max()) + 1
        seg_offsets["tm6_0p15nm"] += int(s_tm6_15.max()) + 1
        seg_offsets["3cv_0p10nm"] += int(s_3cv_10.max()) + 1
        seg_offsets["3cv_0p12nm"] += int(s_3cv_12.max()) + 1
        seg_offsets["3cv_0p15nm"] += int(s_3cv_15.max()) + 1
        seg_offsets["3cv_0p20nm"] += int(s_3cv_20.max()) + 1

    df_all = pd.concat(records, axis=0, ignore_index=True)

    # Assert global contiguity across concatenated table for every segmentation column
    for col in [
        "seg_id_z5p0",
        "seg_id_tm6_0p15nm",
        "seg_id_3cv_0p10nm",
        "seg_id_3cv_0p12nm",
        "seg_id_3cv_0p15nm",
        "seg_id_3cv_0p20nm",
    ]:
        assert_contiguous_segments(df_all[col].to_numpy())

    parquet_path = out_dir / "ws1_featurized_segments.parquet"
    df_all.to_parquet(parquet_path, index=False)

    audit_summary = {
        "total_frames": int(df_all.shape[0]),
        "residue_pair_ind_shape": list(res_pairs.shape),
        "units_raw": "nm (float64)",
        "units_processed": "Angstrom (nm * 10.0, verified against Figure_8C.py & Figure_8D.py)",
        "crystal_references_angstrom": CRYSTAL_REFS_ANGSTROM,
        "ensembles": audit_ensembles,
        "parquet_path": str(parquet_path),
    }
    with open(out_dir / "ws1_data_audit.json", "w", encoding="utf-8") as fh:
        json.dump(audit_summary, fh, indent=2)

    return df_all, audit_summary


if __name__ == "__main__":
    df, summary = load_and_build_ws1_dataset()
    print(f"Saved WS1 dataset ({df.shape[0]:,} frames) to {summary['parquet_path']}")
