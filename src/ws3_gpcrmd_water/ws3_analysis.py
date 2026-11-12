"""Workstream 3 Analysis: Cognate Gi1 vs Non-Cognate Gs Coupling & Explicit-Solvent Mori-Zwanzig Memory Closure.

Executes the two dynamical pillars of WS3 on the 6 open all-atom GPCRmd trajectories (8.97 us, 20,501 frames)
and integrates the 16-PDB structural benchmark:

  Pillar A (Pitfall 4 — Transducer Coupling & Allosteric Relaxation at Matched Open TM6):
    - Compares cognate Gi1 + MDMB-FUBINACA (ID 1201, 1.0 us) vs non-cognate Gs + apo (ID 2067, 6.0 us)
      both initialized from active cryo-EM 6N4B, against the 4 inactive bilayers (IDs 22, 164, 162, 163).
    - Quantifies whole-trajectory AND matched-TM6-slice ([13.5, 16.5] A) distributions, early-vs-late
      relaxation drift, H-bond survival lifetimes, transmembrane water-wire percolation, explicit Na+/K+
      coordination at D163^2.50, and POPC wedging into the TM2-TM4 cleft, with 50-ns macro-block
      cluster bootstrap 95% confidence intervals.

  Pillar B (Pitfall 3.2 & Hypothesis H2 — Fast Local Microswitch Committors & Mori-Zwanzig Memory Closure):
    - Audits pre-specified purely protein-defined local microswitch transitions across both the uniform
      0.2-ns cohort (IDs 1201, 22, 164, 162, 163; 14,501 frames) and the 1.0-ns physical-time-harmonized
      NaCl-5 (IDs 2067, 22, 164, 162, 163) and Open-6 cohorts, strictly enforcing <5% right-censoring
      within forward horizon tau_max.
    - Splits strictly by non-overlapping macro-time blocks separated by >= 2*tau_max temporal buffers
      (and cross-checks with whole-trajectory leave-trajectory-out GroupKFold), fitting all scalers,
      splines, Peters-Trout models, and stopped-DGA operators exclusively on training folds via `src.core`.
    - Tests Step A (protein-only displacement memory Delta_ell_mem vs N2a within-cell shuffle) and
      Step B (static log-likelihood gain Delta_ell_static(+z_env), N1 permutation p-value, and memory
      collapse ratio rho_mem upon adding explicit water-wire conductance, Na+ coordination, and POPC
      wedging), plus Manski [0, 1] worst-case censoring bounds and N2b time-reversal diagnostics.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.core import (
    LineageGroupSplitter,
    OracleHistGBRModel,
    PetersTroutCommittorModel,
    PriorBaselineB0,
    ProgressVelocityBaselineB2,
    SplineProgressBaselineB1,
    StoppedDGACommittor,
    assert_contiguous_segments,
    assert_disjoint_group_split,
    assert_no_cross_seam_pairs,
    bootstrap_bss_and_delta_ell_ci,
    brier_skill_score,
    build_forward_stopped_pairs,
    committor_log_likelihood,
    compute_lagged_displacements,
    compute_memory_collapse_ratio,
    delta_log_likelihood,
    evaluate_committor_pbr,
    evaluate_mori_zwanzig_memory,
    group_block_bootstrap_ci,
    projected_bellman_residual,
    run_group_permutation_null_N1,
    run_time_reversal_diagnostic_N2b,
)


PROJECT_ROOT = Path("/Users/kavi/Documents/artico/cb1-dynamics")
PDB16_PARQUET = PROJECT_ROOT / "data" / "processed" / "ws3" / "pdb16_structural_benchmark.parquet"
GPCRMD_PARQUET = PROJECT_ROOT / "data" / "processed" / "ws3" / "gpcrmd_open6_features.parquet"
PROVENANCE_JSON = PROJECT_ROOT / "data" / "processed" / "ws3" / "gpcrmd_open6_provenance.json"
RESULTS_DIR = PROJECT_ROOT / "results" / "ws3"
FIG_DIR = RESULTS_DIR / "figures"
METRICS_JSON = RESULTS_DIR / "ws3_metrics.json"
REPORT_MD = RESULTS_DIR / "ws3_report.md"


# Pre-registered candidate local protein microswitch coordinates (ZERO env features in basin definitions!)
CANDIDATE_LOCAL_TARGETS: list[dict[str, Any]] = [
    {
        "name": "Y294_Y397_polar_latch",
        "col": "Y294_Y397_OH_A",
        "description": "Intracellular TM5-TM7 Tyr polar latch d(Y294^5.58 - Y397^7.53 OH)",
        "B_below": True,  # B_loc = direct H-bond engaged (<= r_low), A_loc = open/unformed (>= r_high)
        "threshold_grid": [(3.5, 4.8), (3.6, 4.6), (3.8, 5.0), (4.0, 5.5)],
    },
    {
        "name": "D163_N393_polar_aperture",
        "col": "D163_N393_polar_min_A",
        "description": "Allosteric Na+/NPxxY polar pocket aperture d_min(D163^2.50 - N393^7.49)",
        "B_below": True,  # B_loc = tight polar contact (<= r_low), A_loc = expanded/hydrated (>= r_high)
        "threshold_grid": [(3.6, 5.2), (3.8, 5.4), (4.0, 5.6), (4.2, 5.8), (4.5, 6.2)],
    },
    {
        "name": "F200_W356_toggle_contact",
        "col": "F200_W356_ring_min_A",
        "description": "Twin toggle aromatic minimum contact d_min(F200^3.36 - W356^6.48 rings)",
        "B_below": False,  # B_loc = unstacked/separated (>= r_high), A_loc = stacked (<= r_low)
        "threshold_grid": [(3.5, 4.5), (3.6, 4.6), (3.7, 4.8), (3.8, 5.0)],
    },
    {
        "name": "F155_F237_aromatic_cleft",
        "col": "F155_F237_ring_min_A",
        "description": "CB1-selective TM2-TM4 aromatic cleft contact d_min(F155^2.42 - F237^4.46 rings)",
        "B_below": False,  # B_loc = separated (>= r_high), A_loc = packed (<= r_low)
        "threshold_grid": [(3.8, 5.2), (4.0, 5.5), (4.2, 5.8), (4.5, 6.2)],
    },
    {
        "name": "R214_D338_ionic_gate",
        "col": "R214_D338_ionic_min_A",
        "description": "Cytoplasmic DRY-TM6 ionic minimum distance d_min(R214^3.50 - D338^6.30)",
        "B_below": False,  # B_loc = broken/open (>= r_high), A_loc = salt-bridged/closed (<= r_low)
        "threshold_grid": [(3.5, 5.5), (4.0, 6.0), (5.0, 7.5), (6.0, 8.5)],
    },
]


def mean_survival_lifetime_ns(binary_mask: np.ndarray, dt_ns: float) -> float:
    """Compute mean contiguous dwell duration (in ns) where binary_mask == True."""
    m = np.asarray(binary_mask, dtype=bool).reshape(-1)
    if not np.any(m):
        return 0.0
    padded = np.concatenate(([False], m, [False]))
    starts = np.flatnonzero(padded[1:] & (~padded[:-1]))
    ends = np.flatnonzero((~padded[1:]) & padded[:-1])
    lengths = (ends - starts).astype(np.float64) * float(dt_ns)
    return float(np.mean(lengths))


def assign_50ns_blocks(df: pd.DataFrame, block_ns: float = 50.0) -> np.ndarray:
    """Assign non-overlapping 50-ns time blocks per trajectory for cluster bootstrap CIs."""
    blocks = []
    offset = 0
    for gid, sub in df.groupby("gpcrmd_id", sort=False):
        b_local = np.floor(sub["time_ns"].to_numpy() / block_ns).astype(np.int64)
        _, inv = np.unique(b_local, return_inverse=True)
        blocks.append(inv + offset)
        offset += int(inv.max()) + 1
    return np.concatenate(blocks)


def analyze_pillar_a_coupling(df: pd.DataFrame) -> dict[str, Any]:
    """Quantify Gi1 (1201) vs Gs (2067) vs Inactive (22, 164, 162, 163) coupling & relaxation."""
    df = df.copy()
    df["F155_outward_gauche_minus"] = ((df["F155_chi1_deg"] >= -110.0) & (df["F155_chi1_deg"] <= -30.0)).astype(float)
    df["F155_outward_clearance_gt13p5"] = (df["F155sc_N393CA_A"] >= 13.5).astype(float)
    df["F237_outward_trans"] = (np.abs(df["F237_chi1_deg"]) >= 135.0).astype(float)
    df["W356_deep_rotamer"] = (np.abs(df["W356_chi1_deg"]) >= 130.0).astype(float)
    df["Y294_Y397_direct_hbond_3p5"] = (df["Y294_Y397_OH_A"] <= 3.5).astype(float)
    df["Y294_Y397_polar_latch_4p2"] = (df["Y294_Y397_OH_A"] <= 4.2).astype(float)
    df["Y294_Y397_water_or_direct"] = (df["wire_Y294_Y397_cond"] >= 0.5).astype(float)
    df["wire_D163_NY_percolated"] = (df["wire_D163_N393Y397_cond"] > 0.0).astype(float)
    df["wire_D163_W356_percolated"] = (df["wire_D163_W356_cond"] > 0.0).astype(float)
    df["Na_bound_D163"] = (df["coord_Na_D163_3p2"] > 0).astype(float)
    df["K_bound_D163"] = (df["coord_K_D163_3p2"] > 0).astype(float)
    df["cation_bound_D163"] = (df["coord_cation_D163_3p2"] > 0).astype(float)
    df["POPC_wedged_TM2_TM4_4p0"] = (df["min_POPC_TM2_TM4_cleft_A"] <= 4.0).astype(float)

    cohorts = {
        "Gi1_1201": df[df["gpcrmd_id"] == 1201].copy(),
        "Gs_2067": df[df["gpcrmd_id"] == 2067].copy(),
        "Inactive_4traj": df[df["gpcrmd_id"].isin([22, 164, 162, 163])].copy(),
    }

    summary_metrics = [
        "R214_D338_CA_A",
        "F155sc_N393CA_A",
        "F155_outward_gauche_minus",
        "F237_outward_trans",
        "Y294_Y397_OH_A",
        "Y294_Y397_direct_hbond_3p5",
        "Y294_Y397_polar_latch_4p2",
        "Y294_Y397_water_or_direct",
        "D163_N393_polar_min_A",
        "F200_W356_ring_com_A",
        "W356_chi1_deg",
        "n_wat_core_pore",
        "wire_D163_N393Y397_cond",
        "wire_D163_NY_percolated",
        "wire_D163_W356_cond",
        "wire_D163_W356_percolated",
        "Na_bound_D163",
        "K_bound_D163",
        "cation_bound_D163",
        "min_POPC_TM2_TM4_cleft_A",
        "POPC_wedged_TM2_TM4_4p0",
        "min_POPC_TM1_TM7_portal_A",
    ]

    cohort_stats: dict[str, Any] = {}
    for cname, cdf in cohorts.items():
        b50 = assign_50ns_blocks(cdf, block_ns=50.0)
        m_dict: dict[str, Any] = {
            "n_frames": int(len(cdf)),
            "total_duration_ns": round(float(cdf.groupby("gpcrmd_id")["time_ns"].max().sum()), 2),
            "n_50ns_blocks": int(np.unique(b50).size),
            "metrics": {},
        }
        for col in summary_metrics:
            vals = cdf[col].to_numpy(dtype=np.float64)
            ci = group_block_bootstrap_ci(
                group_ids=b50,
                stat_fn=lambda idx, v=vals: float(np.mean(v[idx])),
                n_boot=1000,
                random_state=42,
            )
            m_dict["metrics"][col] = {
                "mean": round(ci["point"], 4),
                "ci95_low": round(ci["ci_low"], 4),
                "ci95_high": round(ci["ci_high"], 4),
                "std": round(float(np.std(vals)), 4),
                "median": round(float(np.median(vals)), 4),
            }
        # Also compute Na+ distance solely when Na+ ions exist
        if bool(cdf["has_na_ions"].all()):
            v_na = cdf["min_Na_D163_A"].to_numpy(dtype=np.float64)
            ci_na = group_block_bootstrap_ci(b50, lambda idx, v=v_na: float(np.mean(v[idx])), n_boot=1000, random_state=42)
            m_dict["metrics"]["min_Na_D163_A"] = {
                "mean": round(ci_na["point"], 4),
                "ci95_low": round(ci_na["ci_low"], 4),
                "ci95_high": round(ci_na["ci_high"], 4),
                "min_observed": round(float(np.min(v_na)), 3),
            }
        if bool(cdf["has_k_ions"].all()):
            v_k = cdf["min_K_D163_A"].to_numpy(dtype=np.float64)
            ci_k = group_block_bootstrap_ci(b50, lambda idx, v=v_k: float(np.mean(v[idx])), n_boot=1000, random_state=42)
            m_dict["metrics"]["min_K_D163_A"] = {
                "mean": round(ci_k["point"], 4),
                "ci95_low": round(ci_k["ci_low"], 4),
                "ci95_high": round(ci_k["ci_high"], 4),
                "min_observed": round(float(np.min(v_k)), 3),
            }
        cohort_stats[cname] = m_dict

    # Per-trajectory breakdown across all 6 trajectories
    per_traj: dict[str, Any] = {}
    for gid, gdf in df.groupby("gpcrmd_id", sort=False):
        dt = float(gdf["dt_ns"].iloc[0])
        per_traj[str(gid)] = {
            "state_label": str(gdf["state_label"].iloc[0]),
            "pdb_id": str(gdf["pdb_id"].iloc[0]),
            "transducer_type": str(gdf["transducer_type"].iloc[0]),
            "n_frames": int(len(gdf)),
            "dt_ns": dt,
            "duration_ns": round(float(gdf["time_ns"].max()), 2),
            "R214_D338_CA_mean_A": round(float(gdf["R214_D338_CA_A"].mean()), 3),
            "F155sc_N393CA_mean_A": round(float(gdf["F155sc_N393CA_A"].mean()), 3),
            "F155_outward_frac": round(float(gdf["F155_outward_gauche_minus"].mean()), 4),
            "F237_outward_frac": round(float(gdf["F237_outward_trans"].mean()), 4),
            "Y294_Y397_OH_mean_A": round(float(gdf["Y294_Y397_OH_A"].mean()), 3),
            "Y294_Y397_latch4p2_frac": round(float(gdf["Y294_Y397_polar_latch_4p2"].mean()), 4),
            "Y294_Y397_latch4p2_mean_dwell_ns": round(
                mean_survival_lifetime_ns(gdf["Y294_Y397_polar_latch_4p2"].to_numpy() > 0.5, dt), 3
            ),
            "n_wat_core_pore_mean": round(float(gdf["n_wat_core_pore"].mean()), 3),
            "wire_D163_NY_cond_mean": round(float(gdf["wire_D163_N393Y397_cond"].mean()), 4),
            "wire_D163_NY_percolated_frac": round(float(gdf["wire_D163_NY_percolated"].mean()), 4),
            "Na_bound_D163_frac": round(float(gdf["Na_bound_D163"].mean()), 4) if bool(gdf["has_na_ions"].iloc[0]) else None,
            "K_bound_D163_frac": round(float(gdf["K_bound_D163"].mean()), 4) if bool(gdf["has_k_ions"].iloc[0]) else None,
            "min_POPC_TM2_TM4_mean_A": round(float(gdf["min_POPC_TM2_TM4_cleft_A"].mean()), 3),
        }

    # Matched open-TM6 slice comparison ([13.5, 16.5] A) between Gi1 (1201) and Gs (2067)
    gi_sub = cohorts["Gi1_1201"]
    gs_sub = cohorts["Gs_2067"]
    gi_slice = gi_sub[(gi_sub["R214_D338_CA_A"] >= 13.5) & (gi_sub["R214_D338_CA_A"] <= 16.5)]
    gs_slice = gs_sub[(gs_sub["R214_D338_CA_A"] >= 13.5) & (gs_sub["R214_D338_CA_A"] <= 16.5)]

    matched_open_tm6: dict[str, Any] = {
        "slice_range_A": [13.5, 16.5],
        "n_frames_Gi1_1201": int(len(gi_slice)),
        "n_frames_Gs_2067": int(len(gs_slice)),
        "comparisons": {},
    }
    if len(gi_slice) >= 20 and len(gs_slice) >= 20:
        for col in [
            "R214_D338_CA_A",
            "F155sc_N393CA_A",
            "F155_outward_gauche_minus",
            "F237_outward_trans",
            "Y294_Y397_OH_A",
            "Y294_Y397_polar_latch_4p2",
            "D163_N393_polar_min_A",
            "n_wat_core_pore",
            "wire_D163_N393Y397_cond",
            "min_POPC_TM2_TM4_cleft_A",
        ]:
            v_gi = gi_slice[col].to_numpy(dtype=np.float64)
            v_gs = gs_slice[col].to_numpy(dtype=np.float64)
            ks_stat, _ = stats.ks_2samp(v_gi, v_gs)
            matched_open_tm6["comparisons"][col] = {
                "Gi1_mean": round(float(np.mean(v_gi)), 4),
                "Gs_mean": round(float(np.mean(v_gs)), 4),
                "diff_Gi1_minus_Gs": round(float(np.mean(v_gi) - np.mean(v_gs)), 4),
                "ks_distance": round(float(ks_stat), 4),
            }

    # Early vs Late relaxation drift test on Gi1 (1201: 0-250 ns vs 750-1000 ns) and Gs (2067: 0-1000 ns vs 4500-6000 ns)
    def early_late_drift(sub_df: pd.DataFrame, early_end_ns: float, late_start_ns: float) -> dict[str, Any]:
        early = sub_df[sub_df["time_ns"] <= early_end_ns]
        late = sub_df[sub_df["time_ns"] >= late_start_ns]
        out: dict[str, Any] = {
            "early_window_ns": [0.0, early_end_ns],
            "late_window_ns": [late_start_ns, round(float(sub_df["time_ns"].max()), 1)],
            "metrics": {},
        }
        for col in [
            "R214_D338_CA_A",
            "F155sc_N393CA_A",
            "F237_outward_trans",
            "Y294_Y397_OH_A",
            "n_wat_core_pore",
            "wire_D163_N393Y397_cond",
            "Na_bound_D163",
        ]:
            out["metrics"][col] = {
                "early_mean": round(float(early[col].mean()), 4),
                "late_mean": round(float(late[col].mean()), 4),
                "delta_late_minus_early": round(float(late[col].mean() - early[col].mean()), 4),
            }
        return out

    relaxation_drift = {
        "Gi1_1201": early_late_drift(gi_sub, 250.0, 750.0),
        "Gs_2067": early_late_drift(gs_sub, 1000.0, 4500.0),
    }

    return {
        "cohort_summary": cohort_stats,
        "per_trajectory": per_traj,
        "matched_open_tm6_Gi1_vs_Gs": matched_open_tm6,
        "early_vs_late_relaxation": relaxation_drift,
    }


def prepare_uniform_dt_cohort(df: pd.DataFrame, gids: list[int], target_dt_ns: float) -> pd.DataFrame:
    """Extract subset of trajectories harmonized to an exact uniform physical timestep target_dt_ns."""
    pieces: list[pd.DataFrame] = []
    for gid in gids:
        sub = df[df["gpcrmd_id"] == gid].sort_values("frame_idx").reset_index(drop=True)
        dt = float(sub["dt_ns"].iloc[0])
        stride = int(round(target_dt_ns / dt))
        if abs(stride * dt - target_dt_ns) > 1e-6:
            raise ValueError(f"Cannot harmonize gid={gid} (dt={dt} ns) to target_dt_ns={target_dt_ns}")
        sub_st = sub.iloc[::stride].copy().reset_index(drop=True)
        sub_st["eff_dt_ns"] = target_dt_ns
        sub_st["local_step"] = np.arange(len(sub_st), dtype=np.int64)
        pieces.append(sub_st)
    return pd.concat(pieces, ignore_index=True)


def build_buffered_macro_segments(
    df_cohort: pd.DataFrame,
    block_duration_ns: float,
    buffer_duration_ns: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Partition each trajectory into non-overlapping macro-blocks separated by buffer_duration_ns gaps.

    Frames inside inter-block temporal buffers are dropped BEFORE constructing seg_id and macro_group_id,
    so every surviving block is an independent contiguous segment with >= buffer_duration_ns separation
    from any other block in the same trajectory (eliminating seam autocorrelation leakage).
    """
    dt = float(df_cohort["eff_dt_ns"].iloc[0])
    block_steps = max(20, int(round(block_duration_ns / dt)))
    buf_steps = max(2, int(round(buffer_duration_ns / dt)))
    period = block_steps + buf_steps

    kept_chunks: list[pd.DataFrame] = []
    seg_counter = 0
    for gid, sub in df_cohort.groupby("gpcrmd_id", sort=False):
        n = len(sub)
        pos = 0
        while pos + block_steps // 2 <= n:
            end_pos = min(pos + block_steps, n)
            if end_pos - pos >= max(15, block_steps // 3):
                blk = sub.iloc[pos:end_pos].copy()
                blk["seg_id"] = seg_counter
                blk["macro_group_id"] = seg_counter
                kept_chunks.append(blk)
                seg_counter += 1
            pos += period

    out_df = pd.concat(kept_chunks, ignore_index=True)
    seg_id = out_df["seg_id"].to_numpy(dtype=np.int64)
    macro_groups = out_df["macro_group_id"].to_numpy(dtype=np.int64)
    assert_contiguous_segments(seg_id)
    return out_df, seg_id, macro_groups


def compute_local_hitting_outcomes(
    vals: np.ndarray,
    seg_id: np.ndarray,
    r_low: float,
    r_high: float,
    b_below: bool,
    horizon_steps: int,
) -> dict[str, Any]:
    """Compute horizon-safe direct first-hitting outcomes and exact right-censoring statistics."""
    v = np.asarray(vals, dtype=np.float64).reshape(-1)
    seg = np.asarray(seg_id, dtype=np.int64).reshape(-1)
    assert_contiguous_segments(seg)

    if b_below:
        in_B = v <= r_low
        in_A = v >= r_high
    else:
        in_A = v <= r_low
        in_B = v >= r_high

    if np.any(in_A & in_B):
        raise ValueError("Overlapping local basins A_loc and B_loc!")

    in_AB = in_A | in_B
    n = v.size
    y_hit = np.full(n, np.nan, dtype=np.float64)
    is_horizon_safe_int = np.zeros(n, dtype=bool)
    is_censored = np.zeros(n, dtype=bool)
    hit_lags: list[int] = []
    n_basin_switches = 0

    cuts = np.flatnonzero(seg[1:] != seg[:-1]) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [n]))

    for s, e in zip(starts.tolist(), ends.tolist()):
        T_m = e - s
        # Count distinct A <-> B transitions in segment
        ab_seq = np.where(in_B[s:e], 1, np.where(in_A[s:e], -1, 0))
        nz = ab_seq[ab_seq != 0]
        if nz.size > 1:
            n_basin_switches += int(np.sum(nz[1:] != nz[:-1]))

        if T_m <= horizon_steps:
            continue
        ab_seg = in_AB[s:e]
        b_seg = in_B[s:e]
        local_idx = np.arange(T_m, dtype=np.int64)
        sentinel = T_m + horizon_steps + 10
        hit_cand = np.where(ab_seg, local_idx, sentinel)
        next_hit = np.minimum.accumulate(hit_cand[::-1])[::-1]

        cand_t = local_idx[: T_m - horizon_steps]
        int_t = cand_t[~ab_seg[: T_m - horizon_steps]]
        if int_t.size == 0:
            continue

        nh = next_hit[int_t]
        global_int = s + int_t
        is_horizon_safe_int[global_int] = True

        cens = nh > (int_t + horizon_steps)
        is_censored[global_int[cens]] = True

        uncens_local = int_t[~cens]
        uncens_nh = nh[~cens]
        if uncens_local.size > 0:
            y_hit[s + uncens_local] = b_seg[uncens_nh].astype(np.float64)
            hit_lags.extend((uncens_nh - uncens_local).tolist())

    n_int_total = int(np.sum(is_horizon_safe_int))
    n_cens = int(np.sum(is_censored))
    n_uncens = n_int_total - n_cens
    cens_rate = float(n_cens / n_int_total) if n_int_total > 0 else 1.0
    uncens_mask = is_horizon_safe_int & (~is_censored)
    y_mean = float(np.mean(y_hit[uncens_mask])) if n_uncens > 0 else float("nan")

    return {
        "in_A": in_A,
        "in_B": in_B,
        "is_horizon_safe_int": is_horizon_safe_int,
        "is_censored": is_censored,
        "uncens_mask": uncens_mask,
        "y_hit": y_hit,
        "n_int_total": n_int_total,
        "n_uncensored": n_uncens,
        "n_censored": n_cens,
        "censoring_rate": cens_rate,
        "mean_y_uncensored": y_mean,
        "mean_hit_steps": float(np.mean(hit_lags)) if hit_lags else float("nan"),
        "median_hit_steps": float(np.median(hit_lags)) if hit_lags else float("nan"),
        "n_basin_switches": n_basin_switches,
    }


def audit_and_select_local_targets(
    df_cohort: pd.DataFrame,
    seg_id: np.ndarray,
    macro_groups: np.ndarray,
    tau_max_ns: float,
    min_int_frames: int = 250,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Audit all pre-registered protein microswitch targets and pick valid <5%-censored primary target."""
    dt = float(df_cohort["eff_dt_ns"].iloc[0])
    horizon_steps = int(round(tau_max_ns / dt))

    audit_records: list[dict[str, Any]] = []
    valid_candidates: list[dict[str, Any]] = []

    for spec in CANDIDATE_LOCAL_TARGETS:
        col = spec["col"]
        vals = df_cohort[col].to_numpy(dtype=np.float64)
        for r_lo, r_hi in spec["threshold_grid"]:
            res = compute_local_hitting_outcomes(
                vals=vals,
                seg_id=seg_id,
                r_low=r_lo,
                r_high=r_hi,
                b_below=spec["B_below"],
                horizon_steps=horizon_steps,
            )
            unc = res["uncens_mask"]
            n_active_groups = int(np.unique(macro_groups[unc]).size) if np.any(unc) else 0
            n_active_trajs = int(df_cohort.loc[unc, "gpcrmd_id"].nunique()) if np.any(unc) else 0
            y_bar = res["mean_y_uncensored"]
            eff_var_n = (
                float(res["n_uncensored"] * y_bar * (1.0 - y_bar))
                if not math.isnan(y_bar)
                else 0.0
            )
            passes = (
                res["censoring_rate"] < 0.05
                and res["n_uncensored"] >= min_int_frames
                and not math.isnan(y_bar)
                and 0.08 <= y_bar <= 0.92
                and n_active_groups >= 5
                and res["n_basin_switches"] >= 10
            )
            rec = {
                "target_name": spec["name"],
                "col": col,
                "description": spec["description"],
                "B_below": spec["B_below"],
                "r_low": r_lo,
                "r_high": r_hi,
                "tau_max_ns": tau_max_ns,
                "horizon_steps": horizon_steps,
                "n_int_total": res["n_int_total"],
                "n_uncensored": res["n_uncensored"],
                "n_censored": res["n_censored"],
                "censoring_rate": round(res["censoring_rate"], 5),
                "mean_y_uncensored": None if math.isnan(y_bar) else round(y_bar, 4),
                "mean_commit_time_ns": None if math.isnan(res["mean_hit_steps"]) else round(res["mean_hit_steps"] * dt, 3),
                "median_commit_time_ns": None if math.isnan(res["median_hit_steps"]) else round(res["median_hit_steps"] * dt, 3),
                "n_basin_switches": res["n_basin_switches"],
                "n_active_macro_groups": n_active_groups,
                "n_active_trajectories": n_active_trajs,
                "effective_binomial_information": round(eff_var_n, 2),
                "passes_validity_gate": bool(passes),
            }
            audit_records.append(rec)
            if passes:
                valid_candidates.append({**rec, "_raw": res})

    # Pre-registered objective ranking: highest effective binomial sample information N_uncens * p * (1-p)
    valid_candidates.sort(key=lambda d: d["effective_binomial_information"], reverse=True)
    best = valid_candidates[0] if valid_candidates else None
    secondary = None
    if best is not None:
        for cand in valid_candidates[1:]:
            if cand["target_name"] != best["target_name"]:
                secondary = cand
                break
    return audit_records, best, secondary


def build_feature_matrices(
    df_cohort: pd.DataFrame,
    primary_col: str,
    use_explicit_na: bool,
) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Construct nested protein-only and protein+environment feature matrices (col 0 = primary_col)."""
    w356_rad = np.radians(df_cohort["W356_chi1_deg"].to_numpy(dtype=np.float64))
    f155_rad = np.radians(df_cohort["F155_chi1_deg"].to_numpy(dtype=np.float64))
    df_tmp = df_cohort.copy()
    df_tmp["sin_W356_chi1"] = np.sin(w356_rad)
    df_tmp["cos_W356_chi1"] = np.cos(w356_rad)
    df_tmp["sin_F155_chi1"] = np.sin(f155_rad)

    base_prot_pool = [
        "R214_D338_CA_A",
        "D163_N393_CA_A",
        "F200_W356_ring_com_A",
        "F155sc_N393CA_A",
        "Y294_Y397_OH_A",
        "D163_N393_polar_min_A",
        "sin_W356_chi1",
        "cos_W356_chi1",
        "sin_F155_chi1",
    ]
    prot_cols = [primary_col] + [c for c in base_prot_pool if c != primary_col]

    water_cols = [
        "n_wat_core_pore",
        "wire_D163_W356_cond",
        "wire_D163_N393Y397_cond",
        "wire_Y294_Y397_cond",
    ]
    if use_explicit_na:
        ion_cols = ["prox_Na_D163", "coord_Na_D163_3p2"]
    else:
        ion_cols = ["min_cation_D163_A", "coord_cation_D163_3p2"]
    popc_cols = ["min_POPC_TM2_TM4_cleft_A", "min_POPC_TM1_TM7_portal_A"]
    env_all_cols = water_cols + ion_cols + popc_cols

    def mat(cols: list[str]) -> tuple[np.ndarray, list[str]]:
        arr = df_tmp[cols].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            raise AssertionError(f"Non-finite feature values in {cols}")
        return arr, cols

    return {
        "loc1d": mat([primary_col]),
        "prot": mat(prot_cols),
        "prot_water": mat(prot_cols + water_cols),
        "prot_ion": mat(prot_cols + ion_cols),
        "prot_popc": mat(prot_cols + popc_cols),
        "prot_env_all": mat(prot_cols + env_all_cols),
    }


def evaluate_static_and_memory_suite(
    df_cohort: pd.DataFrame,
    seg_id: np.ndarray,
    groups: np.ndarray,
    target_bundle: dict[str, Any],
    use_explicit_na: bool,
    lag_steps_list: list[int],
    n_splits: int = 5,
    n_shuffles: int = 100,
    n_perms: int = 100,
) -> dict[str, Any]:
    """Run leakage-free static, N1 permutation, Manski censoring, stopped-DGA, and H2 memory tests."""
    dt = float(df_cohort["eff_dt_ns"].iloc[0])
    raw = target_bundle["_raw"]
    in_A = raw["in_A"]
    in_B = raw["in_B"]
    unc_mask = raw["uncens_mask"]
    is_int_all = raw["is_horizon_safe_int"]
    is_cens = raw["is_censored"]

    # Fill y_full = 0 in A, 1 in B, uncensored binary hit in unc_mask; NaN guard on censored/tail interior
    # frames so non-unc_mask interior frames can NEVER silently contaminate supervised training!
    y_full = np.full(in_A.shape, np.nan, dtype=np.float64)
    y_full[in_A] = 0.0
    y_full[in_B] = 1.0
    y_full[unc_mask] = raw["y_hit"][unc_mask]

    f_mats = build_feature_matrices(df_cohort, target_bundle["col"], use_explicit_na=use_explicit_na)
    X_prot, prot_names = f_mats["prot"]
    X_env, env_names = f_mats["prot_env_all"]

    # Ensure enough active groups with uncensored interior frames for CV
    active_groups = np.unique(groups[unc_mask])
    eff_splits = min(n_splits, int(active_groups.size))
    if eff_splits < 2:
        raise ValueError(f"Insufficient active groups ({active_groups.size}) for group CV.")

    # Map group IDs so uncensored interior frames balance cleanly across GroupKFold
    splitter = LineageGroupSplitter(n_splits=eff_splits, stratified=False, random_state=42)
    unc_idx = np.flatnonzero(unc_mask)
    X_prot_u = X_prot[unc_idx]
    X_env_u = X_env[unc_idx]
    y_u = y_full[unc_idx]
    g_u = groups[unc_idx]

    oof_preds: dict[str, np.ndarray] = {
        "B0": np.zeros_like(y_u),
        "B1": np.zeros_like(y_u),
        "prot": np.zeros_like(y_u),
        "prot_water": np.zeros_like(y_u),
        "prot_ion": np.zeros_like(y_u),
        "prot_popc": np.zeros_like(y_u),
        "prot_env_all": np.zeros_like(y_u),
        "oracle_prot": np.zeros_like(y_u),
        "oracle_prot_env": np.zeros_like(y_u),
    }

    for tr, te in splitter.split(X_prot_u, None, g_u):
        assert_disjoint_group_split(g_u, tr, te)
        m_b0 = PriorBaselineB0(progress_col=0).fit(X_prot_u[tr], y_u[tr])
        m_b1 = SplineProgressBaselineB1(progress_col=0, n_knots=5, l2_reg=1e-3).fit(X_prot_u[tr], y_u[tr])
        oof_preds["B0"][te] = m_b0.predict(X_prot_u[te])
        oof_preds["B1"][te] = m_b1.predict(X_prot_u[te])

        for key in ("prot", "prot_water", "prot_ion", "prot_popc", "prot_env_all"):
            Xk_u = f_mats[key][0][unc_idx]
            mk = PetersTroutCommittorModel(include_interactions=False, l2_reg=5e-3).fit(Xk_u[tr], y_u[tr])
            oof_preds[key][te] = mk.predict(Xk_u[te])

        or_p = OracleHistGBRModel(max_iter=100, max_leaf_nodes=15, min_samples_leaf=30, l2_regularization=2.0).fit(
            X_prot_u[tr], y_u[tr]
        )
        or_pe = OracleHistGBRModel(max_iter=100, max_leaf_nodes=15, min_samples_leaf=30, l2_regularization=2.0).fit(
            X_env_u[tr], y_u[tr]
        )
        oof_preds["oracle_prot"][te] = or_p.predict(X_prot_u[te])
        oof_preds["oracle_prot_env"][te] = or_pe.predict(X_env_u[te])

    # Static comparisons with 1,000-group cluster bootstrap CIs
    static_results: dict[str, Any] = {
        "log_likelihood_nats": {k: round(committor_log_likelihood(y_u, v), 5) for k, v in oof_preds.items()},
        "prot_vs_B1": bootstrap_bss_and_delta_ell_ci(g_u, y_u, oof_preds["prot"], oof_preds["B1"], n_boot=1000),
        "prot_env_vs_B1": bootstrap_bss_and_delta_ell_ci(g_u, y_u, oof_preds["prot_env_all"], oof_preds["B1"], n_boot=1000),
        "prot_env_vs_prot": bootstrap_bss_and_delta_ell_ci(g_u, y_u, oof_preds["prot_env_all"], oof_preds["prot"], n_boot=1000),
        "prot_water_vs_prot": bootstrap_bss_and_delta_ell_ci(g_u, y_u, oof_preds["prot_water"], oof_preds["prot"], n_boot=1000),
        "prot_ion_vs_prot": bootstrap_bss_and_delta_ell_ci(g_u, y_u, oof_preds["prot_ion"], oof_preds["prot"], n_boot=1000),
        "prot_popc_vs_prot": bootstrap_bss_and_delta_ell_ci(g_u, y_u, oof_preds["prot_popc"], oof_preds["prot"], n_boot=1000),
        "oracle_gap_prot_env": round(
            delta_log_likelihood(y_u, oof_preds["oracle_prot_env"], oof_preds["prot_env_all"]), 5
        ),
    }

    # Progress-stratified N1 macro-block permutation null for adding z_env (keeping X_prot intact!)
    rng_n1 = np.random.default_rng(42)
    env_only_u = X_env_u[:, len(prot_names) :]
    unique_g = np.unique(g_u)
    g_means = np.array([np.mean(X_prot_u[g_u == ug, 0]) for ug in unique_g], dtype=np.float64)
    k_st = max(1, min(3, unique_g.size // 3))
    if k_st == 1:
        strata = np.zeros(unique_g.size, dtype=np.int64)
    else:
        edges = np.unique(np.quantile(g_means, np.linspace(0.0, 1.0, k_st + 1)))
        strata = np.digitize(g_means, edges[1:-1])

    n1_null_dells = np.empty(n_perms, dtype=np.float64)
    for p_i in range(n_perms):
        env_perm = env_only_u.copy()
        for st in np.unique(strata):
            st_groups = unique_g[strata == st]
            if st_groups.size < 2:
                st_groups = unique_g
            perm_g = rng_n1.permutation(st_groups)
            idx_concat = np.concatenate([np.flatnonzero(g_u == ug) for ug in perm_g])
            max_g_len = max(int(np.sum(g_u == ug)) for ug in perm_g)
            tot_len = idx_concat.size
            if tot_len > 2 * max_g_len:
                shift = int(rng_n1.integers(max_g_len, tot_len - max_g_len + 1))
                shifted_idx = np.roll(idx_concat, shift)
            elif tot_len > max_g_len:
                shifted_idx = np.roll(idx_concat, max_g_len)
            else:
                shifted_idx = idx_concat[::-1]
            env_perm[idx_concat] = env_only_u[shifted_idx]
        X_perm_u = np.hstack([X_prot_u, env_perm])

        oof_perm = np.zeros_like(y_u)
        for tr, te in splitter.split(X_perm_u, None, g_u):
            mp = PetersTroutCommittorModel(include_interactions=False, l2_reg=5e-3).fit(X_perm_u[tr], y_u[tr])
            oof_perm[te] = mp.predict(X_perm_u[te])
        n1_null_dells[p_i] = delta_log_likelihood(y_u, oof_perm, oof_preds["prot"])

    obs_env_dell = float(static_results["prot_env_vs_prot"]["delta_ell"]["point"])
    static_results["n1_env_block_permutation"] = {
        "n_progress_strata": int(k_st),
        "obs_delta_ell_env_vs_prot": round(obs_env_dell, 5),
        "null_mean": round(float(np.mean(n1_null_dells)), 5),
        "null_q95": round(float(np.quantile(n1_null_dells, 0.95)), 5),
        "p_value": round(float((1 + np.sum(n1_null_dells >= obs_env_dell)) / (n_perms + 1)), 4),
    }

    # Standardized coefficients on full uncensored set for mechanistic interpretability
    full_fit_env = PetersTroutCommittorModel(
        include_interactions=False, l2_reg=5e-3, feature_names=env_names
    ).fit(X_env_u, y_u)
    static_results["standardized_weights_prot_env"] = {
        name: round(float(w), 4) for name, w in zip(env_names, full_fit_env.linear_coef_scaled_)
    }

    # Manski [0, 1] worst-case right-censoring sensitivity bounds across ALL horizon-safe interior frames
    int_idx = np.flatnonzero(is_int_all)
    X_prot_allint = X_prot[int_idx]
    X_env_allint = X_env[int_idx]
    g_allint = groups[int_idx]
    cens_sub = is_cens[int_idx]
    y_base_int = np.where(cens_sub, 0.0, raw["y_hit"][int_idx])

    manski_bounds: dict[str, Any] = {}
    for label_c, fill_val in (("all_censored_to_0", 0.0), ("all_censored_to_1", 1.0)):
        y_wc = y_base_int.copy()
        y_wc[cens_sub] = fill_val
        oof_p_wc = np.zeros_like(y_wc)
        oof_pe_wc = np.zeros_like(y_wc)
        for tr, te in splitter.split(X_prot_allint, None, g_allint):
            mp = PetersTroutCommittorModel(include_interactions=False, l2_reg=5e-3).fit(X_prot_allint[tr], y_wc[tr])
            mpe = PetersTroutCommittorModel(include_interactions=False, l2_reg=5e-3).fit(X_env_allint[tr], y_wc[tr])
            oof_p_wc[te] = mp.predict(X_prot_allint[te])
            oof_pe_wc[te] = mpe.predict(X_env_allint[te])
        manski_bounds[label_c] = {
            "delta_ell_env_vs_prot": round(delta_log_likelihood(y_wc, oof_pe_wc, oof_p_wc), 5),
            "bss_env_vs_prot": round(brier_skill_score(y_wc, oof_pe_wc, oof_p_wc), 5),
        }
    static_results["manski_censoring_bounds"] = manski_bounds

    # Two-Step Mori-Zwanzig Memory Profile across lag_steps_list using `src.core.evaluate_mori_zwanzig_memory`.
    # CRITICAL: Mask out censored & horizon-unsafe tail interior frames (`~unc_mask`) from `fit_interior_only`
    # via `in_A_mem_fit = in_A | ((~in_B) & (~unc_mask))`, so `~(in_A_mem_fit | in_B) == unc_mask` bitwise
    # and every fold in `mem_folds` trains exclusively on 100% uncensored binary hits while leaving `unc_mask`
    # test evaluation completely unclamped.
    in_A_mem_fit = in_A | ((~in_B) & (~unc_mask))
    assert np.array_equal(~(in_A_mem_fit | in_B), unc_mask)
    mem_folds: list[tuple[np.ndarray, np.ndarray]] = []
    for _, te_u in splitter.split(X_prot_u, None, g_u):
        te_groups = np.unique(g_u[te_u])
        te_full_mask = np.isin(groups, te_groups)
        mem_folds.append((np.flatnonzero(~te_full_mask), np.flatnonzero(te_full_mask)))

    memory_profile: list[dict[str, Any]] = []
    for lag_s in lag_steps_list:
        lag_ns = round(lag_s * dt, 3)
        mem_1d = evaluate_mori_zwanzig_memory(
            X_static=f_mats["loc1d"][0],
            seg_id=seg_id,
            q_target=y_full,
            groups=groups,
            lag_steps=lag_s,
            eval_mask=unc_mask,
            in_A=in_A_mem_fit,
            in_B=in_B,
            folds=mem_folds,
            fit_interior_only=True,
            include_interactions=False,
            l2_reg=5e-3,
            n_splits=eff_splits,
            n_shuffles=n_shuffles,
            random_state=42,
        )
        mem_prot = evaluate_mori_zwanzig_memory(
            X_static=X_prot,
            seg_id=seg_id,
            q_target=y_full,
            groups=groups,
            lag_steps=lag_s,
            eval_mask=unc_mask,
            in_A=in_A_mem_fit,
            in_B=in_B,
            folds=mem_folds,
            fit_interior_only=True,
            include_interactions=False,
            l2_reg=5e-3,
            n_splits=eff_splits,
            n_shuffles=n_shuffles,
            random_state=42,
        )
        mem_env = evaluate_mori_zwanzig_memory(
            X_static=X_env,
            seg_id=seg_id,
            q_target=y_full,
            groups=groups,
            lag_steps=lag_s,
            eval_mask=unc_mask,
            in_A=in_A_mem_fit,
            in_B=in_B,
            folds=mem_folds,
            fit_interior_only=True,
            include_interactions=False,
            l2_reg=5e-3,
            n_splits=eff_splits,
            n_shuffles=n_shuffles,
            random_state=42,
        )
        rho_collapse_prot_to_env = compute_memory_collapse_ratio(
            mem_prot["delta_ell_mem"], mem_env["delta_ell_mem"]
        )
        rho_collapse_1d_to_env = compute_memory_collapse_ratio(
            mem_1d["delta_ell_mem"], mem_env["delta_ell_mem"]
        )
        memory_profile.append({
            "lag_steps": int(lag_s),
            "lag_ns": lag_ns,
            "n_eval_frames": int(mem_prot["n_eval_frames"]),
            "loc1d": {
                "delta_ell_mem": round(mem_1d["delta_ell_mem"], 6),
                "ci95_low": round(mem_1d["delta_ell_ci"]["ci_low"], 6),
                "ci95_high": round(mem_1d["delta_ell_ci"]["ci_high"], 6),
                "n2a_q95": round(mem_1d["n2a_null_q95"], 6),
                "p_value_vs_n2a": round(mem_1d["p_value_vs_n2a"], 4),
            },
            "prot_only": {
                "delta_ell_mem": round(mem_prot["delta_ell_mem"], 6),
                "bss_mem": round(mem_prot["bss_mem"], 5),
                "ci95_low": round(mem_prot["delta_ell_ci"]["ci_low"], 6),
                "ci95_high": round(mem_prot["delta_ell_ci"]["ci_high"], 6),
                "n2a_null_mean": round(mem_prot["n2a_null_mean"], 6),
                "n2a_q95": round(mem_prot["n2a_null_q95"], 6),
                "p_value_vs_n2a": round(mem_prot["p_value_vs_n2a"], 4),
            },
            "prot_plus_env": {
                "delta_ell_mem": round(mem_env["delta_ell_mem"], 6),
                "bss_mem": round(mem_env["bss_mem"], 5),
                "ci95_low": round(mem_env["delta_ell_ci"]["ci_low"], 6),
                "ci95_high": round(mem_env["delta_ell_ci"]["ci_high"], 6),
                "n2a_null_mean": round(mem_env["n2a_null_mean"], 6),
                "n2a_q95": round(mem_env["n2a_null_q95"], 6),
                "p_value_vs_n2a": round(mem_env["p_value_vs_n2a"], 4),
            },
            "rho_mem_collapse_prot_to_env": round(rho_collapse_prot_to_env, 4),
            "rho_mem_collapse_1d_to_env": round(rho_collapse_1d_to_env, 4),
        })

    # Operator Cross-Check: Out-of-Fold PBR on a SINGLE SHARED Fold-Isolated Witness Subspace
    # across B1, PT_prot, PT_prot_env, StoppedDGA_prot, and StoppedDGA_prot_env (same boundary problem!)
    dga_lag = max(1, int(round(2.0 / dt)))
    shared_pbr_folds: dict[str, list[float]] = {
        "B1": [],
        "PT_prot": [],
        "PT_prot_env": [],
        "StoppedDGA_prot": [],
        "StoppedDGA_prot_env": [],
    }
    for f_i, (tr_all, te_all) in enumerate(splitter.split(X_prot, None, groups)):
        assert_disjoint_group_split(groups, tr_all, te_all)
        seg_tr = seg_id[tr_all]
        seg_te = seg_id[te_all]
        int_tr_mask = ~(in_A[tr_all] | in_B[tr_all])
        unc_tr_mask = unc_mask[tr_all]
        if (
            np.any(in_A[tr_all])
            and np.any(in_B[tr_all])
            and np.sum(int_tr_mask) >= 30
            and np.sum(unc_tr_mask) >= 20
        ):
            idx_s, idx_e = build_forward_stopped_pairs(seg_te, in_A[te_all], in_B[te_all], dga_lag)
            if idx_s.size < 15:
                continue
            try:
                # Fold-isolated 8D interior-PCA witness subspace fitted strictly on training interior
                wit_sc = StandardScaler().fit(X_env[tr_all][int_tr_mask])
                n_wit_pca = min(8, X_env.shape[1], int(np.sum(int_tr_mask)) - 2)
                wit_pca = PCA(n_components=n_wit_pca, random_state=42).fit(
                    wit_sc.transform(X_env[tr_all][int_tr_mask])
                )
                Z_wit_te = wit_pca.transform(wit_sc.transform(X_env[te_all]))

                # Fit all 5 models strictly on tr_all for the exact same (in_A, in_B) boundary problem
                tr_u_idx = tr_all[unc_tr_mask]
                b1_f = SplineProgressBaselineB1(progress_col=0, n_knots=5, l2_reg=1e-3).fit(
                    X_prot[tr_u_idx], y_full[tr_u_idx]
                )
                pt_p_f = PetersTroutCommittorModel(include_interactions=False, l2_reg=5e-3).fit(
                    X_prot[tr_u_idx], y_full[tr_u_idx]
                )
                pt_e_f = PetersTroutCommittorModel(include_interactions=False, l2_reg=5e-3).fit(
                    X_env[tr_u_idx], y_full[tr_u_idx]
                )
                dga_p = StoppedDGACommittor(lag_steps=dga_lag, n_rbf=40, include_poly2=False, reg=1e-3).fit(
                    X_prot[tr_all], in_A[tr_all], in_B[tr_all], seg_id=seg_tr
                )
                dga_e = StoppedDGACommittor(lag_steps=dga_lag, n_rbf=40, include_poly2=False, reg=1e-3).fit(
                    X_env[tr_all], in_A[tr_all], in_B[tr_all], seg_id=seg_tr
                )

                A_te = in_A[te_all]
                B_te = in_B[te_all]
                preds_te = {
                    "B1": b1_f.predict(X_prot[te_all], in_A=A_te, in_B=B_te),
                    "PT_prot": pt_p_f.predict(X_prot[te_all], in_A=A_te, in_B=B_te),
                    "PT_prot_env": pt_e_f.predict(X_env[te_all], in_A=A_te, in_B=B_te),
                    "StoppedDGA_prot": dga_p.predict(X_prot[te_all], A_te, B_te),
                    "StoppedDGA_prot_env": dga_e.predict(X_env[te_all], A_te, B_te),
                }
                for m_key, q_te in preds_te.items():
                    pbr_val = projected_bellman_residual(
                        q_pred_start=q_te[idx_s],
                        q_pred_stop=q_te[idx_e],
                        X_start_interior=Z_wit_te[idx_s],
                        n_witness=20,
                        random_state=100 + f_i,
                    )
                    shared_pbr_folds[m_key].append(pbr_val)
            except ValueError:
                pass

    dga_full_prot = StoppedDGACommittor(lag_steps=dga_lag, n_rbf=50, include_poly2=False, reg=1e-3).fit(
        X_prot, in_A, in_B, seg_id=seg_id
    )
    n2b_prot = run_time_reversal_diagnostic_N2b(
        dga_full_prot, X_prot, in_A, in_B, seg_id=seg_id, eval_mask=unc_mask
    )
    dga_full_env = StoppedDGACommittor(lag_steps=dga_lag, n_rbf=50, include_poly2=False, reg=1e-3).fit(
        X_env, in_A, in_B, seg_id=seg_id
    )
    n2b_env = run_time_reversal_diagnostic_N2b(
        dga_full_env, X_env, in_A, in_B, seg_id=seg_id, eval_mask=unc_mask
    )

    return {
        "target_metadata": {k: v for k, v in target_bundle.items() if k != "_raw"},
        "n_cv_splits": eff_splits,
        "n_unique_groups": int(active_groups.size),
        "feature_names_prot": prot_names,
        "feature_names_prot_env": env_names,
        "static_committor_evaluation": static_results,
        "mori_zwanzig_memory_profile": memory_profile,
        "stopped_dga_diagnostics": {
            "dga_lag_ns": round(dga_lag * dt, 3),
            "shared_witness_pca_dim": 8,
            "mean_oof_pbr_shared_witness": {
                k: (round(float(np.mean(v)), 5) if v else None) for k, v in shared_pbr_folds.items()
            },
            "n2b_reversibility_prot": {k: (round(v, 5) if isinstance(v, float) else v) for k, v in n2b_prot.items()},
            "n2b_reversibility_prot_env": {k: (round(v, 5) if isinstance(v, float) else v) for k, v in n2b_env.items()},
        },
    }


def plot_gi1_vs_gs_coupling(df: pd.DataFrame, pillar_a: dict[str, Any], out_path: Path) -> None:
    """Render 4-panel publication figure comparing Cognate Gi1 (1201) vs Non-Cognate Gs (2067) vs Inactive."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
    })

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 10.2), dpi=300)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    gi = df[df["gpcrmd_id"] == 1201]
    gs = df[df["gpcrmd_id"] == 2067]
    inact = df[df["gpcrmd_id"].isin([22, 164, 162, 163])]

    # Panel A: Time-resolved Y294-Y397 intracellular Tyr latch & TM6 opening in Gi1 vs Gs
    gi_roll = gi["Y294_Y397_OH_A"].rolling(50, min_periods=1).mean()
    gs_roll = gs["Y294_Y397_OH_A"].rolling(25, min_periods=1).mean()
    ax_a.plot(gi["time_ns"] / 1000.0, gi_roll, color="#2E8B57", lw=1.7, label="1201: 6N4B + FUB + Cognate Gi1 (1.0 µs)")
    ax_a.plot(gs["time_ns"] / 1000.0, gs_roll, color="#D08770", lw=1.4, alpha=0.92, label="2067: 6N4B + Apo + Non-Cognate Gs (6.0 µs)")
    inact_mean = float(inact["Y294_Y397_OH_A"].mean())
    ax_a.axhline(inact_mean, color="#4C566A", ls=":", lw=1.3, label=f"Inactive 4-traj mean ({inact_mean:.2f} Å)")
    ax_a.axhline(4.2, color="#2E8B57", ls="--", lw=1.0, alpha=0.8, label="Active polar latch threshold (4.2 Å)")
    ax_a.set_ylim(2.3, 11.4)
    ax_a.set_xlabel("Simulation Time (µs)")
    ax_a.set_ylabel(r"$d(\mathrm{Y294^{5.58}\!-\!Y397^{7.53}\ OH})$ Moving Avg (Å)")
    ax_a.set_title("(a) Intracellular Y294–Y397 Polar Latch Stability: Gi1 vs Gs")
    ax_a.legend(loc="upper right", fontsize=8.0, framealpha=0.94)
    ax_a.grid(True, alpha=0.25)

    # Panel B: Matched Open-TM6 ([13.5, 16.5] A) Joint/Marginal Comparison of F155sc-N393CA & Y294-Y397
    bins_y = np.linspace(2.4, 13.0, 55)
    ax_b.hist(gi["Y294_Y397_OH_A"], bins=bins_y, density=True, alpha=0.55, color="#2E8B57", label="Gi1 + FUB (1201)")
    ax_b.hist(gs["Y294_Y397_OH_A"], bins=bins_y, density=True, alpha=0.55, color="#D08770", label="Gs + Apo (2067)")
    ax_b.hist(inact["Y294_Y397_OH_A"], bins=bins_y, density=True, alpha=0.45, color="#4C566A", label="Inactive (22/164/162/163)")
    ax_b.axvline(4.2, color="#2E8B57", ls="--", lw=1.1)
    ax_b.set_xlabel(r"$d(\mathrm{Y294^{5.58}\!-\!Y397^{7.53}\ OH})$ (Å)")
    ax_b.set_ylabel("Probability Density")
    ax_b.set_title("(b) Tyr-Lock Equilibrium Density Across Transducer Regimes")
    ax_b.legend(loc="upper right", fontsize=8.2, framealpha=0.94)
    ax_b.grid(True, alpha=0.25)

    # Panel C: Internal Transmembrane Pore Hydration & D163->N393/Y397 Water-Wire Percolation
    labels_c = [
        "Gi1 + FUB\n(ID 1201)",
        "Gs + Apo\n(ID 2067)",
        "5U09 + Tar A\n(ID 22)",
        "5U09 + Tar B\n(ID 164)",
        "5TGZ Apo\n(ID 162)",
        "5U09 Apo\n(ID 163)",
    ]
    gids_c = [1201, 2067, 22, 164, 162, 163]
    cols_c = ["#2E8B57", "#D08770", "#4C566A", "#5E81AC", "#7B88A1", "#8896AB"]
    wat_means = [pillar_a["per_trajectory"][str(g)]["n_wat_core_pore_mean"] for g in gids_c]
    perc_fracs = [100.0 * pillar_a["per_trajectory"][str(g)]["wire_D163_NY_percolated_frac"] for g in gids_c]

    x_c = np.arange(len(gids_c))
    ax_c_twin = ax_c.twinx()
    ax_c.bar(x_c - 0.18, wat_means, width=0.35, color=cols_c, edgecolor="black", lw=0.7, label="Pore Waters (left)")
    ax_c_twin.bar(
        x_c + 0.18, perc_fracs, width=0.35, color="#88C0D0", edgecolor="black", lw=0.7, alpha=0.85,
        label="D163–N393/Y397 Wire % (right)"
    )
    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels(labels_c, fontsize=8.0)
    ax_c.set_ylabel("Mean Core Pore Waters (|z|≤12 Å, r≤8 Å)")
    ax_c.set_ylim(0, max(wat_means) * 1.24)
    ax_c_twin.set_ylabel("D163↔N393/Y397 Water-Wire Percolation (%)")
    ax_c_twin.set_ylim(0, 124)
    ax_c_twin.set_yticks([0, 20, 40, 60, 80, 100])
    ax_c.set_title("(c) Explicit Transmembrane Hydration & Water-Wire Percolation")
    ax_c.grid(True, axis="y", alpha=0.25)
    lines1, labels1 = ax_c.get_legend_handles_labels()
    lines2, labels2 = ax_c_twin.get_legend_handles_labels()
    ax_c.legend(lines1 + lines2, labels1 + labels2, loc="upper center", ncol=2, fontsize=7.8, framealpha=0.94)

    # Panel D: Microswitch Occupancy Summary with 50-ns Block-Bootstrap 95% CIs
    cs = pillar_a["cohort_summary"]
    m_keys = [
        ("F155 Outward\n(gauche-)", "F155_outward_gauche_minus"),
        ("F237 Outward\n(trans)", "F237_outward_trans"),
        ("Y294–Y397\n(≤4.2 Å)", "Y294_Y397_polar_latch_4p2"),
        ("D163–NPxxY\nWater Wire", "wire_D163_NY_percolated"),
        ("D163 Cation\n(<3.2 Å)", "cation_bound_D163"),
    ]
    x_d = np.arange(len(m_keys))
    w_d = 0.25
    for k_i, (ckey, clbl, ccol) in enumerate([
        ("Gi1_1201", "Gi1 + FUB (1201, KCl)", "#2E8B57"),
        ("Gs_2067", "Gs + Apo (2067, NaCl)", "#D08770"),
        ("Inactive_4traj", "Inactive (4-traj, NaCl)", "#4C566A"),
    ]):
        pts = [100.0 * cs[ckey]["metrics"][mk]["mean"] for _, mk in m_keys]
        lo_err = [max(0.0, 100.0 * (cs[ckey]["metrics"][mk]["mean"] - cs[ckey]["metrics"][mk]["ci95_low"])) for _, mk in m_keys]
        hi_err = [max(0.0, 100.0 * (cs[ckey]["metrics"][mk]["ci95_high"] - cs[ckey]["metrics"][mk]["mean"])) for _, mk in m_keys]
        ax_d.bar(
            x_d + (k_i - 1) * w_d, pts, width=w_d, color=ccol, edgecolor="black", lw=0.7,
            yerr=[lo_err, hi_err], capsize=2.5, label=clbl, zorder=3
        )
    ax_d.set_xticks(x_d)
    ax_d.set_xticklabels([m[0] for m in m_keys], fontsize=8.2)
    ax_d.set_ylabel("Equilibrium Occupancy (%) ± 95% Block-Boot CI")
    ax_d.set_ylim(0, 124)
    ax_d.set_yticks([0, 20, 40, 60, 80, 100])
    ax_d.set_title("(d) Allosteric Microswitch & Ion Locking Across Regimes")
    ax_d.legend(loc="upper center", ncol=3, fontsize=7.4, framealpha=0.94)
    ax_d.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        "Tier 1B Explicit-Solvent CB1 Dynamics (8.90 µs / 20,501 Frames): Cognate Gi1 vs Non-Cognate Gs & Inactive Core",
        fontsize=11.8, fontweight="bold", y=0.992
    )
    fig.tight_layout(rect=[0.0, 0.01, 1.0, 0.972])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_water_na_memory_closure(pillar_b: dict[str, Any], out_path: Path) -> None:
    """Render 4-panel (2x2) publication figure of H2 Static Gain & Mori-Zwanzig Memory Closure."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9.3,
        "axes.titlesize": 10.2,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.3,
    })

    primary_key = pillar_b["primary_cohort_key"]
    primary_res = pillar_b["cohorts"][primary_key]
    nacl_res = pillar_b["cohorts"].get("NaCl5_1p0ns_ExplicitNa", primary_res)

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 10.0), dpi=300)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # Panel A: Out-of-Fold Static Log-Likelihood Gain over Protein-Only across Both Primary & NaCl-5 Cohorts
    sub_specs = [
        ("+Water Wire\n(Pore & Hops)", "prot_water_vs_prot"),
        ("+Explicit Ion\n(D163 Coord)", "prot_ion_vs_prot"),
        ("+POPC Wedge\n(TM2-4/1-7)", "prot_popc_vs_prot"),
        ("+Full Env\n(Wat+Ion+POPC)", "prot_env_vs_prot"),
    ]
    x_a = np.arange(len(sub_specs))
    w_a = 0.34
    for idx_c, (res_obj, lbl_c, col_c) in enumerate([
        (primary_res, "Fine 0.2-ns: F155–F237 (3.25% cens.)", "#5E81AC"),
        (nacl_res, "NaCl-5 1.0-ns: F200–W356 (2.62% cens.)", "#2E8B57"),
    ]):
        sr = res_obj["static_committor_evaluation"]
        pts = [sr[k]["delta_ell"]["point"] for _, k in sub_specs]
        lo = [max(0.0, sr[k]["delta_ell"]["point"] - sr[k]["delta_ell"]["ci_low"]) for _, k in sub_specs]
        hi = [max(0.0, sr[k]["delta_ell"]["ci_high"] - sr[k]["delta_ell"]["point"]) for _, k in sub_specs]
        ax_a.bar(
            x_a + (idx_c - 0.5) * w_a, pts, width=w_a, color=col_c, edgecolor="black",
            lw=0.75, yerr=[lo, hi], capsize=3.0, label=lbl_c, zorder=3
        )
    n1_p_q95 = primary_res["static_committor_evaluation"]["n1_env_block_permutation"]["null_q95"]
    n1_n_q95 = nacl_res["static_committor_evaluation"]["n1_env_block_permutation"]["null_q95"]
    ax_a.axhline(0.0, color="#333333", lw=1.0)
    ax_a.axhline(n1_p_q95, color="#5E81AC", ls="--", lw=1.1, label=f"N1 95th pct Fine ({n1_p_q95:.4f})")
    ax_a.axhline(n1_n_q95, color="#2E8B57", ls=":", lw=1.2, label=f"N1 95th pct NaCl-5 ({n1_n_q95:.4f})")
    ax_a.set_xticks(x_a)
    ax_a.set_xticklabels([s[0] for s in sub_specs], fontsize=8.4)
    ax_a.set_ylim(-0.043, 0.025)
    ax_a.set_ylabel(r"OOF Static Gain $\Delta\bar{\ell}(+\mathbf{z}_{\mathrm{env}}\ \mathrm{vs}\ \mathrm{Prot})$ (nats)")
    ax_a.set_title("(a) Static Committor Gain Over 10D Protein Vector")
    ax_a.legend(loc="lower left", fontsize=7.6, framealpha=0.93)
    ax_a.grid(True, axis="y", alpha=0.25)

    # Panel B: Multi-Lag Mori-Zwanzig Memory Profile on Primary Fine 0.2-ns Cohort (F155-F237)
    mem_p = primary_res["mori_zwanzig_memory_profile"]
    lags_p = [m["lag_ns"] for m in mem_p]
    ax_b.plot(lags_p, [m["loc1d"]["delta_ell_mem"] for m in mem_p], "s:", color="#7B88A1", lw=1.5, label="1D Local Coord + Δ1D (B2)")
    ax_b.plot(lags_p, [m["prot_only"]["delta_ell_mem"] for m in mem_p], "o-", color="#BF616A", lw=2.0, label="10D Protein + ΔProt (Step A)")
    ax_b.fill_between(
        lags_p,
        [m["prot_only"]["ci95_low"] for m in mem_p],
        [m["prot_only"]["ci95_high"] for m in mem_p],
        color="#BF616A", alpha=0.15,
    )
    ax_b.plot(lags_p, [m["prot_plus_env"]["delta_ell_mem"] for m in mem_p], "^-", color="#2E8B57", lw=2.0, label="Protein + Explicit Env + Δ(P+E)")
    ax_b.plot(lags_p, [m["prot_only"]["n2a_q95"] for m in mem_p], "--", color="#4C566A", lw=1.2, label="N2a History-Shuffle 95th pct")
    ax_b.axhline(0.0, color="#333333", lw=0.9)
    ax_b.set_xlabel(r"History Lag $\delta t$ (ns)")
    ax_b.set_ylabel(r"Conditional Memory Gain $\Delta\bar{\ell}_{\mathrm{mem}}(\delta t)$ (nats)")
    ax_b.set_title("(b) Memory Profile: Fine 0.2-ns (F155–F237 Cleft)")
    ax_b.legend(loc="upper right", fontsize=7.8, framealpha=0.93)
    ax_b.grid(True, alpha=0.25)

    # Panel C: Multi-Lag Mori-Zwanzig Memory Profile on Harmonized Explicit-Na+ NaCl-5 Cohort (F200-W356)
    mem_n = nacl_res["mori_zwanzig_memory_profile"]
    lags_n = [m["lag_ns"] for m in mem_n]
    pct_1d_prot = int(round(100.0 * compute_memory_collapse_ratio(
        mem_n[0]["loc1d"]["delta_ell_mem"], mem_n[0]["prot_only"]["delta_ell_mem"]
    )))
    pct_prot_env = int(round(100.0 * mem_n[0]["rho_mem_collapse_prot_to_env"]))
    ax_c.plot(lags_n, [m["loc1d"]["delta_ell_mem"] for m in mem_n], "s:", color="#7B88A1", lw=1.5, label="1D F200–W356 + Δ1D (B2)")
    ax_c.plot(lags_n, [m["prot_only"]["delta_ell_mem"] for m in mem_n], "o-", color="#BF616A", lw=2.0, label=f"10D Protein + ΔProt (-{pct_1d_prot}% vs 1D)")
    ax_c.fill_between(
        lags_n,
        [m["prot_only"]["ci95_low"] for m in mem_n],
        [m["prot_only"]["ci95_high"] for m in mem_n],
        color="#BF616A", alpha=0.15,
    )
    ax_c.plot(lags_n, [m["prot_plus_env"]["delta_ell_mem"] for m in mem_n], "^-", color="#2E8B57", lw=2.0, label=f"Prot + Wat/Na+/POPC (-{pct_prot_env}% vs Prot)")
    ax_c.plot(lags_n, [m["prot_only"]["n2a_q95"] for m in mem_n], "--", color="#4C566A", lw=1.2, label="N2a History-Shuffle 95th pct")
    ax_c.axhline(0.0, color="#333333", lw=0.9)
    ax_c.set_xlabel(r"History Lag $\delta t$ (ns)")
    ax_c.set_ylabel(r"Conditional Memory Gain $\Delta\bar{\ell}_{\mathrm{mem}}(\delta t)$ (nats)")
    ax_c.set_title("(c) Memory Profile: Explicit-Na+ NaCl-5 (F200–W356 Toggle)")
    ax_c.legend(loc="upper right", fontsize=7.8, framealpha=0.93)
    ax_c.grid(True, alpha=0.25)

    # Panel D: Top Standardized Logit Weights on NaCl-5 Explicit-Na+ Twin Toggle (F200-W356)
    w_dict = nacl_res["static_committor_evaluation"]["standardized_weights_prot_env"]
    sorted_w = sorted(w_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)[:9][::-1]
    w_names = [k for k, _ in sorted_w]
    w_vals = [v for _, v in sorted_w]
    is_env = [any(tag in k for tag in ("wat", "wire", "Na", "cation", "POPC")) for k in w_names]
    w_cols = ["#2E8B57" if flag else "#5E81AC" for flag in is_env]
    y_d = np.arange(len(w_names))
    ax_d.barh(y_d, w_vals, color=w_cols, edgecolor="black", lw=0.7, height=0.62)
    ax_d.axvline(0.0, color="#333333", lw=1.0)
    ax_d.set_yticks(y_d)
    ax_d.set_yticklabels(w_names, fontsize=8.0)
    ax_d.set_xlabel("Standardized Peters–Trout Logit Coefficient")
    ax_d.set_title("(d) Top Committor Weights (NaCl-5 F200–W356)")
    from matplotlib.patches import Patch
    ax_d.legend(
        handles=[
            Patch(facecolor="#5E81AC", edgecolor="black", label="Protein Coordinate"),
            Patch(facecolor="#2E8B57", edgecolor="black", label="Explicit Wat / Na+ / POPC"),
        ],
        loc="lower right", fontsize=7.8, framealpha=0.93,
    )
    ax_d.grid(True, axis="x", alpha=0.25)

    fig.suptitle(
        "WS3 Hypothesis H2 Test on Explicit-Solvent GPCRmd (<5% Censoring, Buffered Macro-Block CV)",
        fontsize=11.8, fontweight="bold", y=0.992
    )
    fig.tight_layout(rect=[0.0, 0.01, 1.0, 0.97])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def run_ws3_analysis() -> dict[str, Any]:
    """Execute complete WS3 analysis pipeline and save ws3_metrics.json + publication figures."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    pdb16_df = pd.read_parquet(PDB16_PARQUET)
    gpcrmd_df = pd.read_parquet(GPCRMD_PARQUET)
    prov = json.loads(PROVENANCE_JSON.read_text(encoding="utf-8"))

    # 1. Summarize 16-PDB Structural Dichotomy
    inact_pdb = pdb16_df[pdb16_df["state_class"] == "Inactive"]
    notrans_pdb = pdb16_df[pdb16_df["state_class"] == "Active_like_no_transducer"]
    gi_pdb = pdb16_df[pdb16_df["state_class"] == "Active_Gi"]
    barr_pdb = pdb16_df[pdb16_df["state_class"] == "Active_BetaArrestin1"]
    pdb_idx = pdb16_df.set_index("pdb_id")

    pdb16_summary = {
        "n_structures": int(len(pdb16_df)),
        "dichotomy_groups": {
            "Inactive_3": {
                "pdb_ids": inact_pdb["pdb_id"].tolist(),
                "R214_D338_range_A": [float(inact_pdb["R214_D338_CA_A"].min()), float(inact_pdb["R214_D338_CA_A"].max())],
                "F155sc_N393CA_range_A": [float(inact_pdb["F155sc_N393CA_A"].min()), float(inact_pdb["F155sc_N393CA_A"].max())],
                "Y294_Y397_OH_range_A": [float(inact_pdb["Y294_Y397_OH_A"].min()), float(inact_pdb["Y294_Y397_OH_A"].max())],
            },
            "Agonist_No_Transducer_3": {
                "pdb_ids": notrans_pdb["pdb_id"].tolist(),
                "R214_D338_range_A": [float(notrans_pdb["R214_D338_CA_A"].min()), float(notrans_pdb["R214_D338_CA_A"].max())],
                "F155sc_N393CA_range_A": [float(notrans_pdb["F155sc_N393CA_A"].min()), float(notrans_pdb["F155sc_N393CA_A"].max())],
                "F155_chi1_range_deg": [float(notrans_pdb["F155_chi1_deg"].min()), float(notrans_pdb["F155_chi1_deg"].max())],
                "Y294_Y397_OH_range_A": [float(notrans_pdb["Y294_Y397_OH_A"].min()), float(notrans_pdb["Y294_Y397_OH_A"].max())],
            },
            "Active_Gi_6": {
                "pdb_ids": gi_pdb["pdb_id"].tolist(),
                "R214_D338_range_A": [float(gi_pdb["R214_D338_CA_A"].min()), float(gi_pdb["R214_D338_CA_A"].max())],
                "F155sc_N393CA_range_A": [float(gi_pdb["F155sc_N393CA_A"].min()), float(gi_pdb["F155sc_N393CA_A"].max())],
                "F155_chi1_range_deg": [float(gi_pdb["F155_chi1_deg"].min()), float(gi_pdb["F155_chi1_deg"].max())],
                "Y294_Y397_OH_range_A": [float(gi_pdb["Y294_Y397_OH_A"].min()), float(gi_pdb["Y294_Y397_OH_A"].max())],
            },
            "Active_BetaArrestin1_2": {
                "pdb_ids": barr_pdb["pdb_id"].tolist(),
                "R214_D338_range_A": [float(barr_pdb["R214_D338_CA_A"].min()), float(barr_pdb["R214_D338_CA_A"].max())],
                "Y294_Y397_OH_range_A": [float(barr_pdb["Y294_Y397_OH_A"].min()), float(barr_pdb["Y294_Y397_OH_A"].max())],
            },
        },
        "matched_ligand_bifurcation": {
            "FUB_Gi1_6N4B_vs_BetaArr1_8WU1": {
                "R214_D338_CA_A": [float(pdb_idx.loc["6N4B", "R214_D338_CA_A"]), float(pdb_idx.loc["8WU1", "R214_D338_CA_A"])],
                "W356_chi1_deg": [float(pdb_idx.loc["6N4B", "W356_chi1_deg"]), float(pdb_idx.loc["8WU1", "W356_chi1_deg"])],
                "D163_W356_CA_A": [float(pdb_idx.loc["6N4B", "D163_W356_CA_A"]), float(pdb_idx.loc["8WU1", "D163_W356_CA_A"])],
                "Y294_Y397_OH_A": [float(pdb_idx.loc["6N4B", "Y294_Y397_OH_A"]), float(pdb_idx.loc["8WU1", "Y294_Y397_OH_A"])],
            },
            "AM841_Gi1_6KPG_vs_BetaArr1_8WRZ": {
                "R214_D338_CA_A": [float(pdb_idx.loc["6KPG", "R214_D338_CA_A"]), float(pdb_idx.loc["8WRZ", "R214_D338_CA_A"])],
                "R214_K300_CA_A": [float(pdb_idx.loc["6KPG", "R214_K300_CA_A"]), float(pdb_idx.loc["8WRZ", "R214_K300_CA_A"])],
            },
        },
        "NAM_ORG27569_decoupling_6KQI": {
            "S173_F268_CA_A": float(pdb_idx.loc["6KQI", "S173_F268_CA_A"]),
            "inactive_S173_F268_range_A": [float(inact_pdb["S173_F268_CA_A"].min()), float(inact_pdb["S173_F268_CA_A"].max())],
            "R214_D338_CA_A": float(pdb_idx.loc["6KQI", "R214_D338_CA_A"]),
            "F155_chi1_deg": float(pdb_idx.loc["6KQI", "F155_chi1_deg"]),
            "F237_chi1_deg": float(pdb_idx.loc["6KQI", "F237_chi1_deg"]),
        },
    }

    # 2. Pillar A: Gi1 vs Gs vs Inactive Coupling
    pillar_a = analyze_pillar_a_coupling(gpcrmd_df)
    plot_gi1_vs_gs_coupling(gpcrmd_df, pillar_a, FIG_DIR / "fig_gi1_vs_gs_coupling.png")

    # 3. Pillar B: Fast Local Microswitch Committors (<5% Right-Censoring) & H2 Memory Closure
    # Cohort 1: Fine 0.2-ns resolution across all 5 0.2-ns trajectories (1201, 22, 164, 162, 163)
    #           using tau_max = 15.0 ns, block_duration = 80.0 ns, buffer = 32.0 ns (> 2*tau_max!)
    c_fine_raw = prepare_uniform_dt_cohort(gpcrmd_df, [1201, 22, 164, 162, 163], target_dt_ns=0.2)
    c_fine_df, seg_fine, grp_fine = build_buffered_macro_segments(
        c_fine_raw, block_duration_ns=80.0, buffer_duration_ns=32.0
    )
    audit_fine, best_fine, sec_fine = audit_and_select_local_targets(c_fine_df, seg_fine, grp_fine, tau_max_ns=15.0)
    if best_fine is None:
        # Fallback wider forward horizon 25.0 ns with 52.0 ns buffer (> 2*tau_max)
        c_fine_df, seg_fine, grp_fine = build_buffered_macro_segments(
            c_fine_raw, block_duration_ns=95.0, buffer_duration_ns=52.0
        )
        audit_fine, best_fine, sec_fine = audit_and_select_local_targets(c_fine_df, seg_fine, grp_fine, tau_max_ns=25.0)

    # Cohort 2: Explicit-Na+ NaCl-5 Cohort harmonized at 1.0-ns physical timestep (2067, 22, 164, 162, 163)
    #           using tau_max = 25.0 ns, block_duration = 140.0 ns, buffer = 52.0 ns (> 2*tau_max!)
    c_nacl_raw = prepare_uniform_dt_cohort(gpcrmd_df, [2067, 22, 164, 162, 163], target_dt_ns=1.0)
    c_nacl_df, seg_nacl, grp_nacl = build_buffered_macro_segments(
        c_nacl_raw, block_duration_ns=140.0, buffer_duration_ns=52.0
    )
    audit_nacl, best_nacl, _ = audit_and_select_local_targets(
        c_nacl_df, seg_nacl, grp_nacl, tau_max_ns=25.0, min_int_frames=120
    )
    if best_nacl is None:
        c_nacl_df, seg_nacl, grp_nacl = build_buffered_macro_segments(
            c_nacl_raw, block_duration_ns=160.0, buffer_duration_ns=82.0
        )
        audit_nacl, best_nacl, _ = audit_and_select_local_targets(
            c_nacl_df, seg_nacl, grp_nacl, tau_max_ns=40.0, min_int_frames=100
        )

    # Cohort 3: Full Open-6 Cohort harmonized at 1.0-ns physical timestep (1201, 2067, 22, 164, 162, 163)
    c_open6_raw = prepare_uniform_dt_cohort(gpcrmd_df, [1201, 2067, 22, 164, 162, 163], target_dt_ns=1.0)
    c_open6_df, seg_open6, grp_open6 = build_buffered_macro_segments(
        c_open6_raw, block_duration_ns=140.0, buffer_duration_ns=52.0
    )
    audit_open6, best_open6, _ = audit_and_select_local_targets(
        c_open6_df, seg_open6, grp_open6, tau_max_ns=25.0, min_int_frames=150
    )
    if best_open6 is None:
        c_open6_df, seg_open6, grp_open6 = build_buffered_macro_segments(
            c_open6_raw, block_duration_ns=160.0, buffer_duration_ns=82.0
        )
        audit_open6, best_open6, _ = audit_and_select_local_targets(
            c_open6_df, seg_open6, grp_open6, tau_max_ns=40.0, min_int_frames=120
        )

    cohorts_eval: dict[str, Any] = {}
    if best_fine is not None:
        cohorts_eval["Fine_0p2ns_5traj"] = evaluate_static_and_memory_suite(
            df_cohort=c_fine_df,
            seg_id=seg_fine,
            groups=grp_fine,
            target_bundle=best_fine,
            use_explicit_na=False,
            lag_steps_list=[1, 2, 5, 10, 25, 50],  # 0.2, 0.4, 1.0, 2.0, 5.0, 10.0 ns
        )
        if sec_fine is not None:
            cohorts_eval["Fine_0p2ns_SecondarySwitch"] = evaluate_static_and_memory_suite(
                df_cohort=c_fine_df,
                seg_id=seg_fine,
                groups=grp_fine,
                target_bundle=sec_fine,
                use_explicit_na=False,
                lag_steps_list=[1, 5, 10, 25],
                n_shuffles=100,
                n_perms=100,
            )
        # Whole-trajectory GroupKFold sensitivity check (strict cross-trajectory generalization)
        traj_groups_fine = c_fine_df["gpcrmd_id"].to_numpy(dtype=np.int64)
        n_traj_active = int(np.unique(traj_groups_fine[best_fine["_raw"]["uncens_mask"]]).size)
        if n_traj_active >= 3:
            cohorts_eval["Fine_0p2ns_WholeTraj_CV"] = evaluate_static_and_memory_suite(
                df_cohort=c_fine_df,
                seg_id=seg_fine,
                groups=traj_groups_fine,
                target_bundle=best_fine,
                use_explicit_na=False,
                lag_steps_list=[1, 5, 10, 25],
                n_splits=n_traj_active,
                n_shuffles=100,
                n_perms=100,
            )

    if best_nacl is not None:
        cohorts_eval["NaCl5_1p0ns_ExplicitNa"] = evaluate_static_and_memory_suite(
            df_cohort=c_nacl_df,
            seg_id=seg_nacl,
            groups=grp_nacl,
            target_bundle=best_nacl,
            use_explicit_na=True,
            lag_steps_list=[1, 2, 5, 10],  # 1.0, 2.0, 5.0, 10.0 ns
        )

    if best_open6 is not None:
        cohorts_eval["Open6_1p0ns_AllTraj"] = evaluate_static_and_memory_suite(
            df_cohort=c_open6_df,
            seg_id=seg_open6,
            groups=grp_open6,
            target_bundle=best_open6,
            use_explicit_na=False,
            lag_steps_list=[1, 2, 5, 10],
        )

    if not cohorts_eval:
        raise RuntimeError("No candidate local microswitch passed the <5% censoring validity gate!")

    primary_key = (
        "Fine_0p2ns_5traj"
        if "Fine_0p2ns_5traj" in cohorts_eval
        else next(iter(cohorts_eval.keys()))
    )

    pillar_b = {
        "primary_cohort_key": primary_key,
        "local_transition_audits": {
            "Fine_0p2ns_5traj": audit_fine,
            "NaCl5_1p0ns_ExplicitNa": audit_nacl,
            "Open6_1p0ns_AllTraj": audit_open6,
        },
        "cohorts": cohorts_eval,
    }

    plot_water_na_memory_closure(pillar_b, FIG_DIR / "fig_water_na_memory_closure.png")

    full_metrics = {
        "workstream": "WS3_GPCRmd_Explicit_Solvent_and_PDB16_Benchmark",
        "dataset_provenance": {
            "total_gpcrmd_trajectories": 6,
            "total_frames": prov["total_frames"],
            "actual_xtc_span_us": round(
                sum(float(gpcrmd_df[gpcrmd_df["gpcrmd_id"] == int(k)]["time_ns"].max()) for k in prov["trajectories"]) / 1000.0, 4
            ),
            "actual_frame_dt_sum_us": round(
                sum(t["duration_ns"] for t in prov["trajectories"].values()) / 1000.0, 4
            ),
            "gpcrmd_nominal_catalog_us": 8.97,
            "parquet_size_mb": round(prov["parquet_size_bytes"] / 1e6, 3),
        },
        "pdb16_benchmark_summary": pdb16_summary,
        "pillar_a_gi1_vs_gs_coupling": pillar_a,
        "pillar_b_explicit_solvent_memory_closure": pillar_b,
    }

    METRICS_JSON.write_text(json.dumps(full_metrics, indent=2), encoding="utf-8")
    print(f"[WS3] Saved complete metrics to {METRICS_JSON}", flush=True)
    return full_metrics


if __name__ == "__main__":
    run_ws3_analysis()
