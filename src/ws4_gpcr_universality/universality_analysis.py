"""Cross-Receptor Statistical Mechanics Evaluator for Hypotheses U1 & U2 (WS4).

Implements leakage-free buffered macro-block splitting (`>= 2 * tau_max` temporal gaps),
conserved buried microswitch committor closure (`StoppedDGACommittor`, Peters-Trout, B0/B1),
Two-Step Mori-Zwanzig displacement memory tests (`evaluate_mori_zwanzig_memory`, N2a shuffles,
`compute_memory_collapse_ratio`), progress-stratified N1 environment permutation tests,
evolutionary specificity statistics on the 2.42-4.46 cleft (`CB1 Phe-Phe` vs `CB2 Phe-Leu` &
Class A controls), and publication-grade 4-panel superfamily figure generation.
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
    PetersTroutCommittorModel,
    PriorBaselineB0,
    SplineProgressBaselineB1,
    StoppedDGACommittor,
    assert_contiguous_segments,
    assert_disjoint_group_split,
    assert_no_cross_seam_pairs,
    bootstrap_bss_and_delta_ell_ci,
    build_forward_stopped_pairs,
    committor_log_likelihood,
    compute_memory_collapse_ratio,
    delta_log_likelihood,
    evaluate_mori_zwanzig_memory,
    fast_additive_ratio_group_bootstrap_ci,
    projected_bellman_residual,
)


RECEPTOR_DISPLAY_ORDER: tuple[str, ...] = ("CB1", "CB2", "S1P1", "B2AR", "MOR", "A2AR")

CANDIDATE_BURIED_TARGETS: tuple[dict[str, Any], ...] = (
    {
        "name": "D250_N749_polar_aperture",
        "col": "d_250_749_polar_A",
        "description": "Conserved D2.50-N7.49 Na+/water pocket polar aperture",
        "B_below": True,
        "threshold_grid": [(3.5, 5.0), (3.8, 5.4), (4.0, 5.8), (4.5, 6.4)],
    },
    {
        "name": "Y558_Y753_polar_latch",
        "col": "d_558_753_OH_A",
        "description": "Conserved Y5.58-Y7.53 intracellular Tyr polar latch",
        "B_below": True,
        "threshold_grid": [(3.5, 4.8), (3.8, 5.2), (4.2, 5.8), (4.8, 6.8)],
    },
    {
        "name": "BW336_W648_toggle_contact",
        "col": "d_336_648_sc_min_A",
        "description": "Conserved buried 3.36-W6.48 toggle sidechain contact",
        "B_below": False,
        "threshold_grid": [(3.5, 4.6), (3.7, 4.9), (4.0, 5.4), (4.4, 6.0)],
    },
)


def prepare_receptor_uniform_dt(df_receptor: pd.DataFrame) -> pd.DataFrame:
    """Return a physical-time-uniform cohort (`eff_dt_ns`) without mixing unequal lag steps."""
    if len(df_receptor) == 0:
        raise ValueError("Empty receptor DataFrame passed to prepare_receptor_uniform_dt.")
    df = df_receptor.sort_values(["traj_id", "frame_idx"]).reset_index(drop=True).copy()
    dts = np.round(df.groupby("traj_id")["dt_ns"].first().to_numpy(dtype=np.float64), 6)
    min_dt = float(np.min(dts))
    sub_min = df[np.isclose(df["dt_ns"], min_dt, atol=1e-5)].copy()
    # Prefer native finest dt_ns whenever >=2 trajectories (or all trajectories) share it
    if len(np.unique(dts)) == 1 or (sub_min["traj_id"].nunique() >= 2 and len(sub_min) >= 800):
        sub_min["eff_dt_ns"] = min_dt
        return sub_min.reset_index(drop=True)

    # Otherwise harmonize to max_dt via exact integer striding
    target_dt = float(np.max(dts))
    parts: list[pd.DataFrame] = []
    for _, tdf in df.groupby("traj_id", sort=False):
        dt_i = float(tdf["dt_ns"].iloc[0])
        stride = max(1, int(round(target_dt / dt_i)))
        st_df = tdf.iloc[::stride].copy()
        st_df["eff_dt_ns"] = round(dt_i * stride, 6)
        parts.append(st_df)
    out = pd.concat(parts, ignore_index=True)
    return out


def build_buffered_macro_segments(
    df_cohort: pd.DataFrame,
    macro_block_ns: float = 25.0,
    buffer_ns: float = 4.0,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Partition trajectories into contiguous macro-blocks separated by `>= buffer_ns` deleted gaps."""
    dt = float(df_cohort["eff_dt_ns"].iloc[0])
    traj_spans = [
        float(sub["time_ns"].iloc[-1] - sub["time_ns"].iloc[0])
        for _, sub in df_cohort.groupby("traj_id", sort=False)
    ]
    med_span = float(np.median(traj_spans)) if traj_spans else float(macro_block_ns)
    eff_buf_ns = float(buffer_ns)
    min_block_ns = max(8.0 * dt, eff_buf_ns + 4.0 * dt)
    target_block_ns = max(float(macro_block_ns), min_block_ns)
    eff_block_ns = min(target_block_ns, max(min_block_ns, med_span / 3.2))

    kept_parts: list[pd.DataFrame] = []
    seg_ids: list[np.ndarray] = []
    group_ids: list[np.ndarray] = []
    next_group = 0

    for _, sub in df_cohort.groupby("traj_id", sort=False):
        t_arr = sub["time_ns"].to_numpy(dtype=np.float64)
        t_rel = t_arr - float(t_arr[0])
        period = eff_block_ns + eff_buf_ns
        block_k = np.floor((t_rel + 1e-12) / period).astype(np.int64)
        offset_in_period = t_rel - block_k * period
        keep_mask = offset_in_period < (eff_block_ns - 1e-9)
        sub_kept = sub.loc[keep_mask].copy()
        bk_kept = block_k[keep_mask]
        prev_end_t: float | None = None
        for uk in np.unique(bk_kept):
            m_uk = bk_kept == uk
            if int(np.sum(m_uk)) < 8:
                continue
            chunk_df = sub_kept.loc[m_uk].copy()
            t_start = float(chunk_df["time_ns"].iloc[0])
            t_end = float(chunk_df["time_ns"].iloc[-1])
            if prev_end_t is not None and (t_start - prev_end_t) < (eff_buf_ns - 1e-9):
                raise AssertionError(
                    f"Temporal seam gap violation in {chunk_df['traj_id'].iloc[0]}: "
                    f"gap={t_start - prev_end_t:.6f} ns < buffer_ns={eff_buf_ns:.6f} ns"
                )
            prev_end_t = t_end
            kept_parts.append(chunk_df)
            n_c = len(chunk_df)
            seg_ids.append(np.full(n_c, next_group, dtype=np.int64))
            group_ids.append(np.full(n_c, next_group, dtype=np.int64))
            next_group += 1

    if not kept_parts:
        raise ValueError("Zero buffered macro-segments retained; check trajectory durations vs buffer_ns.")

    df_out = pd.concat(kept_parts, ignore_index=True)
    seg_arr = np.concatenate(seg_ids)
    grp_arr = np.concatenate(group_ids)
    assert_contiguous_segments(seg_arr)
    return df_out, seg_arr, grp_arr


def _compute_first_hitting_labels(
    values: np.ndarray,
    seg_id: np.ndarray,
    r_low: float,
    r_high: float,
    B_below: bool,
    tau_max_steps: int,
) -> dict[str, Any]:
    """Compute forward-stopped binary first-hitting outcomes within horizon `tau_max_steps` per segment."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if B_below:
        in_B = x <= r_low
        in_A = x >= r_high
    else:
        in_A = x <= r_low
        in_B = x >= r_high

    n = x.size
    y_hit = np.zeros(n, dtype=np.float64)
    is_horizon_safe_int = np.zeros(n, dtype=bool)
    is_censored = np.zeros(n, dtype=bool)

    for s in np.unique(seg_id):
        idx = np.flatnonzero(seg_id == s)
        m = idx.size
        if m <= tau_max_steps:
            continue
        a_loc = in_A[idx]
        b_loc = in_B[idx]
        stop_loc = a_loc | b_loc
        for i_loc in range(m - tau_max_steps):
            if stop_loc[i_loc]:
                continue
            g_i = idx[i_loc]
            is_horizon_safe_int[g_i] = True
            win_stop = np.flatnonzero(stop_loc[i_loc + 1 : i_loc + 1 + tau_max_steps])
            if win_stop.size == 0:
                is_censored[g_i] = True
            else:
                first_off = int(win_stop[0]) + 1
                y_hit[g_i] = 1.0 if b_loc[i_loc + first_off] else 0.0

    uncens_mask = is_horizon_safe_int & (~is_censored)
    n_safe = int(np.sum(is_horizon_safe_int))
    n_cens = int(np.sum(is_censored))
    n_unc = int(np.sum(uncens_mask))
    censor_frac = float(n_cens / max(n_safe, 1))
    p_hit = float(np.mean(y_hit[uncens_mask])) if n_unc > 0 else 0.0
    eff_info = float(n_unc * p_hit * (1.0 - p_hit))
    return {
        "in_A": in_A,
        "in_B": in_B,
        "y_hit": y_hit,
        "is_horizon_safe_int": is_horizon_safe_int,
        "is_censored": is_censored,
        "uncens_mask": uncens_mask,
        "n_safe_interior": n_safe,
        "n_uncensored": n_unc,
        "censor_fraction": censor_frac,
        "p_hit": p_hit,
        "effective_binomial_info": eff_info,
    }


def select_buried_allosteric_target(
    df_buf: pd.DataFrame,
    seg_id: np.ndarray,
    groups: np.ndarray,
    tau_max_steps: int,
    max_censor_fraction: float = 0.05,
) -> dict[str, Any]:
    """Select purely protein-defined conserved buried microswitch target with `<5%` right-censoring."""
    candidates_valid: list[dict[str, Any]] = []
    fallback_pool: list[dict[str, Any]] = []

    for priority_idx, spec in enumerate(CANDIDATE_BURIED_TARGETS):
        vals = df_buf[spec["col"]].to_numpy(dtype=np.float64)
        q20, q30, q70, q80 = np.quantile(vals, [0.20, 0.30, 0.70, 0.80])
        adaptive_pairs = []
        if q70 - q30 >= 0.80:
            adaptive_pairs.append((round(float(q30), 3), round(float(q70), 3)))
        if q80 - q20 >= 0.90:
            adaptive_pairs.append((round(float(q20), 3), round(float(q80), 3)))

        for r_low, r_high in list(spec["threshold_grid"]) + adaptive_pairs:
            res = _compute_first_hitting_labels(vals, seg_id, r_low, r_high, spec["B_below"], tau_max_steps)
            n_active_groups = int(np.unique(groups[res["uncens_mask"]]).size) if res["n_uncensored"] > 0 else 0
            rec = {
                "priority_idx": priority_idx,
                "target_name": spec["name"],
                "col": spec["col"],
                "description": spec["description"],
                "B_below": bool(spec["B_below"]),
                "r_low": float(r_low),
                "r_high": float(r_high),
                "n_uncensored": res["n_uncensored"],
                "n_active_groups": n_active_groups,
                "censor_fraction": res["censor_fraction"],
                "p_hit": res["p_hit"],
                "effective_binomial_info": res["effective_binomial_info"],
                "_raw": res,
            }
            if (
                res["n_uncensored"] >= 40
                and n_active_groups >= 3
                and 0.05 <= res["p_hit"] <= 0.95
            ):
                fallback_pool.append(rec)
                if res["censor_fraction"] <= max_censor_fraction:
                    candidates_valid.append(rec)

    if candidates_valid:
        candidates_valid.sort(key=lambda d: (d["priority_idx"], -d["effective_binomial_info"]))
        return candidates_valid[0]
    if fallback_pool:
        fallback_pool.sort(key=lambda d: (d["censor_fraction"], d["priority_idx"], -d["effective_binomial_info"]))
        return fallback_pool[0]
    raise ValueError("No valid buried microswitch transition target found; check trajectory sampling.")


def _build_ws4_feature_matrices(
    df_buf: pd.DataFrame,
    primary_col: str,
) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Build nested protein-only and protein+water/ion/lipid feature matrices (`col 0 == primary_col`)."""
    df_tmp = df_buf.copy()
    w648_rad = np.radians(df_tmp["bw648_chi1_deg"].to_numpy(dtype=np.float64))
    bw242_rad = np.radians(df_tmp["bw242_chi1_deg"].to_numpy(dtype=np.float64))
    df_tmp["sin_bw648_chi1"] = np.sin(w648_rad)
    df_tmp["cos_bw648_chi1"] = np.cos(w648_rad)
    df_tmp["sin_bw242_chi1"] = np.sin(bw242_rad)

    base_prot = [
        "d_350_630_CA_A",
        "d_250_749_CA_A",
        "d_336_648_sc_com_A",
        "d_242sc_749CA_A",
        "d_558_753_OH_A",
        "d_250_749_polar_A",
        "d_336_648_sc_min_A",
        "sin_bw648_chi1",
        "cos_bw648_chi1",
        "sin_bw242_chi1",
    ]
    prot_cols = [primary_col] + [c for c in base_prot if c != primary_col]

    water_cols = [
        "n_wat_core_pore",
        "wire_D250_W648_cond",
        "wire_D250_NPxxY_cond",
        "wire_558_753_cond",
    ]
    all_have_na = bool(np.all(df_tmp["has_na_ions"].to_numpy() == 1))
    if all_have_na:
        ion_cols = ["prox_Na_D250", "coord_Na_D250_3p2"]
    else:
        ion_cols = ["prox_cation_D250", "coord_cation_D250_3p2"]
    lipid_cols = ["min_lipid_TM2_TM4_cleft_A", "min_lipid_TM1_TM7_portal_A"]

    def _mat(cols: list[str]) -> tuple[np.ndarray, list[str]]:
        arr = df_tmp[cols].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            raise AssertionError(f"Non-finite values in WS4 feature matrix columns: {cols}")
        return arr, cols

    return {
        "prot": _mat(prot_cols),
        "prot_water_ion": _mat(prot_cols + water_cols + ion_cols),
        "prot_env_all": _mat(prot_cols + water_cols + ion_cols + lipid_cols),
    }


def evaluate_u1_water_na_memory_closure(
    df_receptor: pd.DataFrame,
    receptor_key: str | None = None,
    tau_max_ns: float = 3.0,
    macro_block_ns: float = 25.0,
    lag_steps_list: list[int] | None = None,
    n_splits: int = 5,
    n_shuffles: int = 100,
    n_perms: int = 100,
    n_boot: int = 1000,
) -> dict[str, Any]:
    """Evaluate Hypothesis U1 (conserved D2.50-W6.48-NPxxY water/Na+ committor & memory closure)."""
    rec_name = receptor_key or str(df_receptor["receptor"].iloc[0])
    df_u = prepare_receptor_uniform_dt(df_receptor)
    dt = float(df_u["eff_dt_ns"].iloc[0])
    tau_max_steps = max(3, int(round(tau_max_ns / dt)))
    buffer_ns = max(2.0 * tau_max_steps * dt, 2.0 * dt)

    df_buf, seg_id, groups = build_buffered_macro_segments(
        df_u, macro_block_ns=macro_block_ns, buffer_ns=buffer_ns
    )
    target = select_buried_allosteric_target(df_buf, seg_id, groups, tau_max_steps=tau_max_steps)
    raw = target["_raw"]
    in_A = raw["in_A"]
    in_B = raw["in_B"]
    unc_mask = raw["uncens_mask"]

    y_full = np.zeros(in_A.shape, dtype=np.float64)
    y_full[in_A] = 0.0
    y_full[in_B] = 1.0
    y_full[unc_mask] = raw["y_hit"][unc_mask]
    if not np.all(np.isfinite(y_full)):
        raise AssertionError("Non-finite values in y_full committor labels.")

    f_mats = _build_ws4_feature_matrices(df_buf, target["col"])
    X_prot, prot_cols = f_mats["prot"]
    X_wi, _ = f_mats["prot_water_ion"]
    X_env, env_cols = f_mats["prot_env_all"]

    unc_idx = np.flatnonzero(unc_mask)
    X_prot_u = X_prot[unc_idx]
    X_wi_u = X_wi[unc_idx]
    X_env_u = X_env[unc_idx]
    y_u = y_full[unc_idx]
    g_u = groups[unc_idx]

    eff_splits = min(n_splits, int(np.unique(g_u).size))
    if eff_splits < 2:
        raise ValueError(f"{rec_name}: insufficient active macro-blocks ({eff_splits}) for CV.")
    n_pos_g = int(np.unique(g_u[y_u > 0.5]).size)
    n_neg_g = int(np.unique(g_u[y_u <= 0.5]).size)
    use_strat = (n_pos_g >= eff_splits) and (n_neg_g >= eff_splits)
    y_strata = y_u.astype(np.int64) if use_strat else None
    splitter = LineageGroupSplitter(n_splits=eff_splits, stratified=use_strat, random_state=42)
    eff_l2 = float(np.clip(2.5 / max(unc_idx.size, 20), 1.5e-2, 1.5e-1))

    oof: dict[str, np.ndarray] = {
        "B0": np.zeros_like(y_u),
        "B1": np.zeros_like(y_u),
        "prot": np.zeros_like(y_u),
        "prot_water_ion": np.zeros_like(y_u),
        "prot_env_all": np.zeros_like(y_u),
    }
    for tr, te in splitter.split(X_prot_u, y_strata, g_u):
        assert_disjoint_group_split(g_u, tr, te)
        oof["B0"][te] = PriorBaselineB0(progress_col=0).fit(X_prot_u[tr], y_u[tr]).predict(X_prot_u[te])
        oof["B1"][te] = SplineProgressBaselineB1(progress_col=0, n_knots=5, l2_reg=eff_l2).fit(
            X_prot_u[tr], y_u[tr]
        ).predict(X_prot_u[te])
        oof["prot"][te] = PetersTroutCommittorModel(include_interactions=False, l2_reg=eff_l2).fit(
            X_prot_u[tr], y_u[tr]
        ).predict(X_prot_u[te])
        oof["prot_water_ion"][te] = PetersTroutCommittorModel(include_interactions=False, l2_reg=eff_l2).fit(
            X_wi_u[tr], y_u[tr]
        ).predict(X_wi_u[te])
        oof["prot_env_all"][te] = PetersTroutCommittorModel(include_interactions=False, l2_reg=eff_l2).fit(
            X_env_u[tr], y_u[tr]
        ).predict(X_env_u[te])

    stat_env_vs_prot = bootstrap_bss_and_delta_ell_ci(
        g_u, y_u, oof["prot_env_all"], oof["prot"], n_boot=n_boot
    )
    stat_wi_vs_prot = bootstrap_bss_and_delta_ell_ci(
        g_u, y_u, oof["prot_water_ion"], oof["prot"], n_boot=n_boot
    )

    # Progress-stratified N1 macro-block permutation null for z_env
    rng_n1 = np.random.default_rng(42)
    env_only_u = X_env_u[:, len(prot_cols) :]
    unique_g = np.unique(g_u)
    g_means = np.array([np.mean(X_prot_u[g_u == ug, 0]) for ug in unique_g], dtype=np.float64)
    k_st = max(1, min(3, unique_g.size // 3))
    if k_st == 1:
        strata = np.zeros(unique_g.size, dtype=np.int64)
    else:
        edges = np.unique(np.quantile(g_means, np.linspace(0.0, 1.0, k_st + 1)))
        strata = np.digitize(g_means, edges[1:-1])

    n1_null = np.empty(n_perms, dtype=np.float64)
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
        oof_p = np.zeros_like(y_u)
        for tr, te in splitter.split(X_perm_u, y_strata, g_u):
            oof_p[te] = PetersTroutCommittorModel(include_interactions=False, l2_reg=eff_l2).fit(
                X_perm_u[tr], y_u[tr]
            ).predict(X_perm_u[te])
        n1_null[p_i] = delta_log_likelihood(y_u, oof_p, oof["prot"])

    obs_dell_env = float(stat_env_vs_prot["delta_ell"]["point"])
    n1_p_val = float((1 + np.sum(n1_null >= obs_dell_env)) / (n_perms + 1))

    # Two-Step Mori-Zwanzig Memory Collapse with strict censored-frame masking
    in_A_mem_fit = in_A | ((~in_B) & (~unc_mask))
    if not np.array_equal(~(in_A_mem_fit | in_B), unc_mask):
        raise AssertionError("Uncensored interior mask mismatch in Mori-Zwanzig setup.")
    mem_folds: list[tuple[np.ndarray, np.ndarray]] = []
    for _, te_u in splitter.split(X_prot_u, y_strata, g_u):
        te_groups = np.unique(g_u[te_u])
        te_full_mask = np.isin(groups, te_groups)
        mem_folds.append((np.flatnonzero(~te_full_mask), np.flatnonzero(te_full_mask)))

    if lag_steps_list is None:
        lag_steps_list = sorted({1, max(2, int(round(1.0 / dt)))})

    memory_profile: list[dict[str, Any]] = []
    for lag_s in lag_steps_list:
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
            l2_reg=eff_l2,
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
            l2_reg=eff_l2,
            n_splits=eff_splits,
            n_shuffles=n_shuffles,
            random_state=42,
        )
        rho_collapse = compute_memory_collapse_ratio(mem_prot["delta_ell_mem"], mem_env["delta_ell_mem"])
        memory_profile.append({
            "lag_steps": int(lag_s),
            "lag_ns": round(lag_s * dt, 3),
            "n_eval_frames": int(mem_prot["n_eval_frames"]),
            "prot_only": {
                "delta_ell_mem": round(float(mem_prot["delta_ell_mem"]), 6),
                "ci95_low": round(float(mem_prot["delta_ell_ci"]["ci_low"]), 6),
                "ci95_high": round(float(mem_prot["delta_ell_ci"]["ci_high"]), 6),
                "p_value_vs_n2a": round(float(mem_prot["p_value_vs_n2a"]), 4),
            },
            "prot_plus_env": {
                "delta_ell_mem": round(float(mem_env["delta_ell_mem"]), 6),
                "ci95_low": round(float(mem_env["delta_ell_ci"]["ci_low"]), 6),
                "ci95_high": round(float(mem_env["delta_ell_ci"]["ci_high"]), 6),
                "p_value_vs_n2a": round(float(mem_env["p_value_vs_n2a"]), 4),
            },
            "rho_mem_collapse": round(float(rho_collapse), 4),
        })

    # Shared-witness out-of-block Koopman StoppedDGA Projected Bellman Residual
    dga_lag = max(1, min(tau_max_steps - 1, int(round(1.0 / dt))))
    pbr_prot_folds: list[float] = []
    pbr_env_folds: list[float] = []
    for f_i, (tr_all, te_all) in enumerate(mem_folds):
        assert_disjoint_group_split(groups, tr_all, te_all)
        seg_tr = seg_id[tr_all]
        seg_te = seg_id[te_all]
        int_tr = ~(in_A[tr_all] | in_B[tr_all])
        if np.any(in_A[tr_all]) and np.any(in_B[tr_all]) and int(np.sum(int_tr)) >= 20:
            idx_s, idx_e = build_forward_stopped_pairs(seg_te, in_A[te_all], in_B[te_all], dga_lag)
            assert_no_cross_seam_pairs(seg_te, idx_s, idx_e)
            if idx_s.size < 12:
                continue
            try:
                wit_sc = StandardScaler().fit(X_env[tr_all][int_tr])
                n_pca = min(6, X_env.shape[1], int(np.sum(int_tr)) - 2)
                wit_pca = PCA(n_components=n_pca, random_state=42).fit(
                    wit_sc.transform(X_env[tr_all][int_tr])
                )
                Z_wit_te = wit_pca.transform(wit_sc.transform(X_env[te_all]))

                n_rbf_use = min(35, max(8, int(np.sum(int_tr)) // 4))
                dga_p = StoppedDGACommittor(lag_steps=dga_lag, n_rbf=n_rbf_use, include_poly2=False, reg=1e-3).fit(
                    X_prot[tr_all], in_A[tr_all], in_B[tr_all], seg_id=seg_tr
                )
                dga_e = StoppedDGACommittor(lag_steps=dga_lag, n_rbf=n_rbf_use, include_poly2=False, reg=1e-3).fit(
                    X_env[tr_all], in_A[tr_all], in_B[tr_all], seg_id=seg_tr
                )
                q_p = dga_p.predict(X_prot[te_all], in_A[te_all], in_B[te_all])
                q_e = dga_e.predict(X_env[te_all], in_A[te_all], in_B[te_all])
                pbr_prot_folds.append(
                    projected_bellman_residual(
                        q_p[idx_s], q_p[idx_e], Z_wit_te[idx_s], n_witness=16, random_state=100 + f_i
                    )
                )
                pbr_env_folds.append(
                    projected_bellman_residual(
                        q_e[idx_s], q_e[idx_e], Z_wit_te[idx_s], n_witness=16, random_state=100 + f_i
                    )
                )
            except ValueError:
                pass

    primary_mem = max(memory_profile, key=lambda m: m["prot_only"]["delta_ell_mem"])
    return {
        "receptor": rec_name,
        "n_trajectories": int(df_u["traj_id"].nunique()),
        "n_frames_uniform_dt": int(len(df_u)),
        "eff_dt_ns": dt,
        "n_macro_blocks": int(np.unique(groups).size),
        "selected_target": {
            k: v for k, v in target.items() if k != "_raw"
        },
        "static_committor": {
            "log_likelihood_nats": {k: round(float(committor_log_likelihood(y_u, v)), 5) for k, v in oof.items()},
            "water_ion_vs_prot": stat_wi_vs_prot,
            "env_all_vs_prot": stat_env_vs_prot,
            "n1_permutation_p_value": round(n1_p_val, 4),
            "n1_null_q95": round(float(np.quantile(n1_null, 0.95)), 6),
        },
        "shared_witness_stopped_dga_pbr": {
            "dga_lag_steps": int(dga_lag),
            "n_valid_folds": len(pbr_prot_folds),
            "mean_pbr_prot": round(float(np.mean(pbr_prot_folds)), 5) if pbr_prot_folds else None,
            "mean_pbr_prot_env": round(float(np.mean(pbr_env_folds)), 5) if pbr_env_folds else None,
        },
        "mori_zwanzig_memory_profile": memory_profile,
        "primary_memory_summary": primary_mem,
    }


def _rotamer_well_indices(chi1_deg: np.ndarray) -> np.ndarray:
    """Map dihedral angles `[-180, 180]` to 3 staggered wells: `0=g- [-120,0)`, `1=g+ [0,120)`, `2=trans`."""
    ang = np.asarray(chi1_deg, dtype=np.float64)
    wells = np.full(ang.shape, 2, dtype=np.int64)
    wells[(ang >= -120.0) & (ang < 0.0)] = 0
    wells[(ang >= 0.0) & (ang < 120.0)] = 1
    return wells


def _shannon_entropy_bits(discrete_states: np.ndarray, n_states: int) -> float:
    """Compute Shannon entropy `-sum_k p_k log2(p_k)` in bits."""
    if discrete_states.size == 0:
        return 0.0
    counts = np.bincount(discrete_states, minlength=n_states).astype(np.float64)
    probs = counts / np.sum(counts)
    nz = probs[probs > 0.0]
    return float(-np.sum(nz * np.log2(nz)))


def _assign_time_blocks(df: pd.DataFrame, block_ns: float = 50.0) -> np.ndarray:
    """Assign non-overlapping macro-time block IDs per trajectory for cluster bootstrap CIs."""
    out: list[np.ndarray] = []
    offset = 0
    for _, sub in df.groupby("traj_id", sort=False):
        span = max(float(sub["time_ns"].iloc[-1] - sub["time_ns"].iloc[0]), 1e-6)
        eff_b = min(block_ns, max(span / 4.0, float(sub["dt_ns"].iloc[0]) * 5.0))
        b_loc = np.floor((sub["time_ns"].to_numpy(dtype=np.float64) - float(sub["time_ns"].iloc[0])) / eff_b).astype(np.int64)
        _, inv = np.unique(b_loc, return_inverse=True)
        out.append(inv + offset)
        offset += int(inv.max()) + 1
    return np.concatenate(out)


def evaluate_u2_tm2_tm4_evolutionary_specificity(
    df_by_receptor: dict[str, pd.DataFrame],
    block_ns: float = 50.0,
    n_boot: int = 1000,
) -> dict[str, Any]:
    """Evaluate Hypothesis U2: `CB1` (`F155^2.42-F237^4.46`) aromatic rotamer gate vs `CB2` & Class A controls."""
    per_receptor: dict[str, Any] = {}
    ordered_keys = [k for k in RECEPTOR_DISPLAY_ORDER if k in df_by_receptor] + [
        k for k in df_by_receptor if k not in RECEPTOR_DISPLAY_ORDER
    ]

    for rec_key in ordered_keys:
        df = df_by_receptor[rec_key].sort_values(["traj_id", "frame_idx"]).reset_index(drop=True)
        blocks = _assign_time_blocks(df, block_ns=block_ns)
        ones = np.ones(len(df), dtype=np.float64)

        bw242 = str(df["bw242_resname"].iloc[0])
        bw446 = str(df["bw446_resname"].iloc[0])
        is_arom_pair = int(df["is_aromatic_pair_242_446"].iloc[0])

        d_min = df["d_242_446_sc_min_A"].to_numpy(dtype=np.float64)
        contact_4p2 = (d_min <= 4.2).astype(np.float64)
        arom_lock_4p2 = contact_4p2 * float(is_arom_pair)
        lip_wedge_4p0 = (df["min_lipid_TM2_TM4_cleft_A"].to_numpy(dtype=np.float64) <= 4.0).astype(np.float64)

        w242 = _rotamer_well_indices(df["bw242_chi1_deg"].to_numpy(dtype=np.float64))
        w446 = _rotamer_well_indices(df["bw446_chi1_deg"].to_numpy(dtype=np.float64))
        joint_9 = w242 * 3 + w446

        h242 = _shannon_entropy_bits(w242, 3)
        h446 = _shannon_entropy_bits(w446, 3) if int(df["has_bw446_chi1"].iloc[0]) == 1 else 0.0
        h_joint = _shannon_entropy_bits(joint_9, 9)
        minor_242 = float(1.0 - np.max(np.bincount(w242, minlength=3)) / len(w242))
        minor_446 = (
            float(1.0 - np.max(np.bincount(w446, minlength=3)) / len(w446))
            if int(df["has_bw446_chi1"].iloc[0]) == 1
            else 0.0
        )

        # Spearman coupling between cleft opening d_242_446_sc_min_A and lipid proximity (-min_lipid_cleft)
        lip_d = df["min_lipid_TM2_TM4_cleft_A"].to_numpy(dtype=np.float64)
        if np.std(d_min) > 1e-8 and np.std(lip_d) > 1e-8:
            rho_lip, _ = stats.spearmanr(d_min, -lip_d)
            rho_lip_val = float(rho_lip) if np.isfinite(rho_lip) else 0.0
        else:
            rho_lip_val = 0.0

        # Conserved allosteric water/Na+ occupancy summary for Panel D phase portrait
        wire_occ = (df["wire_D250_NPxxY_cond"].to_numpy(dtype=np.float64) > 0.0).astype(np.float64)
        cation_occ = (df["coord_cation_D250_3p2"].to_numpy(dtype=np.float64) > 0).astype(np.float64)

        per_receptor[rec_key] = {
            "bw242_resname": bw242,
            "bw446_resname": bw446,
            "cleft_pair_label": f"{bw242}2.42-{bw446}4.46",
            "is_aromatic_pair_242_446": bool(is_arom_pair),
            "n_frames": int(len(df)),
            "n_blocks": int(np.unique(blocks).size),
            "d_242_446_sc_min_A": fast_additive_ratio_group_bootstrap_ci(blocks, d_min, ones, n_boot=n_boot),
            "cleft_contact_occupancy_4p2A": fast_additive_ratio_group_bootstrap_ci(
                blocks, contact_4p2, ones, n_boot=n_boot
            ),
            "aromatic_lock_occupancy_4p2A": fast_additive_ratio_group_bootstrap_ci(
                blocks, arom_lock_4p2, ones, n_boot=n_boot
            ),
            "lipid_wedging_occupancy_4p0A": fast_additive_ratio_group_bootstrap_ci(
                blocks, lip_wedge_4p0, ones, n_boot=n_boot
            ),
            "bw242_chi1_entropy_bits": round(h242, 4),
            "bw446_chi1_entropy_bits": round(h446, 4),
            "joint_242_446_rotamer_entropy_bits": round(h_joint, 4),
            "bw242_minor_rotamer_pop": round(minor_242, 4),
            "bw446_minor_rotamer_pop": round(minor_446, 4),
            "spearman_cleft_opening_vs_lipid_entry": round(rho_lip_val, 4),
            "wire_D250_NPxxY_percolation": fast_additive_ratio_group_bootstrap_ci(
                blocks, wire_occ, ones, n_boot=n_boot
            ),
            "cation_D250_coordination": fast_additive_ratio_group_bootstrap_ci(
                blocks, cation_occ, ones, n_boot=n_boot
            ),
        }

    # Direct evolutionary contrast: CB1 (Phe2.42-Phe4.46) vs CB2 (Phe2.42-Leu4.46) and non-aromatic-4.46 pool
    evolutionary_contrasts: dict[str, Any] = {}
    if "CB1" in per_receptor and "CB2" in per_receptor:
        cb1_r = per_receptor["CB1"]
        cb2_r = per_receptor["CB2"]
        evolutionary_contrasts["CB1_vs_CB2"] = {
            "CB1_pair": cb1_r["cleft_pair_label"],
            "CB2_pair": cb2_r["cleft_pair_label"],
            "delta_aromatic_lock_occupancy": round(
                cb1_r["aromatic_lock_occupancy_4p2A"]["point"] - cb2_r["aromatic_lock_occupancy_4p2A"]["point"], 4
            ),
            "delta_joint_rotamer_entropy_bits": round(
                cb1_r["joint_242_446_rotamer_entropy_bits"] - cb2_r["joint_242_446_rotamer_entropy_bits"], 4
            ),
            "delta_cleft_contact_occupancy_4p2A": round(
                cb1_r["cleft_contact_occupancy_4p2A"]["point"] - cb2_r["cleft_contact_occupancy_4p2A"]["point"], 4
            ),
        }

    non_arom_keys = [k for k, v in per_receptor.items() if not v["is_aromatic_pair_242_446"]]
    if "CB1" in per_receptor and non_arom_keys:
        evolutionary_contrasts["CB1_vs_non_aromatic_446_controls"] = {
            "control_receptors": non_arom_keys,
            "cb1_is_unique_aromatic_446": len([k for k, v in per_receptor.items() if v["is_aromatic_pair_242_446"]]) == 1,
            "cb1_aromatic_lock_occ": round(per_receptor["CB1"]["aromatic_lock_occupancy_4p2A"]["point"], 4),
            "controls_max_aromatic_lock_occ": round(
                max(per_receptor[k]["aromatic_lock_occupancy_4p2A"]["point"] for k in non_arom_keys), 4
            ),
            "cb1_joint_rotamer_entropy_bits": per_receptor["CB1"]["joint_242_446_rotamer_entropy_bits"],
            "controls_mean_joint_rotamer_entropy_bits": round(
                float(np.mean([per_receptor[k]["joint_242_446_rotamer_entropy_bits"] for k in non_arom_keys])), 4
            ),
        }

    return {
        "per_receptor": per_receptor,
        "evolutionary_contrasts": evolutionary_contrasts,
    }


def plot_superfamily_universality_figure(
    suite_summary: dict[str, Any],
    df_by_receptor: dict[str, pd.DataFrame],
    out_fig_path: str | Path,
) -> Path:
    """Render publication-grade 4-panel (`2x2`) Class A GPCR Universality vs Specificity figure."""
    out_path = Path(out_fig_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    u1_map = suite_summary["U1_water_na_closure"]["per_receptor"]
    u2_map = suite_summary["U2_tm2_tm4_specificity"]["per_receptor"]
    recs = [k for k in RECEPTOR_DISPLAY_ORDER if k in u1_map] + [
        k for k in u1_map if k not in RECEPTOR_DISPLAY_ORDER
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5), dpi=300)
    plt.subplots_adjust(hspace=0.34, wspace=0.26)

    x = np.arange(len(recs))
    w = 0.36

    # Panel A: U1 Mori-Zwanzig Displacement Memory Collapse across all GPCRs
    ax_a = axes[0, 0]
    mem_p = [u1_map[r]["primary_memory_summary"]["prot_only"]["delta_ell_mem"] for r in recs]
    mem_e = [u1_map[r]["primary_memory_summary"]["prot_plus_env"]["delta_ell_mem"] for r in recs]
    rhos = [u1_map[r]["primary_memory_summary"]["rho_mem_collapse"] for r in recs]

    ax_a.bar(x - w / 2, mem_p, width=w, color="#C0392B", alpha=0.88, label="Protein-only ($\\mathbf{z}_{\\mathrm{prot}}$)")
    ax_a.bar(x + w / 2, mem_e, width=w, color="#1F618D", alpha=0.90, label="+ Explicit Water / $\\mathrm{Na}^+$ / Lipid")
    ax_a.axhline(0.0, color="#333333", lw=0.9, ls="--")
    for idx, r_val in enumerate(rhos):
        y_top = max(mem_p[idx], mem_e[idx], 0.0)
        ax_a.text(
            x[idx],
            y_top + 0.002,
            f"$\\rho={100.0 * r_val:.0f}\\%$",
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
            color="#1B2631",
        )
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(recs, fontweight="bold")
    ax_a.set_ylabel("Mori–Zwanzig Memory Gain $\\Delta\\bar{\\ell}_{\\mathrm{mem}}$ (nats)")
    ax_a.set_title("A. U1: Universal Buried Water/$\\mathrm{Na}^+$ Memory Collapse", fontweight="bold", loc="left")
    ax_a.legend(frameon=False, fontsize=8.5)
    ax_a.grid(axis="y", alpha=0.25)

    # Panel B: U1 Out-of-Block Static Committor Gain (+z_env vs z_prot) with 95% CIs
    ax_b = axes[0, 1]
    dell_pts = [u1_map[r]["static_committor"]["env_all_vs_prot"]["delta_ell"]["point"] for r in recs]
    dell_lo = [
        max(0.0, pt - u1_map[r]["static_committor"]["env_all_vs_prot"]["delta_ell"]["ci_low"])
        for r, pt in zip(recs, dell_pts)
    ]
    dell_hi = [
        max(0.0, u1_map[r]["static_committor"]["env_all_vs_prot"]["delta_ell"]["ci_high"] - pt)
        for r, pt in zip(recs, dell_pts)
    ]
    p_vals = [u1_map[r]["static_committor"]["n1_permutation_p_value"] for r in recs]
    bars_b = ax_b.bar(x, dell_pts, width=0.55, color="#148F77", alpha=0.88, yerr=[dell_lo, dell_hi], capsize=4)
    ax_b.axhline(0.0, color="#333333", lw=0.9, ls="--")
    for idx, (bar, pv) in enumerate(zip(bars_b, p_vals)):
        star = "**" if pv <= 0.02 else ("*" if pv <= 0.05 else f"p={pv:.2f}")
        ax_b.text(
            bar.get_x() + bar.get_width() / 2.0,
            max(dell_pts[idx] + dell_hi[idx], 0.0) + 0.002,
            star,
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(recs, fontweight="bold")
    ax_b.set_ylabel("Out-of-Block $\\Delta\\bar{\\ell}_{\\mathrm{static}}(+\\mathbf{z}_{\\mathrm{env}})$ (nats)")
    ax_b.set_title("B. U1: Explicit Solvent/Ion Committor Predictive Gain", fontweight="bold", loc="left")
    ax_b.grid(axis="y", alpha=0.25)

    # Panel C: U2 Evolutionary Specificity of 2.42-4.46 Cleft (CB1 Phe-Phe vs CB2 Phe-Leu & Controls)
    ax_c = axes[1, 0]
    arom_occ = [100.0 * u2_map[r]["aromatic_lock_occupancy_4p2A"]["point"] for r in recs]
    raw_occ = [100.0 * u2_map[r]["cleft_contact_occupancy_4p2A"]["point"] for r in recs]
    ent_joint = [u2_map[r]["joint_242_446_rotamer_entropy_bits"] for r in recs]
    pair_labels = [f"{r}\n({u2_map[r]['bw242_resname'][0]}2.42-{u2_map[r]['bw446_resname'][0]}4.46)" for r in recs]

    ax_c.bar(x - w / 2, arom_occ, width=w, color="#8E44AD", alpha=0.90, label="Aromatic-Pair Lock (≤ 4.2 Å)")
    ax_c.bar(x + w / 2, raw_occ, width=w, color="#AEB6BF", alpha=0.80, label="Any Sidechain Contact (≤ 4.2 Å)")
    ax_c2 = ax_c.twinx()
    ax_c2.plot(x, ent_joint, color="#D35400", marker="o", lw=2.0, label="Joint $\\chi_1$ Entropy (bits)")
    ax_c2.set_ylabel("Joint $(2.42, 4.46)$ $\\chi_1$ Entropy (bits)", color="#D35400")
    ax_c2.tick_params(axis="y", labelcolor="#D35400")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(pair_labels, fontsize=8.5, fontweight="bold")
    ax_c.set_ylabel("Cleft Contact Occupancy (%)")
    ax_c.set_title("C. U2: Unique CB1 $\\mathrm{F155^{2.42}-F237^{4.46}}$ Aromatic Cleft Lock", fontweight="bold", loc="left")
    ax_c.legend(loc="upper right", frameon=False, fontsize=8.0)
    ax_c.grid(axis="y", alpha=0.25)

    # Panel D: Trajectory-Level Allosteric Phase Space (D2.50 Cation Coordination vs Water-Wire Percolation)
    ax_d = axes[1, 1]
    palette = {
        "CB1": "#8E44AD",
        "CB2": "#2980B9",
        "S1P1": "#16A085",
        "B2AR": "#E67E22",
        "MOR": "#C0392B",
        "A2AR": "#2C3E50",
    }
    for r in recs:
        df_r = df_by_receptor[r]
        for _, t_sub in df_r.groupby("traj_id", sort=False):
            st = str(t_sub["state_class"].iloc[0])
            marker = "^" if st.lower().startswith("act") else "o"
            x_ion = 100.0 * float(np.mean(t_sub["coord_cation_D250_3p2"].to_numpy() > 0))
            y_wire = 100.0 * float(np.mean(t_sub["wire_D250_NPxxY_cond"].to_numpy() > 0.0))
            ax_d.scatter(
                x_ion,
                y_wire,
                s=68,
                marker=marker,
                color=palette.get(r, "#34495E"),
                edgecolor="white",
                linewidth=0.7,
                alpha=0.9,
            )
        ax_d.scatter([], [], s=60, color=palette.get(r, "#34495E"), label=r)
    ax_d.scatter([], [], s=60, color="#555555", marker="o", label="Inactive")
    ax_d.scatter([], [], s=60, color="#555555", marker="^", label="Active")
    ax_d.set_xlabel("$\\mathrm{D^{2.50}}$ Cation ($\\mathrm{Na^+/K^+}$) Coordination Occupancy (%)")
    ax_d.set_ylabel("$\\mathrm{D^{2.50}}$–NPxxY Water-Wire Percolation (%)")
    ax_d.set_title("D. Superfamily Allosteric Ion vs. Water-Wire Phase Space", fontweight="bold", loc="left")
    ax_d.legend(ncol=2, frameon=False, fontsize=8.0, loc="best")
    ax_d.grid(True, alpha=0.25)

    fig.suptitle(
        "Class A GPCR Superfamily Dynamics: Conserved Buried Water/Na+ Closure (U1) vs. CB1-Specific 2.42–4.46 Gate (U2)",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_superfamily_universality_suite(
    df_by_receptor: dict[str, pd.DataFrame],
    out_json_path: str | Path | None = None,
    out_fig_path: str | Path | None = None,
    tau_max_ns: float = 3.0,
    n_splits: int = 5,
    n_shuffles: int = 100,
    n_perms: int = 100,
    n_boot: int = 1000,
) -> dict[str, Any]:
    """Execute complete U1 & U2 multi-GPCR evaluation suite, save summary JSON, and render 4-panel figure."""
    ordered_keys = [k for k in RECEPTOR_DISPLAY_ORDER if k in df_by_receptor] + [
        k for k in df_by_receptor if k not in RECEPTOR_DISPLAY_ORDER
    ]
    u1_per_rec: dict[str, Any] = {}
    for rec_key in ordered_keys:
        u1_per_rec[rec_key] = evaluate_u1_water_na_memory_closure(
            df_by_receptor[rec_key],
            receptor_key=rec_key,
            tau_max_ns=tau_max_ns,
            n_splits=n_splits,
            n_shuffles=n_shuffles,
            n_perms=n_perms,
            n_boot=n_boot,
        )

    n_positive_dell = sum(
        1 for v in u1_per_rec.values()
        if v["static_committor"]["env_all_vs_prot"]["delta_ell"]["point"] > 0.0
    )
    n_positive_collapse = sum(
        1 for v in u1_per_rec.values()
        if v["primary_memory_summary"]["rho_mem_collapse"] > 0.0
    )
    u2_res = evaluate_u2_tm2_tm4_evolutionary_specificity(df_by_receptor, n_boot=n_boot)

    summary: dict[str, Any] = {
        "n_receptors_evaluated": len(ordered_keys),
        "receptors": ordered_keys,
        "U1_water_na_closure": {
            "n_receptors_positive_static_env_gain": n_positive_dell,
            "n_receptors_positive_memory_collapse": n_positive_collapse,
            "median_rho_mem_collapse": round(
                float(np.median([v["primary_memory_summary"]["rho_mem_collapse"] for v in u1_per_rec.values()])), 4
            ),
            "per_receptor": u1_per_rec,
        },
        "U2_tm2_tm4_specificity": u2_res,
    }

    if out_fig_path is not None:
        fig_p = plot_superfamily_universality_figure(summary, df_by_receptor, out_fig_path)
        summary["figure_path"] = str(fig_p)

    if out_json_path is not None:
        jp = Path(out_json_path)
        jp.parent.mkdir(parents=True, exist_ok=True)
        with open(jp, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

    return summary
