"""Prospective statistical readout, survival analysis, and Mori-Zwanzig closure for WS5.

Analyzes partial (`rep01..rep05` interim) or complete (`15 reps x 3 arms = 45 trajectories = 900 ns`)
prospective OpenMM committor shooting data produced by `ShootingCampaignQueue`:
  1. Pure `R214^3.50-D338^6.30` stopping-time & full-horizon commitment classification
     (zero orthogonal microswitches in target definition -> zero circularity, zero t=0 membership):
     - Collapse basin `A_collapsed` (`q_outcome = 0`): first passage `tau_A` to cytoplasmic
       `R214_D338_CA_A <= collapse_tm6_cutoff_A` (`12.80 A`) prior to `tau_B`.
     - Committed active basin `B_committed` (`q_outcome = 1`): either first passage `tau_B < tau_A`
       to full active cavity expansion `R214_D338_CA_A >= expand_tm6_cutoff_A` (`15.30 A`), OR
       uncollapsed full-horizon active persistence (`tau_A = inf`, `t_max >= min_sustained_open_ns`
       [`18.0 ns`], and terminal-quarter mean `R214_D338_CA_A >= sustained_open_tm6_A` [`14.00 A`]).
  2. Non-parametric Kaplan-Meier open-TM6 survival curves with Greenwood log-log 95% CIs
     and Mantel-Cox log-rank tests (`Arm A vs Arm B`, `Arm A vs Arm C`).
  3. Empirical forward committor `q_hat` with exact Clopper-Pearson 95% binomial CIs,
     Fisher's exact test, and 10,000-shuffle permutation test on terminal TM6 opening.
  4. Single-Hamiltonian (WT `Arm A + Arm B` only) `src.core` Mori-Zwanzig two-step memory closure
     strictly restricted to the pre-hitting transition corridor (`time_ns <= t_hit`) with dynamic
     `dt_ns` lag estimation and stratified group cross-validation.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.core.memory_tests import compute_memory_collapse_ratio, evaluate_mori_zwanzig_memory


OKABE_ITO_ARMS = {
    "Arm_A_Fake_Open_Trapped": ("#D55E00", "Arm A: WT Fake-Open Trapped"),
    "Arm_B_Committed_Open": ("#0072B2", "Arm B: WT Committed-Open"),
    "Arm_C_Mutant_F237L": ("#009E73", "Arm C: F237L (4.46 Swap)"),
}

WT_ARMS = ("Arm_A_Fake_Open_Trapped", "Arm_B_Committed_Open")


def compute_clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """Return `(p_hat, ci_low, ci_high)` exact Clopper-Pearson binomial confidence interval."""
    if n <= 0:
        return (float("nan"), float("nan"), float("nan"))
    if k < 0 or k > n:
        raise ValueError(f"Invalid binomial count k={k} for n={n}")
    p_hat = float(k) / float(n)
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return (p_hat, lo, hi)


def compute_kaplan_meier_curve(
    times: np.ndarray,
    events: np.ndarray,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compute exact Kaplan-Meier survival curve `S(t)` with Greenwood log-log 95% CIs."""
    t_arr = np.asarray(times, dtype=np.float64)
    e_arr = np.asarray(events, dtype=bool)
    if t_arr.size == 0:
        return pd.DataFrame(columns=["time_ns", "at_risk", "events", "survival", "ci_low", "ci_high"])

    uniq_t = np.unique(np.concatenate(([0.0], np.sort(t_arr))))
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    surv = 1.0
    greenwood_sum = 0.0
    rows = []

    for t_val in uniq_t:
        at_risk = int(np.sum(t_arr >= t_val))
        d_i = int(np.sum((t_arr == t_val) & e_arr)) if t_val > 0.0 else 0
        if at_risk > 0 and d_i > 0:
            surv *= 1.0 - (d_i / float(at_risk))
            if at_risk > d_i:
                greenwood_sum += d_i / float(at_risk * (at_risk - d_i))

        if surv <= 0.0:
            ci_lo, ci_hi = 0.0, 0.0
        elif surv >= 1.0 or greenwood_sum <= 0.0:
            ci_lo, ci_hi = surv, 1.0
        else:
            se_loglog = np.sqrt(greenwood_sum) / abs(np.log(surv))
            theta = np.exp(z * se_loglog)
            ci_lo = float(surv ** theta)
            ci_hi = float(surv ** (1.0 / theta))

        rows.append(
            {
                "time_ns": float(t_val),
                "at_risk": at_risk,
                "events": d_i,
                "survival": float(surv),
                "ci_low": float(np.clip(ci_lo, 0.0, 1.0)),
                "ci_high": float(np.clip(ci_hi, 0.0, 1.0)),
            }
        )
    return pd.DataFrame(rows)


def compute_logrank_test(
    times_1: np.ndarray,
    events_1: np.ndarray,
    times_2: np.ndarray,
    events_2: np.ndarray,
) -> dict[str, float]:
    """Two-sample Mantel-Cox log-rank test comparing collapse hazard between two shooting arms."""
    t1 = np.asarray(times_1, dtype=np.float64)
    e1 = np.asarray(events_1, dtype=bool)
    t2 = np.asarray(times_2, dtype=np.float64)
    e2 = np.asarray(events_2, dtype=bool)

    all_event_times = np.unique(np.concatenate([t1[e1], t2[e2]]))
    if all_event_times.size == 0:
        return {"chi2": 0.0, "p_value": 1.0, "observed_1": 0.0, "expected_1": 0.0}

    obs_1 = 0.0
    exp_1 = 0.0
    var_1 = 0.0
    for tj in all_event_times:
        n1j = float(np.sum(t1 >= tj))
        n2j = float(np.sum(t2 >= tj))
        nj = n1j + n2j
        if nj <= 1.0:
            continue
        d1j = float(np.sum((t1 == tj) & e1))
        d2j = float(np.sum((t2 == tj) & e2))
        dj = d1j + d2j
        obs_1 += d1j
        exp_1 += dj * (n1j / nj)
        var_1 += (n1j * n2j * dj * (nj - dj)) / (nj * nj * (nj - 1.0))

    if var_1 <= 1e-12:
        return {"chi2": 0.0, "p_value": 1.0, "observed_1": obs_1, "expected_1": exp_1}
    chi2 = ((obs_1 - exp_1) ** 2) / var_1
    pval = float(stats.chi2.sf(chi2, df=1))
    return {
        "chi2": float(chi2),
        "p_value": pval,
        "observed_1": float(obs_1),
        "expected_1": float(exp_1),
    }


def _permutation_diff_means_p(x: np.ndarray, y: np.ndarray, n_perm: int = 10_000, seed: int = 2026) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0 or y.size == 0:
        return {"diff_mean": float("nan"), "p_value": float("nan")}
    obs_diff = float(np.mean(y) - np.mean(x))
    pooled = np.concatenate([x, y])
    nx = x.size
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        d = float(np.mean(perm[nx:]) - np.mean(perm[:nx]))
        if abs(d) >= abs(obs_diff) - 1e-12:
            exceed += 1
    return {
        "diff_mean": obs_diff,
        "p_value": float((exceed + 1) / (n_perm + 1)),
    }


def summarize_prospective_campaign(
    traj_dir: Path,
    collapse_tm6_cutoff_A: float = 12.80,
    expand_tm6_cutoff_A: float = 15.30,
    min_sustained_open_ns: float = 18.0,
    sustained_open_tm6_A: float = 14.00,
    target_memory_lag_ns: float = 0.50,
    n_memory_shuffles: int = 50,
) -> dict[str, Any]:
    """Load all available trajectory Parquet tables and compute survival, committor, and memory metrics."""
    pq_files = sorted(Path(traj_dir).glob("*_observables.parquet"))
    if not pq_files:
        return {"n_trajectories": 0, "frames_df": pd.DataFrame(), "traj_summary_df": pd.DataFrame()}

    dfs = [pd.read_parquet(p) for p in pq_files]
    frames_df = pd.concat(dfs, ignore_index=True).sort_values(["arm_name", "replicate", "step"]).reset_index(drop=True)

    traj_rows: list[dict[str, Any]] = []
    for (arm_name, rep, job_id), g in frames_df.groupby(["arm_name", "replicate", "job_id"], sort=True):
        g = g.sort_values("step").reset_index(drop=True)
        t_max = float(g["time_ns"].iloc[-1])

        # Pure R214_D338_CA_A stopping-time boundaries (strictly t > 0; t=0 seeds are ~14.26 A inside (12.80, 15.30) A)
        post_t0 = g[g["time_ns"] > 0.0] if len(g) > 1 else g
        hit_A = post_t0[post_t0["R214_D338_CA_A"] <= collapse_tm6_cutoff_A]
        hit_B = post_t0[post_t0["R214_D338_CA_A"] >= expand_tm6_cutoff_A]

        tau_A = float(hit_A["time_ns"].iloc[0]) if len(hit_A) > 0 else float("inf")
        tau_B = float(hit_B["time_ns"].iloc[0]) if len(hit_B) > 0 else float("inf")

        q4_mask = g["time_ns"] >= (0.75 * t_max)
        q4 = g[q4_mask] if np.any(q4_mask) else g
        q4_mean_tm6 = float(q4["R214_D338_CA_A"].mean())

        if math.isfinite(tau_A) and tau_A <= tau_B:
            collapsed = True
            t_hit = tau_A
            fate = "A_collapsed"
            q_outcome = 0
        elif math.isfinite(tau_B) and tau_B < tau_A:
            collapsed = False
            t_hit = tau_B
            fate = "B_expanded"
            q_outcome = 1
        elif (not math.isfinite(tau_A)) and t_max >= min_sustained_open_ns and q4_mean_tm6 >= sustained_open_tm6_A:
            collapsed = False
            t_hit = t_max
            fate = "B_sustained_open"
            q_outcome = 1
        else:
            collapsed = False
            t_hit = t_max
            fate = "undecided_open"
            q_outcome = -1

        tail_mask = g["time_ns"] >= (0.5 * t_max)
        tail = g[tail_mask] if np.any(tail_mask) else g
        traj_rows.append(
            {
                "job_id": job_id,
                "arm_name": arm_name,
                "replicate": int(rep),
                "n_frames": len(g),
                "duration_ns": t_max,
                "collapsed_event": bool(collapsed),
                "event_or_censor_time_ns": float(t_hit),
                "t_hit_ns": float(t_hit),
                "fate": fate,
                "q_outcome": q_outcome,
                "init_R214_D338_A": float(g["R214_D338_CA_A"].iloc[0]),
                "final_R214_D338_A": float(g["R214_D338_CA_A"].iloc[-1]),
                "q4_mean_R214_D338_A": q4_mean_tm6,
                "tail_mean_R214_D338_A": float(tail["R214_D338_CA_A"].mean()),
                "tail_mean_Y294_Y397_OH_A": float(tail["Y294_Y397_OH_A"].mean()),
                "tail_mean_F155sc_N393CA_A": float(tail["F155sc_N393CA_A"].mean()),
                "tail_mean_wire_cond": float(tail["wire_conductance_d163_ny"].mean()),
            }
        )

    traj_df = pd.DataFrame(traj_rows)

    arm_stats: dict[str, Any] = {}
    km_curves: dict[str, pd.DataFrame] = {}
    for arm_name, sub in traj_df.groupby("arm_name", sort=True):
        resolved = sub[sub["q_outcome"] >= 0]
        k_b = int(np.sum(resolved["q_outcome"] == 1))
        n_res = len(resolved)
        q_hat, q_lo, q_hi = compute_clopper_pearson_ci(k_b, n_res)

        n_all = len(sub)
        k_persist = int(np.sum(~sub["collapsed_event"]))
        p_pers, pers_lo, pers_hi = compute_clopper_pearson_ci(k_persist, n_all)

        km_df = compute_kaplan_meier_curve(
            sub["event_or_censor_time_ns"].to_numpy(),
            sub["collapsed_event"].to_numpy(),
        )
        km_curves[arm_name] = km_df
        arm_stats[arm_name] = {
            "n_trajectories": n_all,
            "total_simulated_ns": float(sub["duration_ns"].sum()),
            "n_collapsed": int(sub["collapsed_event"].sum()),
            "n_committed_B": k_b,
            "n_resolved": n_res,
            "q_hat_resolved": q_hat,
            "q_hat_ci95": [q_lo, q_hi],
            "open_persistence_rate": p_pers,
            "open_persistence_ci95": [pers_lo, pers_hi],
            "tail_R214_D338_mean_A": float(sub["tail_mean_R214_D338_A"].mean()),
            "tail_R214_D338_sd_A": float(sub["tail_mean_R214_D338_A"].std(ddof=1)) if n_all > 1 else 0.0,
        }

    # Pairwise hypothesis tests: Arm A vs Arm B (single-Hamiltonian causal switch test) & Arm A vs Arm C (F237L rescue)
    comparisons: dict[str, Any] = {}
    for pair_key, a1, a2 in (
        ("Arm_A_vs_Arm_B_Single_Hamiltonian", "Arm_A_Fake_Open_Trapped", "Arm_B_Committed_Open"),
        ("Arm_A_vs_Arm_C_F237L_Rescue", "Arm_A_Fake_Open_Trapped", "Arm_C_Mutant_F237L"),
    ):
        s1 = traj_df[traj_df["arm_name"] == a1]
        s2 = traj_df[traj_df["arm_name"] == a2]
        if len(s1) > 0 and len(s2) > 0:
            lr = compute_logrank_test(
                s1["event_or_censor_time_ns"].to_numpy(),
                s1["collapsed_event"].to_numpy(),
                s2["event_or_censor_time_ns"].to_numpy(),
                s2["collapsed_event"].to_numpy(),
            )
            table_2x2 = [
                [int(np.sum(~s1["collapsed_event"])), int(np.sum(s1["collapsed_event"]))],
                [int(np.sum(~s2["collapsed_event"])), int(np.sum(s2["collapsed_event"]))],
            ]
            _, fisher_p = stats.fisher_exact(table_2x2)
            perm = _permutation_diff_means_p(
                s1["tail_mean_R214_D338_A"].to_numpy(),
                s2["tail_mean_R214_D338_A"].to_numpy(),
            )
            comparisons[pair_key] = {
                "logrank": lr,
                "fisher_exact_persistence_p": float(fisher_p),
                "tail_TM6_diff_A_arm2_minus_arm1": perm["diff_mean"],
                "tail_TM6_permutation_p": perm["p_value"],
            }

    # Primary Single-Hamiltonian Mori-Zwanzig Memory Closure (strictly WT Arm A + Arm B, pre-hitting t <= t_hit!)
    memory_closure: dict[str, Any] = {
        "available": False,
        "hamiltonian_scope": "WT_only_Arm_A_Arm_B",
    }
    wt_resolved = traj_df[
        traj_df["arm_name"].isin(WT_ARMS) & (traj_df["q_outcome"] >= 0)
    ].sort_values("job_id")
    n_pos_wt = int(np.sum(wt_resolved["q_outcome"] == 1))
    n_neg_wt = int(np.sum(wt_resolved["q_outcome"] == 0))

    if len(wt_resolved) >= 4 and min(n_pos_wt, n_neg_wt) >= 2:
        fate_map = dict(zip(wt_resolved["job_id"], wt_resolved["q_outcome"]))
        thit_map = dict(zip(wt_resolved["job_id"], wt_resolved["t_hit_ns"]))
        wt_jobs = list(wt_resolved["job_id"])

        corridor_parts: list[pd.DataFrame] = []
        pos_dt_list: list[float] = []
        for jid in wt_jobs:
            grp = frames_df[frames_df["job_id"] == jid].sort_values("step")
            # Restrict strictly to the pre-hitting transition corridor (time_ns <= t_hit)
            grp_pre = grp[grp["time_ns"] <= thit_map[jid] + 1e-9].copy()
            if len(grp_pre) >= 3:
                dts = np.diff(grp_pre["time_ns"].to_numpy(dtype=np.float64))
                pos_dt_list.extend(dts[dts > 1e-9].tolist())
                corridor_parts.append(grp_pre)

        if corridor_parts and pos_dt_list:
            dt_ns = float(np.median(pos_dt_list))
            lag_steps = max(1, int(round(target_memory_lag_ns / dt_ns)))
            actual_lag_ns = round(lag_steps * dt_ns, 4)

            # Keep only trajectories longer than lag_steps inside the transition corridor
            valid_parts = [p for p in corridor_parts if len(p) > lag_steps + 1]
            valid_jids = [str(p["job_id"].iloc[0]) for p in valid_parts]
            valid_outcomes = [fate_map[j] for j in valid_jids]
            n_pos_v = sum(1 for v in valid_outcomes if v == 1)
            n_neg_v = sum(1 for v in valid_outcomes if v == 0)

            if len(valid_parts) >= 4 and min(n_pos_v, n_neg_v) >= 2:
                f_cor = pd.concat(valid_parts, ignore_index=True)
                job_to_int = {jid: idx for idx, jid in enumerate(valid_jids)}
                seg_id = f_cor["job_id"].map(job_to_int).to_numpy(dtype=np.int64)
                y_all = f_cor["job_id"].map(fate_map).to_numpy(dtype=np.float64)
                tm6_arr = f_cor["R214_D338_CA_A"].to_numpy(dtype=np.float64)
                in_A = tm6_arr <= collapse_tm6_cutoff_A
                in_B = tm6_arr >= expand_tm6_cutoff_A
                interior_count = int(np.sum(~(in_A | in_B)))

                if interior_count >= 12:
                    X1 = f_cor[["R214_D338_CA_A"]].to_numpy(dtype=np.float64)
                    XS3 = f_cor[
                        [
                            "R214_D338_CA_A",
                            "F155sc_N393CA_A",
                            "Y294_Y397_OH_A",
                            "wire_conductance_d163_ny",
                            "min_CHL1_cleft_A",
                        ]
                    ].to_numpy(dtype=np.float64)
                    n_sp = min(5, n_pos_v, n_neg_v)
                    m_1d = evaluate_mori_zwanzig_memory(
                        X1,
                        seg_id,
                        y_all,
                        groups=seg_id,
                        lag_steps=lag_steps,
                        in_A=in_A,
                        in_B=in_B,
                        fit_interior_only=True,
                        n_splits=n_sp,
                        n_shuffles=n_memory_shuffles,
                        stratify_labels=y_all.astype(np.int64),
                    )
                    m_s3 = evaluate_mori_zwanzig_memory(
                        XS3,
                        seg_id,
                        y_all,
                        groups=seg_id,
                        lag_steps=lag_steps,
                        in_A=in_A,
                        in_B=in_B,
                        fit_interior_only=True,
                        n_splits=n_sp,
                        n_shuffles=n_memory_shuffles,
                        stratify_labels=y_all.astype(np.int64),
                    )
                    g1 = float(m_1d["delta_ell_mem"])
                    gs3 = float(m_s3["delta_ell_mem"])
                    rho = compute_memory_collapse_ratio(g1, gs3)
                    memory_closure = {
                        "available": True,
                        "hamiltonian_scope": "WT_only_Arm_A_Arm_B",
                        "n_wt_resolved_trajectories": len(valid_parts),
                        "n_eval_frames": int(m_1d["n_eval_frames"]),
                        "dt_ns": round(dt_ns, 5),
                        "lag_steps": int(lag_steps),
                        "lag_ns": actual_lag_ns,
                        "delta_ll_1d_nats": g1,
                        "delta_ll_stage3_nats": gs3,
                        "n2a_p_1d": float(m_1d["p_value_vs_n2a"]),
                        "n2a_p_stage3": float(m_s3["p_value_vs_n2a"]),
                        "closure_fraction_rho": rho,
                    }

    return {
        "n_trajectories": len(traj_df),
        "total_simulated_ns": float(traj_df["duration_ns"].sum()),
        "frames_df": frames_df,
        "traj_summary_df": traj_df,
        "arm_stats": arm_stats,
        "km_curves": km_curves,
        "comparisons": comparisons,
        "memory_closure": memory_closure,
    }


def plot_prospective_campaign_readout(summary: dict[str, Any], out_png: Path | None = None) -> plt.Figure:
    """Render publication-grade 3-panel prospective shooting readout figure."""
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5), constrained_layout=True)
    frames_df: pd.DataFrame = summary["frames_df"]
    traj_df: pd.DataFrame = summary["traj_summary_df"]
    km_curves: dict[str, pd.DataFrame] = summary.get("km_curves", {})
    arm_stats: dict[str, Any] = summary.get("arm_stats", {})

    if len(traj_df) == 0:
        for ax in axes:
            ax.text(0.5, 0.5, "No completed chunks yet", ha="center", va="center")
        return fig

    # Panel A: Kaplan-Meier Open-TM6 Survival S(t)
    ax0 = axes[0]
    for arm_name, (col, lbl) in OKABE_ITO_ARMS.items():
        if arm_name not in km_curves:
            continue
        km = km_curves[arm_name]
        ax0.step(km["time_ns"], km["survival"], where="post", color=col, lw=2.2, label=lbl)
        ax0.fill_between(km["time_ns"], km["ci_low"], km["ci_high"], step="post", color=col, alpha=0.16)
    ax0.set_ylim(-0.03, 1.05)
    ax0.set_xlabel("Unbiased Shooting Time $t$ (ns)")
    ax0.set_ylabel("Open-TM6 Survival Probability $\\hat{S}(t)$")
    ax0.set_title("A. Prospective Open-TM6 Kaplan–Meier Survival")
    ax0.legend(loc="lower left", frameon=True, fontsize=8.0)

    # Panel B: Cytoplasmic R214-D338 CA Traces & Ensemble Mean
    ax1 = axes[1]
    for arm_name, (col, lbl) in OKABE_ITO_ARMS.items():
        sub = frames_df[frames_df["arm_name"] == arm_name]
        if len(sub) == 0:
            continue
        for _, g in sub.groupby("job_id"):
            ax1.plot(g["time_ns"], g["R214_D338_CA_A"], color=col, lw=0.7, alpha=0.22)
        agg = sub.groupby("time_ns")["R214_D338_CA_A"].agg(["mean", "std", "count"]).reset_index()
        se = np.where(agg["count"] > 1, agg["std"] / np.sqrt(agg["count"]), 0.0)
        ax1.plot(agg["time_ns"], agg["mean"], color=col, lw=2.3, label=lbl)
        ax1.fill_between(agg["time_ns"], agg["mean"] - 1.96 * se, agg["mean"] + 1.96 * se, color=col, alpha=0.18)
    ax1.axhline(14.3, color="black", ls=":", lw=1.0, alpha=0.7, label="Matched $t=0$ (14.3 Å)")
    ax1.axhline(15.3, color="#0072B2", ls="-.", lw=0.9, alpha=0.6, label="Expand $B$ (15.3 Å)")
    ax1.axhline(12.8, color="#CC0000", ls="--", lw=1.0, alpha=0.7, label="Collapse $A$ (12.8 Å)")
    ax1.set_xlabel("Unbiased Shooting Time $t$ (ns)")
    ax1.set_ylabel(r"Cytoplasmic $d(\mathrm{R214}^{3.50}\text{–}\mathrm{D338}^{6.30}\ \mathrm{C}\alpha)$ (Å)")
    ax1.set_title("B. Single-Hamiltonian TM6 Trajectory Bifurcation")
    ax1.legend(loc="best", frameon=True, fontsize=7.5)

    # Panel C: Open Persistence / Committor CI + Stage-3 Phase Portrait
    ax2 = axes[2]
    for arm_name, (col, lbl) in OKABE_ITO_ARMS.items():
        sub_t = traj_df[traj_df["arm_name"] == arm_name]
        if len(sub_t) == 0:
            continue
        st = arm_stats[arm_name]
        lbl_short = f"{lbl.split(':')[0]} (persist={st['open_persistence_rate']:.2f})"
        ax2.scatter(
            sub_t["tail_mean_F155sc_N393CA_A"],
            sub_t["tail_mean_Y294_Y397_OH_A"],
            c=col,
            s=65,
            edgecolors="black",
            alpha=0.9,
            label=lbl_short,
        )
    ax2.axvline(13.5, color="black", ls="--", lw=0.9, alpha=0.6)
    ax2.axhline(4.25, color="black", ls=":", lw=0.9, alpha=0.6)
    ax2.set_xlabel(r"Tail Mean $\mathrm{F155}^{2.42}_{\mathrm{sc}}\text{–}\mathrm{N393}^{7.49}_{\mathrm{C}\alpha}$ (Å)")
    ax2.set_ylabel(r"Tail Mean Latch $d(\mathrm{Y294}^{5.58}\text{–}\mathrm{Y397}^{7.53}\ \mathrm{OH})$ (Å)")
    ax2.set_title("C. Stage-3 Rotamer Flip vs. Intracellular Latch")
    ax2.legend(loc="best", frameon=True, fontsize=8.0)

    if out_png is not None:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=180)
    return fig
