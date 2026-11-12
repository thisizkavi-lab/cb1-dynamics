"""Publication-quality multi-panel figure generator for Workstream 2."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "figure.dpi": 220,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_fig1_msm_tpt(df_micro: pd.DataFrame, metrics: dict, fig_dir: Path) -> Path:
    msm_m = metrics["global_5000run_msm_tpt"]
    its_scan = msm_m["markov_invariants"]["multi_lag_its_scan"]
    ck20 = msm_m["markov_invariants"]["ck_test_20ns_4step"]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))

    # Panel A: Multi-lag Implied Timescales
    ax = axes[0, 0]
    lag_keys = sorted(its_scan.keys(), key=lambda k: its_scan[k]["lag_ns"])
    lags_ns = [its_scan[k]["lag_ns"] for k in lag_keys]
    colors_its = ["#1b4f72", "#2874a6", "#148f77", "#d4ac0d", "#ca6f1e", "#922b21"]
    for m_i in range(6):
        vals_us = [its_scan[k]["its_ns"][m_i] / 1000.0 for k in lag_keys]
        ax.plot(lags_ns, vals_us, marker="o", lw=2.0, color=colors_its[m_i], label=f"Mode {m_i + 1}")
    ax.axvline(35.0, color="#333333", ls="--", lw=1.2, label="Published MSM lag (35 ns)")
    ax.fill_between(lags_ns, 0, [l / 1000.0 for l in lags_ns], color="#dddddd", alpha=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("MSM Lag Time $\\tau$ (ns)")
    ax.set_ylabel("Implied Timescale $t_k(\\tau)$ ($\\mu$s)")
    ax.set_title("A. 5,000-Run (483 $\\mu$s) Implied Timescale Convergence")
    ax.legend(ncol=2, frameon=False, loc="lower right")
    ax.grid(alpha=0.25)

    # Panel B: 6-Macrostate Chapman-Kolmogorov Consistency (20, 40, 60, 80 ns)
    ax = axes[0, 1]
    steps = ["1", "2", "3", "4"]
    step_lags = [ck20["steps"][s]["lag_ns"] for s in steps]
    macro_short = ["U_Sol", "U1_Mem", "U2_Mem", "I1_TM17", "I2_AB", "B1_Ortho"]
    m_cols = ["#c0392b", "#2980b9", "#16a085", "#f39c12", "#8e44ad", "#2c3e50"]
    for m_i, m_lbl in enumerate(macro_short):
        pred_d = [ck20["steps"][s]["pred_diag"][m_i] for s in steps]
        emp_d = [ck20["steps"][s]["emp_diag"][m_i] for s in steps]
        ax.plot(step_lags, pred_d, lw=1.8, color=m_cols[m_i], label=f"{m_lbl} (MSM)")
        ax.scatter(step_lags, emp_d, marker="s", s=34, color=m_cols[m_i], edgecolor="white", zorder=4)
    ax.set_ylim(0.25, 1.03)
    ax.set_xlabel("Chapman–Kolmogorov Horizon $k\\tau_0$ (ns, $\\tau_0=20$ ns)")
    ax.set_ylabel("Self-Transition Probability $P(M_i \\to M_i; k\\tau_0)$")
    ax.set_title("B. Macrostate Chapman–Kolmogorov Test (Lines: $P^k$, Squares: MD)")
    ax.legend(ncol=2, frameon=False, loc="lower left")
    ax.grid(alpha=0.25)

    # Panel C: 2D (X, Y) Ligand COM Microstate Map Colored by Exact TPT q_+^{MSM}
    ax = axes[1, 0]
    act = df_micro[df_micro["is_active"]].copy()
    sizes = np.clip(act["stationary_pi"].to_numpy() * 28000.0, 12.0, 220.0)
    sc = ax.scatter(
        act["lig_com_x_mean_A"],
        act["lig_com_y_mean_A"],
        c=act["q_msm_phys"],
        s=sizes,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        alpha=0.88,
        edgecolors="black",
        linewidths=0.3,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Exact 800-State TPT Committor $q_+^{\\mathrm{MSM}}$")
    ax.set_xlabel("Canonical Ligand COM $X$ ($\\mathrm{\\AA}$)")
    ax.set_ylabel("Canonical Ligand COM $Y$ ($\\mathrm{\\AA}$)")
    ax.set_title("C. 800-Microstate Spatial Projection & TM1–TM7 Portal Entry")
    ax.grid(alpha=0.25)

    # Panel D: q_+^{MSM} vs 1D R_portal showing large orthogonal variance in [8.5, 16] A
    ax = axes[1, 1]
    macro_palette = {
        2: ("#e74c3c", "U_Sol (Ex)"),
        3: ("#3498db", "U1_Mem (Bulk2)"),
        1: ("#1abc9c", "U2_Mem (Bulk1)"),
        0: ("#f39c12", "I1_TM17 (Portal)"),
        4: ("#9b59b6", "I2_TM17 (AB Vestibule)"),
        5: ("#1f2d3d", "B_Pro (B1 Orthosteric)"),
    }
    for m_id, (col, lbl) in macro_palette.items():
        sub = act[act["macrostate_id"] == m_id]
        ax.scatter(
            sub["r_portal_mean_A"],
            sub["q_msm_phys"],
            s=36,
            color=col,
            alpha=0.85,
            label=lbl,
            edgecolors="white",
            linewidths=0.3,
        )
    ax.axvspan(9.5, 18.5, color="#f4d03f", alpha=0.16, label="Matched-$R_{\\mathrm{portal}}$ H1 slice")
    ax.set_xlim(5.0, 52.0)
    ax.set_ylim(-0.03, 1.04)
    ax.set_xlabel("Microstate Mean Pocket Depth $R_{\\mathrm{portal}}$ ($\\mathrm{\\AA}$)")
    ax.set_ylabel("Exact TPT Forward Committor $q_+^{\\mathrm{MSM}}$")
    ax.set_title("D. Overlap of Portal/Vestibule/Bound States at Fixed $R_{\\mathrm{portal}}$")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    out_p = fig_dir / "fig1_global_msm_tpt_landscape.png"
    fig.savefig(out_p)
    plt.close(fig)
    return out_p


def plot_fig2_h1_gating(metrics: dict, fig_dir: Path) -> Path:
    h1_msm = metrics["H1_matched_portal_slice"]["q_msm_phys"]["models"]
    h1_dga = metrics["H1_matched_portal_slice"]["q_dga_oof"]["models"]
    dga_lags = metrics["continuous_stopped_dga_validation"]["lags"]
    abl = metrics["H1_switch_family_ablations"]

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.8))

    # Panel A: Out-of-Fold Matched-R_portal BSS_int across model hierarchy (q_msm & q_dga)
    ax = axes[0, 0]
    m_order = [
        ("B2_spline_plus_vel2ns", "B2 (R + dR)"),
        ("TierA_CA_linear", "Tier A (CA)"),
        ("H1_Switches_additive", "H1 Switches (Add)"),
        ("H1_Switches_full_epistasis", "H1 Switches (Epi)"),
        ("TierB_Full_sidechains", "Tier B (+Rotamers)"),
        ("TierC_Full_lipid_AEA", "Tier C (+Lipid/AEA)"),
        ("Tier3_Oracle_HistGBR", "Tier 3 Oracle"),
    ]
    y_pos = np.arange(len(m_order))
    h = 0.36
    bss_m = [h1_msm[k]["bss_vs_b1"] for k, _ in m_order]
    err_m_lo = [h1_msm[k]["bss_vs_b1"] - h1_msm[k]["bss_ci95"][0] for k, _ in m_order]
    err_m_hi = [h1_msm[k]["bss_ci95"][1] - h1_msm[k]["bss_vs_b1"] for k, _ in m_order]

    bss_d = [h1_dga[k]["bss_vs_b1"] for k, _ in m_order]
    err_d_lo = [h1_dga[k]["bss_vs_b1"] - h1_dga[k]["bss_ci95"][0] for k, _ in m_order]
    err_d_hi = [h1_dga[k]["bss_ci95"][1] - h1_dga[k]["bss_vs_b1"] for k, _ in m_order]

    ax.barh(y_pos + h / 2, bss_m, height=h, xerr=[err_m_lo, err_m_hi], color="#1f618d", alpha=0.9, capsize=2.5, label="Target: Global $q_+^{\\mathrm{MSM}}$")
    ax.barh(y_pos - h / 2, bss_d, height=h, xerr=[err_d_lo, err_d_hi], color="#117a65", alpha=0.85, capsize=2.5, label="Target: Fold-Isolated $q_+^{\\mathrm{DGA}}$")
    ax.axvline(0.0, color="black", lw=1.0)
    ax.axvline(0.15, color="#c0392b", ls="--", lw=1.1, label="H1 confirmation (0.15)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([lbl for _, lbl in m_order])
    ax.set_xlabel("Matched-$R_{\\mathrm{portal}}$ Brier Skill Score $\\mathrm{BSS}_{\\mathrm{int}}$ vs. B1 Spline")
    ax.set_title("A. Out-of-Lineage Predictive Skill in $[9.5, 18.5]\\,\\mathrm{\\AA}$ (95% CI)")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.25)

    # Panel B: Target-free OOF Projected Bellman Residual E_PBR (shared witness subspace) & N2b Reversibility
    ax = axes[0, 1]
    dga_pbr_20 = dga_lags["20.0ns"]["oof_pbr_mean"]
    pbr_entries = [
        ("B1 Spline", h1_dga["B1_spline_1d"]["oof_pbr_mean"], "#95a5a6"),
        ("H1 Switches", h1_dga["H1_Switches_full_epistasis"]["oof_pbr_mean"], "#2e86c1"),
        ("Tier A (CA)", h1_dga["TierA_CA_linear"]["oof_pbr_mean"], "#5dade2"),
        ("Tier B", h1_dga["TierB_Full_sidechains"]["oof_pbr_mean"], "#1b4f72"),
        ("Tier C", h1_dga["TierC_Full_lipid_AEA"]["oof_pbr_mean"], "#148f77"),
        ("Stopped-DGA", dga_pbr_20, "#0e6251"),
        ("Oracle", h1_dga["Tier3_Oracle_HistGBR"]["oof_pbr_mean"], "#7d3c98"),
    ]
    pbr_vals = [v for _, v, _ in pbr_entries]
    bars = ax.bar(
        range(len(pbr_entries)),
        pbr_vals,
        color=[c for _, _, c in pbr_entries],
        width=0.58,
    )
    for rect, v in zip(bars, pbr_vals):
        if v is not None:
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.0005, f"{v:.4f}", ha="center", va="bottom", fontsize=7.8)
    ax.set_ylim(0.0, max(pbr_vals) * 1.18)
    ax.set_xticks(range(len(pbr_entries)))
    ax.set_xticklabels([lbl for lbl, _, _ in pbr_entries], rotation=22, ha="right")
    ax.set_ylabel("Out-of-Sample Bellman Residual $\\mathcal{E}_{\\mathrm{PBR}}$ (Lower = Better)")
    n2b_20 = dga_lags["20.0ns"]["n2b_l2_error_mean"]
    ax.set_title(f"B. Shared-Witness Koopman $\\mathcal{{E}}_{{\\mathrm{{PBR}}}}$ (20 ns; N2b $L^2={n2b_20:.3f}$)")
    ax.grid(axis="y", alpha=0.25)

    # Panel C: Per-0.5 A Bin Conditional Skill & Spearman rho across [9.5, 18.5] A
    ax = axes[1, 0]
    bins_h1 = h1_msm["H1_Switches_full_epistasis"]["per_bin"]
    bins_tc = h1_msm["TierC_Full_lipid_AEA"]["per_bin"]
    centers_h1 = [0.5 * (b["bin_low"] + b["bin_high"]) for b in bins_h1]
    centers_tc = [0.5 * (b["bin_low"] + b["bin_high"]) for b in bins_tc]
    ax.plot(centers_h1, [b["spearman_rho"] for b in bins_h1], marker="o", lw=2.0, color="#2e86c1", label="H1 Core Switches ($\\rho$ inside $0.5\\,\\mathrm{\\AA}$ bin)")
    ax.plot(centers_tc, [b["spearman_rho"] for b in bins_tc], marker="s", lw=2.0, color="#117864", label="Tier C + Lipid/AEA ($\\rho$ inside $0.5\\,\\mathrm{\\AA}$ bin)")
    ax.axhline(0.0, color="black", ls=":", lw=1.0)
    ax.set_ylim(-0.06, 0.92)
    ax.set_xlabel("Matched $R_{\\mathrm{portal}}$ Stratum Center ($\\mathrm{\\AA}$, bin width $= 0.5\\,\\mathrm{\\AA}$)")
    ax.set_ylabel("Within-Bin Spearman $\\rho(\\hat{q}_+, q_+^{\\mathrm{MSM}})$")
    ax.set_title("C. Orthogonal Committor Ranking Inside Narrow $0.5\\,\\mathrm{\\AA}$ Slices")
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, edgecolor="none", loc="lower right")
    ax.grid(alpha=0.25)

    # Panel D: Add-One Microswitch & Environmental Family Decomposition
    ax = axes[1, 1]
    fam_labels = [
        ("TM2_TM4_F155_F237", "TM2–TM4 (F155/F237)"),
        ("Polar_NPxxY_DRY_D163_N393_Y294_Y397", "Polar/NPxxY/DRY"),
        ("TwinToggle_F200_W356", "Twin Toggle (F200/W356)"),
        ("Portal_Vloop_F177_F379_M103", "Portal/V-loop (F177/F379)"),
        ("Neuronal_Lipid_Ceramide_Shell", "Neuronal Lipid/Ceramide"),
        ("AEA_Internal_Orientation", "AEA Head/Tail Orientation"),
    ]
    fy = np.arange(len(fam_labels))
    fbss = [abl[k]["add_one_bss_vs_b1"] for k, _ in fam_labels]
    flo = [abl[k]["add_one_bss_vs_b1"] - abl[k]["add_one_bss_ci95"][0] for k, _ in fam_labels]
    fhi = [abl[k]["add_one_bss_ci95"][1] - abl[k]["add_one_bss_vs_b1"] for k, _ in fam_labels]
    fcols = ["#7f8c8d", "#5dade2", "#2874a6", "#1f618d", "#d68910", "#117864"]
    ax.barh(fy, fbss, xerr=[flo, fhi], color=fcols, height=0.6, capsize=3)
    ax.axvline(0.0, color="black", lw=1.0)
    ax.set_yticks(fy)
    ax.set_yticklabels([lbl for _, lbl in fam_labels])
    ax.set_xlabel("Add-One Gain $\\Delta\\mathrm{BSS}_{\\mathrm{int}}$ over 1D $R_{\\mathrm{portal}}$ Spline")
    ax.set_title("D. Add-One Skill by Switch & Lipid Family")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    out_p = fig_dir / "fig2_h1_matched_slice_gating.png"
    fig.savefig(out_p)
    plt.close(fig)
    return out_p


def plot_fig3_h2_memory_collapse(metrics: dict, fig_dir: Path) -> Path:
    h2_msm = metrics["H2_mori_zwanzig_memory_collapse"]["q_msm_phys"]
    h2_dga = metrics["H2_mori_zwanzig_memory_collapse"]["q_dga_oof"]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.9))
    lag_keys = ["0.5ns", "1.0ns", "2.0ns", "5.0ns", "10.0ns"]
    lags_x = [0.5, 1.0, 2.0, 5.0, 10.0]

    tier_plot_cfg = [
        ("Step0_ReceptorCA_only", "Step 0: Receptor CA Only (No Ligand)", "#922b21", "o"),
        ("TierA_CA_backbone", "Tier A: CA Backbone + 1D R_portal", "#d35400", "s"),
        ("TierB_Sidechains", "Tier B: + Sidechain Rotamers & Switches", "#2874a6", "^"),
        ("TierB_plus_LipidsOnly", "Tier B + Neuronal Lipids Only", "#7d3c98", "D"),
        ("TierC_Full_Lipid_AEA", "Tier C: + Explicit Lipids & AEA Pose", "#117864", "P"),
    ]

    # Panel A: Memory curves on q_msm_phys
    ax = axes[0]
    for t_key, lbl, col, mk in tier_plot_cfg:
        md = h2_msm["memory_by_tier_and_lag"][t_key]
        ys = [md[lk]["delta_ell_mem"] * 1000.0 for lk in lag_keys]
        lo = [(md[lk]["delta_ell_mem"] - md[lk]["delta_ell_ci"]["ci_low"]) * 1000.0 for lk in lag_keys]
        hi = [(md[lk]["delta_ell_ci"]["ci_high"] - md[lk]["delta_ell_mem"]) * 1000.0 for lk in lag_keys]
        ax.errorbar(lags_x, ys, yerr=[lo, hi], marker=mk, lw=1.9, color=col, capsize=2.5, label=lbl)
    n2a_q95 = [h2_msm["memory_by_tier_and_lag"]["TierA_CA_backbone"][lk]["n2a_null_q95"] * 1000.0 for lk in lag_keys]
    ax.fill_between(lags_x, 0.0, np.maximum(0.0, n2a_q95), color="#bdc3c7", alpha=0.4, label="N2a Shuffle 95% Null")
    ax.axhline(0.0, color="black", ls=":", lw=1.0)
    ax.set_xlabel("History Lag $\\delta t$ (ns)")
    ax.set_ylabel("Memory Gain $\\Delta\\bar{\\ell}_{\\mathrm{mem}}(\\delta t)$ ($\\times 10^{-3}$ nats/frame)")
    ax.set_title("A. Mori–Zwanzig Memory ($q_+^{\\mathrm{MSM}}$)")
    ax.legend(frameon=False, fontsize=7.6, loc="best")
    ax.grid(alpha=0.25)

    # Panel B: Memory curves on fold-isolated q_dga_oof
    ax = axes[1]
    for t_key, lbl, col, mk in tier_plot_cfg:
        md = h2_dga["memory_by_tier_and_lag"][t_key]
        ys = [md[lk]["delta_ell_mem"] * 1000.0 for lk in lag_keys]
        lo = [(md[lk]["delta_ell_mem"] - md[lk]["delta_ell_ci"]["ci_low"]) * 1000.0 for lk in lag_keys]
        hi = [(md[lk]["delta_ell_ci"]["ci_high"] - md[lk]["delta_ell_mem"]) * 1000.0 for lk in lag_keys]
        ax.errorbar(lags_x, ys, yerr=[lo, hi], marker=mk, lw=1.9, color=col, capsize=2.5, label=lbl)
    n2a_dga_q95 = [h2_dga["memory_by_tier_and_lag"]["TierA_CA_backbone"][lk]["n2a_null_q95"] * 1000.0 for lk in lag_keys]
    ax.fill_between(lags_x, 0.0, np.maximum(0.0, n2a_dga_q95), color="#bdc3c7", alpha=0.4)
    ax.axhline(0.0, color="black", ls=":", lw=1.0)
    ax.set_xlabel("History Lag $\\delta t$ (ns)")
    ax.set_ylabel("Memory Gain $\\Delta\\bar{\\ell}_{\\mathrm{mem}}(\\delta t)$ ($\\times 10^{-3}$ nats/frame)")
    ax.set_title("B. Mori–Zwanzig Memory (Fold-Isolated $q_+^{\\mathrm{DGA}}$)")
    ax.grid(alpha=0.25)

    # Panel C: Residual 10 ns Memory Retention Across the 5-Tier Hierarchy (MSM vs DGA)
    ax = axes[2]
    short_tiers = ["Step 0\n(Rec CA)", "Tier A\n(CA + R)", "Tier B\n(+Rotamers)", "Tier B\n(+Lipids)", "Tier C\n(+Lipid/AEA)"]
    tx = np.arange(len(tier_plot_cfg))
    w_bar = 0.36
    msm_10 = [h2_msm["memory_by_tier_and_lag"][tk]["10.0ns"]["delta_ell_mem"] * 1000.0 for tk, _, _, _ in tier_plot_cfg]
    dga_10 = [h2_dga["memory_by_tier_and_lag"][tk]["10.0ns"]["delta_ell_mem"] * 1000.0 for tk, _, _, _ in tier_plot_cfg]
    rho_0c_msm = h2_msm["collapse_summary_by_lag"]["10.0ns"]["rho_mem_Step0_to_TierC"]
    rho_ac_dga = h2_dga["collapse_summary_by_lag"]["10.0ns"]["rho_mem_TierA_to_TierC"]
    ax.bar(tx - w_bar / 2, msm_10, width=w_bar, color="#1f618d", alpha=0.9, label=f"Discrete $q_+^{{\\mathrm{{MSM}}}}$ ($\\rho_{{0\\to C}}={rho_0c_msm:.2f}$)")
    ax.bar(tx + w_bar / 2, dga_10, width=w_bar, color="#117a65", alpha=0.9, label=f"Continuous $q_+^{{\\mathrm{{DGA}}}}$ ($\\rho_{{A\\to C}}={rho_ac_dga:.2f}$)")
    ax.axhline(0.0, color="black", lw=1.0)
    ax.set_xticks(tx)
    ax.set_xticklabels(short_tiers, fontsize=8.3)
    ax.set_ylabel("Residual $\\delta t=10$ ns Memory ($\\times 10^{-3}$ nats/frame)")
    ax.set_title("C. Two-Step Memory Collapse at $\\delta t = 10$ ns")
    ax.legend(frameon=False, fontsize=8.0, loc="upper right")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    out_p = fig_dir / "fig3_h2_mori_zwanzig_memory_collapse.png"
    fig.savefig(out_p)
    plt.close(fig)
    return out_p


def plot_fig4_biophysical_mechanism(df: pd.DataFrame, df_oof: pd.DataFrame, fig_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.4))
    m_order = [
        (1, "Bulk1\n(U2_Mem)", "#1abc9c"),
        (0, "Portal\n(I1_TM17)", "#f39c12"),
        (4, "Vestibule\n(I2_AB)", "#9b59b6"),
        (5, "Orthosteric\n(B1)", "#1f2d3d"),
    ]

    # Panel A: AEA Head-minus-Tail Depth Orientation across Portal -> Vestibule -> B1
    ax = axes[0, 0]
    data_a = [df.loc[df["macrostate_id"] == m_id, "env_aea_head_minus_tail_depth_A"].to_numpy() for m_id, _, _ in m_order]
    bp = ax.boxplot(data_a, patch_artist=True, showfliers=False, widths=0.52)
    for patch, (_, _, col) in zip(bp["boxes"], m_order):
        patch.set_facecolor(col)
        patch.set_alpha(0.78)
    ax.axhline(0.0, color="#c0392b", ls="--", lw=1.2, label="Tail-first (>0) vs Head-first (<0)")
    ax.set_xticks(range(1, len(m_order) + 1))
    ax.set_xticklabels([lbl for _, lbl, _ in m_order])
    ax.set_ylabel("AEA Head $-$ Tail Pocket Depth ($\\mathrm{\\AA}$)")
    ax.set_title("A. Tail-First Anandamide Insertion Across Binding States")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", alpha=0.25)

    # Panel B: Neuronal Lipid/Ceramide Wedging into TM1-TM7 Portal & F177-F379 Gate
    ax = axes[0, 1]
    pos = np.arange(len(m_order))
    lip_med = [df.loc[df["macrostate_id"] == m_id, "env_lip_tm17_portal_min_A"].median() for m_id, _, _ in m_order]
    lip_q25 = [np.percentile(df.loc[df["macrostate_id"] == m_id, "env_lip_tm17_portal_min_A"], 25) for m_id, _, _ in m_order]
    lip_q75 = [np.percentile(df.loc[df["macrostate_id"] == m_id, "env_lip_tm17_portal_min_A"], 75) for m_id, _, _ in m_order]

    gate_med = [df.loc[df["macrostate_id"] == m_id, "sc_portal_f177_f379_min_A"].median() for m_id, _, _ in m_order]
    gate_q25 = [np.percentile(df.loc[df["macrostate_id"] == m_id, "sc_portal_f177_f379_min_A"], 25) for m_id, _, _ in m_order]
    gate_q75 = [np.percentile(df.loc[df["macrostate_id"] == m_id, "sc_portal_f177_f379_min_A"], 75) for m_id, _, _ in m_order]

    ax.errorbar(pos - 0.1, lip_med, yerr=[np.array(lip_med) - lip_q25, np.array(lip_q75) - lip_med], fmt="o-", lw=2.0, color="#d35400", capsize=3, label="Neuronal Lipid–TM1/TM7 Min Dist")
    ax.errorbar(pos + 0.1, gate_med, yerr=[np.array(gate_med) - gate_q25, np.array(gate_q75) - gate_med], fmt="s-", lw=2.0, color="#1f618d", capsize=3, label="Portal Gate F177–F379 Min Dist")
    ax.set_xticks(pos)
    ax.set_xticklabels([lbl for _, lbl, _ in m_order])
    ax.set_ylabel("Heavy-Atom Distance ($\\mathrm{\\AA}$, Median & IQR)")
    ax.set_title("B. TM1–TM7 Portal Gate Opening vs. Neuronal Lipid Shell Contact")
    ax.legend(frameon=False, loc="best")
    ax.grid(alpha=0.25)

    # Panel C: Orthosteric Twin Toggle (F200-W356) vs Pre-Gi Quiescent Intracellular Switches (F155-F237, R_TM6)
    ax = axes[1, 0]
    tog_med = [df.loc[df["macrostate_id"] == m_id, "sc_f200_w356_min_A"].median() for m_id, _, _ in m_order]
    tm24_med = [df.loc[df["macrostate_id"] == m_id, "sc_f155_f237_min_A"].median() for m_id, _, _ in m_order]
    tm6_med = [df.loc[df["macrostate_id"] == m_id, "ca_r214_d338_A"].median() for m_id, _, _ in m_order]

    ax.plot(pos, tog_med, "o-", lw=2.1, color="#2e86c1", label="Twin Toggle F200–W356 Min Dist")
    ax.plot(pos, tm24_med, "^--", lw=2.0, color="#7f8c8d", label="TM2–TM4 F155–F237 Min Dist")
    ax.plot(pos, tm6_med, "s:", lw=2.0, color="#8e44ad", label="Cytoplasmic TM6 R214–D338 CA")
    ax.set_xticks(pos)
    ax.set_xticklabels([lbl for _, lbl, _ in m_order])
    ax.set_ylabel("Distance ($\\mathrm{\\AA}$, Median)")
    ax.set_title("C. Orthosteric Toggle Response vs. Pre-Gi Intracellular Quiescence")
    ax.legend(frameon=False, loc="best")
    ax.grid(alpha=0.25)

    # Panel D: Representative Reactive Trajectory Time-Series (Portal -> B1 Commitment)
    ax = axes[1, 1]
    # Pick held-out trajectory with largest sustained forward commitment gain (first 20 ns -> final 20 ns)
    fwd_gains = df_oof.groupby("cohort_idx")["q_msm_phys"].agg(
        lambda s: float(s.to_numpy()[-200:].mean() - s.to_numpy()[:200].mean())
    )
    best_cidx = int(fwd_gains.idxmax())
    sub = df_oof[df_oof["cohort_idx"] == best_cidx].sort_values("frame_idx")
    t_ns = sub["frame_idx"].to_numpy() * 0.1
    ax.plot(t_ns, sub["q_msm_phys"], lw=2.2, color="#1c2833", label="Exact Global $q_+^{\\mathrm{MSM}}$")
    ax.plot(t_ns, sub["q_msm_phys__TierC_Full_lipid_AEA"], lw=1.8, color="#117864", alpha=0.9, label="Held-Out Tier C $\\hat{q}_+$")
    ax.plot(t_ns, sub["q_msm_phys__B1_spline_1d"], lw=1.6, ls="--", color="#e67e22", alpha=0.85, label="Held-Out 1D $R_{\\mathrm{portal}}$ Spline B1")
    ax.set_ylim(-0.04, 1.05)
    ax.set_xlabel("Continuous MD Time along Held-Out Trajectory (ns)")
    ax.set_ylabel("Forward Committor Probability $q_+$")
    ax.set_title(f"D. Single Held-Out Transition Run (Cohort #{best_cidx})")
    ax.legend(frameon=False, loc="best", fontsize=8)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    out_p = fig_dir / "fig4_lipid_portal_biophysical_mechanism.png"
    fig.savefig(out_p)
    plt.close(fig)
    return out_p


def generate_all_ws2_figures(root: Path) -> list[str]:
    _setup_style()
    proc_dir = root / "data/processed/ws2"
    res_dir = root / "results/ws2"
    fig_dir = res_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df_micro = pd.read_parquet(proc_dir / "msm_800_microstates.parquet")
    df_feat = pd.read_parquet(proc_dir / "box_aea_cohort_features.parquet")
    df_oof = pd.read_parquet(proc_dir / "ws2_oof_predictions.parquet")
    metrics = json.loads((res_dir / "ws2_metrics.json").read_text())

    paths = [
        str(plot_fig1_msm_tpt(df_micro, metrics, fig_dir)),
        str(plot_fig2_h1_gating(metrics, fig_dir)),
        str(plot_fig3_h2_memory_collapse(metrics, fig_dir)),
        str(plot_fig4_biophysical_mechanism(df_feat, df_oof, fig_dir)),
    ]
    for p in paths:
        print(f"[WS2 Figures] Saved {p}")
    return paths


if __name__ == "__main__":
    generate_all_ws2_figures(Path("/Users/kavi/Documents/artico/cb1-dynamics"))
