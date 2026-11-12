"""Generate publication-grade multi-panel figures for Workstream 1 (Tier 1C).

Reads strictly from:
- /Users/kavi/Documents/artico/cb1-dynamics/data/processed/ws1/ws1_oof_predictions.parquet
- /Users/kavi/Documents/artico/cb1-dynamics/results/ws1/ws1_metrics.json
Writes figures to:
- /Users/kavi/Documents/artico/cb1-dynamics/results/ws1/figures/
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/Users/kavi/Documents/artico/cb1-dynamics")
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "ws1"
RESULTS_DIR = PROJECT_ROOT / "results" / "ws1"
FIG_DIR = RESULTS_DIR / "figures"

# Colorblind-safe Okabe-Ito palette
COL_AGO = "#0072B2"   # Deep blue
COL_THC = "#D55E00"   # Vermilion / orange-red
COL_DGA = "#009E73"   # Bluish green
COL_PT = "#CC79A7"    # Reddish purple
COL_B1 = "#7F7F7F"    # Neutral gray
COL_GBR = "#E69F00"   # Gold


def _set_pub_style() -> None:
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
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )


def plot_fig1_landscape_and_matched_slice(df_oof: pd.DataFrame, metrics: Dict) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)

    cryst = metrics["data_audit"]["crystal_references_angstrom"]
    r_tm6 = df_oof["r_tm6_angstrom"].to_numpy()
    d_tog = df_oof["d_toggle_angstrom"].to_numpy()
    d_npx = df_oof["d_npxxy_angstrom"].to_numpy()
    q_glob = df_oof["q_dga_global_oof"].to_numpy()
    is_ago = df_oof["ligand"].to_numpy() == "full_agonist"

    # Subsample deterministically for crisp hexbin/scatter rendering
    hb1 = axes[0].hexbin(
        r_tm6, d_tog, C=q_glob, reduce_C_function=np.mean, gridsize=55, cmap="viridis", vmin=0.0, vmax=1.0, mincnt=3
    )
    axes[0].axvline(11.8, color="white", ls="--", lw=1.2)
    axes[0].axvline(15.2, color="white", ls="--", lw=1.2)
    axes[0].axvspan(12.5, 14.5, color="#F0E442", alpha=0.16, label="Matched slice [12.5, 14.5] Å")
    axes[0].scatter(
        [cryst["inactive"]["r_tm6_angstrom"]],
        [cryst["inactive"]["d_toggle_angstrom"]],
        marker="^",
        s=90,
        c="#D55E00",
        edgecolors="black",
        zorder=5,
        label="Inactive crystal ref",
    )
    axes[0].scatter(
        [cryst["active_like"]["r_tm6_angstrom"]],
        [cryst["active_like"]["d_toggle_angstrom"]],
        marker="o",
        s=90,
        c="#56B4E9",
        edgecolors="black",
        zorder=5,
        label="Active crystal ref",
    )
    axes[0].set_xlabel(r"$R_{\mathrm{TM6}} = d(\mathrm{R214}^{3.50}\text{–}\mathrm{K343}^{6.35})$ (Å)")
    axes[0].set_ylabel(r"Toggle switch $d(\mathrm{W356}^{6.48}\text{–}\mathrm{D163}^{2.50})$ (Å)")
    axes[0].set_title("A. Test-Fold Committor on Toggle vs. TM6")
    axes[0].legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9)
    cb1 = fig.colorbar(hb1, ax=axes[0], pad=0.02)
    cb1.set_label(r"Test-fold $q_+^{\mathrm{DGA,global}}$")

    hb2 = axes[1].hexbin(
        r_tm6, d_npx, C=q_glob, reduce_C_function=np.mean, gridsize=55, cmap="viridis", vmin=0.0, vmax=1.0, mincnt=3
    )
    axes[1].axvline(11.8, color="black", ls="--", lw=1.0, alpha=0.6)
    axes[1].axvline(15.2, color="black", ls="--", lw=1.0, alpha=0.6)
    axes[1].axvspan(12.5, 14.5, color="#F0E442", alpha=0.16)
    axes[1].scatter(
        [cryst["inactive"]["r_tm6_angstrom"]],
        [cryst["inactive"]["d_npxxy_angstrom"]],
        marker="^",
        s=90,
        c="#D55E00",
        edgecolors="black",
        zorder=5,
    )
    axes[1].scatter(
        [cryst["active_like"]["r_tm6_angstrom"]],
        [cryst["active_like"]["d_npxxy_angstrom"]],
        marker="o",
        s=90,
        c="#56B4E9",
        edgecolors="black",
        zorder=5,
    )
    axes[1].set_xlabel(r"$R_{\mathrm{TM6}} = d(\mathrm{R214}^{3.50}\text{–}\mathrm{K343}^{6.35})$ (Å)")
    axes[1].set_ylabel(r"NPxxY coupling $d(\mathrm{Y153}^{2.40}\text{–}\mathrm{Y397}^{7.53})$ (Å)")
    axes[1].set_title("B. Test-Fold Committor on NPxxY vs. TM6")
    cb2 = fig.colorbar(hb2, ax=axes[1], pad=0.02)
    cb2.set_label(r"Test-fold $q_+^{\mathrm{DGA,global}}$")

    # Panel C: Matched intermediate R_TM6 slice [12.5, 14.5] A bifurcation
    in_sl = (r_tm6 >= 12.5) & (r_tm6 <= 14.5)
    mean_q_thc = float(np.mean(q_glob[in_sl & (~is_ago)]))
    mean_q_ago = float(np.mean(q_glob[in_sl & is_ago]))
    bins_tog = np.linspace(7.2, 12.4, 52)
    axes[2].hist(
        d_tog[in_sl & (~is_ago)],
        bins=bins_tog,
        density=True,
        alpha=0.65,
        color=COL_THC,
        label=rf"$\Delta^9$-THC ($R_{{\mathrm{{TM6}}}}\in[12.5,14.5]$ Å, $\bar{{q}}_+={mean_q_thc:.2f}$)",
    )
    axes[2].hist(
        d_tog[in_sl & is_ago],
        bins=bins_tog,
        density=True,
        alpha=0.65,
        color=COL_AGO,
        label=rf"Full Agonist ($R_{{\mathrm{{TM6}}}}\in[12.5,14.5]$ Å, $\bar{{q}}_+={mean_q_ago:.2f}$)",
    )
    axes[2].axvline(cryst["inactive"]["d_toggle_angstrom"], color=COL_THC, ls=":", lw=1.5)
    axes[2].axvline(cryst["active_like"]["d_toggle_angstrom"], color=COL_AGO, ls=":", lw=1.5)
    axes[2].set_xlabel(r"Toggle switch $d(\mathrm{W356}^{6.48}\text{–}\mathrm{D163}^{2.50})$ at matched $R_{\mathrm{TM6}}$ (Å)")
    axes[2].set_ylabel("Probability density")
    axes[2].set_title("C. Matched-Slice Bifurcation (12.5–14.5 Å)")
    axes[2].legend(loc="upper left", frameon=True)

    out_path = FIG_DIR / "fig1_ws1_committor_landscape_and_matched_slice.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_fig2_operator_and_h1_benchmark(df_oof: pd.DataFrame, metrics: Dict) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)

    # Panel A: Held-out Test PBR & N2b across lags
    lags_dict = metrics["regime_I_global_full_activation_11p8_to_15p2"]["dga_lag_sweep"]["lags"]
    lag_vals = sorted(int(k) for k in lags_dict.keys())
    pbr_b1 = [lags_dict[str(k)]["mean_test_pbr_b1_1cv"] for k in lag_vals]
    pbr_pt = [lags_dict[str(k)]["mean_test_pbr_pt_3cv"] for k in lag_vals]
    pbr_dga = [lags_dict[str(k)]["mean_test_pbr_dga_3cv"] for k in lag_vals]
    n2b_err = [lags_dict[str(k)]["mean_n2b_reversibility_l2_err"] for k in lag_vals]

    axes[0].plot(lag_vals, pbr_b1, "o-", color=COL_B1, lw=2.0, label="B1: 1D $R_{\\mathrm{TM6}}$ spline (PBR)")
    axes[0].plot(lag_vals, pbr_pt, "s-", color=COL_PT, lw=2.0, label="H1: 3-CV Peters–Trout (PBR)")
    axes[0].plot(lag_vals, pbr_dga, "^-", color=COL_DGA, lw=2.0, label="3-CV Stopped DGA (PBR)")
    axes[0].plot(lag_vals, n2b_err, "d--", color=COL_AGO, lw=1.6, label="N2b Reversibility $\\|q_+ + q_- - 1\\|_{L^2}$")
    axes[0].set_xlabel("Operator lag steps $\\tau$ (frames)")
    axes[0].set_ylabel("Out-of-sample operator error")
    axes[0].set_title("A. Model-Free Test-Fold Bellman Residual & N2b")
    axes[0].legend(loc="best", frameon=True)

    # Panel B: H1 BSS_int comparison across Global vs Within-Agonist vs Within-THC
    reg_glob = metrics["regime_I_global_full_activation_11p8_to_15p2"]["h1_benchmark"]
    reg_ago = metrics["regime_II_within_ligand_natural_transitions"]["full_agonist_13p2_to_15p2"]["h1_benchmark"]
    reg_thc = metrics["regime_II_within_ligand_natural_transitions"]["delta9_thc_11p8_to_13p5"]["h1_benchmark"]

    categories = ["Global Pooled\n[12.5, 14.5] Å", "Full Agonist Only\n[13.7, 14.7] Å", "$\\Delta^9$-THC Only\n[12.2, 13.1] Å"]
    x_pos = np.arange(len(categories))
    width = 0.25

    for m_idx, (m_key, m_lbl, col) in enumerate(
        [
            ("pt_linear_3cv", "PT Linear (3-CV)", "#56B4E9"),
            ("pt_epistasis_3cv", "PT + Epistasis (3-CV)", COL_PT),
            ("oracle_hist_gbr_3cv", "Oracle HistGBR (3-CV)", COL_GBR),
        ]
    ):
        pts = [reg[m_key]["bss_vs_b1"]["point"] for reg in (reg_glob, reg_ago, reg_thc)]
        lo_err = [max(0.0, reg[m_key]["bss_vs_b1"]["point"] - reg[m_key]["bss_vs_b1"]["ci_low"]) for reg in (reg_glob, reg_ago, reg_thc)]
        hi_err = [max(0.0, reg[m_key]["bss_vs_b1"]["ci_high"] - reg[m_key]["bss_vs_b1"]["point"]) for reg in (reg_glob, reg_ago, reg_thc)]
        axes[1].bar(
            x_pos + (m_idx - 1) * width,
            pts,
            width=width,
            yerr=[lo_err, hi_err],
            capsize=3,
            color=col,
            edgecolor="black",
            linewidth=0.6,
            label=m_lbl,
        )

    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(categories)
    axes[1].set_ylim(-0.30, 1.08)
    axes[1].set_ylabel(r"Matched-slice Brier Skill Score $\mathrm{BSS}_{\mathrm{int}}$ vs. B1")
    axes[1].set_title("B. H1 Predictive Gain: Global vs. Within-Ligand")
    axes[1].legend(loc="upper right", frameon=True)

    # Panel C: Calibration reliability curve on Global matched slice [12.5, 14.5] A
    cal_pt = reg_glob["pt_epistasis_3cv"]["calibration"]
    r_tm6 = df_oof["r_tm6_angstrom"].to_numpy()
    in_sl = (r_tm6 >= 12.5) & (r_tm6 <= 14.5)
    q_ref_sl = df_oof.loc[in_sl, "q_dga_global_oof"].to_numpy()
    q_b1_sl = df_oof.loc[in_sl, "q_b1_global_oof"].to_numpy()

    # Compute B1 calibration bins for visual comparison
    edges = np.linspace(0.0, 1.0, 9)
    b_idx = np.clip(np.digitize(q_b1_sl, edges) - 1, 0, 7)
    b1_pm, b1_om = [], []
    for b in range(8):
        m = b_idx == b
        if np.sum(m) >= 5:
            b1_pm.append(float(np.mean(q_b1_sl[m])))
            b1_om.append(float(np.mean(q_ref_sl[m])))

    axes[2].plot([0, 1], [0, 1], "k--", lw=1.2, label="Ideal calibration")
    axes[2].plot(b1_pm, b1_om, "o-", color=COL_B1, lw=1.8, label="B1: 1D $R_{\\mathrm{TM6}}$ spline")
    axes[2].plot(
        cal_pt["pred_mean"],
        cal_pt["obs_mean"],
        "s-",
        color=COL_PT,
        lw=2.2,
        label=f"H1 PT+Epistasis (ECE={cal_pt['ece']:.3f})",
    )
    axes[2].set_xlim(-0.02, 1.02)
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].set_xlabel(r"Predicted committor $\hat{q}_+$ on $[12.5, 14.5]$ Å")
    axes[2].set_ylabel(r"Observed test-fold $q_+^{\mathrm{DGA,test}}$")
    axes[2].set_title("C. Fold-Isolated Reliability Curve on Matched Slice")
    axes[2].legend(loc="upper left", frameon=True)

    out_path = FIG_DIR / "fig2_ws1_operator_validation_and_h1_benchmark.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_fig3_memory_and_segmentation(metrics: Dict) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)

    # Panel A: Autocorrelation decay curves
    ens = metrics["data_audit"]["ensembles"]
    lags_acf = sorted(int(k) for k in ens["full_agonist"]["acf_curves"]["r_tm6"].keys())
    axes[0].plot(
        lags_acf,
        [ens["full_agonist"]["acf_curves"]["r_tm6"][str(k)] for k in lags_acf],
        "o-",
        color=COL_AGO,
        lw=2.0,
        label="Agonist $R_{\\mathrm{TM6}}$",
    )
    axes[0].plot(
        lags_acf,
        [ens["delta9_thc"]["acf_curves"]["r_tm6"][str(k)] for k in lags_acf],
        "s-",
        color=COL_THC,
        lw=2.0,
        label="$\\Delta^9$-THC $R_{\\mathrm{TM6}}$",
    )
    axes[0].plot(
        lags_acf,
        [ens["full_agonist"]["acf_curves"]["d_npxxy"][str(k)] for k in lags_acf],
        "^--",
        color=COL_AGO,
        alpha=0.7,
        label="Agonist NPxxY",
    )
    axes[0].plot(
        lags_acf,
        [ens["delta9_thc"]["acf_curves"]["d_npxxy"][str(k)] for k in lags_acf],
        "v--",
        color=COL_THC,
        alpha=0.7,
        label="$\\Delta^9$-THC NPxxY",
    )
    axes[0].axhline(0.0, color="black", lw=0.8)
    axes[0].axvline(500, color="#7F7F7F", ls=":", lw=1.4, label="500-frame CV buffer")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Lag $\\Delta k$ (frames, log scale)")
    axes[0].set_ylabel("Autocorrelation $C(\\Delta k)$")
    axes[0].set_title("A. Coordinate Decorrelation & 500-Frame CV Buffer")
    axes[0].legend(loc="upper right", frameon=True)

    # Panel B: Jump-threshold sensitivity sweep
    sw = metrics["sensitivity_analyses"]["jump_threshold_sweep"]
    labels = ["z5p0_primary", "tm6_2cv_0p15nm", "3cv_0p20nm", "3cv_0p15nm", "3cv_0p12nm", "3cv_0p10nm"]
    disp_lbls = ["z>5.0\n(Primary)", "2-CV\n0.15nm", "3-CV\n0.20nm", "3-CV\n0.15nm", "3-CV\n0.12nm", "3-CV\n0.10nm"]
    bss_vals = [sw[k]["bss_int_pt_vs_b1"] for k in labels]
    pbr_dga = [sw[k]["test_pbr_dga_3cv"] for k in labels]
    pbr_b1 = [sw[k]["test_pbr_b1_1cv"] for k in labels]

    xp = np.arange(len(labels))
    ax2_twin = axes[1].twinx()
    ax2_twin.spines["top"].set_visible(False)
    b1_bars = axes[1].bar(xp - 0.18, bss_vals, width=0.34, color=COL_PT, edgecolor="black", lw=0.6, label="H1 $\\mathrm{BSS}_{\\mathrm{int}}$")
    l1 = ax2_twin.plot(xp + 0.12, pbr_b1, "o--", color=COL_B1, lw=1.8, label="B1 Test PBR")
    l2 = ax2_twin.plot(xp + 0.12, pbr_dga, "^-", color=COL_DGA, lw=2.0, label="3-CV DGA Test PBR")
    axes[1].set_xticks(xp)
    axes[1].set_xticklabels(disp_lbls, fontsize=8)
    axes[1].set_ylim(0.0, 1.18)
    axes[1].set_ylabel(r"Matched-slice $\mathrm{BSS}_{\mathrm{int}}$", color=COL_PT)
    ax2_twin.set_ylim(0.0, 0.115)
    ax2_twin.set_ylabel("Held-out Test PBR $\\mathcal{E}_{\\mathrm{PBR}}$", color=COL_DGA)
    axes[1].set_title("B. Robustness Across Seam Thresholds")
    lines_all = [b1_bars] + l1 + l2
    axes[1].legend(lines_all, [h.get_label() for h in lines_all], loc="upper right", frameon=True, framealpha=0.92)

    # Panel C: H2 Mori-Zwanzig Memory Profiles (1-CV vs 3-CV)
    h2_glob = metrics["h2_mori_zwanzig_memory_step_A"]["global_pooled_slice_12p5_14p5"]["lags"]
    h2_ago = metrics["h2_mori_zwanzig_memory_step_A"]["within_full_agonist_slice_13p7_14p7"]["lags"]
    h2_thc = metrics["h2_mori_zwanzig_memory_step_A"]["within_delta9_thc_slice_12p2_13p1"]["lags"]
    m_lags = sorted(int(k) for k in h2_glob.keys())

    axes[2].plot(
        m_lags,
        [h2_glob[str(k)]["memory_1cv_rtm6_only"]["delta_ell_mem"] for k in m_lags],
        "o--",
        color=COL_B1,
        lw=2.0,
        label="Global 1-CV ($R_{\\mathrm{TM6}}$ only)",
    )
    axes[2].plot(
        m_lags,
        [h2_glob[str(k)]["memory_3cv_protein_subspace"]["delta_ell_mem"] for k in m_lags],
        "s-",
        color=COL_PT,
        lw=2.2,
        label="Global 3-CV subspace",
    )
    axes[2].plot(
        m_lags,
        [h2_ago[str(k)]["memory_3cv_protein_subspace"]["delta_ell_mem"] for k in m_lags],
        "^-",
        color=COL_AGO,
        lw=1.8,
        label="Within-Agonist 3-CV",
    )
    axes[2].plot(
        m_lags,
        [h2_thc[str(k)]["memory_3cv_protein_subspace"]["delta_ell_mem"] for k in m_lags],
        "v-",
        color=COL_THC,
        lw=1.8,
        label="Within-$\\Delta^9$-THC 3-CV",
    )
    axes[2].fill_between(
        m_lags,
        0.0,
        [max(0.0, h2_glob[str(k)]["memory_3cv_protein_subspace"]["n2a_null_q95"]) for k in m_lags],
        color="#CCCCCC",
        alpha=0.5,
        label="N2a within-cell 95% null",
    )
    axes[2].axhline(0.0, color="black", lw=0.8)
    axes[2].set_xlabel("History lag $\\delta t$ (frames)")
    axes[2].set_ylabel(r"Memory gain $\Delta\bar{\ell}_{\mathrm{mem}}$ (nats/frame)")
    axes[2].set_title("C. H2 Memory Collapse (1-CV to 3-CV)")
    axes[2].legend(loc="center right", frameon=True)

    out_path = FIG_DIR / "fig3_ws1_mori_zwanzig_memory_and_segmentation_forensics.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_fig4_h3_ligand_rotation(df_oof: pd.DataFrame, metrics: Dict) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)

    h3_nat = metrics["h3_ligand_bottleneck_rotation"]["natural_within_ligand_transitions"]
    h3_sh = metrics["h3_ligand_bottleneck_rotation"]["shared_boundary_window_12p8_to_13p8"]

    # Panel A: Orthogonal microswitch gradient components (d_toggle, d_npxxy) in raw Å^-1 units
    orth_feats = ["Toggle\n$d(\\mathrm{W356\\text{–}D163})$", "NPxxY\n$d(\\mathrm{Y153\\text{–}Y397})$"]
    orth_keys = ["d_toggle_angstrom", "d_npxxy_angstrom"]
    xp = np.arange(2)
    w_ago = [h3_nat["w_agonist_raw_per_angstrom"][k] for k in orth_keys]
    w_thc = [h3_nat["w_thc_raw_per_angstrom"][k] for k in orth_keys]

    axes[0].bar(xp - 0.17, w_ago, width=0.32, color=COL_AGO, edgecolor="black", lw=0.7, label="Full Agonist ($13.2\\to 15.2$ Å)")
    axes[0].bar(xp + 0.17, w_thc, width=0.32, color=COL_THC, edgecolor="black", lw=0.7, label="$\\Delta^9$-THC ($11.8\\to 13.5$ Å)")
    axes[0].axhline(0.0, color="black", lw=0.9)
    axes[0].set_xticks(xp)
    axes[0].set_xticklabels(orth_feats)
    axes[0].set_ylabel(r"Mean logit committor gradient $\partial \mathrm{logit}\,\hat{q}_+ / \partial \xi_k$ ($\mathrm{\AA}^{-1}$)")
    axes[0].set_title("A. Conserved Within-Well Orthogonal Couplings")
    axes[0].legend(loc="lower right", frameon=True)

    # Panel B: Scale-invariant S_cos comparison against within-ligand split-half reliability ceilings
    rel_ago = h3_nat["within_ligand_split_half_orth_reliability"]
    rel_thc = h3_nat["within_ligand_split_half_orth_reliability"]
    sc_3cv = h3_nat["s_cos_full_3cv_epistasis"]
    sc_orth = h3_nat["s_cos_orthogonal_2cv_toggle_npxxy"]
    sc_sh_orth = h3_sh["s_cos_orthogonal_2cv_toggle_npxxy"]

    bar_labels = [
        "Agonist\nSplit-Half",
        "$\\Delta^9$-THC\nSplit-Half",
        "Cross-Ligand\nFull 3-CV",
        "Cross-Ligand\nOrth (Nat)",
        "Cross-Ligand\nOrth (Shared)",
    ]
    pts = [
        rel_ago["full_agonist_median"],
        rel_thc["delta9_thc_median"],
        sc_3cv["point"],
        sc_orth["point_epistasis"],
        sc_sh_orth["point_epistasis"],
    ]
    lows = [
        rel_ago["full_agonist_ci"][0],
        rel_thc["delta9_thc_ci"][0],
        sc_3cv["ci_low"],
        sc_orth["ci_low"],
        sc_sh_orth["ci_low"],
    ]
    highs = [
        rel_ago["full_agonist_ci"][1],
        rel_thc["delta9_thc_ci"][1],
        sc_3cv["ci_high"],
        sc_orth["ci_high"],
        sc_sh_orth["ci_high"],
    ]
    colors = [COL_AGO, COL_THC, "#7F7F7F", COL_PT, "#E69F00"]
    xb = np.arange(len(bar_labels))
    yerr_lo = [max(0.0, p - l) for p, l in zip(pts, lows)]
    yerr_hi = [max(0.0, h - p) for p, h in zip(pts, highs)]
    axes[1].bar(xb, pts, yerr=[yerr_lo, yerr_hi], capsize=3, color=colors, edgecolor="black", lw=0.7)
    axes[1].axhline(0.0, color="black", lw=0.9)
    axes[1].axhline(1.0, color="gray", ls=":", lw=1.0)
    axes[1].set_xticks(xb)
    axes[1].set_xticklabels(bar_labels, fontsize=7.8)
    axes[1].set_ylim(-1.08, 1.15)
    axes[1].set_ylabel(r"Scale-invariant gradient cosine $\mathcal{S}_{\cos}$")
    axes[1].set_title("B. H3 Cross-Ligand Gradient Cosine vs. Reliability")

    # Panel C: Within-ligand OOF committor profiles vs R_TM6 for both ensembles
    is_ago = df_oof["ligand"].to_numpy() == "full_agonist"
    r_ago = df_oof.loc[is_ago, "r_tm6_angstrom"].to_numpy()
    q_ago_w = df_oof.loc[is_ago, "q_dga_within_natural_oof"].to_numpy()
    r_thc = df_oof.loc[~is_ago, "r_tm6_angstrom"].to_numpy()
    q_thc_w = df_oof.loc[~is_ago, "q_dga_within_natural_oof"].to_numpy()

    def _binned_profile(r: np.ndarray, q: np.ndarray, lo: float, hi: float, n_bins: int = 30):
        edges = np.linspace(lo, hi, n_bins + 1)
        ctrs, mus, p10, p90 = [], [], [], []
        for k in range(n_bins):
            m = (r >= edges[k]) & (r < edges[k + 1])
            if np.sum(m) >= 15:
                ctrs.append(0.5 * (edges[k] + edges[k + 1]))
                mus.append(float(np.mean(q[m])))
                p10.append(float(np.quantile(q[m], 0.10)))
                p90.append(float(np.quantile(q[m], 0.90)))
        return np.array(ctrs), np.array(mus), np.array(p10), np.array(p90)

    c_p, m_p, l_p, h_p = _binned_profile(r_thc, q_thc_w, 10.5, 14.3, 32)
    c_a, m_a, l_a, h_a = _binned_profile(r_ago, q_ago_w, 12.2, 16.5, 32)

    axes[2].plot(c_p, m_p, "-", color=COL_THC, lw=2.2, label="$\\Delta^9$-THC partial ($11.8\\to 13.5$ Å)")
    axes[2].fill_between(c_p, l_p, h_p, color=COL_THC, alpha=0.20)
    axes[2].plot(c_a, m_a, "-", color=COL_AGO, lw=2.2, label="Full Agonist ($13.2\\to 15.2$ Å)")
    axes[2].fill_between(c_a, l_a, h_a, color=COL_AGO, alpha=0.20)
    axes[2].axvline(14.95, color=COL_THC, ls=":", lw=1.4, label="$\\Delta^9$-THC steric wall (14.95 Å)")
    axes[2].set_xlabel(r"$R_{\mathrm{TM6}} = d(\mathrm{R214}^{3.50}\text{–}\mathrm{K343}^{6.35})$ (Å)")
    axes[2].set_ylabel(r"Within-ligand OOF $q_+^{\mathrm{within}}$ (mean & 10–90% band)")
    axes[2].set_title("C. Two-Stage Sub-Transition Committor Profiles")
    axes[2].legend(loc="lower right", frameon=True)

    out_path = FIG_DIR / "fig4_ws1_h3_ligand_bottleneck_rotation.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def generate_all_ws1_figures() -> Dict[str, str]:
    _set_pub_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df_oof = pd.read_parquet(PROCESSED_DIR / "ws1_oof_predictions.parquet")
    with open(RESULTS_DIR / "ws1_metrics.json", "r", encoding="utf-8") as fh:
        metrics = json.load(fh)

    f1 = plot_fig1_landscape_and_matched_slice(df_oof, metrics)
    f2 = plot_fig2_operator_and_h1_benchmark(df_oof, metrics)
    f3 = plot_fig3_memory_and_segmentation(metrics)
    f4 = plot_fig4_h3_ligand_rotation(df_oof, metrics)

    return {
        "fig1": str(f1),
        "fig2": str(f2),
        "fig3": str(f3),
        "fig4": str(f4),
    }


if __name__ == "__main__":
    paths = generate_all_ws1_figures()
    for k, p in paths.items():
        print(f"Generated {k}: {p}")
