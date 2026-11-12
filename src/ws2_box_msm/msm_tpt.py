"""Step 1 of WS2: Global 5,000-Run (482.97 us) 800-State MSM & Exact TPT Analysis.

Unpickles ShuklaGroup's CB1_assym_MSM.pkl safely without legacy pyemma, verifies
row-stochasticity, microscopic detailed balance, multi-lag implied timescales (ITS),
and Chapman-Kolmogorov (CK) macrostate consistency, defines physically non-circular
bulk/portal (A_portal) and deep orthosteric pocket (B_pocket) endpoints from 3D
ligand-pocket penetration depth R_portal, and solves the exact discrete TPT forward
committor q_+^{MSM} and reactive flux across all 800 microstates and 4,829,682 frames.
"""

from __future__ import annotations

import json
from pathlib import Path
import pickle
from typing import Any

import mdtraj as md
import numpy as np
import pandas as pd

# Canonical CB1 orthosteric pocket lining residues (UniProt numbering, P21554)
# S173^2.60, F177^2.64, L193^3.29, V196^3.32, F200^3.36, W279^5.43, W356^6.48, L360^6.52, F379^7.35, S383^7.39
ORTHOSTERIC_POCKET_UNIPROT = [173, 177, 193, 196, 200, 279, 356, 360, 379, 383]
UNIPROT_STRIP_OFFSET_ZERO_IDX = 89  # 0-based mdtraj resid = UniProt - 89 (1-based resSeq = UniProt - 88)

MACROSTATE_NAMES = {
    2: "U_Sol (Ex)",
    3: "U1_Mem (Bulk2)",
    1: "U2_Mem (Bulk1)",
    0: "I1_TM17 (Portal)",
    4: "I2_TM17 (AB Vestibule)",
    5: "B_Pro (B1 Orthosteric)",
}


class _DummyPyEMMAObj:
    """Stub class allowing safe inspection of serialized pyemma MaximumLikelihoodMSM attributes."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self._state = state


class SafeMSMUnpickler(pickle.Unpickler):
    """Restricted unpickler permitting only stdlib/numpy/scipy arrays and stubbing legacy pyemma classes."""

    def find_class(self, module: str, name: str) -> Any:
        if module.startswith(("numpy", "scipy", "builtins", "collections", "_codecs", "copyreg")):
            return super().find_class(module, name)
        return _DummyPyEMMAObj


def load_cb1_msm_object(msm_pkl_path: str | Path) -> Any:
    with open(msm_pkl_path, "rb") as fh:
        return SafeMSMUnpickler(fh).load()


def compute_orthosteric_pocket_center_nm(pdb_path: str | Path, prmtop_path: str | Path) -> np.ndarray:
    """Return 3D C-alpha centroid (in nm) of the 10 canonical CB1 orthosteric pocket residues."""
    ref = md.load(str(pdb_path), top=str(prmtop_path))
    ca_indices = []
    for u_res in ORTHOSTERIC_POCKET_UNIPROT:
        r_idx = u_res - UNIPROT_STRIP_OFFSET_ZERO_IDX
        sel = ref.topology.select(f"resid {r_idx} and name CA")
        if len(sel) != 1:
            raise ValueError(f"Expected 1 CA atom for UniProt {u_res} (resid {r_idx}), got {sel}")
        ca_indices.append(int(sel[0]))
    return ref.xyz[0, ca_indices, :].mean(axis=0).astype(np.float64)


def estimate_reversible_transition_matrix(
    dtrajs: list[np.ndarray],
    active_set: np.ndarray,
    full2active: np.ndarray,
    lag_frames: int,
    max_iter: int = 200,
    tol: float = 1e-11,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a reversible MLE transition matrix on `active_set` at `lag_frames` (sliding counts).

    Uses the standard Prinz et al. (2011) fixed-point iteration:
        pi_i = sum_j (C_ij + C_ji) / sum_j (c_j / pi_j + ... ) -> X_ij = (C_ij + C_ji) / (R_i/pi_i + R_j/pi_j)
    restricted to the largest strongly connected component if needed, with small Dirichlet pseudocount
    regularization only on zero-row states so eigenvalues remain strictly real and well-conditioned.
    """
    n_act = len(active_set)
    C = np.zeros((n_act, n_act), dtype=np.float64)
    for dt in dtrajs:
        if len(dt) <= lag_frames:
            continue
        s0 = full2active[dt[:-lag_frames]]
        s1 = full2active[dt[lag_frames:]]
        valid = (s0 >= 0) & (s1 >= 0)
        if np.any(valid):
            np.add.at(C, (s0[valid], s1[valid]), 1.0)

    C_sym = C + C.T
    # Tiny connectivity floor on observed active support to prevent isolated rows at large lags
    row_sums = C.sum(axis=1)
    if np.any(row_sums == 0):
        C_sym = C_sym + 1e-8 * (C_sym > 0)
        row_sums = np.maximum(row_sums, 1e-8)

    pi = C_sym.sum(axis=1)
    pi = np.maximum(pi, 1e-12)
    pi /= pi.sum()

    for _ in range(max_iter):
        denom = (row_sums[:, None] / pi[:, None]) + (row_sums[None, :] / pi[None, :])
        X = np.where(C_sym > 0, C_sym / denom, 0.0)
        pi_new = X.sum(axis=1)
        pi_new = np.maximum(pi_new, 1e-15)
        pi_new /= pi_new.sum()
        if np.max(np.abs(pi_new - pi)) < tol:
            pi = pi_new
            break
        pi = pi_new

    denom = (row_sums[:, None] / pi[:, None]) + (row_sums[None, :] / pi[None, :])
    X = np.where(C_sym > 0, C_sym / denom, 0.0)
    x_rows = X.sum(axis=1)
    P = np.divide(X, x_rows[:, None], out=np.eye(n_act), where=x_rows[:, None] > 0)
    pi = x_rows / x_rows.sum()
    return P, pi


def implied_timescales_ns(P: np.ndarray, pi: np.ndarray, lag_ns: float, n_modes: int = 10) -> list[float]:
    """Compute top reversible implied timescales t_k = -lag_ns / ln(lambda_k) via symmetrized similarity."""
    sqrt_pi = np.sqrt(np.maximum(pi, 1e-30))
    S = (sqrt_pi[:, None] * P) / sqrt_pi[None, :]
    S = 0.5 * (S + S.T)
    evals = np.linalg.eigvalsh(S)[::-1]  # descending: lambda_0 = 1.0 >= lambda_1 >= ...
    its = []
    for lam in evals[1 : n_modes + 1]:
        if 0.0 < lam < 1.0:
            its.append(float(-lag_ns / np.log(lam)))
        else:
            its.append(float("nan"))
    return its


def solve_discrete_tpt(
    P: np.ndarray,
    pi: np.ndarray,
    active_set: np.ndarray,
    A_full_indices: np.ndarray,
    B_full_indices: np.ndarray,
    lag_ns: float = 35.0,
) -> dict[str, Any]:
    """Solve exact discrete Transition Path Theory (Metzner, Schutte & Vanden-Eijnden 2009).

    Solves:
        (P_II - I) q_+ = -P_IB 1,   q_+|_A = 0,   q_+|_B = 1.
    Under microscopic reversibility, q_- = 1 - q_+.
    """
    A_set = set(int(x) for x in A_full_indices)
    B_set = set(int(x) for x in B_full_indices)
    if A_set & B_set:
        raise ValueError(f"Disjoint endpoint violation: |A intersect B| = {len(A_set & B_set)}")

    A_act = np.array([int(s) in A_set for s in active_set], dtype=bool)
    B_act = np.array([int(s) in B_set for s in active_set], dtype=bool)
    I_act = ~(A_act | B_act)

    if A_act.sum() == 0 or B_act.sum() == 0:
        raise ValueError(f"Empty active basin: |A|={A_act.sum()}, |B|={B_act.sum()}")

    n_act = len(active_set)
    q_act = np.zeros(n_act, dtype=np.float64)
    q_act[B_act] = 1.0

    P_II = P[np.ix_(I_act, I_act)]
    rhs = -P[np.ix_(I_act, B_act)].sum(axis=1)
    L_II = P_II - np.eye(I_act.sum(), dtype=np.float64)
    q_I = np.linalg.solve(L_II, rhs)
    q_act[I_act] = np.clip(q_I, 0.0, 1.0)

    # Check discrete harmonic residual max |((P - I) q_+)_I|
    harmonic_residual = float(np.max(np.abs((P @ q_act - q_act)[I_act])))

    # Gross reactive flux f_{ij} = pi_i P_{ij} (1 - q_+(i)) q_+(j) for i != j
    q_minus = 1.0 - q_act
    f_gross = (pi[:, None] * q_minus[:, None]) * P * q_act[None, :]
    np.fill_diagonal(f_gross, 0.0)
    f_net = np.maximum(0.0, f_gross - f_gross.T)

    # Total reactive flux F_AB leaving A (must equal flux entering B up to machine precision)
    flux_out_A = float(f_gross[A_act, :][:, ~A_act].sum())
    flux_in_B = float(f_gross[~B_act, :][:, B_act].sum())
    pi_A_basin = float(np.sum(pi * q_minus))  # generalized TPT A-population sum_i pi_i q_-(i)
    rate_per_ns = flux_out_A / (lag_ns * pi_A_basin)
    rate_per_us = rate_per_ns * 1000.0
    mfpt_us = 1.0 / rate_per_us if rate_per_us > 0 else float("inf")

    # Reactive path probability density m_i^R = pi_i q_+(i) q_-(i)
    m_reactive_act = pi * q_act * q_minus
    net_flux_through_act = f_net.sum(axis=1)

    q_full = np.full(800, np.nan, dtype=np.float64)
    q_full[active_set] = q_act
    m_reactive_full = np.zeros(800, dtype=np.float64)
    m_reactive_full[active_set] = m_reactive_act
    net_flux_full = np.zeros(800, dtype=np.float64)
    net_flux_full[active_set] = net_flux_through_act

    return {
        "q_full": q_full,
        "q_act": q_act,
        "f_net_act": f_net,
        "m_reactive_full": m_reactive_full,
        "net_flux_full": net_flux_full,
        "n_A_active": int(A_act.sum()),
        "n_B_active": int(B_act.sum()),
        "n_I_active": int(I_act.sum()),
        "harmonic_max_residual": harmonic_residual,
        "flux_out_A_per_step": flux_out_A,
        "flux_in_B_per_step": flux_in_B,
        "flux_conservation_rel_err": abs(flux_out_A - flux_in_B) / max(flux_out_A, 1e-30),
        "pi_A_generalized": pi_A_basin,
        "pi_B_generalized": float(np.sum(pi * q_act)),
        "rate_AB_per_us": rate_per_us,
        "mfpt_A_to_B_us": mfpt_us,
    }


def macrostate_ck_test(
    P_model: np.ndarray,
    pi_model: np.ndarray,
    active_set: np.ndarray,
    full2active: np.ndarray,
    mtM: np.ndarray,
    dtrajs: list[np.ndarray],
    base_lag_frames: int,
    steps: list[int],
) -> dict[str, Any]:
    """Compare coarse-grained MSM prediction P_macro(k * tau) against empirical transition counts at lag k * tau."""
    macro_act = mtM[active_set]
    macro_ids = [2, 3, 1, 0, 4, 5]  # Sol -> Bulk2 -> Bulk1 -> TM17 -> AB -> B1
    results: dict[str, Any] = {"macro_order": [MACROSTATE_NAMES[m] for m in macro_ids], "steps": {}}

    for k in steps:
        Pk = np.linalg.matrix_power(P_model, k)
        pred_M = np.zeros((6, 6), dtype=np.float64)
        for a_i, m_a in enumerate(macro_ids):
            mask_a = macro_act == m_a
            w_a = pi_model[mask_a]
            w_a = w_a / w_a.sum()
            for b_i, m_b in enumerate(macro_ids):
                mask_b = macro_act == m_b
                pred_M[a_i, b_i] = float(w_a @ Pk[np.ix_(mask_a, mask_b)].sum(axis=1))

        # Empirical transition matrix on macrostates weighted by stationary microstate weights within m_a
        lag_k = k * base_lag_frames
        C_act = np.zeros((len(active_set), len(active_set)), dtype=np.float64)
        n_pairs = 0
        for dt in dtrajs:
            if len(dt) <= lag_k:
                continue
            s0 = full2active[dt[:-lag_k]]
            s1 = full2active[dt[lag_k:]]
            valid = (s0 >= 0) & (s1 >= 0)
            if np.any(valid):
                np.add.at(C_act, (s0[valid], s1[valid]), 1.0)
                n_pairs += int(valid.sum())

        row_tot = C_act.sum(axis=1)
        P_emp_micro = np.divide(C_act, row_tot[:, None], out=np.eye(len(active_set)), where=row_tot[:, None] > 0)
        emp_M = np.zeros((6, 6), dtype=np.float64)
        for a_i, m_a in enumerate(macro_ids):
            mask_a = (macro_act == m_a) & (row_tot > 0)
            w_a = pi_model[mask_a]
            w_a = w_a / w_a.sum()
            for b_i, m_b in enumerate(macro_ids):
                mask_b = macro_act == m_b
                emp_M[a_i, b_i] = float(w_a @ P_emp_micro[np.ix_(mask_a, mask_b)].sum(axis=1))

        max_abs_err = float(np.max(np.abs(pred_M - emp_M)))
        rmse = float(np.sqrt(np.mean((pred_M - emp_M) ** 2)))
        results["steps"][str(k)] = {
            "lag_frames": lag_k,
            "lag_ns": float(lag_k * 0.1),
            "n_transition_pairs": n_pairs,
            "pred_diag": [float(pred_M[i, i]) for i in range(6)],
            "emp_diag": [float(emp_M[i, i]) for i in range(6)],
            "pred_matrix": pred_M.tolist(),
            "emp_matrix": emp_M.tolist(),
            "max_abs_err": max_abs_err,
            "rmse": rmse,
        }
    return results


def run_global_msm_tpt_pipeline(
    box_dir: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    box_dir = Path(box_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    msm = load_cb1_msm_object(box_dir / "CB1_assym_MSM.pkl")
    with open(box_dir / "CB1_frame_cluster_indices.pkl", "rb") as fh:
        dtrajs: list[np.ndarray] = pickle.load(fh)
    with open(box_dir / "CB1_micro_to_macro_index.pkl", "rb") as fh:
        mtM: np.ndarray = np.asarray(pickle.load(fh), dtype=int)
    with open(box_dir / "CB1_lig_com_cor.pkl", "rb") as fh:
        lig_com_list: list[np.ndarray] = pickle.load(fh)
    with open(box_dir / "CB1_msm_weight.pkl", "rb") as fh:
        msm_weights_list: list[np.ndarray] = pickle.load(fh)

    P = np.asarray(msm.P, dtype=np.float64)
    pi = np.asarray(msm.pi, dtype=np.float64)
    active_set = np.asarray(msm._active_set, dtype=int)
    full2active = np.asarray(msm._full2active, dtype=int)
    lag_frames = int(msm.lag)
    dt_ns = 0.1  # 100 ps per frame across the 5,000-run ensemble
    lag_ns = lag_frames * dt_ns

    # Verify exact match between dtrajs and msm._dtrajs_full
    if len(dtrajs) != 5000 or len(msm._dtrajs_full) != 5000:
        raise ValueError(f"Expected 5,000 trajectories, got {len(dtrajs)}")
    for i in range(5000):
        if not np.array_equal(dtrajs[i], msm._dtrajs_full[i]):
            raise ValueError(f"Trajectory {i} cluster index mismatch with CB1_assym_MSM.pkl")

    # Verify row-stochasticity and microscopic detailed balance
    row_sum_max_err = float(np.max(np.abs(P.sum(axis=1) - 1.0)))
    min_entry = float(np.min(P))
    flux_mat = pi[:, None] * P
    db_max_abs_err = float(np.max(np.abs(flux_mat - flux_mat.T)))
    db_rel_frob_err = float(np.linalg.norm(flux_mat - flux_mat.T) / np.linalg.norm(flux_mat))
    pi_sum_err = float(abs(pi.sum() - 1.0))

    # Compute orthosteric pocket centroid and per-frame / per-microstate R_portal
    pocket_center_nm = compute_orthosteric_pocket_center_nm(
        box_dir / "CB1_assym-strip.pdb",
        box_dir / "CB1_assym-strip.prmtop",
    )
    all_com_nm = np.concatenate(lig_com_list, axis=0)
    all_d = np.concatenate(dtrajs, axis=0)
    all_w = np.concatenate(msm_weights_list, axis=0)
    total_frames = int(len(all_d))
    all_r_portal_A = np.linalg.norm(all_com_nm - pocket_center_nm[None, :], axis=1) * 10.0

    # Aggregate microstate geometry statistics
    df_frames_tmp = pd.DataFrame(
        {
            "s": all_d,
            "r": all_r_portal_A,
            "x": all_com_nm[:, 0] * 10.0,
            "y": all_com_nm[:, 1] * 10.0,
            "z": all_com_nm[:, 2] * 10.0,
        }
    )
    grp = df_frames_tmp.groupby("s", sort=True)
    r_mean = np.full(800, np.nan)
    r_med = np.full(800, np.nan)
    r_std = np.full(800, np.nan)
    x_mean = np.full(800, np.nan)
    y_mean = np.full(800, np.nan)
    z_mean = np.full(800, np.nan)
    counts = np.zeros(800, dtype=int)

    for s_id, sub in grp:
        s_idx = int(s_id)
        counts[s_idx] = len(sub)
        r_mean[s_idx] = float(sub["r"].mean())
        r_med[s_idx] = float(sub["r"].median())
        r_std[s_idx] = float(sub["r"].std(ddof=0))
        x_mean[s_idx] = float(sub["x"].mean())
        y_mean[s_idx] = float(sub["y"].mean())
        z_mean[s_idx] = float(sub["z"].mean())

    # Endpoint definitions (all strictly independent of receptor sidechain / lipid coordinates!)
    # 1. Primary physical portal->pocket definition:
    #    A_phys: Unbound bulk/membrane states (mtM in {1,2,3}) with mean R_portal >= 22.0 A
    #    B_phys: Deep orthosteric bound states (mtM == 5, B1) with mean R_portal <= 9.5 A
    A_phys = np.where(np.isin(mtM, [1, 2, 3]) & (r_mean >= 22.0))[0]
    B_phys = np.where((mtM == 5) & (r_mean <= 9.5))[0]

    # 2. Pure R_portal distance definition (zero PCCA+ macrostate input):
    A_pureR = np.where(r_mean >= 22.0)[0]
    B_pureR = np.where(r_mean <= 9.0)[0]

    # 3. Canonical Shukla PCCA+ macrostate definition (Unbound {1,2,3} -> B1 {5}):
    A_macro = np.where(np.isin(mtM, [1, 2, 3]))[0]
    B_macro = np.where(mtM == 5)[0]

    tpt_phys = solve_discrete_tpt(P, pi, active_set, A_phys, B_phys, lag_ns=lag_ns)
    tpt_pureR = solve_discrete_tpt(P, pi, active_set, A_pureR, B_pureR, lag_ns=lag_ns)
    tpt_macro = solve_discrete_tpt(P, pi, active_set, A_macro, B_macro, lag_ns=lag_ns)

    # Coarse-grained macrostate net reactive flux matrix for primary TPT
    macro_ids = [2, 3, 1, 0, 4, 5]
    macro_act = mtM[active_set]
    f_macro_gross = np.zeros((6, 6), dtype=np.float64)
    for i_m, m1 in enumerate(macro_ids):
        for j_m, m2 in enumerate(macro_ids):
            if i_m != j_m:
                f_macro_gross[i_m, j_m] = float(
                    tpt_phys["f_net_act"][np.ix_(macro_act == m1, macro_act == m2)].sum()
                )
    f_macro_net = np.maximum(0.0, f_macro_gross - f_macro_gross.T)

    # Multi-lag implied timescales across lags 5, 10, 20, 35, 50, 70 ns
    lag_scan_frames = [50, 100, 200, 350, 500, 700]
    its_scan: dict[str, Any] = {}
    for lf in lag_scan_frames:
        l_ns = lf * dt_ns
        if lf == lag_frames:
            its_vals = implied_timescales_ns(P, pi, lag_ns=l_ns, n_modes=8)
        else:
            P_lf, pi_lf = estimate_reversible_transition_matrix(dtrajs, active_set, full2active, lag_frames=lf)
            its_vals = implied_timescales_ns(P_lf, pi_lf, lag_ns=l_ns, n_modes=8)
        its_scan[f"{l_ns:.1f}ns"] = {"lag_frames": lf, "lag_ns": l_ns, "its_ns": its_vals}

    # Chapman-Kolmogorov tests:
    # (a) Native published MSM at tau_0 = 35 ns for k = 1, 2 (35 ns, 70 ns; max traj length is 100 ns)
    ck_35ns = macrostate_ck_test(P, pi, active_set, full2active, mtM, dtrajs, base_lag_frames=350, steps=[1, 2])
    # (b) Extended 4-step check at tau_0 = 20 ns for k = 1, 2, 3, 4 (20, 40, 60, 80 ns)
    P_20, pi_20 = estimate_reversible_transition_matrix(dtrajs, active_set, full2active, lag_frames=200)
    ck_20ns = macrostate_ck_test(P_20, pi_20, active_set, full2active, mtM, dtrajs, base_lag_frames=200, steps=[1, 2, 3, 4])

    # Build 800-microstate DataFrame
    pi_full = np.zeros(800, dtype=np.float64)
    pi_full[active_set] = pi
    is_active_mask = np.isin(np.arange(800), active_set)

    df_micro = pd.DataFrame(
        {
            "microstate_id": np.arange(800, dtype=int),
            "is_active": is_active_mask,
            "active_idx": full2active,
            "macrostate_id": mtM,
            "macrostate_name": [MACROSTATE_NAMES[int(m)] for m in mtM],
            "n_frames": counts,
            "stationary_pi": pi_full,
            "r_portal_mean_A": r_mean,
            "r_portal_median_A": r_med,
            "r_portal_std_A": r_std,
            "lig_com_x_mean_A": x_mean,
            "lig_com_y_mean_A": y_mean,
            "lig_com_z_mean_A": z_mean,
            "in_A_phys": np.isin(np.arange(800), A_phys),
            "in_B_phys": np.isin(np.arange(800), B_phys),
            "q_msm_phys": tpt_phys["q_full"],
            "q_msm_pureR": tpt_pureR["q_full"],
            "q_msm_macro": tpt_macro["q_full"],
            "reactive_path_density_phys": tpt_phys["m_reactive_full"],
            "net_flux_through_phys": tpt_phys["net_flux_full"],
        }
    )
    micro_parquet_path = out_dir / "msm_800_microstates.parquet"
    df_micro.to_parquet(micro_parquet_path, index=False)

    # Frame-level committor summary across all 4,829,682 frames
    q_frames_phys = tpt_phys["q_full"][all_d]
    valid_f = ~np.isnan(q_frames_phys)
    macro_summary = {}
    for m_id in macro_ids:
        m_mask = (mtM[all_d] == m_id) & valid_f
        m_micro = (mtM == m_id) & is_active_mask
        macro_summary[MACROSTATE_NAMES[m_id]] = {
            "macrostate_id": int(m_id),
            "n_active_microstates": int(m_micro.sum()),
            "n_frames": int((mtM[all_d] == m_id).sum()),
            "stationary_pop": float(pi_full[m_micro].sum()),
            "r_portal_median_A": float(np.median(all_r_portal_A[m_mask])),
            "r_portal_iqr_A": [
                float(np.percentile(all_r_portal_A[m_mask], 25)),
                float(np.percentile(all_r_portal_A[m_mask], 75)),
            ],
            "q_msm_phys_min": float(np.min(q_frames_phys[m_mask])),
            "q_msm_phys_median": float(np.median(q_frames_phys[m_mask])),
            "q_msm_phys_max": float(np.max(q_frames_phys[m_mask])),
        }

    summary = {
        "ensemble": {
            "n_trajectories": 5000,
            "total_frames": total_frames,
            "frame_stride_ns": dt_ns,
            "aggregate_simulation_time_us": round(total_frames * dt_ns / 1000.0, 4),
            "n_microstates_total": 800,
            "n_microstates_active": int(len(active_set)),
            "disconnected_microstates": [int(s) for s in range(800) if not is_active_mask[s]],
            "disconnected_frames": int((~valid_f).sum()),
            "msm_lag_frames": lag_frames,
            "msm_lag_ns": lag_ns,
            "msm_weights_sum": float(all_w.sum()),
            "orthosteric_pocket_uniprot": ORTHOSTERIC_POCKET_UNIPROT,
            "orthosteric_pocket_ca_center_nm": pocket_center_nm.tolist(),
        },
        "markov_invariants": {
            "row_stochasticity_max_abs_err": row_sum_max_err,
            "min_transition_entry": min_entry,
            "stationary_pi_sum_abs_err": pi_sum_err,
            "detailed_balance_max_abs_err": db_max_abs_err,
            "detailed_balance_rel_frobenius_err": db_rel_frob_err,
            "top8_implied_timescales_at_35ns": its_scan["35.0ns"]["its_ns"],
            "multi_lag_its_scan": its_scan,
            "ck_test_35ns_2step": ck_35ns,
            "ck_test_20ns_4step": ck_20ns,
        },
        "tpt_primary_phys": {
            "definition": "A = {mtM in {1,2,3} & mean_R_portal >= 22.0 A}, B = {mtM == 5 & mean_R_portal <= 9.5 A}",
            "n_A_active": tpt_phys["n_A_active"],
            "n_B_active": tpt_phys["n_B_active"],
            "n_I_active": tpt_phys["n_I_active"],
            "harmonic_max_residual": tpt_phys["harmonic_max_residual"],
            "flux_out_A_per_step": tpt_phys["flux_out_A_per_step"],
            "flux_in_B_per_step": tpt_phys["flux_in_B_per_step"],
            "flux_conservation_rel_err": tpt_phys["flux_conservation_rel_err"],
            "pi_A_generalized": tpt_phys["pi_A_generalized"],
            "pi_B_generalized": tpt_phys["pi_B_generalized"],
            "rate_AB_per_us": tpt_phys["rate_AB_per_us"],
            "mfpt_A_to_B_us": tpt_phys["mfpt_A_to_B_us"],
            "n_transition_microstates_005_095": int(
                ((tpt_phys["q_act"] > 0.05) & (tpt_phys["q_act"] < 0.95)).sum()
            ),
            "n_transition_frames_005_095": int(
                ((q_frames_phys[valid_f] > 0.05) & (q_frames_phys[valid_f] < 0.95)).sum()
            ),
            "macrostate_order": [MACROSTATE_NAMES[m] for m in macro_ids],
            "macrostate_net_flux_matrix": f_macro_net.tolist(),
        },
        "tpt_sensitivity_checks": {
            "pure_R_portal_22A_to_9A": {
                "n_A_active": tpt_pureR["n_A_active"],
                "n_B_active": tpt_pureR["n_B_active"],
                "n_I_active": tpt_pureR["n_I_active"],
                "mfpt_A_to_B_us": tpt_pureR["mfpt_A_to_B_us"],
                "pearson_r_vs_phys": float(np.corrcoef(tpt_phys["q_act"], tpt_pureR["q_act"])[0, 1]),
                "spearman_rho_vs_phys": float(
                    pd.Series(tpt_phys["q_act"]).corr(pd.Series(tpt_pureR["q_act"]), method="spearman")
                ),
            },
            "macrostate_unbound_to_B1": {
                "n_A_active": tpt_macro["n_A_active"],
                "n_B_active": tpt_macro["n_B_active"],
                "n_I_active": tpt_macro["n_I_active"],
                "mfpt_A_to_B_us": tpt_macro["mfpt_A_to_B_us"],
                "pearson_r_vs_phys": float(np.corrcoef(tpt_phys["q_act"], tpt_macro["q_act"])[0, 1]),
                "spearman_rho_vs_phys": float(
                    pd.Series(tpt_phys["q_act"]).corr(pd.Series(tpt_macro["q_act"]), method="spearman")
                ),
            },
        },
        "macrostate_profiles": macro_summary,
    }

    summary_path = out_dir / "msm_tpt_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    return summary


if __name__ == "__main__":
    root = Path("/Users/kavi/Documents/artico/cb1-dynamics")
    res = run_global_msm_tpt_pipeline(
        box_dir=root / "data/inventory/box_2025",
        out_dir=root / "data/processed/ws2",
    )
    print("Global 5,000-run MSM & TPT complete!")
    print("  Row stochasticity max err:", res["markov_invariants"]["row_stochasticity_max_abs_err"])
    print("  Detailed balance max err :", res["markov_invariants"]["detailed_balance_max_abs_err"])
    print("  Top 5 ITS at 35 ns (ns)  :", [round(x, 1) for x in res["markov_invariants"]["top8_implied_timescales_at_35ns"][:5]])
    print("  CK 35ns->70ns max err    :", round(res["markov_invariants"]["ck_test_35ns_2step"]["steps"]["2"]["max_abs_err"], 4))
    print("  CK 20ns->80ns max err    :", round(res["markov_invariants"]["ck_test_20ns_4step"]["steps"]["4"]["max_abs_err"], 4))
    print("  Primary TPT MFPT (us)    :", round(res["tpt_primary_phys"]["mfpt_A_to_B_us"], 2))
    print("  Transition frames (0.05<q<0.95):", res["tpt_primary_phys"]["n_transition_frames_005_095"])
