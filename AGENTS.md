# CB1 Activation Dynamics — Cross-Machine Agent Continuity & Rules (`AGENTS.md`)

This repository (`https://github.com/thisizkavi-lab/cb1-dynamics`) investigates from first principles whether human Cannabinoid Receptor 1 ($\text{CB}_1$, `CNR1`, UniProt `P21554`) activation commitment can be predicted from molecular configuration, explicit solvent/lipid gating, and Mori–Zwanzig non-Markovian memory, targeting a no-compromise *Nature*-family empirical standard.

Antigravity conversations are machine-local (`~/.gemini/antigravity/brain/`), so this `AGENTS.md` synchronizes full project state, verified empirical findings, Google Drive/Colab coordinates, and strict methodological guardrails across the user's Mac and lab Linux machines.

---

## 1. User Communication & Operational Rules

1. **Plain, direct English:** Communicate concisely without dense academic jargon walls or sycophancy. Preserve ruthless statistical honesty—keep every falsified hypothesis and negative control intact.
2. **Never use the `browser` subagent to operate Google Colab:** Always give the user direct Colab URLs + 2–3 plain click steps, and monitor outputs via the Google Drive plugin (`gdrive readonly ...`) instead.
3. **Google Drive safety:** Always use `gdrive readonly ...` for listing, checking progress, and downloading artifacts. Never run `gdrive mutate` (especially deletes or permission changes) unless the user explicitly asks.
4. **Python environment:** Run repo tests and scripts with Python 3.11+ (`uv run --no-project --python 3.12 --with-requirements requirements.txt pytest -q` passes `40/40` tests). Never commit raw multi-GB `.xtc`/`.dcd` trajectory binaries to Git.

---

## 2. Live Phase 4 Cloud & Google Drive State (`sonykavindra@gmail.com`)

- **Google Drive Root Folder (`cb1-dynamics-phase4`):**
  - **Folder ID:** `1ObdIu5EhvgptKXOFBCJiU7B4-gl5pcGu`
  - Inspect anytime via Google Drive plugin skill (`gdrive readonly tree 1ObdIu5EhvgptKXOFBCJiU7B4-gl5pcGu`).
- **Workstream 4 (`WS4` — 6-Receptor Class A GPCR Superfamily Universality):**
  - **Notebook:** `notebooks/02_Multi_GPCR_Universality_Colab.ipynb`
  - **Colab Link:** `https://colab.research.google.com/github/thisizkavi-lab/cb1-dynamics/blob/main/notebooks/02_Multi_GPCR_Universality_Colab.ipynb`
  - **Drive Path:** `cb1-dynamics-phase4/ws4_universality/`
  - **Current Status:** All `23/23` explicit-solvent GPCRmd trajectories across all `6/6` Class A GPCRs (`CB1`, `CB2`, `S1P1`, `B2AR`, `MOR`, `A2AR`) have already been streamed and saved to Drive (`CB1_features.parquet`, `CB2_features.parquet`, `S1P1_features.parquet`, `B2AR_features.parquet`, `MOR_features.parquet`, `A2AR_features.parquet`). Check Drive for `ws4_universality_summary.json` and `fig_ws4_gpcr_superfamily_universality.png`; if the Colab tab disconnected before the final summary cell finished, you can download those 6 `.parquet` files (`~few MB` total) locally and run Sections 3–5 (`evaluate_u1_water_na_memory_closure`, `evaluate_u2_tm2_tm4_evolutionary_specificity`, `plot_superfamily_universality_figure`) on CPU in `< 2 minutes` without re-downloading a single `.xtc`!
- **Workstream 5 (`WS5` — Prospective Single-Hamiltonian OpenMM Shooting Swarm, `900 ns`):**
  - **Notebook:** `notebooks/03_Prospective_OpenMM_Shooting_Colab.ipynb`
  - **Colab Link:** `https://colab.research.google.com/github/thisizkavi-lab/cb1-dynamics/blob/main/notebooks/03_Prospective_OpenMM_Shooting_Colab.ipynb`
  - **Drive Path:** `cb1-dynamics-phase4/ws5_shooting/`
  - **Design:** `15 replicates x 20 ns x 3 arms = 45 trajectories = 900 ns` (`4 fs` HMR, `310 K`, explicit POPC/water/150 mM NaCl, CHARMM36m) comparing at matched `~14.3 Å` intracellular `TM6` (`R214^3.50–D338^6.30`):
    - `Arm_A_Fake_Open_Trapped`: WT (`system_wt_ab.xml.gz`), inward `F155^2.42/F237^4.46`, unlatched `Y294^5.58–Y397^7.53`.
    - `Arm_B_Committed_Open`: WT (`system_wt_ab.xml.gz` — byte-identical Hamiltonian `SHA-256 = 82afc08e...`), outward `F155^2.42/F237^4.46`, latched `Y294^5.58–Y397^7.53`.
    - `Arm_C_Mutant_F237L`: CB2-inspired `F237L^4.46` mutant (`system_mut_f237l_c.xml.gz`) initialized from the exact Arm A inward-trapped backbone.
  - **Crash-Safe Queue (`src/ws5_openmm_shooting/queue_engine.py`):** Checkpoints every `2.0 ns` (`checkpoints/<job_id>.state.xml.gz` + `trajectories/<job_id>_observables.parquet`). Supports 2–3 simultaneous Colab `L4 GPU` tabs (`~383 ns/day` per L4 tab; stale locks `> 25 min` auto-reclaim).
  - **Linux Local GPU Option:** When taking over on the lab Linux machine, run `nvidia-smi` first! If the Linux machine has a fast local NVIDIA GPU (`RTX 3090/4090/A5000/A6000` etc.) with CUDA/OpenMM, `src.ws5_openmm_shooting.queue_engine.run_worker_loop` can also execute locally in the background for **0 Colab Compute Units** (or run alongside Colab tabs). Otherwise, keep 2–3 Colab `L4 GPU` tabs open in Chrome/Firefox signed into `sonykavindra@gmail.com`.

---

## 3. Completed & Audited Phases 1–3 Summary (`40/40` `pytest` Passing)

All Phase 1–3 code, `.parquet` feature tables (`50.4 MB` total), reports, and audits are committed in this repo:

1. **Stage 1 — Orthosteric Lipid-Portal Entry (`WS2`, `results/ws2/ws2_report.md`):**
   - Across `5,000` unbiased runs (`482.97 µs` 800-state `AEA` MSM + `160` streamed `.dcd`s), at matched $R_{\text{portal}} \in [9.5, 18.5]\ \text{\AA}$, `Tier C` (`TM1–TM7` `I119–F379` `+2.37 Å` dilation + tail-first `AEA` `S383` lock + annular ceramide/cholesterol clearance) achieves `BSS_int = +0.7859` (`leave-round-out MSM +0.7731`, `p = 0.00498`) and **100.0% continuous `q_dga` Mori–Zwanzig memory closure (`0.5–10 ns`, `N2a p = 0.975–1.000`)**, while intracellular switches remain quiescent (`+0.1051`) without $G_i$.
2. **Stage 2 — Buried Toggle & Explicit Water/$\text{Na}^+$ Closure (`WS1` & `WS3`, `results/ws1/ws1_report.md`, `results/ws3/ws3_report.md`):**
   - In `278,196` frames (`Shukla 2022`), smooth committor-gradient rotation (`H3`) is **falsified** ($\mathcal{S}_{\cos} = 0.9997$), and within-ligand 3-CV DGA skill is **zero** (`-0.0175` Full Agonist, `-0.0387` $\Delta^9\text{-THC}$) once cross-fold target leakage is eliminated; pooled `BSS_int = +0.9180` reflects **$\Delta^9\text{-THC}$ toggle-locked steric arrest at $R_{\text{TM6}} = 14.95\ \text{\AA}$** (`0% >= 15.2 Å`).
   - Across `8.90 µs` of explicit-solvent `GPCRmd` trajectories (`WS3`), adding explicit `D163^2.50–W356^6.48–NPxxY` water-wire conductance + `Na+` coordination cuts `8D` `StoppedDGACommittor` `PBR` by `-52.0%` vs `B1` (`-15.2%` vs `Prot`) and closes `50.0%–79.3%` of 1D toggle memory (`12.2%–55.6%` incremental solvent closure).
3. **Stage 3 — `TM2–TM4` Aromatic Gate & Effector Latch (`WS3` `16` PDBs):**
   - `100%` (`16/16`) structural separation: all `3` open-TM6 agonist-only crystals (`5XRA`, `5XR8`, `7V3Z`: `TM6 = 14.18–14.66 Å`) trap `F155^2.42/F237^4.46` inward with `Y294^5.58–Y397^7.53` unlatched (`5.10–5.45 Å`), whereas all `8` `Gi`/`βarr1` complexes flip `F155/F237` outward (`14.67–16.72 Å`) and latch `Y294–Y397` (`2.92–4.17 Å`), with a `-118.47°` `W356^6.48` $\chi_1$ switch between `6N4B` ($G_{i1}$) and `8WU1` ($\beta\text{arr1}$).
4. **Full Defect & Verification Ledger:** See `docs/04_phase1_verification_audit.md` and `docs/05_phase2_experimental_audit.md` (`D1`–`D12` remediations + adversarial regression tests in `tests/`).

---

## 4. Next Steps After Taking Over on Linux

1. **Verify `WS4` Completion:** Check Google Drive `1ObdIu5EhvgptKXOFBCJiU7B4-gl5pcGu` (`ws4_universality/`). Download `ws4_universality_summary.json` and `fig_ws4_gpcr_superfamily_universality.png` into `results/ws4/` (or if only the 6 `*_features.parquet` files exist because the Mac Colab tab closed early, download the 6 parquets and run `evaluate_u1_water_na_memory_closure` + `evaluate_u2_tm2_tm4_evolutionary_specificity` + `plot_superfamily_universality_figure` locally in ~90 seconds).
2. **Complete & Audit `WS5` (`900 ns` OpenMM Shooting):** Check `nvidia-smi` on Linux; either run `03_Prospective_OpenMM_Shooting_Colab.ipynb` across 2–3 `L4 GPU` tabs on Linux Chrome/Firefox (enable **Settings -> App Settings -> Keep computer awake** in Antigravity or `systemd-inhibit`), or run `queue_engine.py` on a local Linux NVIDIA GPU. Once `45/45` trajectories hit `completed/`, run `evaluate_ws5_shooting_campaign` + `plot_ws5_prospective_shooting_figure`, audit every number, commit `results/{ws4,ws5}/`, and draft the *Nature*-format / bioRxiv `v1` manuscript (`paper/`).
