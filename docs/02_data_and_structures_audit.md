# 02. Ground-Truth Audit of Publicly Accessible CB1 Molecular Dynamics Datasets, Markov State Models, and Reference Structures

> **Audit Scope & Verification Standard**
> Every repository URL, shared cloud link, API endpoint, file byte size, binary trajectory header (`XTC magic=1995`, `DCD CORD`), topology atom count, and residue-numbering convention reported in this document was **empirically tested via live HTTP/API queries and local file inspection** on 2026-11-12 for human Cannabinoid Receptor 1 (`CNR1`, UniProt `P21554`). Machine-readable metadata is saved in [`data/inventory/datasets_manifest.json`](../data/inventory/datasets_manifest.json).

---

## Executive Summary: What Is Actually Downloadable Right Now?

An academic researcher working today with Python, `curl`, GitHub, and Google Colab faces a sharply bifurcated CB1 data landscape: **almost all 2022–2023 shared trajectory links in the flagship CB1 activation papers have rotted (`HTTP 404`), yet three high-value, genuinely time-resolved atomistic sources remain 100% open and programmatically streamable without login or CAPTCHA**, complemented by a 24-replicate `GPCR-MD` collection on `MDRepo` unlockable via a 1-minute human ORCID sign-in.

### Master Availability Matrix

| Tier | Dataset / Repository | Access Status (Headless `curl` / Colab) | Functional Regime Covered | Time-Resolved Volume | Key Scientific Capability & Primary Caveat |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1A** *(Immediate Open)* | **ShuklaGroup 2025 Live Box** (`Endocannabinoid_selectivity_paper_github`, `q9w1f99mybxo6hrgo8sfr3sbm3l8xrmu`) | **OPEN (`HTTP 200/206`)** — zero auth; stream any individual `.dcd` (`67–84 MB`) or MSM `.pkl` (`19–116 MB`) | CB1 + **Anandamide (AEA)** in asymmetric neuronal membrane (`89–411`, 323 aa, full ICL3) | **5,000 `.dcd` runs** (`800–1,000` frames each, $\Delta t = 100\text{ ps}$) = **4,829,682 frames (`482.97 µs`)** + 800-state MSM | **True multi-thousand-trajectory adaptive-sampling ensemble** with lineage tags (`pr_<round>`). **Crucial scope limit:** samples membrane-portal (`TM1–TM7`) entry & pre-activation pocket coupling around **pre-active CB1** ($R_{3.50\text{–}6.30} \approx 9.7\text{–}11.1\text{ \AA}$), **not** full TM6 outward opening ($\sim 15.5\text{ \AA}$). |
| **Tier 1B** *(Immediate Open)* | **GPCRmd Open CB1 Bilayer Simulations** (`IDs 22, 162, 163, 164, 1201, 2067`) | **OPEN (`HTTP 200/206`)** — zero auth; binary `XTC` (`magic=1995`) verified (`0.57–3.80 GB`/file) | Inactive Apo (`5TGZ`, `5U09`), Inverse Agonist (`5U09`+Taranabant ×2), **Active+Gi1+MDMB-FUBINACA** (`6N4B`), **Active Apo+Gs** (`6N4B`) | **6 open `.xtc` runs** = **~20,501 frames (`8.97 µs` aggregate)** in explicit POPC + water + ions | **Full explicit solvent, ions ($\text{Na}^+$/$\text{K}^+$), POPC, and G-protein ($\text{G}_{i1}$ vs $\text{G}_s$) dynamics.** **Caveats:** equilibrium runs (`0.47–6.0 µs`) stay within starting macrobasins; Replicates 2 & 3 of `22/162/163/164` redirect to HTML `/accounts/login/`. |
| **Tier 1C** *(Immediate Local/GitHub)* | **`ShuklaGroup/THC_binding` Fig. 8 `.npy` + `Cannabinoid_activation` Macrostate PDBs** | **OPEN & ALREADY CLONED LOCALLY** (`data/inventory/`) | Full agonist vs partial agonist ($\Delta^9$-THC) across **full CB1 activation span** ($R_{3.50\text{–}6.35} \in [7, 20]\text{ \AA}$) + 6 apo pathway PDBs | **278,196 time-ordered frames** (`173,860` agonist + `104,336` THC) × **3 CVs** (`W356–D163`, `R214–K343`, `NPxxY`) | Covers complete inactive$\leftrightarrow$active switch range ($\text{ac}_1 = 0.85\text{–}0.94$). **Caveat:** saved via `np.concatenate` without explicit trajectory-boundary indices (`~1,349` / `224` discontinuities must be segmented into contiguous blocks). |
| **Tier 2** *(1-Min Human ORCID)* | **MDRepo `GPCR-MD` CB1 Collection** (`MDR00000029, 105, 239, 476, 1109, 1124, 1146`) | **JSON Metadata + 1,250-pt RMSD OPEN**; `.tar`/`.xtc` requires free **human ORCID login** (`mdrepo get`) | Adds **`5XRA`+AM11542+Chol**, **`5XRA` apo**, **`5XR8` apo**, **`6KQI`+CP55940+ORG27569 (NAM)**, plus all 3–6 reps of `5TGZ`/`5U09` | **24 replicates (`11.9 µs`)**; stripped `minimal.tar` is only **`134–291 MB`/system (`1.13 GB` total)**! | Unlocks agonist-alone (`5XRA`) and allosteric NAM (`6KQI`) triplicate trajectories that are still embargoed on `gpcrmd.org`. Cannot be fetched by an unauthenticated headless bot. |
| **Dead / Gated / Invalid** | **24 Shukla 2022–2023 Box links** (`404`); **Dryad `4f4qrfjq5`** (`401`, 218 GB); **IDB-6705697** (`403` WAF); **MISATO/PLAS-20K/DDD** (no lipid bilayer!); **Zenodo `19555568`** (Martini CG); **Zenodo `3242751`** (images only) | **UNUSABLE** without author re-upload, Dryad bearer token + HPC storage, or physical disqualifier | See Section 1.3 & Section 2.3 | — | **Critical scientific safeguard:** Never use `MISATO`, `PLAS-20K`, or `DDD-03B` CB1 entries on MDRepo/Zenodo—all four simulate transmembrane CB1 in **pure water without a lipid bilayer**. |

---

## 1. Deep Audit of `ShuklaGroup` Repositories & External Data Deposits

### 1.1 Organization-Wide GitHub Census (`ShuklaGroup`)
Querying `https://api.github.com/orgs/ShuklaGroup/repos?per_page=100` returned **53 public repositories**. Every GPCR/cannabinoid repository was inspected, and four were cloned into `/Users/kavi/Documents/artico/cb1-dynamics/data/inventory/`:

1. **`ShuklaGroup/Cannabinoid_activation`** (*cloned: 145.68 MB unpacked*, 105 files) — Dutta & Shukla, *Commun. Biol.* **6**, 485 (2023), `doi:10.1038/s42003-023-04868-1` (*CB1 vs CB2 apo/holo activation & subtype selectivity*).
2. **`ShuklaGroup/THC_binding`** (*cloned: 68.02 MB unpacked*, 47 files) — Dutta, Selvam, Das & Shukla, *J. Biol. Chem.* **298**, 101764 (2022), `doi:10.1016/j.jbc.2022.101764` (*Mechanistic origin of $\Delta^9$-THC partial agonism at CB1*).
3. **`ShuklaGroup/Dutta_Shukla_Cannabinoid_2023a`** (*cloned: 0.06 MB*, 19 files) — Dutta & Shukla, *eLife* **13**, RP98798 (2025), `doi:10.7554/eLife.98798` (*Binding kinetics and signaling bias of synthetic cannabinoid new psychoactive substances via TRAM*).
4. **`ShuklaGroup/Dutta_Shukla_EndoCannabinoid_2025`** (*cloned: 0.05 MB*, 17 files) — Dutta & Shukla, *J. Biol. Chem.* **302**, 111434 (2026), `doi:10.1016/j.jbc.2026.111434` (preprint `doi:10.1101/2024.10.25.620304`; *Dynamic mechanism for subtype selectivity of endocannabinoids in CB1 and CB2*).
5. **`ShuklaGroup/Sodium_binding`** (*inspected via GitHub Trees API*, ~65 MB) — Dutta, Selvam & Shukla, *ACS Chem. Neurosci.* **13**, 379–389 (2022), `doi:10.1021/acschemneuro.1c00760` (*Allosteric $\text{Na}^+$ binding pathway in CB1 and CB2*).

*(Other GPCR repos in `ShuklaGroup`—e.g. `GPCR-IPL_Score` and `serotonin_receptor`—contain static docking models or 5-HT receptors only, zero CB1 trajectories.)*

---

### 1.2 File-by-File Audit of Cloned `Cannabinoid_activation` & `THC_binding`

#### A. `data/inventory/Cannabinoid_activation/`
- **`Initial_Coordinates/` (All-Atom Solvated Starting Systems — Preserved Locally)**:
  Contains complete AMBER `.pdb` + `.prmtop` starting coordinates in explicit POPC (`PA/PC/OL`), TIP3P water (`WAT`), and $\text{Na}^+/\text{Cl}^-$ for all 4 CB1 simulation ensembles (plus 4 CB2 ensembles):
  - `CB1-APO_active.{pdb,prmtop}` (`51,938` atoms) & `CB1-APO_inactive.{pdb,prmtop}` (`48,998` atoms)
  - `CB1-HOLO_active.{pdb,prmtop}` (`51,080` atoms, agonist `KCA` = MDMB-FUBINACA) & `CB1-HOLO_inactive.{pdb,prmtop}` (`46,774` atoms, inverse agonist `7DY` = taranabant)
- **`Main_Figure_3/CB1_feature_calculator.py` & `Main_Figure_5/CB1-APO_vampnet_activation_feature_bar_plt.py` (Exact MSM/VAMPnet Feature Definitions)**:
  Defines the 7 core intramolecular CB1 activation features (in `mdtraj` 0-based topology indexing where UniProt numbers match script labels):
  1. `Helix2M_Nloop`: $\text{C}\alpha(\text{M103}^{\text{N-term}})\text{–}\text{C}\alpha(\text{D163}^{2.50})$ distance
  2. `Ex16_distance`: $\text{C}\alpha(116^{1.32})\text{–}\text{C}\alpha(366^{6.58})$ extracellular TM1–TM6 distance
  3. `In6_chi2`: $\text{W356}^{6.48}$ toggle-switch $\chi_2$ dihedral (`CA-CB-CG-CD1`)
  4. `TG_diff_z`: $\langle z \rangle_{\text{sidechain}(\text{W356}^{6.48})} - \langle z \rangle_{\text{sidechain}(\text{F200}^{3.36})}$ twin-toggle vertical separation
  5. `In57_distance`: $\text{C}\alpha(\text{L290}^{5.54})\text{–}\text{OH}(\text{Y397}^{7.53})$ intracellular TM5–NPxxY distance
  6. `In36_distance`: $\text{C}\alpha(\text{R214}^{3.50})\text{–}\text{C}\alpha(\text{K343}^{6.35})$ intracellular TM3–TM6 opening distance
  7. `In52_distance`: $\text{OH}(\text{Y294}^{5.58})\text{–}\text{C}\alpha(\text{I156}^{2.43})$ intracellular TM5–TM2 distance
- **`Main_Figure_4/*.pkl` & `Main_Figure_6..8/` (Static Summaries Only)**:
  Inspection of every `.pkl` and `.npy` in `Main_Figure_3..8` confirms they hold **only static post-processed histogram/bootstrap arrays** (e.g. `Main_Figure_4/*.pkl` has shape `(4, 200, 2)` representing 200 bootstrap conditional-probability curves; `Main_Figure_7` holds static residue KL-divergence vectors). **Zero time-ordered trajectories are stored inside `Cannabinoid_activation` itself.**
- **`Main_Figure_5/CB1_ref_inactive_{2,I1,I2,I3,I4,active}_b.pdb` (6 Apo Activation Pathway Macrostates — Preserved Locally)**:
  Contains 285 $\text{C}\alpha$ atoms per state with per-residue Kullback–Leibler divergence encoded in the B-factor column.
  > **CRITICAL RESIDUE-INDEXING WARNING (`Cannabinoid_activation/Main_Figure_5/*.pdb`):** AMBER sequentially renumbered residues after truncating the N-terminus (`97` aa) **and** deleting `26` residues of ICL3! Consequently, **TM1–TM5 have `resSeq = UniProt - 97`** (`F155=58`, `D163=66`, `F200=103`, `R214=117`, `F237=140`, `Y294=197`), whereas **TM6–H8 have `resSeq = UniProt - 123`** (`D338=215`, `K343=220`, `W356=233`, `Y397=274`). Verified by conserved `DRY` (`116–118`), `CWGP` (`232–235`), and `NPIIY` (`270–274`) motifs:

  | Apo Pathway Macrostate File | $\text{R214}^{3.50}\text{–D338}^{6.30}$ ($\text{\AA}$) | $\text{R214}^{3.50}\text{–K343}^{6.35}$ ($\text{\AA}$) | $\text{F200}^{3.36}\text{–W356}^{6.48}$ ($\text{\AA}$) | $\text{D163}^{2.50}\text{–W356}^{6.48}$ ($\text{\AA}$) | $\text{Y294}^{5.58}\text{–Y397}^{7.53}$ ($\text{\AA}$) | $\text{F155}^{2.42}\text{–F237}^{4.46}$ ($\text{\AA}$) | Mechanistic Milestone |
  | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
  | `CB1_ref_inactive_2_b.pdb` (*Inactive*) | 7.42 | 10.79 | 8.04 | 10.93 | 21.24 | 7.97 | Salt bridge intact; NPxxY extended |
  | `CB1_ref_inactive_I1_b.pdb` (*I1*) | 7.31 | 9.68 | 9.43 | 13.52 | 17.14 | 7.87 | Initial $\text{D163–W356}$ & NPxxY shift |
  | `CB1_ref_inactive_I2_b.pdb` (*I2*) | 8.20 | 10.86 | 8.42 | 13.01 | 16.87 | 8.39 | Pre-transition intermediate |
  | `CB1_ref_inactive_I3_b.pdb` (*I3*) | **8.20** | **14.84** | **11.99** | **14.93** | **12.45** | **6.61** | **Toggle, $\text{F155–F237}$, NPxxY & $\text{K343}$ switch BEFORE $\text{R214–D338}$ breaks!** |
  | `CB1_ref_inactive_I4_b.pdb` (*I4*) | **13.39** | **15.49** | 10.07 | 14.33 | 13.26 | 6.84 | Full ionic-lock breakage |
  | `CB1_ref_inactive_active_b.pdb` (*Activeapo*) | 10.45 | 13.93 | 9.42 | 12.82 | 16.03 | 7.39 | Uncoupled apo active-like basin |

#### B. `data/inventory/THC_binding/` — Real Featurized Frames Preserved in `Figure_8/`
- While `Figure_3..7` and `mutual_information_calculation/` contain static MSM histograms/bars, **`THC_binding/Figure_8/` preserves 278,196 raw featurized simulation frames**:
  - `agonist_toggle_switch_TM6_movement.npy` (`shape=(173860, 2)`, `float64`, `2.78 MB`) & `agonist_toggle_switch_NPxxY_movement.npy` (`shape=(173860, 2)`, `2.78 MB`)
  - `partial_agonist_toggle_switch_TM6_movement.npy` (`shape=(104336, 2)`, `1.67 MB`) & `partial_agonist_toggle_switch_NPxxY_movement.npy` (`shape=(104336, 2)`, `1.67 MB`)
- **Empirical Verification**: Column `0` ($d(\text{W356}^{6.48}\text{–D163}^{2.50})$ in $\text{nm}$) is **bit-for-bit identical** (`max |diff| = 0.0`) between each paired `TM6` and `NPxxY` file, reconstructing an exact **3-coordinate trajectory matrix** $[d(\text{W356–D163}),\, d(\text{R214–K343}),\, d(\text{NPxxY})]$ across **173,860 full-agonist frames** and **104,336 $\Delta^9$-THC partial-agonist frames**.
- **Time-Ordering & Leakage Audit**: Lag-1 autocorrelation is $\text{ac}_1 = 0.852\text{–}0.943$ (confirming consecutive time-ordered trajectory segments rather than shuffled frames), **but** the arrays were saved via `np.concatenate` without trajectory-boundary indices (`1,349` frame-to-frame steps $>0.15\text{ nm}$ in agonist; `224` in partial agonist). Any downstream modeling MUST split strictly on contiguous segments bounded by step discontinuities—never via random frame splits.

---

### 1.3 Empirical HTTP Audit of All 25 ShuklaGroup External Data Links (Box, Dryad, Illinois Data Bank)

Every external URL across all 5 ShuklaGroup repositories was tested via `curl` (`GET --range 0-511`) and REST metadata endpoints:

1. **24 of 25 `uofi.box.com` Shares (All 2022–2023 Repos) Are Dead (`HTTP 404 Not Found`)**:
   - `Cannabinoid_activation` (10 links): `ix0hhvbwgwhipqrhk1oe1iwpl021vyw1`, `xoiuicdpjhrohpjputgnk5ppuwlcfp35`, `iw1wlcdg44gjs7ufx4rriuulindibj5t`, `vyakobq2zbk5xyoxqvccxngif6jzeo0z`, plus all 6 stripped trajectory folders (`p0ivhqihimh8cr3mp4tua5gd7ygsijo8` CB1-APO, `a8dvtgmw4sjajcjbvmkbjbtfcpy9dy2i` CB1-HOLO-active, `9khutxgdsjc7mriqktch5ifbxcda4fs7` CB1-HOLO-inactive, and 3 CB2 folders) $\rightarrow$ **all `HTTP 404`**.
   - `THC_binding` (5 links): `g1x5vfukl6492jtuzpx7u1rbyki2t85z`, `hzzgzq0xw1tcgsjlnecklajpjdtmm4ma`, `88nzwam5ygout7x3wvswxzlbcfkqxky3`, `y92093ay6t8ma0p12jgyxpwxxjf2ibh8`, `mm3yvufwfkkovja78ziv4jt1kzgw6fjn` $\rightarrow$ **all `HTTP 404`**.
   - `Sodium_binding` (1 link): `n8a15pcabkh8fzd42hecxh2mkhz1fzh3` $\rightarrow$ **`HTTP 404`**.
   - `Dutta_Shukla_Cannabinoid_2023a` (8 links, incl. `tram_building.ipynb` `/shared/static/` links): `4lxxdwpj6rf1tcka9yort729lsoww19b`, `i6kjcozct79j4brsmmt5htpvq68i33ri`, `nibqvi0spcd4dfs2q6tybhmg5tyb0g4s`, `vn85ycljd3k5tn91rv08jhcg8ra897dr`, `4ruu5vd4ghlpkqh7wcqj96ngc84wbp5b` $\rightarrow$ **all `HTTP 404`**.

2. **LIVE, OPEN-ACCESS 406.37 GB SHUKLAGROUP BOX REPOSITORY (`Dutta_Shukla_EndoCannabinoid_2025`)**:
   - **Shared Folder**: `https://uofi.box.com/s/q9w1f99mybxo6hrgo8sfr3sbm3l8xrmu` (`Endocannabinoid_selectivity_paper_github`, **`HTTP 200`**).
   - **Direct Unauthenticated Download Endpoint** (verified in `curl -L` and Python `urllib` with `HTTP 200 / 206` binary headers, zero login/CAPTCHA):
     ```text
     https://uofi.app.box.com/index.php?rm=box_download_shared_file&shared_name=q9w1f99mybxo6hrgo8sfr3sbm3l8xrmu&file_id=f_<FILE_ID>
     ```
   - **Verified Contents & Locally Downloaded Metadata (`data/inventory/box_endocannabinoid_2025/`, 26.63 MB)**:
     - **Topology & Residue Mapping (Downloaded Locally)**: `CB1_assym-strip.pdb` (`567,847 B`, `7,005` atoms: CB1 UniProt `89–411` [`323` residues, `resSeq 1–323` = `F89–M411`, exact uniform offset `UniProt = resSeq + 88`, **full ICL3 intact**] + anandamide `LIG` + asymmetric neuronal lipid shell `ER1, GAL, GLC, LPI, NE5, OSM, PPS`), `CB1_assym-strip.prmtop` (`3,661,085 B`), `CB1_resid.csv` (`3,611 B`, explicit Ballesteros–Weinstein table).
     - **Complete 5,000-Trajectory Discrete MSM & Featurizations**:
       - `CB1_frame_cluster_indices.pkl` (`f_1371314214130`, `19,491,396 B` — **downloaded & verified locally**: list of **5,000 discrete time-ordered arrays** over `800` k-means microstates, totaling **4,829,682 frames = 482.97 µs** at `100 ps/frame`!).
       - `CB1_micro_to_macro_index.pkl` (`6,557 B` — **downloaded locally**: maps 800 microstates $\rightarrow$ 6 macrostates: `Sol/Bulk`, `Mem`, `Pro`, `I1^TM17`, `I2^TM17`, `B1`).
       - `CB1_assym_MSM.pkl` (`f_1399682267779`, `34.85 MB`), `CB1_msm_weight.pkl` (`f_1371313667747`, `38.81 MB`), `CB1_lig_com_cor.pkl` (`f_1371313547127`, `116.11 MB` — 3D ligand COM time-series across all 5,000 runs).
     - **5,000 Raw Continuous `.dcd` Trajectories (`folder_id=236736306087`, `406.37 GB` total across 251 JSON pages)**:
       - Named `CB1_assym_pr_<round>_frame_<idx>-strip.dcd` (preserving adaptive-sampling round `pr_<round>` for lineage-grouped `GroupKFold`!).
       - Binary header check on `f_1371479495522` (`CB1_assym_pr_9_frame_1-strip.dcd`, `67,312,196 B`): valid 32-bit `CORD` DCD, `n_frames = 800` (`80 ns`); 1,000-frame runs are `84,140,176 B` (`100 ns`). Any stratified subset of $K$ trajectories can be streamed one-by-one into Python/Colab!
     - **Two Essential Scientific Caveats**:
       1. `macrostate_rep_dcd` (`folder_id=241599702647`, 15 files) stores 1,000 **randomly sampled frames per macrostate** (per `Main_Figure_5A.py`), **not** continuous trajectories—always pull from `CB1_anandamide_binding` (`236736306087`) for dynamical analysis.
       2. Direct $\text{C}\alpha$ measurement on the 4 downloaded macrostate PDBs (`Bulk1_0`, `TM17_0`, `AB_0`, `B1_0`) shows $\text{R214}^{3.50}\text{–D338}^{6.30} = 9.71\text{–}11.06\text{ \AA}$, $\text{R214}^{3.50}\text{–K343}^{6.35} = 12.57\text{–}12.81\text{ \AA}$, and $\text{F200}^{3.36}\text{–W356}^{6.48} = 7.74\text{–}8.83\text{ \AA}$. Thus, **this ensemble resolves endocannabinoid lipid-portal association and pre-activation pocket reorganization around pre-active CB1, NOT full TM6 outward opening ($\sim 15.5\text{ \AA}$)**.

3. **Dryad `doi:10.5061/dryad.4f4qrfjq5` (218.32 GB, CC0-1.0, Dutta & Shukla 2025 *eLife*) & Illinois Data Bank `IDB-6705697`**:
   - Dryad REST v2 (`/api/v2/datasets/doi%3A10.5061%2Fdryad.4f4qrfjq5/versions/.../files`) is open (`HTTP 200`) and lists 6 trajectory tarballs: `CB1_classical_Unbiased.tar.gz` (`89.12 GB`), `CB1_nps_Unbiased.tar.gz` (`78.44 GB`), `CB1_classical_Umbrella_sampling.tar.gz` (`22.89 GB`), `CB1_nps_Umbrella_sampling.tar.gz` (`22.92 GB`), `CB1_classical_Well-tempered_metadynamics.tar.gz` (`2.30 GB`), and `CB1_nps_Well-tempered_metadynamics.tar.gz` (`2.65 GB`). However, `/api/v2/files/<id>/download` returns **`HTTP 401 Unauthorized`** without a personal Dryad OAuth bearer token, and monolithic `78–89 GB` `.tar.gz` archives are infeasible on standard Google Colab disks.
   - `https://databank.illinois.edu/datasets/IDB-6705697` returns **`HTTP 403`** (`awselb/2.0` AWS WAF) to automated `curl`; per strict access rules we did not probe further, and its exact dataset is openly served by the live Box repository above.

---

## 2. Deep Audit of GPCRmd, MDRepo, Zenodo, and Open Biomolecular MD Archives

### 2.1 GPCRmd (`www.gpcrmd.org`) — All 11 Human CB1 (`P21554`) Entries Audited

Scraping and verifying every CB1 dynamics entry on GPCRmd uncovered **11 simulation IDs**: **6 are immediately downloadable via unauthenticated `curl`/Python (`8.97 µs` total)**, while **5 pre-publication entries + Replicates 2 & 3 of the 2020 entries redirect to `/accounts/login/`**.

| GPCRmd ID | PDB & Construct | Functional State & Bound Molecules | Saved Atomistic Environment (`natoms`) | Engine / FF / Stride | Open Length & Frames | Verified Direct Unauthenticated URLs & Exact Byte Sizes |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- |
| **`1201`** | `6N4B` *(WT T210)* | **Active CB1 + Agonist `MDMB-FUBINACA` + Cognate $\text{G}_{i1}$ Heterotrimer** *(Sandhu et al. 2022)* | `203,557` atoms: CB1 + $\text{G}\alpha_{i1}\beta_1\gamma_2$ + 352 POPC + 47,309 TIP3P + 131 $\text{K}^+$ / 136 $\text{Cl}^-$ *(no $\text{Na}^+$)* | GROMACS 2019.3, CHARMM36, `200 ps`/frame | **1.0002 µs** (`5,001` frames, 1/1 rep open) | PDB (`16,488,149 B`): `.../dyn1366/tmp_dyn_0_1366.pdb`<br>TOP (`831 B`): `.../dyn1366/18984_dyn_1201.top`<br>XTC (`3,796,878,084 B` = **3.80 GB**): `https://www.gpcrmd.org/dynadb/files/Dynamics/dyn1366/18986_trj_1201.xtc` |
| **`2067`** | `6N4B` *(WT T210)* | **Active CB1 (Apoform) + Non-Cognate/Promiscuous $\text{G}_s$ Complex** *(Berghella et al. 2025)* | `94,976` atoms: CB1 + $\text{G}_s$ + 201 POPC + 19,916 TIP3P + 52 $\text{Na}^+$ / 74 $\text{Cl}^-$ | ACEMD 3.0, CHARMM36m, `1,000 ps`/frame | **6.000 µs** (`6,000` frames, 1/1 rep open) | PDB (`7,598,157 B`): `.../dyn2336/tmp_dyn_0_2336.pdb`<br>PSF (`18,033,557 B`): `.../dyn2336/24047_dyn_2067.psf`<br>XTC (`2,118,682,836 B` = **2.12 GB**): `https://www.gpcrmd.org/dynadb/files/Dynamics/dyn2336/24048_trj_2067.xtc` |
| **`22`** | `5U09.A` *(T210A)* | **Inactive + Inverse Agonist `Taranabant`** *(Entry A)* | `69,095` atoms: CB1 + `7DY` + POPC + TIP3P + $\text{Na}^+/\text{Cl}^-$ | ACEMD 3.0, CHARMM36m, `200 ps`/frame | **500 ns** (`2,500` frames, Rep 1 of 3 open) | PDB (`5,596,781 B`): `.../10276_dyn_22.pdb`<br>PSF (`13,058,650 B`): `.../10277_dyn_22.psf`<br>XTC Rep 1 (`639,453,028 B` = **639.5 MB**): `https://www.gpcrmd.org/dynadb/files/Dynamics/10280_trj_22.xtc` |
| **`164`** | `5U09.A` *(T210A)* | **Inactive + Inverse Agonist `Taranabant`** *(Independent Entry B)* | `69,095` atoms: CB1 + `7DY` + POPC + TIP3P + $\text{Na}^+/\text{Cl}^-$ | ACEMD 3.0, CHARMM36m, `200 ps`/frame | **500 ns** (`2,500` frames, Rep 1 of 3 open) | PDB (`5,596,781 B`): `.../11529_dyn_164.pdb`<br>PSF (`13,058,650 B`): `.../11528_dyn_164.psf`<br>XTC Rep 1 (`639,543,056 B` = **639.5 MB**): `https://www.gpcrmd.org/dynadb/files/Dynamics/11524_trj_164.xtc` |
| **`162`** | `5TGZ.A` *(T210A)* | **Inactive Apoform** *(AM6538 removed; $\text{Na}^+$ assay)* | `77,514` atoms: CB1 + POPC + TIP3P + 46 $\text{Na}^+$ / 60 $\text{Cl}^-$ | ACEMD 3.0, CHARMM36m, `200 ps`/frame | **~467 ns** (`~2,000–2,333` frames, Rep 1 of 3 open) | PDB (`6,278,712 B`): `.../11514_dyn_162.pdb`<br>PSF (`11,622,801 B`): `.../11512_dyn_162.psf`<br>XTC Rep 1 (`574,074,404 B` = **574.1 MB**): `https://www.gpcrmd.org/dynadb/files/Dynamics/11508_trj_162.xtc` |
| **`163`** | `5U09.A` *(T210A)* | **Inactive Apoform** *(taranabant removed; $\text{Na}^+$ assay)* | `69,092` atoms: CB1 + POPC + TIP3P + 41 $\text{Na}^+$ / 55 $\text{Cl}^-$ | ACEMD 3.0, CHARMM36m, `200 ps`/frame | **500 ns** (`2,500` frames, Rep 1 of 3 open) | PDB (`5,596,530 B`): `.../11520_dyn_163.pdb`<br>PSF (`10,331,506 B`): `.../11521_dyn_163.psf`<br>XTC Rep 1 (`639,350,584 B` = **639.4 MB**): `https://www.gpcrmd.org/dynadb/files/Dynamics/11516_trj_163.xtc` |
| `852`, `896`, `897`, `1002`, `1136` | `5XR8` apo, `5XRA` apo, `5XRA`+AM11542, `6KQI`+CP+ORG, `6KQI` apo | Login-gated on `gpcrmd.org` | — | ACEMD 3.0, CHARMM36m | **0 open on GPCRmd** *(see MDRepo below!)* | Redirect (`HTTP 302 -> 200`, `22.75 KB` HTML) to `/accounts/login/`. |

> **Mandatory GPCRmd Binary-Header Guard**: Replicates 2 & 3 of IDs `22, 162, 163, 164` (`10281, 10282, 11509, 11510, 11517, 11518, 11525, 11526`) return `HTTP 200` after redirecting to a `22.75 KB` HTML login page instead of `403`. Every download script MUST assert `struct.unpack(">i", f.read(4))[0] == 1995` (GROMACS XTC magic integer).

---

### 2.2 MDRepo (`mdrepo.org`) — Major Discovery: 7 Unembargoed GPCR-MD Ensembles + 4 Non-Bilayer Traps

Streaming MDRepo's nightly archive (`https://static.mdrepo.org/metadata/mdrepo-metadata-latest.tar.gz`) and querying `https://mdrepo.org/api/v1/simulation_view/<id>` (cached locally in [`data/inventory/mdrepo_cb1_simulation_views.json`](../data/inventory/mdrepo_cb1_simulation_views.json)) revealed **11 human CB1 (`P21554`) records**, all with `"is_embargoed": false`:

1. **7 Valid Bilayer `GPCR-MD` Ensembles (`24 replicates = 11.9 µs`)**:
   - **`MDR00000029`** (`ID 29` = GPCRmd `896`): `5XRA` active-like **apoform**, $3 \times 500\text{ ns}$ (`1,499.4 ns`), `minimal.tar = 138.1 MB`, `full.tar = 2.08 GB`.
   - **`MDR00000105`** (`ID 105` = GPCRmd `897`): `5XRA` active-like + **AM11542 (`Chembl1683648`) + Cholesterol**, $3 \times 500\text{ ns}$ (`1,499.4 ns`), `minimal.tar = 142.6 MB`, `full.tar = 2.08 GB`.
   - **`MDR00000476`** (`ID 476` = GPCRmd `852`): `5XR8` active-like **apoform**, $3 \times 500\text{ ns}$ (`1,499.4 ns`), `minimal.tar = 139.4 MB`, `full.tar = 2.05 GB`.
   - **`MDR00001109`** (`ID 1109` = GPCRmd `1002`): `6KQI` NAM-locked + **CP55940 + ORG27569 (NAM)**, $3 \times 500\text{ ns}$, `minimal.tar = 143.7 MB`, `full.tar = 1.89 GB`.
   - **`MDR00000239`** (`ID 239` = GPCRmd `162`, `5TGZ` apo, 3 reps, `minimal.tar = 134.1 MB`), **`MDR00001124`** (`ID 1124` = GPCRmd `22+164`, `5U09`+Taranabant, **all 6 reps**, `minimal.tar = 290.7 MB`), **`MDR00001146`** (`ID 1146` = GPCRmd `163`, `5U09` apo, 3 reps, `minimal.tar = 143.5 MB`).
   - **Access Rule** (verified via OpenAPI schema `/api/schema/` and `TravisWheelerLab/mdrepo-readthedocs/docs/src/ai_agents.md`): Unauthenticated scripts can read metadata, 1,250-point $\text{C}\alpha$ `rmsd_values`, and per-residue `rmsf_values` directly via JSON (`already cached locally`), whereas binary `.tar`/`.xtc` retrieval (`POST /api/v1/downloads`) requires a **human researcher to sign in with free ORCID OAuth** at `mdrepo.org` and run `mdrepo get <token>`.

2. **4 Automated-Pipeline Non-Bilayer CB1 Entries on MDRepo — DO NOT USE FOR GPCR DYNAMICS**:
   - `MDR00019886` (`MISATO` `6kqi`, `8 ns`, `100` frames), `MDR00020024` (`MISATO` `6n4b`, `8 ns`, `100` frames), `MDR00090076` (`PLAS-20K` `6n4b`+`KCA`, $5 \times 3.9\text{ ns}$, `40` frames/rep), and `MDR00104079` (`DDD-03B` `6kqi`+CP55940 MetaD egress, `50` runs, `24.11 ns` total) were **all simulated in pure aqueous TIP3P water without a lipid bilayer**! Exposing hydrophobic transmembrane helices `TM1–TM7` to bulk water induces unphysical hydrophobic collapse and forces aqueous ligand egress instead of CB1's physiological lipid-portal pathway.

---

### 2.3 Zenodo, OSF, Figshare, MDDB, Dataverse, MemProtMD, and ATLAS
- **Zenodo `19555568` (`10.5281/zenodo.19555568`, `CHAMP_trjs.zip`, `30.25 GB`, `HTTP 200`)**: Coarse-grained Martini MD of CB1 in POPC:cholesterol with THC and ORG27569. Martini's elastic network restrains tertiary backbone geometry, disqualifying it from microswitch/activation committor modeling (usable solely for coarse cholesterol occupancy).
- **Zenodo `3242751` (`10.5281/zenodo.3242751`, `HTTP 200`)**: Despite its title (*"MD simulation trajectory of cannabidiol in the cannabinoid type 1 receptor"*), file inspection confirms it contains **only 6 rendered `.tga` images and 2 `.conf` files—zero trajectories**.
- **MDDB (`irb.mddbr.eu`), OSF, Figshare, Harvard Dataverse, MemProtMD, and ATLAS (`dsimb.inserm.fr`)**: Queried directly via REST APIs; **zero all-atom human CB1 trajectory datasets** exist (`1CB1` in MDDB is soluble calbindin $\text{D}_{9\text{k}}$, not `CNR1`; MemProtMD holds only short coarse-grained lipid-insertion snapshots).

---

## 3. Experimental Reference Structures Ensemble (`data/inventory/pdb_structures/`)

Out of **35 human CB1 (`P21554`) entries** currently in RCSB PDB, **15 representative structures** spanning every known pharmacological and transducer state were downloaded into [`data/inventory/pdb_structures/`](../data/inventory/pdb_structures/) (`29` `.cif`/`.pdb` files, `22.30 MB`) and parsed to compute exact $\text{C}\alpha\text{–C}\alpha$ microswitch reference coordinates:

| PDB ID | CB1 Chain | TM6–H8 `auth_seq_id` Offset | Pos `210` ($3.46$) | Method & Res. | Functional State, Bound Ligands & Transducer | $\text{R214–D338}$ ($3.50\text{–}6.30$) | $\text{R214–K343}$ ($3.50\text{–}6.35$) | $\text{F200–W356}$ ($3.36\text{–}6.48$) | $\text{D163–W356}$ ($2.50\text{–}6.48$) | $\text{Y294–Y397}$ ($5.58\text{–}7.53$) | $\text{F155–F237}$ ($2.42\text{–}4.46$) |
| :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`5TGZ`** | `A` | `0` | `ALA` | X-ray `2.80 Å` | Inactive + Antagonist `AM6538` (`ZDG`) | 8.69 | 11.95 | 8.46 | 12.29 | 17.67 | 8.12 |
| **`5U09`** | `A` | `0` | `ALA` | X-ray `2.60 Å` | Inactive + Inverse Agonist `Taranabant` (`7DY`) | **6.75** | **10.05** | 8.59 | 13.10 | **18.20** | 8.37 |
| **`6KQI`** | `A` | `0` | `ALA` | X-ray `3.25 Å` | **NAM-Locked Inactive** + Agonist `CP55940` (`9GF`) + NAM `ORG27569` (`9GL`) | **7.20** | **10.12** | **8.96** | 12.77 | **18.07** | 8.33 |
| **`7FEE`** | `A` | `0` | `ALA` | X-ray `2.70 Å` | **Mutant-Locked Inactive-like** (`S203K,T210A,E273K,T283V,R340E,N393D`) + `CP55940` (`9GF`) + PAM `ZCZ011` (`7IC`) *(no Gi)* | **9.22** | **12.54** | **9.79** | 12.83 | 17.26 | 8.07 |
| **`5XRA`** | `A` | `0` | `ALA` | X-ray `2.80 Å` | Active-like + Agonist `AM11542` (`8D3`) *(no Gi; note 2025 Nature notice)* | 14.18 | 15.47 | 10.86 | 14.73 | 14.69 | 8.32 |
| **`5XR8`** | `A` | **`+122`** | `ALA` | X-ray `2.95 Å` | Active-like + Agonist `AM841` (`8D0`) *(no Gi; `W6.48=478`!)* | 14.43 | 15.22 | 11.10 | 14.94 | 14.78 | 8.31 |
| **`7V3Z`** | `A` | **`+122`** | `ALA` | X-ray `3.29 Å` | Active-like + Agonist `CP55940` (`9GF`) + `CLR` *(`H154L,T210A,E273K,T283V,R340E`; no Gi; `W6.48=478`!)* | 14.66 | 15.01 | 11.32 | 15.03 | 14.11 | 8.17 |
| **`6N4B`** | `R` | `0` | `THR` (WT) | Cryo-EM `3.00 Å` | **Active + `MDMB-FUBINACA` (`KCA`) + $\text{G}_{i1}$ + `scFv16`** | **14.94** | **15.77** | 11.29 | **15.29** | 13.36 | 7.65 |
| **`6KPG`** | `R` | `0` | `THR` (WT) | Cryo-EM `3.00 Å` | Active + `AM841` (`8D0`) + $\text{G}_{i1}$ + `scFv16` | 14.45 | 15.75 | 10.62 | 14.30 | 13.61 | **6.67** |
| **`7WV9`** | `R` | `0` | `THR` (WT) | Cryo-EM `3.36 Å` | Active + `CP55940` (`9GF`) + PAM `ZCZ011` (`7IC`) + **$\text{G}_{i2}$ + `scFv16`** | 14.99 | 16.07 | 11.56 | 15.21 | 13.34 | 7.88 |
| **`8GHV`** | `D` | `0` | `THR` (WT) | Cryo-EM `2.80 Å` | Active + Endocannabinoid mimic `AMG315` (`ZI5`) + $\text{G}_{i1}$ + `scFv16` | 15.53 | 15.94 | 11.63 | 14.87 | 14.33 | 7.80 |
| **`9ERX`** | `R` | `0` | `THR` (WT) | Cryo-EM `2.90 Å` | Active + Agonist `HU-210` (`A1H66`) + $\text{G}_{i1}$ + `scFv16` *(mmCIF only)* | 13.99 | 15.28 | 11.27 | 14.57 | 13.33 | 7.07 |
| **`8K8J`** | `R` | `0` | **`ILE` (`T210I`)** | Cryo-EM `2.88 Å` | Active + `Fenofibrate` (`J3O`) + **Non-cognate $\text{MiniG}_{sq}$ + `Nb35`** | 16.12 | 16.08 | 10.53 | 14.09 | 13.25 | 7.16 |
| **`8WRZ`** | `R` | `0` | **`ILE` (`T210I`)** | Cryo-EM `3.60 Å` | Active + `AM841` (`8D0`) + **$\beta$-Arrestin-1 + `scFv30`** | 17.13 | 17.08 | 11.09 | 15.02 | 14.24 | 7.94 |
| **`8WU1`** | `R` | `0` | `THR` (WT) | Cryo-EM `3.20 Å` | **Active + `MDMB-FUBINACA` (`KCA`) + $\beta$-Arrestin-1 + `Fab30`** | **15.19** | **15.73** | 10.82 | **13.36** | 13.50 | 7.51 |

### Critical Structural Pitfalls & Contrasts Caught by Direct Parsing
1. **Chain & Numbering Traps**: X-ray monomers use Chain `A` with thermostabilizing `T210A`; cryo-EM complexes use Chain `R` (`D` in `8GHV`) with WT `T210`—except `8K8J` and `8WRZ`, which carry the **constitutively super-active `T210I` mutation** (inflating TM6 opening to `16.1–17.1 Å`). Worse, **`5XR8` and `7V3Z` shift `auth_seq_id` by `+122` across TM6–H8** (`D6.30 = 460`, `W6.48 = 478`, `Y7.53 = 519`) because residue numbering runs continuously through their ICL3 flavodoxin fusion!
2. **Matched-Ligand, Matched-WT $\text{G}_{i1}$ vs $\beta$-Arrestin-1 Contrast (`6N4B` vs `8WU1`)**: Both bind identical `KCA` (MDMB-FUBINACA) on WT `T210` and exhibit virtually identical intracellular TM6 opening (`14.94` vs `15.19 Å`) and NPxxY inward shift (`13.36` vs `13.50 Å`), yet **`D163^2.50–W356^6.48` differs by `1.93 Å` (`15.29 Å` in $\text{G}_{i1}$ vs `13.36 Å` in $\beta$-arrestin-1)**—directly pointing to sodium-pocket / toggle coupling as an orthogonal discriminator at matched $R_{\text{TM6}}$.
3. **Transducer Requirement for $\text{F155}^{2.42}\text{–F237}^{4.46}$ Compaction**: Agonist-alone crystals (`5XRA`, `5XR8`, `7V3Z`) open TM6 (`14.2–14.7 Å`) while keeping `F155–F237` at inactive-like `8.17–8.32 Å`, whereas transducer-coupled cryo-EM structures (`6N4B`, `6KPG`, `9ERX`, `8K8J`, `8WU1`) and Shukla's `I3/I4` pathway intermediates compact `F155–F237` to `6.61–7.65 Å`.

---

## 4. Brutally Honest Feasibility & Sampling Assessment

### 4.1 The Central Ground-Truth Constraint
**No unauthenticated public repository currently hosts raw atomistic trajectories that spontaneously traverse the full inactive$\leftrightarrow$active CB1 TM6 opening barrier ($R_{3.50\text{–}6.30}: 7\text{ \AA} \leftrightarrow 15\text{ \AA}$)** because all 2022–2023 ShuklaGroup Box shares (`Cannabinoid_activation`, `THC_binding`) return `HTTP 404`. Pretending that `500 ns` GPCRmd equilibrium runs or `80–100 ns` pre-active AEA-binding runs sample full global receptor activation would be scientifically dishonest.

### 4.2 Exact Mapping: What Each Accessible Dataset CAN and CANNOT Test

| Dynamical Question / Hypothesis Class | Can Open Public Data Answer It Rigorously Right Now? | Which Dataset(s) to Use & Why | What CANNOT Be Claimed |
| :--- | :--- | :--- | :--- |
| **1. Lineage-split committor / stopped-DGA prediction & Mori–Zwanzig memory collapse on an adaptive-sampling MD ensemble** | **YES — Rigorously on Pre-Activation / Orthosteric Association** | **Tier 1A: Shukla 2025 Live Box (`CB1_anandamide_binding`)** — `CB1_frame_cluster_indices.pkl` (`5,000` runs / `4.83M` frames, local!) + `CB1_assym_MSM.pkl` (`35 MB`) + `CB1_lig_com_cor.pkl` (`116 MB`) + streamable subset of $50\text{–}150$ continuous `.dcd` files (`pr_<round>` grouped) in asymmetric neuronal lipids. | Cannot claim this dataset measures full intracellular TM6 outward opening (`R214–D338` stays `9.7–11.1 Å`); target is `Bulk/TM17` $\rightarrow$ orthosteric `B1` commitment & pre-activation toggle/pocket reorganization. |
| **2. Explicit-solvent / $\text{Na}^+$ (`D163^2.50`) / POPC memory-kernel & cognate $\text{G}_{i1}$ vs promiscuous $\text{G}_s$ coupling at fixed macrostate** | **YES — Rigorously at Sub-Microsecond to `6 µs` Equilibrium** | **Tier 1B: 6 Open GPCRmd Runs (`8.97 µs`)** — Compare Inactive Apo (`162, 163`) vs Taranabant-locked (`22, 164`) vs Active+$\text{G}_{i1}$+Agonist (`1201`, `1.0 µs`) vs Active Apo+$\text{G}_s$ (`2067`, `6.0 µs`). Directly tests within-basin microswitch relaxation, $\text{Na}^+$ pocket hydration, and $\text{G}_{i1}$ vs $\text{G}_s$ allosteric coupling at matched `6N4B` backbone! | Cannot estimate global inactive$\leftrightarrow$active transition rates or barrier committors from single-basin equilibrium runs; note `ID 1201` uses $\text{KCl}$ (no $\text{Na}^+$) and `ID 2067` has `1.0 ns` stride vs `0.2 ns` on others. |
| **3. Conditional heterogeneity across the FULL $7\text{–}20\text{ \AA}$ TM6 activation range (Full Agonist vs Partial Agonist $\Delta^9$-THC)** | **YES — With Mandatory Concatenation-Boundary Disclosure** | **Tier 1C: `THC_binding/Figure_8/*.npy` (`278,196` frames, local!) + 6 `Cannabinoid_activation` Macrostate PDBs + 15 Experimental PDBs** — Tests $\operatorname{Var}(d_{\text{toggle}}, d_{\text{NPxxY}} \mid R_{\text{TM6}})$ and short-lag predictive skill across the full activation span. | Only 3 CVs preserved (`W356–D163`, `R214–K343`, `NPxxY`); missing explicit trajectory IDs require splitting on jump-segmented contiguous blocks rather than true seeding-tree lineages. |
| **4. Allosteric NAM (`ORG27569`, `6KQI`) orthosteric-intracellular decoupling & agonist-alone (`5XRA`) relaxation across triplicates** | **CONDITIONAL — Requires 1-Minute Human ORCID Login on MDRepo** | **Tier 2: MDRepo `minimal.tar` (`MDR00001109` `6KQI`+CP+ORG, `MDR00000105` `5XRA`+AM11542, `MDR00000029` `5XRA` apo)** — Only `138–144 MB` per 3-replicate `1.5 µs` bundle! | Unauthenticated headless scripts can only read MDRepo's 1,250-point `rmsd_values` and `rmsf_values` until a user runs `mdrepo get <token>`. |

---

### 4.3 Local Python / `uv` & Google Colab Execution Blueprint

- **Local Workspace Readiness (`/Users/kavi/Documents/artico/cb1-dynamics/`)**:
  - Verified macOS `arm64` host with **`671 GiB` free disk**, `/Users/kavi/.local/bin/uv` (`0.13.7`), and local `cpython-3.12.16`.
  - Zero global package modifications needed; run any trajectory/MSM analysis in an ephemeral cached environment via:
    ```bash
    /Users/kavi/.local/bin/uv run --python 3.12 \
      --with mdtraj --with numpy --with scipy --with scikit-learn \
      python scripts/featurize_cb1.py
    ```
- **Google Colab Disk & Bandwidth Strategy**:
  - Standard Colab instances provide `~70–100 GB` ephemeral disk and `12.7 GB` RAM:
    1. **GPCRmd**: All 6 open `.xtc` + `.pdb`/`.psf` files total **`8.41 GB`** (`574 MB` to `3.80 GB` each)—every single trajectory fits comfortably on standard Colab disk and loads via `mdtraj.iterload(xtc_path, top=pdb_path, chunk=500)` using `<1.5 GB` RAM.
    2. **Shukla 2025 Live Box (`406.37 GB` total)**: Never download all 5,000 `.dcd` files at once in Colab. Instead: (a) load `CB1_frame_cluster_indices.pkl` (`19.5 MB`), `CB1_assym_MSM.pkl` (`34.9 MB`), and `CB1_lig_com_cor.pkl` (`116.1 MB`) directly into RAM (`<300 MB`), and (b) stream stratified batches of $50\text{–}150$ individual `.dcd` files (`67.3 MB` each), compute low-dimensional structural/lipid features immediately with `mdtraj`, save the `~50 KB` `.npz` feature array, and `os.unlink()` each raw `.dcd` file—keeping peak Colab disk usage **`<1.0 GB`**.
