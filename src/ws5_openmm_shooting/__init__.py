"""Phase 4 (WS5): Prospective Single-Hamiltonian OpenMM Committor Shooting Swarm.

Provides:
  - `seed_builder`: Deterministic builder & Hamiltonian-identity verifier for the 3 matched
    open-TM6 (`R214-D338 ~ 14.3 A`) human CB1 bilayer seeds (`Arm_A_Fake_Open_Trapped`,
    `Arm_B_Committed_Open` sharing a bitwise-identical WT OpenMM `System`, plus
    `Arm_C_Mutant_F237L` evolutionary `4.46` swap).
  - `observables`: PBC-aware explicit-solvent physical observable extractor (`R214-D338`,
    `R214-K343`, `F155/X237/F200/W356` chi1/chi2, `Y294-Y397`, `D163-NPxxY` water-wire
    conductance, `Na+-D163` coordination, and `CHL1`/`POPC` TM2-TM4 cleft wedging).
  - `queue_engine`: Crash-proof, multi-tab atomic Google Drive queue with portable
    `State` XML (`2 ns` chunk) checkpointing and stale-heartbeat (`>25 min`) recovery.
  - `analysis`: Prospective Kaplan-Meier survival, Clopper-Pearson committor CIs,
    log-rank / Fisher / permutation hypothesis tests, and `src.core` Mori-Zwanzig closure.
"""

from .observables import build_ws5_topology_indices, extract_ws5_frame_observables
from .queue_engine import (
    ARM_SPECS,
    CampaignConfig,
    ShootingCampaignQueue,
    build_interleaved_manifest,
    create_production_simulation,
    load_seed_bundle,
)
from .analysis import (
    compute_clopper_pearson_ci,
    compute_kaplan_meier_curve,
    compute_logrank_test,
    summarize_prospective_campaign,
    plot_prospective_campaign_readout,
)

__all__ = [
    "ARM_SPECS",
    "CampaignConfig",
    "ShootingCampaignQueue",
    "build_interleaved_manifest",
    "build_ws5_topology_indices",
    "compute_clopper_pearson_ci",
    "compute_kaplan_meier_curve",
    "compute_logrank_test",
    "create_production_simulation",
    "extract_ws5_frame_observables",
    "load_seed_bundle",
    "plot_prospective_campaign_readout",
    "summarize_prospective_campaign",
]
