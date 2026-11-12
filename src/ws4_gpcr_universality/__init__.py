"""Workstream 4 (WS4): Family-Wide Multi-GPCR Universality & Evolutionary Specificity Suite.

Tests two pre-registered statistical-mechanical hypotheses across 6 Class A GPCRs
(CB1, CB2, S1P1, B2AR, MOR, A2AR) using open explicit-solvent bilayer MD trajectories from GPCRmd:

  - Hypothesis U1 (Universal Buried Water/Na+ Closure):
      Conserved D2.50-W6.48-NPxxY internal water-wire conductance, transmembrane pore hydration,
      and Na+/cation coordination at D2.50 collapse buried microswitch non-Markovian displacement
      memory (rho_mem > 0) and improve out-of-block committor log-likelihood across Class A GPCRs.

  - Hypothesis U2 (Evolutionary Specificity of the 2.42-4.46 Cleft):
      While 2.42 is conserved aromatic (Phe/Tyr) across all 6 Class A GPCRs, 4.46 is uniquely
      aromatic Phe in CB1 (F155^2.42 - F237^4.46), forming a bimodal rotamer-gated aromatic lock
      coupled to lipid cleft wedging, whereas CB2 (F72^2.42 - L154^4.46) and non-aromatic 4.46
      Class A receptors (S1P1, B2AR, MOR, A2AR) lack aromatic-pair steric locking.
"""

from .featurizer import (
    AROMATIC_RESNAMES,
    CANONICAL_BW_POSITIONS,
    build_bw_topology_index,
    download_parallel_http206,
    featurize_receptor_chunk,
    load_gpcr_family_manifest,
    pbc_wrap_receptor_angstrom,
    probe_xtc_http206_magic1995,
    stream_and_featurize_receptor,
    verify_manifest_live_probes,
    verify_xtc_header_1995,
)
from .universality_analysis import (
    evaluate_u1_water_na_memory_closure,
    evaluate_u2_tm2_tm4_evolutionary_specificity,
    plot_superfamily_universality_figure,
    run_superfamily_universality_suite,
)

__all__ = [
    "AROMATIC_RESNAMES",
    "CANONICAL_BW_POSITIONS",
    "build_bw_topology_index",
    "download_parallel_http206",
    "evaluate_u1_water_na_memory_closure",
    "evaluate_u2_tm2_tm4_evolutionary_specificity",
    "featurize_receptor_chunk",
    "load_gpcr_family_manifest",
    "pbc_wrap_receptor_angstrom",
    "plot_superfamily_universality_figure",
    "probe_xtc_http206_magic1995",
    "run_superfamily_universality_suite",
    "stream_and_featurize_receptor",
    "verify_manifest_live_probes",
    "verify_xtc_header_1995",
]
