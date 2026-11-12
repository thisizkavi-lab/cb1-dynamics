"""Phase 3 verification tests for GitHub packaging, executed Colab notebook, and guardrails.

Audits:
1. Repository configuration files (`pyproject.toml`, `requirements.txt`, `.gitignore`, `README.md`)
   and mandatory anti-overclaiming / residue-indexing documentation.
2. Executed `notebooks/CB1_Activation_Committor_Discovery.ipynb`: valid nbformat v4 JSON,
   0 error outputs, every code cell executed (`execution_count != None`) with populated stdout/figures,
   strict 1-to-1 code->markdown analysis cell pairing, Colab bootstrap + `RUN_LIVE_STREAMING_DEMO`,
   and mandatory negative-control synthesis.
3. Compact Parquet store hygiene (all 7 `.parquet` tables `< 15 MB` each, zero retained `.dcd`/`.xtc` in processed/results).
4. Git repository boundary & GitHub LFS-free size budget (`rev-parse --show-toplevel` strictly equals
   `cb1-dynamics`, zero nested gitlinks, zero tracked files `> 25 MB`).
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "CB1_Activation_Committor_Discovery.ipynb"


def test_repository_config_and_readme_guardrails() -> None:
    for rel in ("pyproject.toml", "requirements.txt", ".gitignore", "README.md"):
        p = REPO_ROOT / rel
        assert p.is_file() and p.stat().st_size > 100, f"Missing or empty packaging file: {rel}"

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    required_tokens = [
        # Core datasets, Phase 4 Colab links, PDB benchmarks & quickstart
        "CB1_Activation_Committor_Discovery.ipynb",
        "colab.research.google.com/github/thisizkavi-lab/cb1-dynamics/blob/main/notebooks/02_Multi_GPCR_Universality_Colab.ipynb",
        "colab.research.google.com/github/thisizkavi-lab/cb1-dynamics/blob/main/notebooks/03_Prospective_OpenMM_Shooting_Colab.ipynb",
        "uv run",
        "Colab",
        "5XRA",
        "6KQI",
        "6N4B",
        "8WU1",
        # Mandatory Phase 2 Section 4 negative controls & falsifications
        "0.9997",
        "-0.0175",
        "-0.0387",
        "14.95",
        "+0.1051",
        "+0.0049",
        "0.3187",
        # Residue-indexing trap documentation
        "88",
        "122",
        "8GHV",
        "KCl",
        # Verified primary literature PMIDs & exact JSON parity guards
        "+0.3066",
        "128",
        "35227761",
        "37147497",
        "41962866",
        "32661425",
        "40866700",
        "38101408",
        "38102480",
        "33908762",
        "38391020",
    ]
    for tok in required_tokens:
        assert tok in readme, f"README.md missing required token/guardrail: {tok!r}"

    # Explicitly reject misattributed/fabricated PMIDs and stale values
    for forbidden_tok in ("33887951", "37147420", "+0.0282", "0.0562"):
        assert forbidden_tok not in readme, f"README.md contains forbidden/stale token: {forbidden_tok!r}"


def test_executed_colab_notebook_structure_and_outputs() -> None:
    assert NOTEBOOK_PATH.is_file(), f"Missing notebook: {NOTEBOOK_PATH}"
    assert NOTEBOOK_PATH.stat().st_size < 25_000_000, "Executed notebook exceeds 25 MB GitHub budget."

    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert nb.get("nbformat") == 4
    cells = nb.get("cells", [])
    assert len(cells) >= 13, f"Expected >=13 cells, found {len(cells)}"

    code_indices = [i for i, c in enumerate(cells) if c.get("cell_type") == "code"]
    md_indices = [i for i, c in enumerate(cells) if c.get("cell_type") == "markdown"]
    assert len(code_indices) >= 7, f"Expected >=7 code cells, found {len(code_indices)}"
    assert len(md_indices) >= 8, f"Expected >=8 markdown cells, found {len(md_indices)}"

    # Final cell must be comprehensive conclusion markdown cell
    assert cells[-1]["cell_type"] == "markdown", "Final cell must be the comprehensive conclusion markdown cell."

    png_output_count = 0
    for idx in code_indices:
        cell = cells[idx]
        # 1. Every code cell must be executed cleanly
        exec_count = cell.get("execution_count")
        assert isinstance(exec_count, int) and exec_count >= 1, (
            f"Code cell {idx} was not executed (execution_count={exec_count!r})."
        )
        outputs = cell.get("outputs", [])
        assert len(outputs) >= 1, f"Code cell {idx} produced zero outputs."

        for out in outputs:
            assert out.get("output_type") != "error", (
                f"Code cell {idx} raised an execution error: {out.get('ename')}: {out.get('evalue')}"
            )
            data = out.get("data", {})
            if "image/png" in data:
                b64_img = "".join(data["image/png"]) if isinstance(data["image/png"], list) else data["image/png"]
                assert len(b64_img) > 5_000, f"Inline PNG in code cell {idx} is unexpectedly tiny."
                png_output_count += 1

        # 2. Strict cell pairing rule: every code cell MUST be immediately followed by a quantitative markdown cell
        assert idx + 1 < len(cells), f"Code cell {idx} has no following analysis cell."
        next_cell = cells[idx + 1]
        assert next_cell.get("cell_type") == "markdown", (
            f"Code cell {idx} is followed by {next_cell.get('cell_type')!r} instead of 'markdown'."
        )
        next_md_text = "".join(next_cell.get("source", []))
        assert len(next_md_text.strip()) >= 300, (
            f"Markdown analysis cell {idx + 1} following code cell {idx} is too short ({len(next_md_text)} chars)."
        )

    assert png_output_count >= 6, f"Expected >=6 inline figure PNGs across notebook, found {png_output_count}"

    # Verify Colab bootstrap + live HTTP 206 streaming toggle in first code cell
    first_code_src = "".join(cells[code_indices[0]].get("source", []))
    for bootstrap_tok in ("google.colab", "RUN_LIVE_STREAMING_DEMO", "Range", "1995", "CORD"):
        assert bootstrap_tok in first_code_src, f"Bootstrap code cell missing {bootstrap_tok!r}"

    # Verify mandatory negative controls & falsifications in conclusion/markdown narrative
    all_md_text = "\n".join("".join(cells[i].get("source", [])) for i in md_indices)
    for guardrail_tok in ("0.9997", "-0.0175", "-0.0387", "14.95", "+0.1051", "+0.0049", "0.3187", "0.0349"):
        assert guardrail_tok in all_md_text, f"Notebook markdown narrative missing guardrail {guardrail_tok!r}"
    for forbidden_nb_tok in ("33887951", "37147420", "N2b < 0.07"):
        assert forbidden_nb_tok not in all_md_text, f"Notebook markdown contains stale/wrong token {forbidden_nb_tok!r}"


def test_processed_parquet_budget_and_zero_raw_trajectories() -> None:
    expected_parquets = [
        REPO_ROOT / "data/processed/ws1/ws1_featurized_segments.parquet",
        REPO_ROOT / "data/processed/ws1/ws1_oof_predictions.parquet",
        REPO_ROOT / "data/processed/ws2/msm_800_microstates.parquet",
        REPO_ROOT / "data/processed/ws2/box_aea_cohort_features.parquet",
        REPO_ROOT / "data/processed/ws2/ws2_oof_predictions.parquet",
        REPO_ROOT / "data/processed/ws3/pdb16_structural_benchmark.parquet",
        REPO_ROOT / "data/processed/ws3/gpcrmd_open6_features.parquet",
    ]
    total_bytes = 0
    for pq in expected_parquets:
        assert pq.is_file(), f"Missing processed Parquet artifact: {pq}"
        sz = pq.stat().st_size
        assert sz < 15_000_000, f"Parquet {pq.name} exceeds 15 MB per-file budget ({sz} bytes)."
        total_bytes += sz
    assert 40_000_000 <= total_bytes <= 60_000_000, f"Unexpected total Parquet footprint: {total_bytes} bytes"

    for folder in (REPO_ROOT / "data/processed", REPO_ROOT / "results", REPO_ROOT / "notebooks"):
        leaked_bins = [
            p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in {".dcd", ".xtc", ".trr", ".part", ".tmp"}
        ]
        assert not leaked_bins, f"Found uncleaned raw/temp trajectory binaries: {leaked_bins}"


def test_git_repository_scope_and_tracked_file_size_budget() -> None:
    top = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert Path(top).resolve() == REPO_ROOT.resolve(), (
        f"Git repository root ({top}) is NOT scoped strictly to {REPO_ROOT}!"
    )

    ls_out = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "-s"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()
    assert len(ls_out) >= 40, f"Expected >=40 tracked files in cb1-dynamics git repo, found {len(ls_out)}"

    tracked_rel_paths: set[str] = set()
    max_bytes = 25 * 1024 * 1024  # Strict 25 MiB cap (<25 MB without Git LFS)
    for line in ls_out:
        meta, rel_path = line.split("\t", 1)
        mode = meta.split()[0]
        assert mode != "160000", f"Nested gitlink/submodule detected at {rel_path}; must be ignored!"
        tracked_rel_paths.add(rel_path)

        fpath = REPO_ROOT / rel_path
        assert fpath.suffix.lower() not in {".dcd", ".xtc", ".trr", ".part", ".tmp"}, (
            f"Raw trajectory binary tracked in git: {rel_path}"
        )
        size_b = fpath.stat().st_size
        assert size_b < 25_000_000 and size_b < max_bytes, (
            f"Tracked file {rel_path} ({size_b / 1e6:.2f} MB) exceeds 25 MB GitHub budget!"
        )

    for must_track in (
        "README.md",
        "pyproject.toml",
        "requirements.txt",
        ".gitignore",
        "notebooks/CB1_Activation_Committor_Discovery.ipynb",
        "data/processed/ws1/ws1_featurized_segments.parquet",
        "data/processed/ws2/box_aea_cohort_features.parquet",
        "data/processed/ws3/gpcrmd_open6_features.parquet",
        "data/processed/ws3/pdb16_structural_benchmark.parquet",
    ):
        assert must_track in tracked_rel_paths, f"Required deliverable not tracked in git: {must_track}"
