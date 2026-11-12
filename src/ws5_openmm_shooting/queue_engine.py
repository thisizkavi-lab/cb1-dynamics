"""Crash-proof, multi-tab Google Drive queue & 2-ns chunk checkpoint engine for WS5.

Designed for Google Colab GPUs (`L4` / `A100`) backed by `/content/drive/MyDrive/cb1-dynamics-phase4/ws5_shooting/`
and fast local CPU/OpenCL smoke-testing:
  - Interleaved round-robin job ordering (`rep01_A, rep01_B, rep01_C, rep02_A, ...`) across
    `15 replicates x 20 ns x 3 arms = 45 unbiased shooting trajectories = 900 ns` total.
  - Portable compressed OpenMM `State` XML (`checkpoints/<job_id>.state.xml.gz`) + atomic
    `100 ps`-strided `.parquet` observable tables saved after every `2 ns` chunk (`10 chunks/job`).
  - Atomic claim locks (`locks/<job_id>.lock.json`) with read-back verification and automatic
    reclamation of stale locks (`heartbeat_age > 25 min = 1500 s`), enabling 2-3 simultaneous
    Colab GPU tabs to run in parallel without conflict.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gzip
import json
import math
import os
from pathlib import Path
import random
import socket
import time
from typing import Any
import uuid

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import pandas as pd

from .observables import build_ws5_topology_indices, extract_ws5_frame_observables


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEEDS_DIR = REPO_ROOT / "data" / "openmm_seeds"
DEFAULT_DRIVE_ROOT = Path("/content/drive/MyDrive/cb1-dynamics-phase4/ws5_shooting")

ARM_SPECS: dict[str, dict[str, Any]] = {
    "Arm_A_Fake_Open_Trapped": {
        "short_code": "A",
        "label": "Arm A: WT Fake-Open Trapped (F155/F237 Inward, Unlatched)",
        "pdb_gz": "arm_a_fake_open_trapped.pdb.gz",
        "system_xml_gz": "system_wt_ab.xml.gz",
        "arm_seed_offset": 11,
        "genotype": "WT_P21554",
    },
    "Arm_B_Committed_Open": {
        "short_code": "B",
        "label": "Arm B: WT Committed-Open (F155/F237 Outward, Y294-Y397 Latched)",
        "pdb_gz": "arm_b_committed_open.pdb.gz",
        "system_xml_gz": "system_wt_ab.xml.gz",
        "arm_seed_offset": 23,
        "genotype": "WT_P21554",
    },
    "Arm_C_Mutant_F237L": {
        "short_code": "C",
        "label": "Arm C: F237L Evolutionary Swap (Inward Start, Deleted 4.46 Phenyl)",
        "pdb_gz": "arm_c_mutant_f237l.pdb.gz",
        "system_xml_gz": "system_mut_f237l_c.xml.gz",
        "arm_seed_offset": 37,
        "genotype": "F237L_4p46",
    },
}

ARM_ORDER = (
    "Arm_A_Fake_Open_Trapped",
    "Arm_B_Committed_Open",
    "Arm_C_Mutant_F237L",
)


@dataclass(frozen=True)
class CampaignConfig:
    """Configuration for the prospective OpenMM shooting campaign."""

    n_replicates_per_arm: int = 15
    traj_duration_ns: float = 20.0
    chunk_duration_ns: float = 2.0
    dt_ps: float = 0.004
    report_stride_ps: float = 100.0
    temperature_K: float = 310.0
    friction_per_ps: float = 1.0
    base_random_seed: int = 20260400
    stale_lock_timeout_s: float = 1500.0  # 25 minutes

    @property
    def steps_per_report(self) -> int:
        val = round(self.report_stride_ps / self.dt_ps)
        if abs(val * self.dt_ps - self.report_stride_ps) > 1e-6:
            raise ValueError(f"report_stride_ps ({self.report_stride_ps}) not divisible by dt_ps ({self.dt_ps})")
        return int(val)

    @property
    def steps_per_chunk(self) -> int:
        chunk_ps = self.chunk_duration_ns * 1000.0
        val = round(chunk_ps / self.dt_ps)
        if val % self.steps_per_report != 0:
            raise ValueError("chunk_duration_ns must be an exact multiple of report_stride_ps.")
        return int(val)

    @property
    def total_chunks_per_traj(self) -> int:
        val = round(self.traj_duration_ns / self.chunk_duration_ns)
        if abs(val * self.chunk_duration_ns - self.traj_duration_ns) > 1e-6:
            raise ValueError("traj_duration_ns must be an exact multiple of chunk_duration_ns.")
        return int(val)

    @property
    def total_steps_per_traj(self) -> int:
        return self.total_chunks_per_traj * self.steps_per_chunk


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_gzip_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as fh:
        fh.write(text)
    os.replace(tmp, path)


def _atomic_write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def build_interleaved_manifest(cfg: CampaignConfig) -> list[dict[str, Any]]:
    """Build interleaved round-robin replicate schedule (`rep01_A, rep01_B, rep01_C, rep02_A, ...`)."""
    jobs: list[dict[str, Any]] = []
    order_idx = 0
    for rep in range(1, cfg.n_replicates_per_arm + 1):
        for arm_name in ARM_ORDER:
            spec = ARM_SPECS[arm_name]
            order_idx += 1
            code = spec["short_code"]
            job_id = f"rep{rep:02d}_Arm_{code}"
            vel_seed = cfg.base_random_seed + rep * 1009 + spec["arm_seed_offset"]
            int_seed = vel_seed + 500_000
            bar_seed = vel_seed + 900_000
            jobs.append(
                {
                    "queue_order": order_idx,
                    "job_id": job_id,
                    "replicate": rep,
                    "arm_name": arm_name,
                    "arm_code": code,
                    "genotype": spec["genotype"],
                    "pdb_gz": spec["pdb_gz"],
                    "system_xml_gz": spec["system_xml_gz"],
                    "vel_seed": int(vel_seed),
                    "integrator_seed": int(int_seed),
                    "barostat_seed": int(bar_seed),
                    "traj_duration_ns": float(cfg.traj_duration_ns),
                    "chunk_duration_ns": float(cfg.chunk_duration_ns),
                    "total_chunks": int(cfg.total_chunks_per_traj),
                }
            )
    return jobs


def load_seed_bundle(arm_name: str, seeds_dir: Path = DEFAULT_SEEDS_DIR) -> tuple[app.PDBFile, mm.System, dict[str, Any]]:
    """Load compressed `.pdb.gz` coordinates/topology and `.xml.gz` unrestrained OpenMM `System`."""
    if arm_name not in ARM_SPECS:
        raise KeyError(f"Unknown arm_name {arm_name!r}; expected one of {list(ARM_SPECS)}")
    spec = ARM_SPECS[arm_name]
    pdb_path = seeds_dir / spec["pdb_gz"]
    xml_path = seeds_dir / spec["system_xml_gz"]
    if not pdb_path.is_file() or not xml_path.is_file():
        raise FileNotFoundError(f"Missing WS5 seed files for {arm_name}: {pdb_path} / {xml_path}")

    with gzip.open(pdb_path, "rt", encoding="utf-8") as fhp:
        pdb = app.PDBFile(fhp)
    with gzip.open(xml_path, "rt", encoding="utf-8") as fhx:
        system = mm.XmlSerializer.deserialize(fhx.read())
    t_idx = build_ws5_topology_indices(pdb.topology)
    return pdb, system, t_idx


def select_execution_platform(prefer_gpu: bool = True) -> tuple[mm.Platform, dict[str, str]]:
    """Select CUDA (`mixed` precision) on Colab GPUs, or OpenCL/CPU locally."""
    available = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
    if prefer_gpu and "CUDA" in available:
        return mm.Platform.getPlatformByName("CUDA"), {"Precision": "mixed"}
    if prefer_gpu and "OpenCL" in available:
        return mm.Platform.getPlatformByName("OpenCL"), {"Precision": "mixed"}
    if "CPU" in available:
        return mm.Platform.getPlatformByName("CPU"), {}
    return mm.Platform.getPlatform(0), {}


def create_production_simulation(
    pdb: app.PDBFile,
    system: mm.System,
    cfg: CampaignConfig,
    integrator_seed: int,
    barostat_seed: int,
    prefer_gpu: bool = True,
) -> tuple[app.Simulation, str]:
    """Instantiate unrestrained `4 fs` HMR `LangevinMiddleIntegrator` + `MonteCarloMembraneBarostat`."""
    available = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
    candidates: list[tuple[str, dict[str, str]]] = []
    if prefer_gpu:
        if "CUDA" in available:
            candidates.append(("CUDA", {"Precision": "mixed"}))
            candidates.append(("CUDA", {}))
        if "OpenCL" in available:
            candidates.append(("OpenCL", {"Precision": "mixed"}))
            candidates.append(("OpenCL", {}))
    if "CPU" in available:
        candidates.append(("CPU", {}))
    candidates.append((mm.Platform.getPlatform(0).getName(), {}))

    last_err: Exception | None = None
    for plat_name, props in candidates:
        try:
            sys_copy = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
            for i in range(sys_copy.getNumForces()):
                f = sys_copy.getForce(i)
                if isinstance(f, mm.MonteCarloMembraneBarostat):
                    f.setRandomNumberSeed(int(barostat_seed))

            integrator = mm.LangevinMiddleIntegrator(
                cfg.temperature_K * unit.kelvin,
                cfg.friction_per_ps / unit.picosecond,
                cfg.dt_ps * unit.picoseconds,
            )
            integrator.setRandomNumberSeed(int(integrator_seed))
            plat = mm.Platform.getPlatformByName(plat_name)
            sim = app.Simulation(pdb.topology, sys_copy, integrator, plat, props)
            return sim, plat.getName()
        except mm.OpenMMException as exc:
            last_err = exc
            continue
    raise RuntimeError(f"Failed to initialize OpenMM Simulation across {candidates}: {last_err}")


class ShootingCampaignQueue:
    """Crash-proof, multi-worker atomic Google Drive queue for the WS5 OpenMM shooting swarm."""

    def __init__(
        self,
        work_dir: Path = DEFAULT_DRIVE_ROOT,
        seeds_dir: Path = DEFAULT_SEEDS_DIR,
        cfg: CampaignConfig | None = None,
        worker_id: str | None = None,
        prefer_gpu: bool = True,
    ) -> None:
        self.work_dir = Path(work_dir)
        self.seeds_dir = Path(seeds_dir)
        self.cfg = cfg or CampaignConfig()
        self.prefer_gpu = prefer_gpu
        self.worker_id = worker_id or f"{socket.gethostname()}_pid{os.getpid()}_{uuid.uuid4().hex[:6]}"

        self.locks_dir = self.work_dir / "locks"
        self.ckpt_dir = self.work_dir / "checkpoints"
        self.traj_dir = self.work_dir / "trajectories"
        self.done_dir = self.work_dir / "completed"
        for d in (self.locks_dir, self.ckpt_dir, self.traj_dir, self.done_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.manifest_path = self.work_dir / "campaign_manifest.json"
        self.jobs = build_interleaved_manifest(self.cfg)
        if not self.manifest_path.exists():
            _atomic_write_json(
                self.manifest_path,
                {
                    "config": asdict(self.cfg),
                    "total_jobs": len(self.jobs),
                    "total_campaign_ns": round(len(self.jobs) * self.cfg.traj_duration_ns, 2),
                    "jobs": self.jobs,
                },
            )
        self._seed_cache: dict[str, tuple[app.PDBFile, mm.System, dict[str, Any]]] = {}

    def _get_seed(self, arm_name: str) -> tuple[app.PDBFile, mm.System, dict[str, Any]]:
        if arm_name not in self._seed_cache:
            self._seed_cache[arm_name] = load_seed_bundle(arm_name, self.seeds_dir)
        return self._seed_cache[arm_name]

    def _lock_path(self, job_id: str) -> Path:
        return self.locks_dir / f"{job_id}.lock.json"

    def _done_path(self, job_id: str) -> Path:
        return self.done_dir / f"{job_id}.done.json"

    def _progress_path(self, job_id: str) -> Path:
        return self.ckpt_dir / f"{job_id}.progress.json"

    def _state_path(self, job_id: str) -> Path:
        return self.ckpt_dir / f"{job_id}.state.xml.gz"

    def _parquet_path(self, job_id: str) -> Path:
        return self.traj_dir / f"{job_id}_observables.parquet"

    def try_claim_job(self, job: dict[str, Any], now_epoch: float | None = None) -> bool:
        """Atomically claim `job` if incomplete and unlocked (or if existing lock heartbeat is stale >25 min)."""
        job_id = job["job_id"]
        if self._done_path(job_id).exists():
            return False

        now = time.time() if now_epoch is None else float(now_epoch)
        lpath = self._lock_path(job_id)
        reclaimed_stale = False
        if lpath.exists():
            try:
                existing = json.loads(lpath.read_text(encoding="utf-8"))
                age = now - float(existing.get("heartbeat_epoch", 0.0))
                if existing.get("worker_id") != self.worker_id and age <= self.cfg.stale_lock_timeout_s:
                    return False
                reclaimed_stale = existing.get("worker_id") != self.worker_id
            except Exception:
                reclaimed_stale = True

        claim_payload = {
            "job_id": job_id,
            "worker_id": self.worker_id,
            "claimed_at_epoch": now,
            "heartbeat_epoch": now,
            "reclaimed_stale_lock": reclaimed_stale,
        }
        _atomic_write_json(lpath, claim_payload)
        # Brief jittered read-back verification to guard against simultaneous FUSE tab claims
        time.sleep(0.02 + random.uniform(0.0, 0.03))
        try:
            verify = json.loads(lpath.read_text(encoding="utf-8"))
            return verify.get("worker_id") == self.worker_id
        except Exception:
            return False

    def refresh_heartbeat(self, job_id: str, chunk_idx: int, now_epoch: float | None = None) -> None:
        lpath = self._lock_path(job_id)
        now = time.time() if now_epoch is None else float(now_epoch)
        payload = {
            "job_id": job_id,
            "worker_id": self.worker_id,
            "heartbeat_epoch": now,
            "active_chunk": int(chunk_idx),
        }
        _atomic_write_json(lpath, payload)

    def release_lock(self, job_id: str) -> None:
        lpath = self._lock_path(job_id)
        if lpath.exists():
            try:
                cur = json.loads(lpath.read_text(encoding="utf-8"))
                if cur.get("worker_id") == self.worker_id:
                    lpath.unlink(missing_ok=True)
            except Exception:
                lpath.unlink(missing_ok=True)

    def run_job(
        self,
        job: dict[str, Any],
        max_chunks_this_call: int | None = None,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Run or resume `job` in crash-safe chunks, checkpointing State XML & Parquet after each chunk."""
        job_id = job["job_id"]
        arm_name = job["arm_name"]
        pdb, system, t_idx = self._get_seed(arm_name)

        prog_path = self._progress_path(job_id)
        state_path = self._state_path(job_id)
        pq_path = self._parquet_path(job_id)
        resuming = prog_path.exists() and state_path.exists() and pq_path.exists()

        if resuming:
            prog = json.loads(prog_path.read_text(encoding="utf-8"))
            start_chunk = int(prog["completed_chunks"])
            current_step = int(prog["current_step"])
            prior_wall_s = float(prog.get("wall_seconds_elapsed", 0.0))
        else:
            start_chunk = 0
            current_step = 0
            prior_wall_s = 0.0

        # Offset RNG seeds by start_chunk so resumed chunks never replay Chunk 0's Langevin/barostat RNG stream
        eff_int_seed = int(job["integrator_seed"]) + start_chunk * 7919
        eff_bar_seed = int(job["barostat_seed"]) + start_chunk * 104729

        sim, plat_name = create_production_simulation(
            pdb,
            system,
            self.cfg,
            integrator_seed=eff_int_seed,
            barostat_seed=eff_bar_seed,
            prefer_gpu=self.prefer_gpu,
        )

        if resuming:
            with gzip.open(state_path, "rt", encoding="utf-8") as fhs:
                saved_state = mm.XmlSerializer.deserialize(fhs.read())
            sim.context.setState(saved_state)
            existing_rows = pd.read_parquet(pq_path).to_dict(orient="records")
            if verbose:
                print(
                    f"[{job_id}] Resuming from chunk {start_chunk}/{self.cfg.total_chunks_per_traj} "
                    f"(step={current_step}, t={current_step * self.cfg.dt_ps / 1000.0:.3f} ns, rng_offset={start_chunk}) on {plat_name}"
                )
        else:
            bv0 = pdb.topology.getPeriodicBoxVectors()
            if bv0 is not None:
                sim.context.setPeriodicBoxVectors(*bv0)
            sim.context.setPositions(pdb.positions)
            sim.context.setVelocitiesToTemperature(
                self.cfg.temperature_K * unit.kelvin,
                int(job["vel_seed"]),
            )
            # Record exact t = 0 frame before stepping!
            st0 = sim.context.getState(getPositions=True, getEnergy=True, enforcePeriodicBox=True)
            bv_arr = st0.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
            box_diag = np.array([float(bv_arr[0, 0]), float(bv_arr[1, 1]), float(bv_arr[2, 2])])
            obs0 = extract_ws5_frame_observables(
                st0.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
                box_diag,
                t_idx,
                time_ns=0.0,
                step=0,
            )
            obs0.update(
                {
                    "job_id": job_id,
                    "arm_name": arm_name,
                    "arm_code": job["arm_code"],
                    "replicate": int(job["replicate"]),
                    "potential_energy_kJ_mol": float(st0.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)),
                    "kinetic_energy_kJ_mol": float(st0.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)),
                }
            )
            existing_rows = [obs0]
            if verbose:
                print(
                    f"[{job_id}] Initialized fresh Maxwell-Boltzmann shooting run ({arm_name}, seed={job['vel_seed']}) on {plat_name}"
                )

        reports_per_chunk = self.cfg.steps_per_chunk // self.cfg.steps_per_report
        steps_per_report = self.cfg.steps_per_report
        target_end_chunk = self.cfg.total_chunks_per_traj
        if max_chunks_this_call is not None:
            target_end_chunk = min(target_end_chunk, start_chunk + int(max_chunks_this_call))

        for chunk_idx in range(start_chunk, target_end_chunk):
            t_chunk0 = time.time()
            self.refresh_heartbeat(job_id, chunk_idx + 1)

            for _ in range(reports_per_chunk):
                sim.step(steps_per_report)
                current_step += steps_per_report
                t_ns = round(current_step * self.cfg.dt_ps / 1000.0, 6)
                st = sim.context.getState(getPositions=True, getEnergy=True, enforcePeriodicBox=True)
                pe = float(st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))
                ke = float(st.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole))
                if not (math.isfinite(pe) and math.isfinite(ke)):
                    raise RuntimeError(f"Non-finite energy in {job_id} at step {current_step}: PE={pe}, KE={ke}")
                bv = st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
                box_diag = np.array([float(bv[0, 0]), float(bv[1, 1]), float(bv[2, 2])])
                row = extract_ws5_frame_observables(
                    st.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
                    box_diag,
                    t_idx,
                    time_ns=t_ns,
                    step=current_step,
                )
                row.update(
                    {
                        "job_id": job_id,
                        "arm_name": arm_name,
                        "arm_code": job["arm_code"],
                        "replicate": int(job["replicate"]),
                        "potential_energy_kJ_mol": pe,
                        "kinetic_energy_kJ_mol": ke,
                    }
                )
                existing_rows.append(row)

            # End of chunk: atomically checkpoint portable OpenMM State XML + Parquet + Progress JSON
            chunk_wall = time.time() - t_chunk0
            prior_wall_s += chunk_wall
            ckpt_state = sim.context.getState(
                getPositions=True,
                getVelocities=True,
                getParameters=True,
                enforcePeriodicBox=True,
            )
            _atomic_write_gzip_text(state_path, mm.XmlSerializer.serialize(ckpt_state))

            df_out = pd.DataFrame(existing_rows).drop_duplicates(subset=["step"], keep="last").sort_values("step").reset_index(drop=True)
            _atomic_write_parquet(pq_path, df_out)

            ns_simulated_chunk = self.cfg.chunk_duration_ns
            ns_per_day = (ns_simulated_chunk / max(chunk_wall, 1e-6)) * 86400.0
            completed_chunks = chunk_idx + 1
            prog_payload = {
                "job_id": job_id,
                "arm_name": arm_name,
                "replicate": int(job["replicate"]),
                "completed_chunks": completed_chunks,
                "total_chunks": self.cfg.total_chunks_per_traj,
                "current_step": current_step,
                "current_time_ns": round(current_step * self.cfg.dt_ps / 1000.0, 6),
                "wall_seconds_elapsed": round(prior_wall_s, 3),
                "last_chunk_ns_per_day": round(ns_per_day, 2),
                "platform": plat_name,
                "worker_id": self.worker_id,
                "updated_at_epoch": time.time(),
            }
            _atomic_write_json(prog_path, prog_payload)
            self.refresh_heartbeat(job_id, completed_chunks)
            if verbose:
                last_r = df_out.iloc[-1]
                print(
                    f"  [{job_id}] Chunk {completed_chunks}/{self.cfg.total_chunks_per_traj} saved "
                    f"({prog_payload['current_time_ns']:.2f} ns | {ns_per_day:.1f} ns/day | "
                    f"R214-D338={last_r['R214_D338_CA_A']:.2f} A | Y294-Y397={last_r['Y294_Y397_OH_A']:.2f} A)"
                )

        is_complete = (current_step >= self.cfg.total_steps_per_traj)
        if is_complete:
            _atomic_write_json(
                self._done_path(job_id),
                {
                    "job_id": job_id,
                    "arm_name": arm_name,
                    "replicate": int(job["replicate"]),
                    "completed_at_epoch": time.time(),
                    "total_steps": current_step,
                    "total_time_ns": round(current_step * self.cfg.dt_ps / 1000.0, 6),
                    "wall_seconds_elapsed": round(prior_wall_s, 3),
                    "platform": plat_name,
                    "worker_id": self.worker_id,
                },
            )
            self.release_lock(job_id)

        return {
            "job_id": job_id,
            "completed": is_complete,
            "current_step": current_step,
            "current_time_ns": round(current_step * self.cfg.dt_ps / 1000.0, 6),
            "parquet_path": str(pq_path),
            "platform": plat_name,
        }

    def run_worker_loop(
        self,
        max_jobs: int | None = None,
        max_wall_hours: float | None = None,
        verbose: bool = True,
    ) -> list[dict[str, Any]]:
        """Continuously claim and execute interleaved shooting replicates until campaign or budget ends."""
        t0 = time.time()
        results: list[dict[str, Any]] = []
        for job in self.jobs:
            if max_jobs is not None and len(results) >= max_jobs:
                break
            if max_wall_hours is not None and (time.time() - t0) / 3600.0 >= max_wall_hours:
                if verbose:
                    print(f"[Worker {self.worker_id}] Reached max_wall_hours={max_wall_hours}; stopping cleanly.")
                break
            if not self.try_claim_job(job):
                continue
            try:
                res = self.run_job(job, verbose=verbose)
                results.append(res)
            finally:
                if self._done_path(job["job_id"]).exists():
                    self.release_lock(job["job_id"])
        return results

    def status_summary(self) -> pd.DataFrame:
        """Return a per-arm progress table across completed, in-progress, and pending replicates."""
        rows = []
        now = time.time()
        for job in self.jobs:
            jid = job["job_id"]
            done = self._done_path(jid).exists()
            prog_p = self._progress_path(jid)
            lock_p = self._lock_path(jid)
            ns_done = 0.0
            speed = np.nan
            if prog_p.exists():
                p = json.loads(prog_p.read_text(encoding="utf-8"))
                ns_done = float(p.get("current_time_ns", 0.0))
                speed = float(p.get("last_chunk_ns_per_day", np.nan))
            active_lock = False
            if not done and lock_p.exists():
                try:
                    lk = json.loads(lock_p.read_text(encoding="utf-8"))
                    active_lock = (now - float(lk.get("heartbeat_epoch", 0.0))) <= self.cfg.stale_lock_timeout_s
                except Exception:
                    active_lock = False
            state = "completed" if done else ("running" if active_lock else ("partial" if ns_done > 0 else "queued"))
            rows.append(
                {
                    "queue_order": job["queue_order"],
                    "job_id": jid,
                    "arm_name": job["arm_name"],
                    "replicate": job["replicate"],
                    "status": state,
                    "simulated_ns": ns_done,
                    "target_ns": job["traj_duration_ns"],
                    "ns_per_day": speed,
                }
            )
        return pd.DataFrame(rows)
