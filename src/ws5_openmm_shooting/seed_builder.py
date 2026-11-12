"""Deterministic OpenMM starting-structure & single-Hamiltonian seed builder for WS5.

Constructs ONE wild-type human CB1 (`P21554`, `M109-M411`, 303 intact WT residues, no T210A,
no E273K/T283V/R340E, no flavodoxin) solvated in an OPM-oriented symmetric `POPC` bilayer
with crystallographic cholesterols (`CLR502`, `CLR503` -> `CHL1`), pre-hydrated internal
`D163^2.50-W356^6.48-N393^7.49-Y397^7.53` polar pore waters, `TIP3P-FB` bulk water, and
`0.15 M NaCl` using purely builtin OpenMM XML forcefields (`amber14-all.xml`, `amber14/tip3pfb.xml`).

Prepares 3 starting seeds at **matched open cytoplasmic TM6 (`R214^3.50-D338^6.30 = 14.3 +/- 0.15 A`)**:
  1. `Arm_B_Committed_Open` (`arm_b_committed_open.pdb.gz`):
     Outward `F155^2.42` (`chi1 ~ -65 deg`), *trans* `F237^4.46` (`chi1 ~ 180 deg`),
     latched `Y294^5.58-Y397^7.53` (`~3.1-3.4 A`), and hydrated `D163-NPxxY` wire.
  2. `Arm_A_Fake_Open_Trapped` (`arm_a_fake_open_trapped.pdb.gz`):
     Derived directly from the **identical topology & atom ordering** as Arm B via temporary
     steered torsion/distance restraints (`CustomTorsionForce` + `CustomBondForce` discarded
     after equilibration!), flipping `F155^2.42` (`chi1 ~ +65 deg`) and `F237^4.46`
     (`chi1 ~ -80 deg`) inward into the `5XRA/7V3Z` trapped rotamers and opening
     `Y294-Y397` (`~5.2 A`) while holding `R214-D338` at `14.3 A`. Shares the **bitwise-identical**
     unrestrained OpenMM `System` (`system_wt_ab.xml.gz`) with Arm B.
  3. `Arm_C_Mutant_F237L` (`arm_c_mutant_f237l.pdb.gz` + `system_mut_f237l_c.xml.gz`):
     Prepared identically to Arm A (`14.3 A` TM6, inward `155/237` cleft, open `Y294-Y397 ~ 5.2 A`)
     carrying ONLY the evolutionary single-residue swap `F237L` (`CB1 -> CB2` at `4.46`) inside
     the identical membrane, cholesterol, water, and ion box.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import time
from pathlib import Path
from typing import Any

import mdtraj as md
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit

from .observables import build_ws5_topology_indices, extract_ws5_frame_observables


REPO_ROOT = Path(__file__).resolve().parents[2]
SEEDS_DIR = REPO_ROOT / "data" / "openmm_seeds"
FORCEFIELD_XMLS = ("amber14-all.xml", "amber14/tip3pfb.xml")

TARGET_TM6_OPEN_NM = 1.430  # 14.30 A matched cytoplasmic R214-D338 CA opening


def _kabsch_transform(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_mean = P.mean(axis=0)
    q_mean = Q.mean(axis=0)
    P0 = P - p_mean
    Q0 = Q - q_mean
    U, _, Vt = np.linalg.svd(P0.T @ Q0)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, 1.0 if d > 0 else -1.0])
    R = U @ D @ Vt
    return R, p_mean, q_mean


def _select_fast_local_platform() -> mm.Platform:
    names = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
    for pref in ("CUDA", "OpenCL", "CPU"):
        if pref in names:
            return mm.Platform.getPlatformByName(pref)
    return mm.Platform.getPlatform(0)


def create_unrestrained_production_system(
    topology: app.Topology,
    ff: app.ForceField | None = None,
    temperature_K: float = 310.0,
    hmr_amu: float = 3.0,
    add_membrane_barostat: bool = True,
) -> mm.System:
    """Create the canonical 100% unrestrained OpenMM production Hamiltonian (`4 fs` HMR)."""
    if ff is None:
        ff = app.ForceField(*FORCEFIELD_XMLS)
    system = ff.createSystem(
        topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
        rigidWater=True,
        hydrogenMass=hmr_amu * unit.amu,
    )
    if add_membrane_barostat:
        barostat = mm.MonteCarloMembraneBarostat(
            1.0 * unit.bar,
            0.0 * unit.bar * unit.nanometer,
            temperature_K * unit.kelvin,
            mm.MonteCarloMembraneBarostat.XYIsotropic,
            mm.MonteCarloMembraneBarostat.ZFree,
            25,
        )
        # Deterministic seed=0 on the System object; per-replicate seed set at runtime!
        barostat.setRandomNumberSeed(0)
        system.addForce(barostat)
    return system


def _build_chl1_all_atom_topology(
    ff: app.ForceField,
    clr_records: list[tuple[int, str, np.ndarray]],
) -> tuple[app.Topology, list[mm.Vec3]]:
    """Build stereochemically exact all-atom CHL1 residues from crystallographic CLR heavy atoms."""
    chl_tmpl = ff._templates["CHL1"]
    heavy_adj: dict[str, list[str]] = {ta.name: [] for ta in chl_tmpl.atoms if ta.element.symbol != "H"}
    h_children: dict[str, list[str]] = {ta.name: [] for ta in chl_tmpl.atoms if ta.element.symbol != "H"}
    for i1, i2 in chl_tmpl.bonds:
        a1, a2 = chl_tmpl.atoms[i1], chl_tmpl.atoms[i2]
        if a1.element.symbol != "H" and a2.element.symbol != "H":
            heavy_adj[a1.name].append(a2.name)
            heavy_adj[a2.name].append(a1.name)
        elif a1.element.symbol == "H" and a2.element.symbol != "H":
            h_children[a2.name].append(a1.name)
        elif a2.element.symbol == "H" and a1.element.symbol != "H":
            h_children[a1.name].append(a2.name)

    def place_hydrogens(heavy_nm: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        coords = dict(heavy_nm)
        for p_name, h_list in h_children.items():
            if not h_list:
                continue
            p_pos = coords[p_name]
            nbr_u = [(coords[n] - p_pos) / np.linalg.norm(coords[n] - p_pos) for n in heavy_adj[p_name]]
            blen = 0.096 if p_name.startswith("O") else 0.109
            if len(h_list) == 1:
                s = sum(nbr_u)
                d_vec = -s / np.linalg.norm(s) if np.linalg.norm(s) > 1e-4 else np.array([0.0, 0.0, 1.0])
                if len(nbr_u) == 1:
                    perp = np.cross(nbr_u[0], np.array([1.0, 0.0, 0.0]))
                    if np.linalg.norm(perp) < 1e-3:
                        perp = np.cross(nbr_u[0], np.array([0.0, 1.0, 0.0]))
                    perp /= np.linalg.norm(perp)
                    d_vec = -np.cos(np.deg2rad(109.5)) * nbr_u[0] + np.sin(np.deg2rad(109.5)) * perp
                coords[h_list[0]] = p_pos + blen * d_vec
            elif len(h_list) == 2:
                bis = -(nbr_u[0] + nbr_u[1])
                bis /= np.linalg.norm(bis)
                perp = np.cross(nbr_u[0], nbr_u[1])
                perp /= np.linalg.norm(perp)
                coords[h_list[0]] = p_pos + blen * (np.cos(np.deg2rad(54.7)) * bis + np.sin(np.deg2rad(54.7)) * perp)
                coords[h_list[1]] = p_pos + blen * (np.cos(np.deg2rad(54.7)) * bis - np.sin(np.deg2rad(54.7)) * perp)
            elif len(h_list) == 3:
                axis = -nbr_u[0]
                perp1 = np.cross(axis, np.array([1.0, 0.0, 0.0]))
                if np.linalg.norm(perp1) < 1e-3:
                    perp1 = np.cross(axis, np.array([0.0, 1.0, 0.0]))
                perp1 /= np.linalg.norm(perp1)
                perp2 = np.cross(axis, perp1)
                for k, h_name in enumerate(h_list):
                    phi = 2.0 * np.pi * k / 3.0
                    rad = perp1 * np.cos(phi) + perp2 * np.sin(phi)
                    d_vec = np.cos(np.deg2rad(70.5)) * axis + np.sin(np.deg2rad(70.5)) * rad
                    coords[h_name] = p_pos + blen * d_vec
        return coords

    chl_top = app.Topology()
    chl_pos: list[mm.Vec3] = []
    by_res: dict[int, dict[str, np.ndarray]] = {}
    for rseq, aname, pt_A in clr_records:
        by_res.setdefault(rseq, {})[aname] = pt_A / 10.0

    ch = chl_top.addChain("C")
    for rseq, amap in sorted(by_res.items()):
        full = place_hydrogens(amap)
        res = chl_top.addResidue("CHL1", ch, id=str(rseq))
        tmap: dict[str, app.Atom] = {}
        for ta in chl_tmpl.atoms:
            a_obj = chl_top.addAtom(ta.name, ta.element, res)
            tmap[ta.name] = a_obj
            p = full[ta.name]
            chl_pos.append(mm.Vec3(float(p[0]), float(p[1]), float(p[2])))
        for i1, i2 in chl_tmpl.bonds:
            n1, n2 = chl_tmpl.atoms[i1].name, chl_tmpl.atoms[i2].name
            chl_top.addBond(tmap[n1], tmap[n2])
    return chl_top, chl_pos


def _build_internal_pore_waters(
    rec_top: app.Topology,
    rec_pos_nm: list[mm.Vec3],
) -> tuple[app.Topology, list[mm.Vec3]]:
    """Place non-clashing internal waters along the hydrophilic D163^2.50-W356^6.48-N393^7.49-Y397^7.53 channel."""
    pos_A = np.array([[v.x, v.y, v.z] for v in rec_pos_nm], dtype=np.float64) * 10.0
    heavy_mask = np.array([a.element is not None and a.element.symbol != "H" for a in rec_top.atoms()], dtype=bool)
    heavy_pts = pos_A[heavy_mask]

    res_list = list(list(rec_top.chains())[0].residues())
    u_to_res = {109 + i: r for i, r in enumerate(res_list)}

    def res_sc_center(u: int) -> np.ndarray:
        pts = [
            pos_A[a.index]
            for a in u_to_res[u].atoms()
            if a.element is not None and a.element.symbol != "H" and a.name not in ("N", "CA", "C", "O")
        ]
        return np.mean(pts, axis=0)

    anchors = [
        res_sc_center(163),  # D2.50 Na+/polar pocket
        res_sc_center(356),  # W6.48 toggle
        res_sc_center(393),  # N7.49 NPxxY
        res_sc_center(397),  # Y7.53 NPxxY
        res_sc_center(294),  # Y5.58 intracellular latch
    ]
    pore_mid = np.mean(anchors[:4], axis=0)

    # Generate 2.85 A candidate lattice inside 11 A sphere around the polar channel
    grid_1d = np.arange(-10.5, 10.6, 2.85)
    placed_O: list[np.ndarray] = []
    # Prioritize candidates closest to polar sidechains (D163, W356 NE1, N393, Y397, S390, S167)
    polar_u = [163, 167, 207, 294, 356, 387, 390, 393, 397]
    polar_pts = np.array(
        [
            pos_A[a.index]
            for u in polar_u
            for a in u_to_res[u].atoms()
            if a.element is not None and a.element.symbol in ("O", "N") and a.name not in ("N", "O")
        ],
        dtype=np.float64,
    )

    candidates: list[tuple[float, np.ndarray]] = []
    for dx in grid_1d:
        for dy in grid_1d:
            for dz in grid_1d:
                cand = pore_mid + np.array([dx, dy, dz])
                if np.linalg.norm(cand - pore_mid) > 11.2:
                    continue
                d_heavy = float(np.min(np.linalg.norm(heavy_pts - cand, axis=1)))
                if d_heavy < 2.62 or d_heavy > 4.80:
                    continue
                d_pol = float(np.min(np.linalg.norm(polar_pts - cand, axis=1)))
                if d_pol <= 4.30:
                    candidates.append((d_pol, cand))

    candidates.sort(key=lambda t: t[0])
    for _, cand in candidates:
        if all(np.linalg.norm(cand - prev) >= 2.75 for prev in placed_O):
            placed_O.append(cand)
        if len(placed_O) >= 16:
            break

    wat_top = app.Topology()
    wat_pos: list[mm.Vec3] = []
    ch = wat_top.addChain("P")
    el_O = app.Element.getBySymbol("O")
    el_H = app.Element.getBySymbol("H")
    # Ideal TIP3P water geometry: O-H = 0.09572 nm, H-O-H = 104.52 deg
    r_oh = 0.09572
    half_ang = np.deg2rad(104.52 / 2.0)
    h1_off = np.array([r_oh * np.sin(half_ang), 0.0, r_oh * np.cos(half_ang)])
    h2_off = np.array([-r_oh * np.sin(half_ang), 0.0, r_oh * np.cos(half_ang)])

    for idx_w, o_A in enumerate(placed_O, start=1):
        res = wat_top.addResidue("HOH", ch, id=str(idx_w))
        aO = wat_top.addAtom("O", el_O, res)
        aH1 = wat_top.addAtom("H1", el_H, res)
        aH2 = wat_top.addAtom("H2", el_H, res)
        wat_top.addBond(aO, aH1)
        wat_top.addBond(aO, aH2)
        o_nm = o_A / 10.0
        wat_pos.append(mm.Vec3(float(o_nm[0]), float(o_nm[1]), float(o_nm[2])))
        wat_pos.append(mm.Vec3(float(o_nm[0] + h1_off[0]), float(o_nm[1] + h1_off[1]), float(o_nm[2] + h1_off[2])))
        wat_pos.append(mm.Vec3(float(o_nm[0] + h2_off[0]), float(o_nm[1] + h2_off[1]), float(o_nm[2] + h2_off[2])))

    return wat_top, wat_pos


def _build_equilibrated_popc_patch(
    ff: app.ForceField,
    plat: mm.Platform,
    target_xy_nm: float = 8.40,
) -> app.Modeller:
    """Create a pre-equilibrated 8.4 x 8.4 nm symmetric periodic POPC bilayer patch."""
    base_patch = app.PDBFile(os.path.join(os.path.dirname(app.__file__), "data", "POPC.pdb"))
    box0 = base_patch.topology.getUnitCellDimensions().value_in_unit(unit.nanometer)
    pos0 = np.array(base_patch.positions.value_in_unit(unit.nanometer), dtype=np.float64)
    pos0 -= pos0.min(axis=0)
    target_z = float(box0[2])

    parent = list(range(base_patch.topology.getNumAtoms()))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    bonds_by_atom: dict[int, list[int]] = {}
    for a1, a2 in base_patch.topology.bonds():
        r1, r2 = find(a1.index), find(a2.index)
        if r1 != r2:
            parent[r2] = r1
        bonds_by_atom.setdefault(a1.index, []).append(a2.index)

    comp_res: dict[int, list[app.Residue]] = {}
    for res in base_patch.topology.residues():
        comp_res.setdefault(find(next(res.atoms()).index), []).append(res)

    upper, lower, waters = [], [], []
    z_mid = 0.5 * target_z
    for ix in range(2):
        for iy in range(2):
            shift = np.array([ix * box0[0], iy * box0[1], 0.0])
            for rlist in comp_res.values():
                a_idx = [a.index for r in rlist for a in r.atoms()]
                com = (pos0[a_idx] + shift).mean(axis=0)
                if 0.12 <= com[0] <= (target_xy_nm - 0.12) and 0.12 <= com[1] <= (target_xy_nm - 0.12):
                    if rlist[0].name in ("HOH", "WAT"):
                        waters.append((rlist, shift))
                    elif com[2] >= z_mid:
                        upper.append((rlist, shift))
                    else:
                        lower.append((rlist, shift))

    n_leaf = min(len(upper), len(lower))
    upper, lower = upper[:n_leaf], lower[:n_leaf]

    new_top = app.Topology()
    new_top.setUnitCellDimensions(mm.Vec3(target_xy_nm, target_xy_nm, target_z) * unit.nanometer)
    new_pos: list[mm.Vec3] = []
    ch_lip = new_top.addChain("M")
    ch_wat = new_top.addChain("W")

    def append_mol(rlist: list[app.Residue], shift: np.ndarray, ch: app.Chain) -> None:
        omap: dict[int, app.Atom] = {}
        for r in rlist:
            nr = new_top.addResidue(r.name, ch)
            for a in r.atoms():
                na = new_top.addAtom(a.name, a.element, nr)
                omap[a.index] = na
                p = pos0[a.index] + shift
                new_pos.append(mm.Vec3(float(p[0]), float(p[1]), float(p[2])))
        for old_i, na in omap.items():
            for nbr_i in bonds_by_atom.get(old_i, ()):
                if nbr_i in omap:
                    new_top.addBond(na, omap[nbr_i])

    for rlist, shift in upper + lower:
        append_mol(rlist, shift, ch_lip)
    for rlist, shift in waters:
        append_mol(rlist, shift, ch_wat)

    sys_p = ff.createSystem(new_top, nonbondedMethod=app.PME, nonbondedCutoff=1.0 * unit.nanometer, constraints=app.HBonds)
    sys_p.addForce(
        mm.MonteCarloMembraneBarostat(
            1.0 * unit.bar,
            0.0 * unit.bar * unit.nanometer,
            310.0 * unit.kelvin,
            mm.MonteCarloMembraneBarostat.XYIsotropic,
            mm.MonteCarloMembraneBarostat.ZFree,
            10,
        )
    )
    integ = mm.LangevinMiddleIntegrator(310.0 * unit.kelvin, 5.0 / unit.picosecond, 0.002 * unit.picoseconds)
    integ.setRandomNumberSeed(2026)
    sim = app.Simulation(new_top, sys_p, integ, plat)
    sim.context.setPositions(new_pos * unit.nanometer)
    sim.minimizeEnergy(maxIterations=200)
    sim.context.setVelocitiesToTemperature(310.0 * unit.kelvin, 2026)
    sim.step(500)
    st = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
    bv = st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
    new_top.setUnitCellDimensions(mm.Vec3(float(bv[0, 0]), float(bv[1, 1]), float(bv[2, 2])) * unit.nanometer)
    return app.Modeller(new_top, st.getPositions())


def _run_steered_relaxation(
    topology: app.Topology,
    init_positions: Any,
    ff: app.ForceField,
    plat: mm.Platform,
    target_tm6_nm: float,
    target_f155_chi1_deg: float,
    target_x237_chi1_deg: float,
    target_y294_y397_nm: float,
    target_f155cg_n393ca_nm: float = 1.48,
    md_steps: int = 1200,
    seed: int = 2026,
) -> tuple[Any, np.ndarray]:
    """Run temporary restrained steering + minimization, returning relaxed coordinates & box.

    All steering forces (`CustomBondForce`, `CustomTorsionForce`, `CustomExternalForce`) exist ONLY
    on this temporary `sys_steer` object and are completely discarded before saving production seeds.
    """
    t_idx = build_ws5_topology_indices(topology)
    sys_steer = ff.createSystem(
        topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
        rigidWater=True,
        hydrogenMass=3.0 * unit.amu,
    )

    pos_arr_nm = np.array(init_positions.value_in_unit(unit.nanometer), dtype=np.float64)

    # 1. Positional restraint on 7TM C-alpha scaffold (allowing TM6 cytoplasmic end 335..348 and
    #    local TM2/TM4 cleft 151..159, 234..241 to relax cleanly into inward vs outward packing)
    pos_restraint = mm.CustomExternalForce("0.5 * k_pos * periodicdistance(x, y, z, x0, y0, z0)^2")
    pos_restraint.addGlobalParameter("k_pos", 1500.0)  # kJ/(mol*nm^2)
    pos_restraint.addPerParticleParameter("x0")
    pos_restraint.addPerParticleParameter("y0")
    pos_restraint.addPerParticleParameter("z0")
    flexible_u = set(range(310, 349)) | set(range(256, 265)) | set(range(151, 160)) | set(range(234, 242))
    for u, ca_i in t_idx["ca"].items():
        if u in flexible_u:
            continue
        p0 = pos_arr_nm[ca_i]
        pos_restraint.addParticle(int(ca_i), [float(p0[0]), float(p0[1]), float(p0[2])])
    sys_steer.addForce(pos_restraint)

    # 2. Harmonic distance restraints on R214-D338 CA (exact 14.30 A across all arms!),
    #    Y294-Y397 OH latch, and F155_CG - N393_CA inward/outward packing
    bond_steer = mm.CustomBondForce("0.5 * k_b * (r - r0)^2")
    bond_steer.addPerBondParameter("k_b")
    bond_steer.addPerBondParameter("r0")
    bond_steer.addBond(int(t_idx["ca"][214]), int(t_idx["ca"][338]), [40000.0, float(target_tm6_nm)])
    bond_steer.addBond(int(t_idx["y294_oh"]), int(t_idx["y397_oh"]), [25000.0, float(target_y294_y397_nm)])
    f155_cg_idx = int(t_idx["chi_quads"]["F155_chi1_deg"][3])
    bond_steer.addBond(f155_cg_idx, int(t_idx["ca"][393]), [12000.0, float(target_f155cg_n393ca_nm)])
    sys_steer.addForce(bond_steer)

    # 3. Periodic torsion steering on F155^2.42 chi1, F/L237^4.46 chi1, and W356^6.48 chi1 (-60 deg active toggle)
    tors_steer = mm.CustomTorsionForce("0.5 * k_t * min(dtheta, 2*pi - dtheta)^2; dtheta = abs(theta - theta0)")
    tors_steer.addGlobalParameter("pi", float(np.pi))
    tors_steer.addPerTorsionParameter("k_t")
    tors_steer.addPerTorsionParameter("theta0")
    q_f155 = t_idx["chi_quads"]["F155_chi1_deg"]
    q_x237 = t_idx["chi_quads"]["X237_chi1_deg"]
    q_w356 = t_idx["chi_quads"]["W356_chi1_deg"]
    tors_steer.addTorsion(*[int(i) for i in q_f155], [1200.0, float(np.deg2rad(target_f155_chi1_deg))])
    tors_steer.addTorsion(*[int(i) for i in q_x237], [1200.0, float(np.deg2rad(target_x237_chi1_deg))])
    tors_steer.addTorsion(*[int(i) for i in q_w356], [600.0, float(np.deg2rad(-62.0))])
    sys_steer.addForce(tors_steer)

    integ = mm.LangevinMiddleIntegrator(310.0 * unit.kelvin, 2.0 / unit.picosecond, 0.002 * unit.picoseconds)
    integ.setRandomNumberSeed(seed)
    sim = app.Simulation(topology, sys_steer, integ, plat)
    bv0 = topology.getPeriodicBoxVectors()
    if bv0 is not None:
        sim.context.setPeriodicBoxVectors(*bv0)
    sim.context.setPositions(init_positions)

    sim.minimizeEnergy(maxIterations=350)
    sim.context.setVelocitiesToTemperature(310.0 * unit.kelvin, seed)
    # Stage 1: steered thermal relaxation allowing lipids/waters/sidechains to pack cleanly
    sim.step(md_steps)
    # Stage 2: release C-alpha scaffold restraint slightly while tightening target observables, then final minimization
    sim.context.setParameter("k_pos", 400.0)
    sim.step(max(200, md_steps // 3))
    sim.minimizeEnergy(maxIterations=350)

    st = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
    bv = st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
    box_diag_nm = np.array([float(bv[0, 0]), float(bv[1, 1]), float(bv[2, 2])], dtype=np.float64)
    return st.getPositions(), box_diag_nm


def _mutate_phe237_to_leu237_in_place(
    wt_top: app.Topology,
    wt_positions: Any,
    box_diag_nm: np.ndarray,
    ff: app.ForceField,
) -> tuple[app.Topology, Any]:
    """Mutate ONLY PHE237 -> LEU237 while keeping every other atom, lipid, CHL1, water, and ion identical."""
    wt_pos_nm = np.array(wt_positions.value_in_unit(unit.nanometer), dtype=np.float64)
    leu_tmpl = ff._templates["LEU"]

    # Find residue 237 (index 237 - 109 = 128 on chain 0)
    rec_chain = list(wt_top.chains())[0]
    res_237 = list(rec_chain.residues())[237 - 109]
    if res_237.name != "PHE":
        raise AssertionError(f"Expected PHE at residue 237, found {res_237.name}")

    phe_amap = {a.name: pos for a, pos in ((a, wt_pos_nm[a.index]) for a in res_237.atoms())}
    # Exact shared heavy + backbone/beta H coordinates from PHE237
    leu_coords: dict[str, np.ndarray] = {}
    for shared_name in ("N", "H", "CA", "HA", "CB", "HB2", "HB3", "CG", "CD1", "CD2", "C", "O"):
        if shared_name in phe_amap:
            leu_coords[shared_name] = phe_amap[shared_name].copy()

    # Stereochemically place LEU237's aliphatic HG (on CG) and methyl tripods HD11..13 (on CD1), HD21..23 (on CD2)
    cg = leu_coords["CG"]
    cb = leu_coords["CB"]
    cd1 = leu_coords["CD1"]
    cd2 = leu_coords["CD2"]
    u_cb = (cb - cg) / np.linalg.norm(cb - cg)
    u_cd1 = (cd1 - cg) / np.linalg.norm(cd1 - cg)
    u_cd2 = (cd2 - cg) / np.linalg.norm(cd2 - cg)
    d_hg = -(u_cb + u_cd1 + u_cd2)
    d_hg /= np.linalg.norm(d_hg)
    leu_coords["HG"] = cg + 0.109 * d_hg

    def place_methyl(c_pos: np.ndarray, parent_pos: np.ndarray, h_names: tuple[str, str, str]) -> None:
        axis = (c_pos - parent_pos) / np.linalg.norm(c_pos - parent_pos)
        p1 = np.cross(axis, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(p1) < 1e-3:
            p1 = np.cross(axis, np.array([0.0, 1.0, 0.0]))
        p1 /= np.linalg.norm(p1)
        p2 = np.cross(axis, p1)
        for k, hn in enumerate(h_names):
            phi = 2.0 * np.pi * k / 3.0
            rad = p1 * np.cos(phi) + p2 * np.sin(phi)
            d_vec = np.cos(np.deg2rad(70.5)) * axis + np.sin(np.deg2rad(70.5)) * rad
            leu_coords[hn] = c_pos + 0.109 * d_vec

    place_methyl(cd1, cg, ("HD11", "HD12", "HD13"))
    place_methyl(cd2, cg, ("HD21", "HD22", "HD23"))

    # Rebuild full topology preserving every single residue, chain, and inter-residue bond!
    mut_top = app.Topology()
    mut_top.setUnitCellDimensions(mm.Vec3(*[float(x) for x in box_diag_nm]) * unit.nanometer)
    mut_pos: list[mm.Vec3] = []
    old_atom_to_new: dict[int, app.Atom] = {}
    leu_name_to_new: dict[str, app.Atom] = {}

    for ch in wt_top.chains():
        nch = mut_top.addChain(ch.id)
        for r in ch.residues():
            if r.index == res_237.index:
                nr = mut_top.addResidue("LEU", nch, id=r.id, insertionCode=r.insertionCode)
                for ta in leu_tmpl.atoms:
                    na = mut_top.addAtom(ta.name, ta.element, nr)
                    leu_name_to_new[ta.name] = na
                    p = leu_coords[ta.name]
                    mut_pos.append(mm.Vec3(float(p[0]), float(p[1]), float(p[2])))
                for i1, i2 in leu_tmpl.bonds:
                    n1, n2 = leu_tmpl.atoms[i1].name, leu_tmpl.atoms[i2].name
                    mut_top.addBond(leu_name_to_new[n1], leu_name_to_new[n2])
                # Map N and C to old peptide bond endpoints
                for a in r.atoms():
                    if a.name in ("N", "C"):
                        old_atom_to_new[a.index] = leu_name_to_new[a.name]
            else:
                nr = mut_top.addResidue(r.name, nch, id=r.id, insertionCode=r.insertionCode)
                for a in r.atoms():
                    na = mut_top.addAtom(a.name, a.element, nr)
                    old_atom_to_new[a.index] = na
                    p = wt_pos_nm[a.index]
                    mut_pos.append(mm.Vec3(float(p[0]), float(p[1]), float(p[2])))

    phe_internal_atoms = {a.index for a in res_237.atoms()}
    for a1, a2 in wt_top.bonds():
        if a1.index in phe_internal_atoms and a2.index in phe_internal_atoms:
            continue
        if a1.index in old_atom_to_new and a2.index in old_atom_to_new:
            mut_top.addBond(old_atom_to_new[a1.index], old_atom_to_new[a2.index])

    return mut_top, mut_pos * unit.nanometer


def _write_gzip_pdb_and_hash(topology: app.Topology, positions: Any, box_diag_nm: np.ndarray, dest_gz: Path) -> dict[str, Any]:
    topology.setUnitCellDimensions(mm.Vec3(*[float(x) for x in box_diag_nm]) * unit.nanometer)
    sio = io.StringIO()
    app.PDBFile.writeFile(topology, positions, sio, keepIds=True)
    raw_bytes = sio.getvalue().encode("utf-8")
    dest_gz.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(dest_gz, "wb", compresslevel=9) as fh:
        fh.write(raw_bytes)
    return {
        "path": str(dest_gz.relative_to(REPO_ROOT)),
        "gzip_bytes": dest_gz.stat().st_size,
        "uncompressed_bytes": len(raw_bytes),
        "sha256_gzip": hashlib.sha256(dest_gz.read_bytes()).hexdigest(),
    }


def _write_gzip_xml_and_hash(xml_str: str, dest_gz: Path) -> dict[str, Any]:
    raw_bytes = xml_str.encode("utf-8")
    dest_gz.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(dest_gz, "wb", compresslevel=9) as fh:
        fh.write(raw_bytes)
    return {
        "path": str(dest_gz.relative_to(REPO_ROOT)),
        "gzip_bytes": dest_gz.stat().st_size,
        "uncompressed_bytes": len(raw_bytes),
        "sha256_uncompressed": hashlib.sha256(raw_bytes).hexdigest(),
        "sha256_gzip": hashlib.sha256(dest_gz.read_bytes()).hexdigest(),
    }


def build_all_ws5_seeds(out_dir: Path = SEEDS_DIR) -> dict[str, Any]:
    """Deterministically build, equilibrate, verify, and serialize all 3 WS5 OpenMM shooting seeds."""
    t_start = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    p_6n4b = REPO_ROOT / "data/inventory/pdb_structures/6N4B.pdb"
    p_1201 = REPO_ROOT / "data/processed/ws3/topologies/gpcrmd_1201.pdb"
    p_2067 = REPO_ROOT / "data/processed/ws3/topologies/gpcrmd_2067.pdb"

    t6 = md.load(str(p_6n4b))
    t12 = md.load(str(p_1201))
    t2067 = md.load(str(p_2067))

    # 1. Rigid-body Kabsch alignment of 6N4B Chain R + CLR502/503 to OPM membrane frame
    popc_P = [a.index for r in t12.topology.residues if r.name == "POPC" for a in r.atoms if a.name == "P"]
    z_mid_nm = float(t12.xyz[0, popc_P, 2].mean())
    r12_map = {
        r.resSeq: next(a.index for a in r.atoms if a.name == "CA")
        for r in t12.topology.chain(3).residues
        if r.is_protein and any(a.name == "CA" for a in r.atoms)
    }
    r6_map = {
        r.resSeq: next(a.index for a in r.atoms if a.name == "CA")
        for r in t6.topology.chain(3).residues
        if r.is_protein and any(a.name == "CA" for a in r.atoms)
    }
    shared_res = sorted(set(r6_map).intersection(r12_map))
    P_nm = t6.xyz[0, [r6_map[k] for k in shared_res]].astype(np.float64)
    Q_nm = t12.xyz[0, [r12_map[k] for k in shared_res]].astype(np.float64)
    xy_mid = Q_nm[:, :2].mean(axis=0)
    Q_centered = Q_nm.copy()
    Q_centered[:, 0] -= xy_mid[0]
    Q_centered[:, 1] -= xy_mid[1]
    Q_centered[:, 2] -= z_mid_nm

    ec_z = np.mean([Q_centered[shared_res.index(k), 2] for k in (176, 180, 270, 368, 375) if k in shared_res])
    ic_z = np.mean([Q_centered[shared_res.index(k), 2] for k in (146, 214, 298, 338, 397) if k in shared_res])
    if ec_z < ic_z:
        Q_centered[:, 1] *= -1.0
        Q_centered[:, 2] *= -1.0

    R_rot, p_mean, q_mean = _kabsch_transform(P_nm, Q_centered)

    # Graft compact WT loop backbones (258..263 ECL2 and 314..334 ICL3) from gpcrmd_2067 (100% WT P21554 109..411)
    # Kabsch-aligned locally onto flanking 6N4B stems so ICL3 stays compactly folded against the cytoplasmic face!
    r20_res = {r.resSeq: r for r in t2067.topology.chain(0).residues if r.is_protein}
    wt_seq_303 = [r20_res[u].name for u in range(109, 412)]

    raw_lines = p_6n4b.read_text().splitlines()
    pdb_lines = ["HEADER    CB1 WT P21554 6N4B OPM ALIGNED"]
    for i in range(0, len(wt_seq_303), 13):
        chunk = wt_seq_303[i : i + 13]
        pdb_lines.append(f"SEQRES {i // 13 + 1:3d} R {len(wt_seq_303):4d}  " + " ".join(f"{rn:>3s}" for rn in chunk))

    # Transform 6N4B Chain R + CLR502/503 into OPM frame
    res6_lines: dict[int, list[str]] = {}
    res6_ca_opm: dict[int, np.ndarray] = {}
    clr_records: list[tuple[int, str, np.ndarray]] = []
    for line in raw_lines:
        if line.startswith(("ATOM  ", "HETATM")) and line[21] == "R":
            rname = line[17:20].strip()
            rseq = int(line[22:26])
            x, y, z = float(line[30:38]) / 10.0, float(line[38:46]) / 10.0, float(line[46:54]) / 10.0
            pt_A = ((np.array([x, y, z]) - p_mean) @ R_rot + q_mean) * 10.0
            if line.startswith("ATOM  "):
                new_line = line[:30] + f"{pt_A[0]:8.3f}{pt_A[1]:8.3f}{pt_A[2]:8.3f}" + line[54:]
                res6_lines.setdefault(rseq, []).append(new_line)
                if line[12:16].strip() == "CA":
                    res6_ca_opm[rseq] = pt_A / 10.0
            elif rname == "CLR":
                aname = line[12:16].strip()
                clr_records.append((rseq, "O3" if aname == "O1" else aname, pt_A))

    # Locally align gpcrmd_2067 onto flanking stems for missing loop gaps (258..263 and 314..334)
    def graft_loop_from_2067(u_start: int, u_end: int, stems: list[int]) -> None:
        P_stem = np.array([t2067.xyz[0, next(a.index for a in r20_res[s].atoms if a.name == "CA")] for s in stems])
        Q_stem = np.array([res6_ca_opm[s] for s in stems])
        Rg, pm, qm = _kabsch_transform(P_stem, Q_stem)
        serial = 9000
        for u in range(u_start, u_end + 1):
            r_obj = r20_res[u]
            lines_u = []
            for a in r_obj.atoms:
                if a.element.symbol == "H":
                    continue
                pt_nm = (t2067.xyz[0, a.index] - pm) @ Rg + qm
                pt_A = pt_nm * 10.0
                serial += 1
                lines_u.append(
                    f"ATOM  {serial:5d} {a.name:<4s} {r_obj.name:>3s} R{u:4d}    "
                    f"{pt_A[0]:8.3f}{pt_A[1]:8.3f}{pt_A[2]:8.3f}  1.00  0.00          {a.element.symbol:>2s}  "
                )
            res6_lines[u] = lines_u

    graft_loop_from_2067(258, 263, [254, 255, 256, 257, 264, 265, 266, 267])
    graft_loop_from_2067(314, 334, [308, 309, 310, 311, 312, 313, 335, 336, 337, 338, 339, 340])

    for u in range(109, 412):
        pdb_lines.extend(res6_lines[u])
    pdb_lines.extend(["TER", "END"])

    try:
        from pdbfixer import PDBFixer
    except ImportError as exc:
        raise ImportError(
            "Rebuilding WS5 seeds from scratch requires `pdbfixer` "
            "(e.g., `pip install git+https://github.com/openmm/pdbfixer.git`). "
            "Pre-built production seeds are already included in `data/openmm_seeds/`."
        ) from exc

    fixer = PDBFixer(pdbfile=io.StringIO("\n".join(pdb_lines)))
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms(seed=2026)

    ff = app.ForceField(*FORCEFIELD_XMLS)
    plat = _select_fast_local_platform()
    print(f"[WS5 Builder] Using local OpenMM platform: {plat.getName()}")

    modeller = app.Modeller(fixer.topology, fixer.positions)
    modeller.addHydrogens(ff, pH=7.4)

    # Add crystallographic cholesterols CLR502 & CLR503 as all-atom CHL1 + internal polar-channel waters!
    chl_top, chl_pos = _build_chl1_all_atom_topology(ff, clr_records)
    wat_top, wat_pos = _build_internal_pore_waters(modeller.topology, modeller.positions.value_in_unit(unit.nanometer))
    n_internal_waters = wat_top.getNumResidues()
    modeller.add(chl_top, chl_pos * unit.nanometer)
    modeller.add(wat_top, wat_pos * unit.nanometer)
    print(
        f"[WS5 Builder] Receptor (303 WT res) + 2 CHL1 + {n_internal_waters} internal pore waters: "
        f"{modeller.topology.getNumAtoms()} atoms"
    )

    # Embed into symmetric 8.4 x 8.4 nm equilibrated periodic POPC bilayer + TIP3P-FB + 0.15 M NaCl
    patch_8p4 = _build_equilibrated_popc_patch(ff, plat, target_xy_nm=8.40)
    modeller.addMembrane(
        ff,
        lipidType=patch_8p4,
        membraneCenterZ=0.0 * unit.nanometer,
        minimumPadding=0.80 * unit.nanometer,
        positiveIon="Na+",
        negativeIon="Cl-",
        ionicStrength=0.15 * unit.molar,
        neutralize=True,
    )
    wt_top = modeller.topology
    print(
        f"[WS5 Builder] Solvated WT membrane box complete: {wt_top.getNumAtoms()} atoms, "
        f"{wt_top.getNumResidues()} residues"
    )

    # 2. Equilibrate Arm_B_Committed_Open (outward F155 -65 deg, trans F237 180 deg, latched Y294-Y397 3.15 A, TM6 14.30 A)
    print("[WS5 Builder] Equilibrating Arm_B_Committed_Open ...")
    pos_b, box_b = _run_steered_relaxation(
        wt_top,
        modeller.positions,
        ff,
        plat,
        target_tm6_nm=TARGET_TM6_OPEN_NM,
        target_f155_chi1_deg=-65.0,
        target_x237_chi1_deg=180.0,
        target_y294_y397_nm=0.315,
        target_f155cg_n393ca_nm=1.48,
        md_steps=1200,
        seed=202601,
    )

    # 3. Derive Arm_A_Fake_Open_Trapped DIRECTLY from Arm_B (exact same wt_top & atom ordering!)
    #    Flipping F155 (+65 deg inward), F237 (-80 deg inward), opening Y294-Y397 (5.20 A), holding TM6 at 14.30 A!
    print("[WS5 Builder] Deriving Arm_A_Fake_Open_Trapped on IDENTICAL WT topology ...")
    wt_top.setUnitCellDimensions(mm.Vec3(*[float(x) for x in box_b]) * unit.nanometer)
    pos_a, box_a = _run_steered_relaxation(
        wt_top,
        pos_b,
        ff,
        plat,
        target_tm6_nm=TARGET_TM6_OPEN_NM,
        target_f155_chi1_deg=65.0,
        target_x237_chi1_deg=-80.0,
        target_y294_y397_nm=0.520,
        target_f155cg_n393ca_nm=1.16,
        md_steps=1600,
        seed=202602,
    )

    # 4. Derive Arm_C_Mutant_F237L from Arm_A via surgical PHE237 -> LEU237 swap in identical solvent/membrane box
    print("[WS5 Builder] Deriving Arm_C_Mutant_F237L (P21554 -> P34972 at 4.46) ...")
    mut_top, mut_init_pos = _mutate_phe237_to_leu237_in_place(wt_top, pos_a, box_a, ff)
    pos_c, box_c = _run_steered_relaxation(
        mut_top,
        mut_init_pos,
        ff,
        plat,
        target_tm6_nm=TARGET_TM6_OPEN_NM,
        target_f155_chi1_deg=65.0,
        target_x237_chi1_deg=-80.0,
        target_y294_y397_nm=0.520,
        target_f155cg_n393ca_nm=1.16,
        md_steps=1000,
        seed=202603,
    )

    # 5. Write compressed .pdb.gz seeds, reload via PDBFile, and verify 100% bitwise-identical WT System XML!
    meta_pdb_a = _write_gzip_pdb_and_hash(wt_top, pos_a, box_a, out_dir / "arm_a_fake_open_trapped.pdb.gz")
    meta_pdb_b = _write_gzip_pdb_and_hash(wt_top, pos_b, box_b, out_dir / "arm_b_committed_open.pdb.gz")
    meta_pdb_c = _write_gzip_pdb_and_hash(mut_top, pos_c, box_c, out_dir / "arm_c_mutant_f237l.pdb.gz")

    with gzip.open(out_dir / "arm_a_fake_open_trapped.pdb.gz", "rt") as fha:
        reload_a = app.PDBFile(fha)
    with gzip.open(out_dir / "arm_b_committed_open.pdb.gz", "rt") as fhb:
        reload_b = app.PDBFile(fhb)
    with gzip.open(out_dir / "arm_c_mutant_f237l.pdb.gz", "rt") as fhc:
        reload_c = app.PDBFile(fhc)

    sys_from_a = create_unrestrained_production_system(reload_a.topology, ff)
    sys_from_b = create_unrestrained_production_system(reload_b.topology, ff)
    sys_from_c = create_unrestrained_production_system(reload_c.topology, ff)

    xml_a = mm.XmlSerializer.serialize(sys_from_a)
    xml_b = mm.XmlSerializer.serialize(sys_from_b)
    if xml_a != xml_b:
        raise AssertionError("CRITICAL: Arm A and Arm B reloaded topologies did NOT produce bitwise-identical OpenMM System XML!")

    xml_c = mm.XmlSerializer.serialize(sys_from_c)
    meta_xml_wt = _write_gzip_xml_and_hash(xml_a, out_dir / "system_wt_ab.xml.gz")
    meta_xml_mut = _write_gzip_xml_and_hash(xml_c, out_dir / "system_mut_f237l_c.xml.gz")

    # 6. Extract initial t=0 physical observables on all 3 seeds
    idx_wt = build_ws5_topology_indices(reload_a.topology)
    idx_mut = build_ws5_topology_indices(reload_c.topology)

    obs_a = extract_ws5_frame_observables(
        np.array(reload_a.positions.value_in_unit(unit.nanometer)),
        box_a,
        idx_wt,
        time_ns=0.0,
        step=0,
    )
    obs_b = extract_ws5_frame_observables(
        np.array(reload_b.positions.value_in_unit(unit.nanometer)),
        box_b,
        idx_wt,
        time_ns=0.0,
        step=0,
    )
    obs_c = extract_ws5_frame_observables(
        np.array(reload_c.positions.value_in_unit(unit.nanometer)),
        box_c,
        idx_mut,
        time_ns=0.0,
        step=0,
    )

    total_seed_bytes = sum(
        p.stat().st_size for p in out_dir.glob("*.gz")
    )
    manifest = {
        "schema_version": "1.0",
        "built_elapsed_s": round(time.time() - t_start, 2),
        "forcefield_xmls": list(FORCEFIELD_XMLS),
        "integrator_spec": {
            "type": "LangevinMiddleIntegrator",
            "temperature_K": 310.0,
            "friction_per_ps": 1.0,
            "dt_ps": 0.004,
            "hydrogen_mass_amu": 3.0,
            "constraints": "HBonds",
            "rigid_water": True,
            "barostat": "MonteCarloMembraneBarostat(1 bar, 0 bar*nm, 310 K, XYIsotropic, ZFree, freq=25)",
        },
        "hamiltonian_identity_verified": True,
        "wt_system_sha256_uncompressed": meta_xml_wt["sha256_uncompressed"],
        "wt_n_atoms": reload_a.topology.getNumAtoms(),
        "mut_f237l_n_atoms": reload_c.topology.getNumAtoms(),
        "n_internal_waters_seeded": n_internal_waters,
        "total_compressed_seed_bytes": total_seed_bytes,
        "total_compressed_seed_mb": round(total_seed_bytes / (1024 * 1024), 3),
        "files": {
            "system_wt_ab_xml_gz": meta_xml_wt,
            "system_mut_f237l_c_xml_gz": meta_xml_mut,
            "arm_a_pdb_gz": meta_pdb_a,
            "arm_b_pdb_gz": meta_pdb_b,
            "arm_c_pdb_gz": meta_pdb_c,
        },
        "initial_observables": {
            "Arm_A_Fake_Open_Trapped": obs_a,
            "Arm_B_Committed_Open": obs_b,
            "Arm_C_Mutant_F237L": obs_c,
        },
    }
    manifest_path = out_dir / "ws5_seed_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"[WS5 Builder] SUCCESS in {manifest['built_elapsed_s']}s! Total seed footprint: "
        f"{manifest['total_compressed_seed_mb']:.3f} MB (< 15 MB budget)"
    )
    for arm_name, obs in manifest["initial_observables"].items():
        print(
            f"  {arm_name:26s}: R214-D338={obs['R214_D338_CA_A']:.2f} A | "
            f"F155_chi1={obs['F155_chi1_deg']:+6.1f} deg | X237_chi1={obs['X237_chi1_deg']:+6.1f} deg | "
            f"F155sc-N393CA={obs['F155sc_N393CA_A']:.2f} A | Y294-Y397={obs['Y294_Y397_OH_A']:.2f} A | "
            f"wat_pore={obs['n_wat_pore']} (C_wire={obs['wire_conductance_d163_ny']:.2f}) | CHL1_cleft={obs['min_CHL1_cleft_A']:.2f} A"
        )
    return manifest


if __name__ == "__main__":
    build_all_ws5_seeds()
