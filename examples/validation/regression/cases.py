"""Regression case manifest and builders.

Every case is a dict in :data:`CASES` listing the on-disk inputs and the run
settings, so a reader sees exactly what is run. :func:`build` turns a case
into a :class:`Run` (solver + evolve + per-step / per-sample hooks) that
``regress.py`` marches with its own instrumentation. Taichi must be
initialised before calling :func:`build`.

Case setups are copied from, or import, the scripts they reproduce:

* ``tracyarm_gen``  -> ``calibration/TracyArm/generate.py`` (defaults).
* coastal cases     -> ``setrun_web.py`` (celeris-format loader, config.json).
* ``breach``        -> ``examples/validation/breach/breach_validation.py``.
* ``malpasset*``    -> ``examples/validation/malpasset/{prep,run}_malpasset.py``.
"""

import importlib.util
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from math import radians
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from celeris.runner import Evolve
    from celeris.solver import Solver

CELERIS_DIR = Path(__file__).resolve().parents[3]
CALIBRATION_DIR = CELERIS_DIR.parent / "calibration"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import data_root

VALIDATION_OUT = data_root() / "celeris"
BREACH_SCRIPT = CELERIS_DIR / "examples/validation/breach/breach_validation.py"
MALPASSET_DIR = CELERIS_DIR / "examples/validation/malpasset"
TRACYARM_GENERATE = CALIBRATION_DIR / "TracyArm/generate.py"
TRACYARM_GOLDEN = CALIBRATION_DIR / "TracyArm/data/gen_v1.npz"

DRY_M = 0.01  # wet threshold for frames, envelopes and volume
RUNUP_DEPTH_M = 0.5  # TracyArm runup/trough wet threshold (RESULTS.md / NOTES.md)
MALPASSET_ARRIVAL_M = 0.05  # compare_malpasset.py ARRIVAL_DEPTH

CASES: dict[str, dict[str, object]] = {
    "tracyarm_gen": {
        "kind": "tracyarm",
        "path": CALIBRATION_DIR / "TracyArm/cases/Endicott_slide",
        "model": "SWE",
        "source": "MovingBodySlide, Lynett Endicott_slide config (generate.py defaults: "
        "wet_dhdt_only=1, emerge 1.0 m, delta 0.01, az-sign -1, carve_scar off)",
        "duration_s": 150.0,
        "frames_s": (30.0, 60.0, 100.0, 150.0),
        "sample_s": 1.0,
        "moving_bed": True,
    },
    "balboa": {
        "kind": "coastal",
        "path": CELERIS_DIR / "examples/Balboa",
        "model": "config.json (Bouss)",
        "duration_s": 120.0,
        "frames_s": (30.0, 60.0, 90.0, 120.0),
        "sample_s": 1.0,
    },
    "crescentcity": {
        "kind": "coastal",
        "path": CELERIS_DIR / "examples/CrescentCity",
        "model": "config.json (NLSW_or_Bous=0 -> SWE, no breaking key)",
        "duration_s": 120.0,
        "frames_s": (30.0, 60.0, 90.0, 120.0),
        "sample_s": 1.0,
    },
    "mavericks": {
        "kind": "coastal",
        "path": CELERIS_DIR / "examples/Mavericks",
        "model": "config.json (Bouss)",
        "duration_s": 120.0,
        "frames_s": (30.0, 60.0, 90.0, 120.0),
        "sample_s": 1.0,
    },
    "ventura": {
        "kind": "coastal",
        "path": CELERIS_DIR / "examples/Ventura",
        "model": "config.json (Bouss)",
        "duration_s": 120.0,
        "frames_s": (30.0, 60.0, 90.0, 120.0),
        "sample_s": 1.0,
    },
    "breach": {
        "kind": "breach",
        "path": VALIDATION_OUT / "breach",
        "model": "SWE, HydrographSource inlet, outlet stage kernel (breach_validation.py)",
        "duration_s": 2700.0,
        "frames_s": (1000.0, 2000.0, 2700.0),
        "sample_s": 10.0,  # breach_validation.SAMPLE_S
    },
    "malpasset": {
        "kind": "malpasset",
        "path": VALIDATION_OUT / "malpasset",
        "model": "SWE, breaking on, Manning 1/30, dx 15 m (run_malpasset.py plain)",
        "duration_s": 4000.0,
        "frames_s": (500.0, 1000.0, 2000.0, 4000.0),
        "sample_s": 1.0,  # run_malpasset.py --sample-dt
        "seed_wetting_m": 0.0,
    },
    "malpasset_seed": {
        "kind": "malpasset",
        "path": VALIDATION_OUT / "malpasset",
        "model": "as malpasset, plus run_malpasset.py --seed-wetting 0.02",
        "duration_s": 4000.0,
        "frames_s": (500.0, 1000.0, 2000.0, 4000.0),
        "sample_s": 1.0,
        "seed_wetting_m": 0.02,
    },
}


@dataclass
class Run:
    """A built case ready to march.

    Attributes:
        solver: The Celeris solver (state at t = 0 after ``Evolve_0``).
        evolve: The runner whose ``Evolve_Steps(step)`` advances one step.
        dt: Time step (s).
        n_steps: Steps to run.
        sample_every: Readback cadence in steps (the reproduced script's own).
        moving_bed: Re-read the bed at every sample (bed changes in time).
        post_step: Called after ``Evolve_Steps(step)`` (case-side kernels).
        sample: ``(eta, bed) -> {name: value}`` per-sample case series.
        metrics: ``series -> {name: scalar}`` from the collected series.
        arrays: Static arrays stored alongside the results.
    """

    solver: "Solver"
    evolve: "Evolve"
    dt: float
    n_steps: int
    sample_every: int
    moving_bed: bool = False
    post_step: Callable[[int], None] | None = None
    sample: Callable[[np.ndarray, np.ndarray], dict[str, object]] | None = None
    metrics: Callable[[dict[str, np.ndarray]], dict[str, float]] | None = None
    arrays: dict[str, np.ndarray] = field(default_factory=dict)


def _import(path: Path) -> ModuleType:
    """Import a script by path without running its ``__main__`` block."""
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cap(n_steps: int, max_steps: int | None) -> int:
    return n_steps if max_steps is None else min(n_steps, max_steps)


def _first_exceed(t: np.ndarray, depth: np.ndarray, thr: float) -> float:
    k = np.nonzero(depth > thr)[0]
    return float(t[k[0]]) if len(k) else float("nan")


def build(*, name: str, max_steps: int | None = None) -> Run:
    """Build a case by name (Taichi already initialised).

    Args:
        name: Key of :data:`CASES`.
        max_steps: Cap on the step count (smoke tests).

    Returns:
        The runnable case.
    """
    cfg = CASES[name]
    kind = str(cfg["kind"])
    builders = {
        "tracyarm": _build_tracyarm,
        "coastal": _build_coastal,
        "breach": _build_breach,
        "malpasset": _build_malpasset,
    }
    return builders[kind](cfg, max_steps)


def _build_coastal(cfg: dict[str, object], max_steps: int | None) -> Run:
    """Upstream example folder, driven as ``setrun_web.py`` does (headless)."""
    from celeris.domain import BoundaryConditions, Domain, Topodata
    from celeris.runner import Evolve
    from celeris.solver import Solver

    path = str(cfg["path"])
    topo = Topodata(datatype="celeris", path=path)
    bc = BoundaryConditions(celeris=True, path=path)
    dom = Domain(topodata=topo)
    solver = Solver(domain=dom, boundary_conditions=bc)  # model from config.json
    evolve = Evolve(solver=solver, maxsteps=1)
    evolve.Evolve_0()
    dt = float(solver.dt)
    n = _cap(round(float(cfg["duration_s"]) / dt), max_steps)
    return Run(solver, evolve, dt, n, max(1, round(float(cfg["sample_s"]) / dt)))


def _build_tracyarm(cfg: dict[str, object], max_steps: int | None) -> Run:
    """``calibration/TracyArm/generate.py`` with its defaults."""
    from celeris.domain import BoundaryConditions, Domain, Topodata
    from celeris.landslide import LandslideParams, MovingBodySlide
    from celeris.runner import Evolve
    from celeris.solver import Solver

    gen = _import(TRACYARM_GENERATE)
    path = str(cfg["path"])
    topo = Topodata(datatype="celeris", path=path)
    bc = BoundaryConditions(celeris=True, path=path)
    dom = Domain(topodata=topo)
    solver = Solver(
        domain=dom, boundary_conditions=bc, model="SWE", useBreakingModel=True
    )
    solver.delta = max(float(solver.delta), 0.01)  # generate.py --delta default
    solver.epsilon = solver.delta**2
    az_sign = -1.0  # generate.py --az-sign default
    slide = LandslideParams(
        thickness_m=gen._peak_thickness(),
        length_m=gen.LENGTH_M,
        width_m=gen.WIDTH_M,
        x0_m=gen.X0_M,
        y0_m=gen.Y0_M,
        azimuth_rad=az_sign * radians(gen.TRAJ_INIT_DEG),
        final_azimuth_rad=az_sign * radians(gen.TRAJ_FINAL_DEG),
        travel_distance_m=gen.TRAVEL_M,
        timescale_s=gen.TIMESCALE_S,
        time_shift_s=gen.TIME_SHIFT_S,
        expo=gen.EXPO,
        wet_dhdt_only=True,  # --wet-only 1
        emerge_min_depth_m=1.0,  # --emerge 1.0
        carve_scar=False,
    )
    solver.landslide = MovingBodySlide(solver, slide)
    evolve = Evolve(solver=solver, maxsteps=1)
    evolve.Evolve_0()
    bed0 = solver.Bottom.to_numpy()[2].copy()
    land = bed0 > 0.0

    def sample(eta: np.ndarray, bed: np.ndarray) -> dict[str, object]:
        wet = (eta - bed) > RUNUP_DEPTH_M
        on_land = wet & land
        return {
            "runup_m": float(eta[on_land].max()) if on_land.any() else np.nan,
            "trough_m": float(eta[wet].min()) if wet.any() else np.nan,
        }

    def metrics(series: dict[str, np.ndarray]) -> dict[str, float]:
        return {
            "max_runup_m": float(np.nanmax(series["runup_m"])),
            "min_trough_m": float(np.nanmin(series["trough_m"])),
        }

    dt = float(solver.dt)
    return Run(
        solver,
        evolve,
        dt,
        _cap(int(float(cfg["duration_s"]) / dt), max_steps),
        max(1, int(gen.FRAME_DT_S / dt)),
        moving_bed=True,
        sample=sample,
        metrics=metrics,
    )


def _build_breach(cfg: dict[str, object], max_steps: int | None) -> Run:
    """``breach_validation.py`` stages prep + run, instrumented (setup copied)."""
    import taichi as ti

    from celeris.domain import BoundaryConditions, Domain, Topodata
    from celeris.hydrograph import HydrographSource
    from celeris.runner import Evolve
    from celeris.solver import Solver

    bv = _import(BREACH_SCRIPT)
    out = Path(str(cfg["path"]))
    bv.prep(out)
    g = dict(np.load(out / "grid.npz"))
    dx = float(g["dx"])
    bed_abs = g["bed"].astype(np.float64)
    nx, ny = bed_abs.shape
    xg, yg = np.meshgrid(np.arange(nx) * dx, np.arange(ny) * dx, indexing="ij")
    np.savetxt(
        out / "bathy.xyz",
        np.column_stack([xg.ravel(), yg.ravel(), (bv.DATUM_M - bed_abs).ravel()]),
    )
    topo = Topodata(filename="bathy.xyz", path=str(out), datatype="xyz")
    bc = BoundaryConditions(celeris=False, North=0, East=0, South=0, West=0)
    dom = Domain(
        topodata=topo,
        x1=0.0,
        x2=nx * dx,
        y1=0.0,
        y2=ny * dx,
        Nx=nx,
        Ny=ny,
        isManning=1,
        friction=bv.MANNING_N,
        Courant=bv.COURANT,
    )
    solver = Solver(
        domain=dom, boundary_conditions=bc, model="SWE", infiltrationRate=0.0
    )
    solver.landslide = HydrographSource(
        solver, g["inlet"], bv.LIQ_T_S, bv.LIQ_Q_M3S, min_depth_m=0.2
    )
    evolve = Evolve(solver=solver, maxsteps=1)
    state = np.zeros((nx, ny, 4), dtype=np.float32)
    state[:, :, 0] = g["eta0"] - bv.DATUM_M
    state[:, :, 1] = g["hu0"]
    state[:, :, 2] = g["hv0"]
    solver.InitStates()
    for f in (
        solver.State,
        solver.stateUVstar,
        solver.NewState,
        solver.current_stateUVstar,
    ):
        f.from_numpy(state)
    solver.InitStates = lambda: None  # keep Evolve_0 from zeroing the state again
    evolve.Evolve_0()

    outlet = ti.field(ti.i32, shape=(nx, ny))
    outlet.from_numpy(g["outlet"].astype(np.int32))

    @ti.kernel
    def impose_outlet_stage(eta_rel: ti.f32):  # type: ignore[no-untyped-def]
        for i, j in outlet:
            if outlet[i, j] == 1:
                b = solver.Bottom[2, i, j]
                e = ti.max(eta_rel, b)
                solver.State[i, j][0] = e
                solver.stateUVstar[i, j][0] = e

    dt = float(solver.dt)

    def post_step(step: int) -> None:
        t_next = (step + 1) * dt
        impose_outlet_stage(
            float(np.interp(t_next, bv.LIQ_T_S, bv.LIQ_SL_M)) - bv.DATUM_M
        )

    probes = [tuple(int(v) for v in ij) for ij in g["probes_ij"]]
    fp, inside = g["floodplain"], g["inside"]

    def sample(eta: np.ndarray, bed: np.ndarray) -> dict[str, object]:
        depth = np.where(inside, eta - bed, 0.0)
        return {
            "probes_eta_masl": np.array([eta[i, j] + bv.DATUM_M for i, j in probes]),
            "floodplain_wet_fraction": float((depth[fp] > bv.DRY_M).mean()),
        }

    def metrics(series: dict[str, np.ndarray]) -> dict[str, float]:
        wet = series["floodplain_wet_fraction"]
        t = series["t_s"]
        m = {
            "floodplain_wet_fraction_final": float(wet[-1]),
            "floodplain_first_wet_s": _first_exceed(t, wet, 0.0),
        }
        for k, px in enumerate(bv.PROBES_X_M):
            m[f"eta_probe_x{int(px)}_final_masl"] = float(
                series["probes_eta_masl"][-1, k]
            )
        return m

    return Run(
        solver,
        evolve,
        dt,
        _cap(int(bv.DURATION_S / dt), max_steps),
        max(1, int(bv.SAMPLE_S / dt)),
        post_step=post_step,
        sample=sample,
        metrics=metrics,
        arrays={"probes_x_m": np.array(bv.PROBES_X_M), "datum_m": np.array(bv.DATUM_M)},
    )


def _build_malpasset(cfg: dict[str, object], max_steps: int | None) -> Run:
    """``run_malpasset.py`` (plain or ``--seed-wetting``), instrumented (setup copied)."""
    import taichi as ti

    from celeris.domain import BoundaryConditions, Domain, Topodata
    from celeris.runner import Evolve
    from celeris.solver import Solver

    wd = Path(str(cfg["path"]))
    if not (wd / "grid.npz").exists() or not (wd / "bathy.xyz").exists():
        logger.info("malpasset grid missing -> running prep_malpasset.py --dx 15")
        subprocess.run(
            [sys.executable, str(MALPASSET_DIR / "prep_malpasset.py"), "--dx", "15"],
            check=True,
            cwd=CELERIS_DIR,
        )
    rm = _import(MALPASSET_DIR / "run_malpasset.py")
    g = dict(np.load(wd / "grid.npz"))
    bed, eta0, dx = (
        g["bed"].astype(np.float64),
        g["eta0"].astype(np.float64),
        float(g["dx"]),
    )
    nx, ny = bed.shape
    gi, gj = g["gauge_i"], g["gauge_j"]
    hmax0 = float((eta0 - bed).max())

    topo = Topodata(filename="bathy.xyz", path=str(wd), datatype="xyz")
    bc = BoundaryConditions(
        celeris=False, North=0, East=0, South=0, West=0, BoundaryWidth=20
    )
    dom = Domain(
        topodata=topo,
        x1=0.0,
        x2=nx * dx,
        y1=0.0,
        y2=ny * dx,
        Nx=nx,
        Ny=ny,
        isManning=1,
        friction=rm.MANNING_N,
        Courant=0.2,
        base_depth=hmax0,
    )
    solver = Solver(
        domain=dom,
        boundary_conditions=bc,
        model="SWE",
        useBreakingModel=True,
        infiltrationRate=0.0,
        show_window=False,
    )
    evolve = Evolve(solver=solver, maxsteps=1)
    evolve.Evolve_0()
    bed_solver = solver.Bottom.to_numpy()[2].astype(np.float64)
    if not np.allclose(bed_solver, bed, atol=1e-3):
        raise RuntimeError("solver bed != grid bed (xyz round trip)")
    eta0 = np.maximum(eta0, bed_solver)
    s0 = np.zeros((nx, ny, 4), dtype=np.float32)
    s0[:, :, 0] = eta0
    for f in (
        solver.State,
        solver.stateUVstar,
        solver.NewState,
        solver.current_stateUVstar,
    ):
        f.from_numpy(s0)

    seed_h = float(cfg["seed_wetting_m"])  # type: ignore[arg-type]
    delta = float(solver.delta)

    @ti.kernel
    def seed_wetting():  # type: ignore[no-untyped-def]
        for i, j in solver.State:
            if i > 1 and j > 1 and i < nx - 2 and j < ny - 2:
                B = solver.Bottom[2, i, j]
                if solver.State[i, j][0] - B <= delta:
                    best = -1.0e9
                    bi, bj = -1, -1
                    for di, dj in ti.static(((1, 0), (-1, 0), (0, 1), (0, -1))):
                        e = solver.State[i + di, j + dj][0]
                        if (
                            e - solver.Bottom[2, i + di, j + dj] > 4.0 * seed_h
                            and e > B + 2.0 * seed_h
                            and e > best
                        ):
                            best, bi, bj = e, i + di, j + dj
                    if bi >= 0:
                        solver.State[i, j][0] = B + seed_h
                        solver.stateUVstar[i, j][0] = B + seed_h
                        solver.State[bi, bj][0] -= seed_h
                        solver.stateUVstar[bi, bj][0] -= seed_h

    def post_step(step: int) -> None:
        seed_wetting()

    names = [str(n) for n in g["gauge_names"]]
    bed_g = bed_solver[gi, gj]

    def sample(eta: np.ndarray, bed: np.ndarray) -> dict[str, object]:
        return {"gauge_eta_m": eta[gi, gj].astype(np.float64)}

    def metrics(series: dict[str, np.ndarray]) -> dict[str, float]:
        t = series["t_s"]
        depth = series["gauge_eta_m"] - bed_g
        arr = {
            n: _first_exceed(t, depth[:, k], MALPASSET_ARRIVAL_M)
            for k, n in enumerate(names)
        }
        m = {
            "arrival_A_s": arr["A"],
            "A_to_B_s": arr["B"] - arr["A"],
            "A_to_C_s": arr["C"] - arr["A"],
        }
        for k, n in enumerate(names):
            if n.startswith("P"):
                m[f"hmax_{n}_m"] = float(depth[:, k].max())
        return m

    dt = float(solver.dt)
    return Run(
        solver,
        evolve,
        dt,
        _cap(int(np.ceil(float(cfg["duration_s"]) / dt)), max_steps),
        max(1, round(float(cfg["sample_s"]) / dt)),
        post_step=post_step if seed_h > 0 else None,
        sample=sample,
        metrics=metrics,
        arrays={"gauge_names": np.array(names), "gauge_i": gi, "gauge_j": gj},
    )
