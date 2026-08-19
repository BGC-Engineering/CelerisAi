"""Acceptance tests for the NetCDF prescribed-bed-motion source.

The reference is the analytic ``MovingBodySlide``: sampling its mound into
frames and replaying them through ``PrescribedBedSlide`` must reproduce the
same wave field. All cases run the Boussinesq model headless on the CPU
backend with solid walls (same basin as ``test_landslide.py``).
"""

from pathlib import Path

import numpy as np
import pytest
import taichi as ti

from celeris.domain import BoundaryConditions, Domain, Topodata
from celeris.landslide import (
    BedMotion,
    LandslideParams,
    MovingBodySlide,
    PrescribedBedSlide,
    peak_speed_timescale,
)
from celeris.runner import Evolve
from celeris.solver import Solver


def _build_solver(tmp_path: Path) -> tuple[Solver, Evolve]:
    """Assemble the 40 x 8 m walled basin of test_landslide (1 m deep)."""
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    xs = np.linspace(0.0, 40.0, 81)
    ys = np.linspace(0.0, 8.0, 17)
    xg, yg = np.meshgrid(xs, ys)
    z = np.full_like(xg, 1.0)
    np.savetxt(
        tmp_path / "basin.xyz", np.column_stack([xg.ravel(), yg.ravel(), z.ravel()])
    )
    topo = Topodata(filename="basin.xyz", path=str(tmp_path), datatype="xyz")
    bc = BoundaryConditions(celeris=False, North=0, South=0, East=0, West=0)
    domain = Domain(topodata=topo, x1=0.0, x2=40.0, y1=0.0, y2=8.0, Nx=160, Ny=32)
    solver = Solver(domain=domain, boundary_conditions=bc, model="Bouss")
    run = Evolve(solver=solver, maxsteps=1)
    return solver, run


def _slide_params() -> LandslideParams:
    """The submerged translating mound of test_landslide."""
    return LandslideParams(
        thickness_m=0.3,
        length_m=2.0,
        width_m=2.0,
        x0_m=10.0,
        y0_m=4.0,
        azimuth_rad=0.0,
        travel_distance_m=10.0,
        timescale_s=peak_speed_timescale(10.0, 2.0),
    )


def _analytic_motion(params: LandslideParams, frame_dt: float) -> BedMotion:
    """Sample the analytic mound of ``params`` into BedMotion frames."""
    x = np.arange(160) * 0.25
    y = np.arange(32) * 0.25
    xg, yg = np.meshgrid(x, y)
    times = np.arange(0.0, params.end_time_s + frame_dt, frame_dt)
    dz = np.zeros((times.size, y.size, x.size), dtype=np.float32)
    for k, t in enumerate(times):
        disp = (
            params.travel_distance_m
            * (1.0 + np.tanh((t - params.time_shift_s) / params.timescale_s))
            / 2.0
        )
        along = xg - params.x0_m - disp
        across = yg - params.y0_m
        arg = (
            np.abs(along / params.length_m) ** params.expo
            + np.abs(across / params.width_m) ** params.expo
        ) / 2.0
        dz[k] = params.thickness_m * np.exp(-arg)
    return BedMotion(x=x, y=y, time=times, dz=dz, source="analytic sample")


def _run(solver: Solver, run: Evolve, end_time: float) -> np.ndarray:
    """Step to ``end_time`` and return eta over the wet region."""
    n_steps = int(end_time / float(solver.dt)) + 50
    for i in range(n_steps):
        run.Evolve_Steps(i)
    state = solver.State.to_numpy()[:, :, 0]
    bed = solver.Bottom.to_numpy()[2, :, :]
    assert np.isfinite(state).all()
    return np.where(bed < 0.0, state, 0.0)


def test_netcdf_round_trip(tmp_path: Path) -> None:
    """to_netcdf / from_netcdf preserve coords, frames and attributes."""
    motion = _analytic_motion(_slide_params(), frame_dt=1.0)
    path = str(tmp_path / "slide.nc")
    motion.to_netcdf(path)
    back = BedMotion.from_netcdf(path)
    np.testing.assert_allclose(back.x, motion.x)
    np.testing.assert_allclose(back.y, motion.y)
    np.testing.assert_allclose(back.time, motion.time)
    np.testing.assert_allclose(back.dz, motion.dz, atol=1e-6)
    assert back.source == "analytic sample"
    assert back.crs == "local"


def test_bed_motion_validation() -> None:
    """Shape mismatches and non-monotonic time are rejected."""
    x = np.arange(4.0)
    y = np.arange(3.0)
    t = np.array([0.0, 1.0])
    dz = np.zeros((2, 3, 4))
    with pytest.raises(ValueError):
        BedMotion(x=x, y=y, time=t, dz=np.zeros((2, 4, 3)))
    with pytest.raises(ValueError):
        BedMotion(x=x, y=y, time=np.array([1.0, 0.0]), dz=dz)
    with pytest.raises(ValueError):
        BedMotion(x=x, y=y, time=t, dz=dz + np.nan)


def test_prescribed_matches_analytic(tmp_path: Path) -> None:
    """Frames sampled from the analytic mound reproduce its wave field."""
    params = _slide_params()
    solver_a, run_a = _build_solver(tmp_path)
    solver_a.landslide = MovingBodySlide(solver_a, params)
    run_a.Evolve_0()
    eta_a = _run(solver_a, run_a, params.end_time_s)

    motion = _analytic_motion(params, frame_dt=0.5)
    path = str(tmp_path / "slide.nc")
    motion.to_netcdf(path)
    solver_n, run_n = _build_solver(tmp_path)
    solver_n.landslide = PrescribedBedSlide(solver_n, BedMotion.from_netcdf(path))
    run_n.Evolve_0()
    eta_n = _run(solver_n, run_n, params.end_time_s)

    peak = np.abs(eta_a).max()
    assert peak > 0.005
    assert np.abs(eta_n - eta_a).max() < 0.15 * peak


def test_bed_tracks_frames_and_zero_outside_coverage(tmp_path: Path) -> None:
    """The bed change equals the frame dz; cells beyond coverage stay put."""
    params = _slide_params()
    # Motion grid covering only x < 20 m: the mound starts inside coverage.
    motion = _analytic_motion(params, frame_dt=0.5)
    keep = motion.x < 20.0
    clipped = BedMotion(
        x=motion.x[keep], y=motion.y, time=motion.time, dz=motion.dz[:, :, keep]
    )
    solver, run = _build_solver(tmp_path)
    slide = PrescribedBedSlide(solver, clipped)
    solver.landslide = slide
    run.Evolve_0()
    bed0 = solver.Bottom.to_numpy()[2, :, :].copy()
    k = 6
    slide.update(float(motion.time[k]))
    change = solver.Bottom.to_numpy()[2, :, :] - bed0
    np.testing.assert_allclose(change, slide.frames[k], atol=1e-5)
    outside = slide.frames[:, solver.nx // 2 :, :]
    assert np.abs(outside).max() == 0.0
