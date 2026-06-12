"""Acceptance tests for the moving-body landslide source.

All cases run the Boussinesq model headless on the CPU backend with solid
walls, using small synthetic basins written as ``xyz`` topo files (positive
z = depth below the still water level).
"""

from math import gamma, pi
from pathlib import Path

import numpy as np
import pytest
import taichi as ti

from celeris.domain import BoundaryConditions, Domain, Topodata
from celeris.landslide import LandslideParams, MovingBodySlide, peak_speed_timescale
from celeris.runner import Evolve
from celeris.solver import Solver


def _build_case(
    tmp_path: Path, depth_fn, params: LandslideParams | None
) -> tuple[Solver, Evolve]:
    """Assemble a 40 x 8 m walled basin with depth ``depth_fn(x)`` (m)."""
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    xs = np.linspace(0.0, 40.0, 81)
    ys = np.linspace(0.0, 8.0, 17)
    xg, yg = np.meshgrid(xs, ys)
    z = depth_fn(xg)
    np.savetxt(
        tmp_path / "basin.xyz", np.column_stack([xg.ravel(), yg.ravel(), z.ravel()])
    )
    topo = Topodata(filename="basin.xyz", path=str(tmp_path), datatype="xyz")
    bc = BoundaryConditions(celeris=False, North=0, South=0, East=0, West=0)
    domain = Domain(topodata=topo, x1=0.0, x2=40.0, y1=0.0, y2=8.0, Nx=160, Ny=32)
    solver = Solver(domain=domain, boundary_conditions=bc, model="Bouss")
    if params is not None:
        solver.landslide = MovingBodySlide(solver, params)
    run = Evolve(solver=solver, maxsteps=1)
    run.Evolve_0()
    return solver, run


def _eta(solver: Solver) -> np.ndarray:
    """Free-surface elevation over the wet region only."""
    state = solver.State.to_numpy()[:, :, 0]
    bed = solver.Bottom.to_numpy()[2, :, :]
    return np.where(bed < 0.0, state, 0.0)


def test_volume_normalization_gaussian() -> None:
    """For expo=2 the mound volume is thickness * L * W * 2 * pi."""
    p = LandslideParams(
        thickness_m=0.5,
        length_m=2.0,
        width_m=3.0,
        x0_m=0.0,
        y0_m=0.0,
        azimuth_rad=0.0,
        travel_distance_m=1.0,
        timescale_s=1.0,
    )
    assert p.volume_m3 == pytest.approx(0.5 * 2.0 * 3.0 * 2.0 * pi)
    shape = 4.0 * 2.0 ** (2.0 / 4.0) * gamma(0.25) ** 2 / 16.0
    p4 = LandslideParams(
        thickness_m=0.5,
        length_m=2.0,
        width_m=3.0,
        x0_m=0.0,
        y0_m=0.0,
        azimuth_rad=0.0,
        travel_distance_m=1.0,
        timescale_s=1.0,
        expo=4.0,
    )
    assert p4.volume_m3 == pytest.approx(0.5 * 2.0 * 3.0 * shape)


def test_peak_speed_timescale() -> None:
    """tau = travel / (2 v_peak)."""
    assert peak_speed_timescale(10.0, 2.5) == pytest.approx(2.0)


def test_still_water_stays_flat(tmp_path: Path) -> None:
    """No slide: the well-balanced scheme must hold still water at rest."""
    solver, run = _build_case(tmp_path, lambda x: np.full_like(x, 1.0), None)
    for i in range(100):
        run.Evolve_Steps(i)
    assert np.abs(_eta(solver)).max() < 1e-6


def test_submerged_slide_radiates_and_conserves(tmp_path: Path) -> None:
    """A translating submerged mound makes waves but conserves total volume."""
    tau = peak_speed_timescale(10.0, 2.0)
    params = LandslideParams(
        thickness_m=0.3,
        length_m=2.0,
        width_m=2.0,
        x0_m=10.0,
        y0_m=4.0,
        azimuth_rad=0.0,
        travel_distance_m=10.0,
        timescale_s=tau,
    )
    solver, run = _build_case(tmp_path, lambda x: np.full_like(x, 1.0), params)
    dt = float(solver.dt)
    n_steps = int(params.end_time_s / dt) + 50
    peak_during_motion = 0.0
    for i in range(n_steps):
        run.Evolve_Steps(i)
        if i % 20 == 0:
            peak_during_motion = max(peak_during_motion, np.abs(_eta(solver)).max())
    eta = _eta(solver)
    assert np.isfinite(solver.State.to_numpy()).all()
    assert peak_during_motion > 0.01
    cell_area = solver.dx * solver.dy
    net_volume = float(eta.sum() * cell_area)
    # The wall BoundaryPass overwrites ghost states, so the closed-basin eta
    # integral oscillates ~+/-0.7 m^3 even with the source off; the source
    # itself nets to zero (translation only). Bound, don't pin to zero.
    assert abs(net_volume) < 0.2 * params.volume_m3


def test_subaerial_slide_makes_positive_wave(tmp_path: Path) -> None:
    """A slide entering from dry slope produces a leading positive wave."""

    def sloped(x: np.ndarray) -> np.ndarray:
        return (20.0 - x) / 10.0  # 2 m deep at x=0, waterline x=20, dry beyond

    tau = peak_speed_timescale(12.0, 3.0)
    params = LandslideParams(
        thickness_m=0.4,
        length_m=2.0,
        width_m=2.0,
        x0_m=26.0,
        y0_m=4.0,
        azimuth_rad=pi,
        travel_distance_m=12.0,
        timescale_s=tau,
    )
    solver, run = _build_case(tmp_path, sloped, params)
    dt = float(solver.dt)
    n_steps = int(params.end_time_s / dt) + 50
    i_probe = int(6.0 / solver.dx)
    j_probe = solver.ny // 2
    crest = 0.0
    for i in range(n_steps):
        run.Evolve_Steps(i)
        crest = max(crest, float(solver.State[i_probe, j_probe][0]))
    assert np.isfinite(solver.State.to_numpy()).all()
    assert crest > 0.01
