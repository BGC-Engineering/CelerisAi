"""Acceptance tests for the hydrograph (inflow) continuity source.

All cases run the SWE model headless on the CPU backend in small synthetic
basins written as ``xyz`` topo files (positive z = depth below the datum).
"""

from pathlib import Path

import numpy as np
import pytest
import taichi as ti
from scipy.optimize import brentq

from celeris.domain import BoundaryConditions, Domain, Topodata
from celeris.hydrograph import HydrographSource
from celeris.runner import Evolve
from celeris.solver import Solver

G = 9.81


def _build_basin(
    tmp_path: Path,
    depth_fn,
    size=(40.0, 8.0),
    cells=(160, 32),
    east: int = 0,
    boundary_width: int = 10,
) -> tuple[Solver, Evolve]:
    """Walled basin with depth ``depth_fn(x)`` (m); ``east=1`` opens a sponge."""
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    lx, ly = size
    nx, ny = cells
    xs = np.linspace(0.0, lx, nx // 2 + 1)
    ys = np.linspace(0.0, ly, ny // 2 + 1)
    xg, yg = np.meshgrid(xs, ys)
    z = depth_fn(xg)
    np.savetxt(
        tmp_path / "basin.xyz", np.column_stack([xg.ravel(), yg.ravel(), z.ravel()])
    )
    topo = Topodata(filename="basin.xyz", path=str(tmp_path), datatype="xyz")
    bc = BoundaryConditions(
        celeris=False, North=0, South=0, East=east, West=0, BoundaryWidth=boundary_width
    )
    domain = Domain(topodata=topo, x1=0.0, x2=lx, y1=0.0, y2=ly, Nx=nx, Ny=ny)
    solver = Solver(domain=domain, boundary_conditions=bc, model="SWE")
    run = Evolve(solver=solver, maxsteps=1)
    return solver, run


def _eta(solver: Solver) -> np.ndarray:
    return solver.State.to_numpy()[:, :, 0]


def _bed(solver: Solver) -> np.ndarray:
    return solver.Bottom.to_numpy()[2, :, :]


INNER = (slice(2, -2), slice(2, -2))  # BoundaryPass owns a 2-cell ghost rim


def _wet_volume(solver: Solver) -> float:
    """Water volume over the interior; the ghost rim mirrors it and must not count."""
    depth = np.maximum(_eta(solver) - _bed(solver), 0.0)[INNER]
    return float(depth.sum() * solver.dx * solver.dy)


def _strip_mask(solver: Solver, i0: int, i1: int) -> np.ndarray:
    mask = np.zeros((solver.nx, solver.ny), dtype=bool)
    mask[i0:i1, 2:-2] = True
    return mask


def _set_eta(solver: Solver, eta: np.ndarray) -> None:
    """Overwrite the free surface in every state copy (momenta zero)."""
    state = np.zeros((solver.nx, solver.ny, 4), dtype=np.float32)
    state[:, :, 0] = eta
    for f in (
        solver.State,
        solver.stateUVstar,
        solver.NewState,
        solver.current_stateUVstar,
    ):
        f.from_numpy(state)


def test_rejects_bad_inputs(tmp_path: Path) -> None:
    solver, _ = _build_basin(tmp_path, lambda x: np.full_like(x, 1.0))
    good = _strip_mask(solver, 10, 14)
    with pytest.raises(ValueError):
        HydrographSource(solver, good[:-1], [0.0, 10.0], [1.0, 1.0])
    with pytest.raises(ValueError):
        HydrographSource(solver, np.zeros_like(good), [0.0, 10.0], [1.0, 1.0])
    with pytest.raises(ValueError):
        HydrographSource(solver, good, [0.0, 10.0, 5.0], [1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        HydrographSource(solver, good, [0.0, 10.0], [1.0])


def test_rejects_dry_inlet(tmp_path: Path) -> None:
    """A source on dry cells is lost by Pass3, so it must be refused up front."""
    solver, run = _build_basin(tmp_path, lambda x: (20.0 - x) / 10.0)  # dry x > 20
    solver.landslide = HydrographSource(
        solver, _strip_mask(solver, 120, 124), [0.0, 10.0], [1.0, 1.0]
    )
    with pytest.raises(ValueError, match="dry"):
        run.Evolve_0()


@pytest.mark.parametrize(
    "times,q",
    [([0.0, 30.0], [1.0, 1.0]), ([0.0, 15.0, 30.0], [0.0, 2.0, 0.0])],
    ids=["constant", "triangular"],
)
def test_closed_basin_gains_hydrograph_volume(tmp_path: Path, times, q) -> None:
    """Injected volume equals the hydrograph integral and then holds."""
    solver, run = _build_basin(tmp_path, lambda x: np.full_like(x, 1.0))
    src = HydrographSource(solver, _strip_mask(solver, 10, 14), times, q)
    solver.landslide = src
    run.Evolve_0()
    v0 = _wet_volume(solver)
    dt = float(solver.dt)
    n_on = round(times[-1] / dt)
    for i in range(n_on):
        run.Evolve_Steps(i)
    assert np.isfinite(solver.State.to_numpy()).all()
    v_on = _wet_volume(solver) - v0
    for i in range(n_on, 2 * n_on):
        run.Evolve_Steps(i)
    v_after = _wet_volume(solver) - v0
    assert src.volume_m3 == pytest.approx(30.0)
    assert v_on == pytest.approx(src.volume_m3, rel=0.03)
    assert v_after == pytest.approx(v_on, rel=0.02)
    area = (solver.nx - 4) * (solver.ny - 4) * solver.dx * solver.dy
    assert _eta(solver)[INNER].mean() == pytest.approx(src.volume_m3 / area, rel=0.03)


def test_open_channel_carries_the_discharge(tmp_path: Path) -> None:
    """Steady inflow at one end, sponge at the other: mid-channel flux = Q."""
    q_in = 0.5
    solver, run = _build_basin(tmp_path, lambda x: np.full_like(x, 1.0), east=1)
    solver.landslide = HydrographSource(
        solver, _strip_mask(solver, 6, 10), [0.0, 400.0], [q_in, q_in]
    )
    run.Evolve_0()
    dt = float(solver.dt)
    i_mid = solver.nx // 2
    flux = []
    for i in range(int(300.0 / dt)):
        run.Evolve_Steps(i)
        if i * dt > 200.0 and i % 20 == 0:
            hu = solver.State.to_numpy()[i_mid, :, 1]
            flux.append(float(hu.sum() * solver.dy))
    assert np.isfinite(solver.State.to_numpy()).all()
    assert np.mean(flux) == pytest.approx(q_in, rel=0.15)


def stoker_depth(
    x: np.ndarray, t: float, x0: float, hl: float, hr: float
) -> np.ndarray:
    """Exact wet-bed dam-break depth (Stoker 1957), SWASHES formulation."""
    cl = np.sqrt(G * hl)

    def f(cm: float) -> float:
        return -8.0 * G * hr * cm**2 * (cl - cm) ** 2 + (cm**2 - G * hr) ** 2 * (
            cm**2 + G * hr
        )

    cm = brentq(f, np.sqrt(G * hr) * (1 + 1e-9), cl * (1 - 1e-9))
    hm = cm**2 / G
    um = 2.0 * (cl - cm)
    s = hm * um / (hm - hr)
    xi = (x - x0) / t
    h = np.where(xi <= -cl, hl, (2.0 * cl - xi) ** 2 / (9.0 * G))
    h = np.where(xi >= um - cm, hm, h)
    return np.where(xi >= s, hr, h)


def test_stoker_dam_break_matches_analytic(tmp_path: Path) -> None:
    """TELEMAC dambreak geometry: 16 m channel, dam at x = 8 m, 1 m over 0.2 m."""
    hl, hr, x0, t_end = 1.0, 0.2, 8.0, 1.5
    solver, run = _build_basin(
        tmp_path, lambda x: np.full_like(x, hl), size=(16.0, 1.0), cells=(320, 20)
    )
    run.Evolve_0()
    x = (np.arange(solver.nx) + 0.5) * solver.dx
    eta0 = np.where(x < x0, 0.0, hr - hl)[:, None] * np.ones((1, solver.ny))
    _set_eta(solver, eta0)
    dt = float(solver.dt)
    n = round(t_end / dt)
    for i in range(n):
        run.Evolve_Steps(i)
    depth = (_eta(solver) - _bed(solver))[:, solver.ny // 2]
    exact = stoker_depth(x, n * dt, x0, hl, hr)
    interior = slice(4, -4)
    l1 = np.abs(depth[interior] - exact[interior]).mean()
    assert np.isfinite(depth).all()
    assert l1 < 0.02  # 2 % of the upstream depth
