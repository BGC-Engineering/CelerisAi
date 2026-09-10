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
    """Water volume inside the walls.

    The ghost rim (two cells) mirrors the interior and must not count, and the
    solid wall passes through the CENTRE of the first interior cell (index 2 and
    n-3), so those cells count half. With this accounting the scheme conserves
    volume to 1e-4 (checked on a disk source in a closed basin).
    """
    depth = np.maximum(_eta(solver) - _bed(solver), 0.0)
    w = np.ones_like(depth)
    w[[2, -3], :] *= 0.5
    w[:, [2, -3]] *= 0.5
    return float((depth * w)[INNER].sum() * solver.dx * solver.dy)


def _section_q(state: np.ndarray, i: int, comp: int, dy: float) -> float:
    """Discharge through column ``i``: interior rows, wall rows (2, n-3) weighted half."""
    hu = state[i, 2:-2, comp].astype(np.float64)
    hu[0] *= 0.5
    hu[-1] *= 0.5
    return float(hu.sum() * dy)


def _strip_mask(solver: Solver, i0: int, i1: int) -> np.ndarray:
    mask = np.zeros((solver.nx, solver.ny), dtype=bool)
    mask[i0:i1, 3:-3] = True  # off the wall cells (rows 2 and n-3)
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
    area = (
        (solver.nx - 5) * (solver.ny - 5) * solver.dx * solver.dy
    )  # walls through cell centres
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
            flux.append(_section_q(solver.State.to_numpy(), i_mid, 1, solver.dy))
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


# ---------------------------------------------------------------------------
# wetdry_scheme="conserving" (see Solver): behaviours the legacy scheme lacks.
# ---------------------------------------------------------------------------


def _build_channel(tmp_path: Path, scheme: str) -> tuple[Solver, Evolve]:
    """Flat 40 x 8 m basin, 1 m deep, walled; the tests then overwrite the bed."""
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    xs, ys = np.linspace(0.0, 40.0, 81), np.linspace(0.0, 8.0, 17)
    xg, yg = np.meshgrid(xs, ys)
    np.savetxt(
        tmp_path / "basin.xyz",
        np.column_stack([xg.ravel(), yg.ravel(), np.ones(xg.size)]),
    )
    topo = Topodata(filename="basin.xyz", path=str(tmp_path), datatype="xyz")
    bc = BoundaryConditions(celeris=False, North=0, South=0, East=0, West=0)
    domain = Domain(topodata=topo, x1=0.0, x2=40.0, y1=0.0, y2=8.0, Nx=160, Ny=32)
    solver = Solver(
        domain=domain, boundary_conditions=bc, model="SWE", wetdry_scheme=scheme
    )
    return solver, Evolve(solver=solver, maxsteps=1)


def _start(
    solver: Solver, run: Evolve, bed: np.ndarray, level: float, dry=None
) -> None:
    """Install ``bed`` and still water at ``level`` (dry where bed is higher,
    and wherever ``dry`` is True: a dam-break start with a dry low side)."""
    bed32 = bed.astype(np.float32)
    solver.Bottom.from_numpy(np.stack([bed32, bed32, bed32, np.zeros_like(bed32)]))
    solver.fill_bottom_field()
    solver.InitStates()
    eta = np.where(bed < level, level, bed)
    if dry is not None:
        eta = np.where(dry, bed, eta)
    _set_eta(solver, eta)
    solver.InitStates = lambda: None  # Evolve_0 would zero the state again
    run.Evolve_0()


def _banked_bed(solver: Solver) -> np.ndarray:
    """Channel bed at -1 m with 1:1 banks rising to +0.5 m at both y edges."""
    j = np.arange(solver.ny)
    rise = np.clip(np.abs(j - (solver.ny - 1) / 2.0) * solver.dy - 2.0, 0.0, 1.5)
    return np.broadcast_to((-1.0 + rise)[None, :], (solver.nx, solver.ny)).copy()


@pytest.mark.parametrize("scheme,tol", [("conserving", 0.01), ("legacy", 0.20)])
def test_sloping_banks_conserve_injected_volume(tmp_path: Path, scheme, tol) -> None:
    """Inflow into a channel with wet/dry banks. Conserving keeps the volume to
    1 %; legacy loses several percent at the banks (bounded, not pinned)."""
    solver, run = _build_channel(tmp_path, scheme)
    bed = _banked_bed(solver)
    inlet = _strip_mask(solver, 10, 14) & (bed < -0.7)
    src = HydrographSource(solver, inlet, [0.0, 30.0], [1.0, 1.0])
    solver.landslide = src
    _start(solver, run, bed, -0.3)
    v0 = _wet_volume(solver)
    dt = float(solver.dt)
    for i in range(round(30.0 / dt)):
        run.Evolve_Steps(i)
    assert np.isfinite(solver.State.to_numpy()).all()
    assert _wet_volume(solver) - v0 == pytest.approx(src.volume_m3, rel=tol)


def test_still_water_above_dry_shelf_floods_it(tmp_path: Path) -> None:
    """Water at rest whose surface stands above a dry shelf must flood it
    (dam-break onto dry bed). Legacy is frozen: no momentum, no flux."""
    wet_frac = {}
    for scheme in ("legacy", "conserving"):
        solver, run = _build_channel(tmp_path, scheme)
        bed = np.full((solver.nx, solver.ny), -1.0)
        shelf_cols = np.arange(solver.nx) >= round(30.0 / solver.dx)
        bed[shelf_cols, :] = 0.4  # shelf above the datum
        # Dam-break start: water at 0.6 m on the left, the shelf dry although
        # it lies 0.2 m below that surface.
        _start(solver, run, bed, 0.6, dry=shelf_cols[:, None])
        v0 = _wet_volume(solver)
        dt = float(solver.dt)
        for i in range(round(40.0 / dt)):
            run.Evolve_Steps(i)
        eta, depth = _eta(solver), _eta(solver) - bed
        shelf = np.zeros_like(bed, bool)
        shelf[round(30.0 / solver.dx) : -2, 2:-2] = True
        wet_frac[scheme] = float(np.mean(depth[shelf] > 0.01))
        # final flat level: 30*8*1.6 m3 over 40*8 m2 with the 0.4 m shelf -> 0.55 m
        if scheme == "conserving":
            assert _wet_volume(solver) == pytest.approx(v0, rel=0.01)
            assert eta[INNER].std() < 0.02  # settles to one flat surface
    assert wet_frac["conserving"] > 0.95, wet_frac
    assert wet_frac["legacy"] < 0.05, wet_frac


def test_still_water_below_datum_stays_dry_on_land(tmp_path: Path) -> None:
    """Land below the datum must not fill with water (legacy writes eta = 0,
    the datum, into fully dry cells)."""
    solver, run = _build_channel(tmp_path, "conserving")
    bed = np.full((solver.nx, solver.ny), -3.0)
    bed[:, 16:] = -1.0  # land at -1 m, below the datum; water at -2 m
    _start(solver, run, bed, -2.0)
    dt = float(solver.dt)
    for i in range(round(30.0 / dt)):
        run.Evolve_Steps(i)
    eta = _eta(solver)[INNER]
    b = bed[INNER]
    land = b > -1.5
    assert np.abs(eta - b)[land].max() < 1e-4
    assert np.abs(eta + 2.0)[~land].max() < 1e-4


# ---------------------------------------------------------------------------
# DischargeBoundary (type 5 edge inflow) and the series file loader.
# ---------------------------------------------------------------------------


def _build_river_basin(tmp_path: Path, east: int) -> tuple[Solver, Evolve]:
    """Flat 1 m deep basin with a discharge boundary on the west edge."""
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    xs, ys = np.linspace(0.0, 40.0, 81), np.linspace(0.0, 8.0, 17)
    xg, yg = np.meshgrid(xs, ys)
    np.savetxt(
        tmp_path / "basin.xyz",
        np.column_stack([xg.ravel(), yg.ravel(), np.ones(xg.size)]),
    )
    topo = Topodata(filename="basin.xyz", path=str(tmp_path), datatype="xyz")
    bc = BoundaryConditions(
        celeris=False, North=0, South=0, East=east, West=5, BoundaryWidth=10
    )
    domain = Domain(topodata=topo, x1=0.0, x2=40.0, y1=0.0, y2=8.0, Nx=160, Ny=32)
    solver = Solver(domain=domain, boundary_conditions=bc, model="SWE")
    return solver, Evolve(solver=solver, maxsteps=1)


def test_read_discharge_series_telemac_style(tmp_path: Path) -> None:
    from celeris.hydrograph import read_discharge_series

    f = tmp_path / "inflow.liq"
    f.write_text(
        "# hydrograph\nT Q(2)\ns m3/s\n0.0 50.0\n7200.0 1000.0\n18800.0, 50.0\n"
    )
    t, q = read_discharge_series(str(f))
    assert t.tolist() == [0.0, 7200.0, 18800.0]
    assert q.tolist() == [50.0, 1000.0, 50.0]


def test_discharge_boundary_rejects_wrong_type(tmp_path: Path) -> None:
    from celeris.hydrograph import DischargeBoundary

    solver, _ = _build_river_basin(tmp_path, east=0)
    with pytest.raises(ValueError, match="type"):
        DischargeBoundary(solver, "east", [0.0, 10.0], [1.0, 1.0])
    with pytest.raises(ValueError):
        DischargeBoundary(solver, "west", [0.0, 10.0, 5.0], [1.0, 1.0, 1.0])


def test_discharge_boundary_closed_basin_volume(tmp_path: Path) -> None:
    """Water entering through the west edge shows up as volume, to 3 %."""
    from celeris.hydrograph import DischargeBoundary

    solver, run = _build_river_basin(tmp_path, east=0)
    river = DischargeBoundary(solver, "west", [0.0, 30.0], [1.0, 1.0])
    solver.inflows.append(river)
    run.Evolve_0()

    def vol() -> (
        float
    ):  # west column is the inflow face, a full cell; other edges are walls
        depth = np.maximum(_eta(solver) - _bed(solver), 0.0)
        w = np.ones_like(depth)
        w[-3, :] *= 0.5
        w[:, [2, -3]] *= 0.5
        return float((depth * w)[INNER].sum() * solver.dx * solver.dy)

    v0 = vol()
    dt = float(solver.dt)
    for i in range(round(30.0 / dt)):
        run.Evolve_Steps(i)
    assert np.isfinite(solver.State.to_numpy()).all()
    assert vol() - v0 == pytest.approx(river.volume_m3, rel=0.03)


def test_discharge_boundary_channel_flux(tmp_path: Path) -> None:
    """Steady river inflow, sponge outlet: mid-channel flux equals Q and the
    water arrives with velocity, no inlet hump (surface within 2 cm of flat)."""
    from celeris.hydrograph import DischargeBoundary

    q_in = 0.5
    solver, run = _build_river_basin(tmp_path, east=1)
    solver.inflows.append(DischargeBoundary(solver, "west", [0.0, 400.0], [q_in, q_in]))
    run.Evolve_0()
    dt = float(solver.dt)
    i_mid = solver.nx // 2
    flux = []
    for i in range(int(300.0 / dt)):
        run.Evolve_Steps(i)
        if i * dt > 200.0 and i % 20 == 0:
            flux.append(_section_q(solver.State.to_numpy(), i_mid, 1, solver.dy))
    eta = _eta(solver)[INNER]
    assert np.isfinite(solver.State.to_numpy()).all()
    assert np.mean(flux) == pytest.approx(q_in, rel=0.15)
    assert eta[:8].mean() - eta[-8:].mean() < 0.02  # no hump at the inlet


# ---------------------------------------------------------------------------
# Several interior sources at once, and the location helper.
# ---------------------------------------------------------------------------


def test_two_sources_add_up_and_overlap_is_refused(tmp_path: Path) -> None:
    from celeris.hydrograph import inlet_mask

    solver, run = _build_basin(tmp_path, lambda x: np.full_like(x, 1.0))
    run.Evolve_0()  # zero state = still water at the datum, so wetness is known
    m1 = inlet_mask(solver, 8.0, 4.0, 1.5)
    m2 = inlet_mask(solver, 30.0, 4.0, 1.5)
    assert m1.sum() == pytest.approx(np.pi * 1.5**2 / (solver.dx * solver.dy), rel=0.15)
    assert not (m1 & m2).any()
    s1 = HydrographSource(solver, m1, [0.0, 30.0], [1.0, 1.0])
    solver.inflows.append(s1)
    s2 = HydrographSource(solver, m2, [0.0, 15.0, 30.0], [0.0, 2.0, 0.0])
    solver.inflows.append(s2)
    with pytest.raises(ValueError, match="overlaps"):
        HydrographSource(
            solver, inlet_mask(solver, 9.0, 4.0, 1.5), [0.0, 1.0], [1.0, 1.0]
        )
    v0 = _wet_volume(solver)
    dt = float(solver.dt)
    for i in range(round(30.0 / dt)):
        run.Evolve_Steps(i)
    gained = _wet_volume(solver) - v0
    assert gained == pytest.approx(s1.volume_m3 + s2.volume_m3, rel=0.03)


def test_inlet_mask_rejects_dry_location(tmp_path: Path) -> None:
    from celeris.hydrograph import inlet_mask

    solver, run = _build_basin(tmp_path, lambda x: (20.0 - x) / 10.0)  # dry for x > 20
    run.Evolve_0()
    with pytest.raises(ValueError, match="no wet cell"):
        inlet_mask(solver, 30.0, 4.0, 1.0)
