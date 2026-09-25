"""Acceptance tests for the level-dependent outlet (``celeris.spillway``).

Small synthetic basins, SWE, CPU backend, conserving wet/dry. The exact
reference is a flat tank draining through a Poleni weir: with
``Q = k (h - z)^1.5`` and area ``A``, ``(h - z)^-1/2 = (h0 - z)^-1/2 + k t / (2 A)``.
"""

import math
from pathlib import Path

import numpy as np
import pytest
from test_hydrograph import (
    INNER,
    G,
    _build_basin,
    _build_channel,
    _build_river_basin,
    _eta,
    _start,
    _strip_mask,
    _wet_volume,
)

from celeris.hydrograph import DischargeBoundary, HydrographSource
from celeris.spillway import SpillwaySink, poleni, rating_table, read_rating_table


def _pool_area(solver) -> float:
    """Wet area inside the walls (walls pass through the centres of cells 2, n-3)."""
    return (solver.nx - 5) * (solver.ny - 5) * solver.dx * solver.dy


def _exact_drawdown(h0: float, sill: float, k: float, area: float, t: np.ndarray):
    return sill + (1.0 / math.sqrt(h0 - sill) + k * t / (2.0 * area)) ** -2


def test_poleni_law() -> None:
    q = poleni(sill_m=0.0, width_m=2.0, mu=0.4)
    k = 0.4 * math.sqrt(2.0 * G) * 2.0
    assert q(0.5, -1.0) == pytest.approx(k * 0.5**1.5)
    assert q(0.5, 0.3) == pytest.approx(k * 0.5**1.5)  # h_down at the 2/3 limit: free
    assert q(0.5, 0.4) == pytest.approx(2.598 * k * 0.4 * math.sqrt(0.1))  # drowned
    assert q(-0.1, -1.0) == 0.0
    assert q(-1.0, 0.5) == pytest.approx(-k * 0.5**1.5)  # reversed head, reversed sign
    with pytest.raises(ValueError):
        poleni(0.0, 0.0)


def test_rating_table(tmp_path: Path) -> None:
    r = rating_table([0.0, 1.0, 2.0], [0.0, 5.0, 20.0])
    assert r(-0.5, 0.0) == 0.0
    assert r(0.5, 0.0) == pytest.approx(2.5)
    assert r(3.0, 0.0) == pytest.approx(20.0)  # held above the table
    (tmp_path / "rating.txt").write_text("# level Q\n0.0 0.0\n1.0 5.0\n")
    assert read_rating_table(str(tmp_path / "rating.txt"))(0.5, 0.0) == pytest.approx(
        2.5
    )
    for h, q in (([0.0, 0.0], [0.0, 1.0]), ([0.0, 1.0], [1.0, 0.0]), ([0.0], [0.0])):
        with pytest.raises(ValueError):
            rating_table(h, q)


def test_rejects_bad_inputs(tmp_path: Path) -> None:
    solver, _ = _build_basin(tmp_path, lambda x: np.full_like(x, 1.0))
    good = _strip_mask(solver, 10, 14)
    q = poleni(0.0, 1.0)
    with pytest.raises(ValueError):
        SpillwaySink(solver, good[:-1], q)
    with pytest.raises(ValueError):
        SpillwaySink(solver, np.zeros_like(good), q)
    rim = np.zeros_like(good)
    rim[1, 10] = True
    with pytest.raises(ValueError):
        SpillwaySink(solver, rim, q)
    with pytest.raises(ValueError):
        SpillwaySink(solver, good, q, outlet_mask=good)
    solver.inflows.append(HydrographSource(solver, good, [0.0, 10.0], [1.0, 1.0]))
    with pytest.raises(ValueError):
        SpillwaySink(solver, _strip_mask(solver, 12, 16), q)
    with pytest.raises(ValueError):
        SpillwaySink(solver, _strip_mask(solver, 20, 24), q, outlet_mask=good)


def test_tank_drawdown_matches_exact_poleni(tmp_path: Path) -> None:
    """A flat 1 m tank draining through a Poleni weir follows the closed form."""
    solver, run = _build_basin(tmp_path, lambda x: np.full_like(x, 1.0))
    sill, mu, width = -0.4, 0.4, 0.5
    sink = SpillwaySink(solver, _strip_mask(solver, 10, 20), poleni(sill, width, mu))
    solver.inflows.append(sink)
    run.Evolve_0()
    v0 = _wet_volume(solver)
    area = _pool_area(solver)
    k = mu * math.sqrt(2.0 * G) * width
    dt = float(solver.dt)
    checks = {}
    for i in range(round(200.0 / dt)):
        run.Evolve_Steps(i)
        t = (i + 1) * dt
        for t_chk in (50.0, 100.0, 200.0):
            if abs(t - t_chk) < 0.5 * dt:
                checks[t_chk] = (float(_eta(solver)[INNER].mean()), _wet_volume(solver))
    assert np.isfinite(solver.State.to_numpy()).all()
    t_arr = np.array(sorted(checks))
    h_exact = _exact_drawdown(0.0, sill, k, area, t_arr)
    h_model = np.array([checks[t][0] for t in t_arr])
    drop = -h_exact  # exact drop from the start (about 7 cm at 200 s)
    assert np.all(np.abs(h_model - h_exact) < 0.03 * drop)
    # the plumbing is exact: removed volume == the sink's own tally
    assert v0 - checks[200.0][1] == pytest.approx(sink.volume_out_m3, rel=0.01)
    assert sink.q_now > 0.0


def test_weir_transfers_between_two_basins(tmp_path: Path) -> None:
    """Sink + outlet across a dry ridge: mass conserved, levels equalise."""
    solver, run = _build_channel(tmp_path, "conserving")
    i = np.arange(solver.nx)[:, None]
    bed = np.where((i >= 76) & (i <= 84), 0.5, -1.0) * np.ones((1, solver.ny))
    level = np.where(i < 80, 0.3, 0.0) * np.ones((1, solver.ny))
    _start(solver, run, bed, 0.0)
    from test_hydrograph import _set_eta

    _set_eta(solver, np.where(bed < level, level, bed))
    intake = _strip_mask(solver, 60, 70)
    outlet = _strip_mask(solver, 90, 100)
    sink = SpillwaySink(solver, intake, poleni(0.0, 1.0, 0.4), outlet_mask=outlet)
    solver.inflows.append(sink)
    v0 = _wet_volume(solver)
    dt = float(solver.dt)
    for n in range(round(400.0 / dt)):
        run.Evolve_Steps(n)
    eta = _eta(solver)
    assert np.isfinite(eta).all()
    assert _wet_volume(solver) == pytest.approx(v0, rel=1e-3)
    left = float(eta[3:70, 3:-3].mean())
    right = float(eta[90:-3, 3:-3].mean())
    assert abs(left - right) < 0.01
    assert left == pytest.approx(0.15, abs=0.01)
    assert sink.volume_out_m3 > 0.0
    assert abs(sink.q_now) < 0.02


def test_clamp_keeps_patch_wet(tmp_path: Path) -> None:
    """An absurd rating cannot pull the intake below ``min_depth_m``."""
    solver, run = _build_basin(tmp_path, lambda x: np.full_like(x, 1.0))
    intake = _strip_mask(solver, 10, 14)
    sink = SpillwaySink(
        solver, intake, rating_table([-0.9, -0.8], [1.0e4, 1.0e4]), min_depth_m=0.2
    )
    solver.inflows.append(sink)
    run.Evolve_0()
    dt = float(solver.dt)
    for n in range(round(5.0 / dt)):
        run.Evolve_Steps(n)
    eta = _eta(solver)
    assert np.isfinite(eta).all()
    depth = eta - solver.Bottom.to_numpy()[2]
    assert depth[intake].min() > 0.2 - 0.02
    assert sink.q_now < 1.0e4


def test_steady_inflow_settles_at_the_rating_head(tmp_path: Path) -> None:
    """River in, weir out: the pool settles where the rating passes the inflow."""
    solver, run = _build_river_basin(tmp_path, east=0)
    q_in, mu, width, sill = 0.5, 0.4, 4.0, 0.0
    solver.inflows.append(DischargeBoundary(solver, "west", [0.0, 1.0e6], [q_in, q_in]))
    sink = SpillwaySink(solver, _strip_mask(solver, 130, 150), poleni(sill, width, mu))
    solver.inflows.append(sink)
    run.Evolve_0()
    dt = float(solver.dt)
    for n in range(round(400.0 / dt)):
        run.Evolve_Steps(n)
    assert np.isfinite(solver.State.to_numpy()).all()
    k = mu * math.sqrt(2.0 * G) * width
    head_exact = (q_in / k) ** (2.0 / 3.0)
    assert sink.q_now == pytest.approx(q_in, rel=0.03)
    pool = float(_eta(solver)[20:120, 3:-3].mean())
    assert pool - sill == pytest.approx(head_exact, rel=0.05)


def test_head_mask_reads_the_far_field(tmp_path: Path) -> None:
    """With a head patch the rating sees that patch's level, not the intake's."""
    solver, run = _build_basin(tmp_path, lambda x: np.full_like(x, 1.0))
    intake = _strip_mask(solver, 10, 12)
    head = _strip_mask(solver, 100, 140)
    sink = SpillwaySink(solver, intake, poleni(-0.4, 4.0, 0.4), head_mask=head)
    solver.inflows.append(sink)
    run.Evolve_0()
    dt = float(solver.dt)
    for n in range(round(20.0 / dt)):
        run.Evolve_Steps(n)
    eta = _eta(solver)
    h_up = sink.levels()[0]
    assert h_up == pytest.approx(float(eta[head].mean()), abs=1e-4)
    assert np.isfinite(eta).all()
