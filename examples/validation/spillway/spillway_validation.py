"""TELEMAC-2D ``weirs2`` example: five weirs between six ponds, as SpillwaySinks.

Model-to-model check of :class:`celeris.spillway.SpillwaySink`. TELEMAC's
generic weirs (``TYPE OF WEIRS = 2``) move water across a line the mesh does
not resolve, at the rate of Poleni's law on the local levels (``LOI_W_INC``,
``mu = 0.4``), and write the discharge of every weir segment to a CSV. Celeris
gets the same six flat ponds (beds 100, 95, 90, 85, 80, 75 m in an L, 50 m
walls between them), the same inflow (0 to 500 m3/s in 200 s on the west edge
of pond 1), the same outlet (pond 6 held at 76 m) and one ``SpillwaySink`` per
weir whose rating is the TELEMAC segment law summed over the weir's segments
(``weirs2.txt``). Compared: the mean level of every pond and the discharge of
every weir over the 12 h run.

    uv run python examples/validation/spillway/spillway_validation.py telemac
    uv run python examples/validation/spillway/spillway_validation.py run
    uv run python examples/validation/spillway/spillway_validation.py compare

Large products go to the validation data root (``paths.py``); only the metrics
CSV and the figure land in ``results/`` next to this file.
"""

import argparse
import csv
import math
import os
import shutil
import subprocess
import sys
import time
from itertools import pairwise
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "malpasset"))
sys.path.insert(0, str(HERE.parent))
from paths import celeris_out, telemac_example, telemac_ref
from read_selafin import read_selafin

MESH_DIR = telemac_example("weirs")
TELEMAC_OUT = telemac_ref() / "weirs2"
OUT_DEFAULT = celeris_out("spillway")
RESULTS = HERE / "results"

G = 9.81
DX_M = 10.0
PAD = 2  # ghost rim; the wall passes through the centre of cell 2 (= x 0)
DATUM_M = 76.0  # the outlet level: pond 6 sits at eta 0
WALL_M = 120.0
DURATION_S = 43200.0
SAMPLE_S = 300.0  # TELEMAC graphic printout period (30 steps of 10 s)
INFLOW_T_S = np.array([0.0, 200.0, 86400.0])  # t2d_weirs2.liq
INFLOW_Q_M3S = np.array([0.0, 500.0, 500.0])
OUTLET_LEVEL_M = 76.0
MANNING_N = 1.0 / 50.0  # Strickler 50
COURANT = 0.2
PHI = 0.4  # hard-coded in TELEMAC's calcul_q_weir.f
TELEMAC_DT_S = 10.0
RELAX = 0.5  # weirs2.txt line 2; Q = (1 - r) Q_new + r Q_old each 10 s step
SMOOTHING_S = TELEMAC_DT_S / math.log(1.0 / RELAX)  # same e-folding time
STRIP_M = 30.0  # patch depth into each pond along the weir
OUTLET_X_M = (20.0, 70.0)  # pond-6 columns held at the outlet level
DRY_WEST_M = 20.0  # pond-6 columns walled off so only pond 1 wets the inflow edge

# (x0, x1, y0, y1, bed) of the six ponds in flow order
PONDS = (
    (0.0, 1000.0, 0.0, 1000.0, 100.0),
    (1050.0, 2050.0, 0.0, 1000.0, 95.0),
    (2100.0, 3100.0, 0.0, 1000.0, 90.0),
    (2100.0, 3100.0, 1050.0, 2050.0, 85.0),
    (1050.0, 2050.0, 1050.0, 2050.0, 80.0),
    (0.0, 1000.0, 1050.0, 2050.0, 75.0),
)
INITIAL_LEVEL_M = (101.0, 95.1, 90.1, 85.1, 80.1, 76.0)


# ----------------------------------------------------------------------------
# TELEMAC weir law (sources/telemac2d/loi_w_inc.f) and the weirs2.txt file
# ----------------------------------------------------------------------------
def loi_w_inc(yam: float, yav: float, ys1: float, ys2: float, width: float) -> float:
    """Discharge over one inclined weir segment (sill from ``ys1`` to ``ys2``),
    upstream level ``yam`` >= downstream ``yav``; TELEMAC's ``LOI_W_INC``."""
    if width < 1e-3:
        return 0.0
    slope = abs(ys1 - ys2) / width
    if slope > 1e-4:
        ysmin = min(ys1, ys2)
        xpd = (yam - ysmin) / slope
    else:
        ysmin = ys1
        xpd = width
    xd = min(xpd, width)
    xpn = 3.0 * yav - 2.0 * yam - ysmin
    if xpn <= 0.0:
        xn = 0.0
    elif xpn <= width * slope:
        xn = xpn / slope
    else:
        xn = width
    if yam < ysmin and yav < ysmin:
        return 0.0
    qn = ((yav - ysmin) * xn - 0.5 * slope * xn**2) * math.sqrt(max(yam - yav, 0.0))
    if slope > 1e-4:
        aux0 = yam - ysmin
        aux1 = max(yam - ysmin - xd * slope, 0.0)
        aux2 = max(yam - ysmin - xn * slope, 0.0)
        aux3 = aux1**1.5 - aux2**1.5
        aux4 = (
            xd * aux1**1.5
            + 2.0 / (5.0 * slope) * aux1**2.5
            - xn * aux2**1.5
            - 2.0 / (5.0 * slope) * aux2**2.5
        )
        qd = PHI * 2.0 / (3.0 * slope) * (slope * aux4 - aux0 * aux3)
    else:
        qd = PHI * (yam - ysmin) * math.sqrt(max(yam - ysmin, 0.0)) * (xd - xn)
    return math.sqrt(2.0 * G) * (qn + qd)


def read_weirs(path: Path) -> list[dict]:
    """The five weirs: segments ``(xa, ya, za, xb, yb, zb)`` and the left/right
    lines of the levee header (side 1 = left, side 2 = right)."""
    weirs, cur = [], None
    for line in path.read_text().splitlines():
        s = line.strip()
        if s.startswith("#Levee") and "element count" in s:
            cur = {"segments": [], "lines": None}
            weirs.append(cur)
        elif cur is not None and s.startswith("#") and cur["lines"] is None:
            vals = s[1:].split()
            if len(vals) == 8 and all(_isnum(v) for v in vals):
                cur["lines"] = np.array([float(v) for v in vals]).reshape(2, 2, 2)
        elif cur is not None and s and _isnum(s.split()[0]) and len(s.split()) > 15:
            v = s.split()
            cur["segments"].append([float(v[k]) for k in (1, 2, 3, 8, 9, 10)])
    for w in weirs:
        w["segments"] = np.array(w["segments"])
        assert w["lines"] is not None and len(w["segments"]) > 0
    assert [len(w["segments"]) for w in weirs] == [20, 40, 10, 24, 32]
    return weirs


def _isnum(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def weir_rating(segments: np.ndarray):
    """``Q(h_side1, h_side2)`` of one weir: TELEMAC's per-segment law summed,
    positive from side 1 to side 2 (``calcul_q_weir.f``)."""
    widths = np.hypot(segments[:, 3] - segments[:, 0], segments[:, 4] - segments[:, 1])

    def rating(h1: float, h2: float) -> float:
        up, down, sign = (h1, h2, 1.0) if h1 > h2 else (h2, h1, -1.0)
        q = 0.0
        for (za, zb), w in zip(segments[:, [2, 5]], widths, strict=True):
            q += loi_w_inc(up, down, za, zb, w)
        return sign * q

    return rating


def pond_of(x: float, y: float) -> int:
    for k, (x0, x1, y0, y1, _) in enumerate(PONDS):
        if x0 - 1.0 <= x <= x1 + 1.0 and y0 - 1.0 <= y <= y1 + 1.0:
            return k
    raise ValueError(f"({x}, {y}) is in no pond")


def weir_sides(w: dict) -> tuple[int, int]:
    """Ponds on side 1 (left line) and side 2 (right line) of a weir."""
    left, right = w["lines"]
    lm, rm = left.mean(axis=0), right.mean(axis=0)
    step = 5.0 * (lm - rm) / np.linalg.norm(lm - rm)
    return pond_of(*(lm + step)), pond_of(*(rm - step))


# ----------------------------------------------------------------------------
# TELEMAC reference run
# ----------------------------------------------------------------------------
def _telemac_env() -> dict[str, str]:
    root = Path(os.environ.get("HOMETEL", Path.home() / "telemac-mascaret"))
    cfg = os.environ.get("USETELCFG", "gfortran")
    scripts, build = root / "scripts/python3", root / "builds" / cfg
    env = os.environ.copy()
    env.update(
        HOMETEL=str(root),
        SYSTELCFG=str(root / "configs/systel.local.cfg"),
        USETELCFG=cfg,
        PATH=f"{scripts}:{env.get('PATH', '')}",
        PYTHONPATH=str(scripts),
        LD_LIBRARY_PATH=f"{build / 'lib'}:{build / 'wrap_api/lib'}",
        OMP_NUM_THREADS="1",
    )
    return env


def telemac() -> None:
    """Run ``t2d_weirs2.cas`` (serial) and reduce it to pond levels and weir Q."""
    TELEMAC_OUT.mkdir(parents=True, exist_ok=True)
    for f in (
        "t2d_weirs2.cas",
        "t2d_weirs2.liq",
        "init_weirs2.slf",
        "weirs2.txt",
        "geo_weirs2.slf",
        "geo_weirs2.cli",
    ):
        shutil.copy2(MESH_DIR / f, TELEMAC_OUT / f)
    env = _telemac_env()
    cmd = [
        sys.executable,
        f"{env['HOMETEL']}/scripts/python3/telemac2d.py",
        "t2d_weirs2.cas",
        "--ncsize",
        "0",
        "--workdirectory",
        "solver",
    ]
    with open(TELEMAC_OUT / "run.log", "w") as log:
        subprocess.run(
            cmd, cwd=TELEMAC_OUT, env=env, stdout=log, stderr=log, check=True
        )
    reduce_telemac()


def reduce_telemac() -> None:
    res = read_selafin(TELEMAC_OUT / "res_weir2.slf")
    bed = res["variables"]["BOTTOM"][0]
    fs = res["variables"]["FREE SURFACE"]
    levels = np.stack(
        [fs[:, np.isclose(bed, p[4])].mean(axis=1) for p in PONDS], axis=1
    )
    q = np.genfromtxt(TELEMAC_OUT / "Qweirs2.csv", delimiter=",", skip_header=2)
    counts = [len(w["segments"]) for w in read_weirs(TELEMAC_OUT / "weirs2.txt")]
    edges = np.cumsum([1, *counts])
    q_weirs = np.stack(
        [q[:, a:b].sum(axis=1) for a, b in pairwise(edges)],
        axis=1,
    )
    np.savez_compressed(
        TELEMAC_OUT / "telemac_weirs2.npz",
        t_s=res["times"],
        levels=levels,
        q_t_s=q[:, 0],
        q_weirs=q_weirs,
    )
    print(
        f"TELEMAC: {len(res['times'])} frames to {res['times'][-1]:.0f} s | final pond "
        f"levels {np.round(levels[-1], 3).tolist()} | final weir Q "
        f"{np.round(q_weirs[-1], 1).tolist()}"
    )


# ----------------------------------------------------------------------------
# Celeris twin
# ----------------------------------------------------------------------------
def build_grid() -> dict:
    nx = round(3100.0 / DX_M) + 2 * PAD + 1
    ny = round(2050.0 / DX_M) + 2 * PAD + 1
    xg, yg = np.meshgrid(
        (np.arange(nx) - PAD) * DX_M, (np.arange(ny) - PAD) * DX_M, indexing="ij"
    )
    bed = np.full((nx, ny), WALL_M)
    eta = bed.copy()
    pond = np.full((nx, ny), -1, dtype=np.int64)
    for k, (x0, x1, y0, y1, z) in enumerate(PONDS):
        m = (xg >= x0) & (xg < x1) & (yg >= y0) & (yg < y1)
        bed[m], eta[m], pond[m] = z, INITIAL_LEVEL_M[k], k
    # pond 6: wall off the west columns so the type-5 edge only sees pond 1, and
    # mark the outlet strip held at 76 m
    p6 = pond == 5
    dry = p6 & (xg < DRY_WEST_M)
    bed[dry], eta[dry], pond[dry] = WALL_M, WALL_M, -1
    outlet = p6 & (xg >= OUTLET_X_M[0]) & (xg < OUTLET_X_M[1])
    weirs = read_weirs(MESH_DIR / "weirs2.txt")
    # the sink refuses the ghost rim and the wall cells (three outer rows/columns);
    # the pond edges at y = 0 sit on that wall row, so the strips stop one cell short
    interior = np.zeros_like(p6)
    interior[3:-3, 3:-3] = True
    used = np.zeros_like(p6)  # two weirs meet in a pond corner: first one keeps it
    patches = []
    for w in weirs:
        s1, s2 = weir_sides(w)
        left, right = w["lines"]
        seg = w["segments"]
        lo = np.minimum(seg[:, :2].min(axis=0), seg[:, 3:5].min(axis=0)) - DX_M
        hi = np.maximum(seg[:, :2].max(axis=0), seg[:, 3:5].max(axis=0)) + DX_M
        pa = (pond == s1) & (_dist_to_line(xg, yg, left) <= STRIP_M)
        pb = (pond == s2) & (_dist_to_line(xg, yg, right) <= STRIP_M)
        # the weir's along-line extent, but the lines run the full pond edge
        # when the weir is shorter (weir 3, 4): keep the strip to the weir
        axis = int(np.argmax(hi - lo))  # 0 = weir runs along x, 1 = along y
        span = xg if axis == 0 else yg
        pa &= (span >= lo[axis]) & (span <= hi[axis]) & interior & ~used
        pb &= (span >= lo[axis]) & (span <= hi[axis]) & interior & ~used
        used |= pa | pb
        assert pa.any() and pb.any(), "empty weir patch"
        patches.append((s1, s2, pa, pb))
    for arr in (bed, eta):
        arr[:PAD, :] = arr[PAD, :]
        arr[nx - PAD :, :] = arr[nx - PAD - 1, :]
        arr[:, :PAD] = arr[:, PAD][:, None]
        arr[:, ny - PAD :] = arr[:, ny - PAD - 1][:, None]
    return {
        "bed": bed,
        "eta0": eta,
        "pond": pond,
        "outlet": outlet,
        "patches": patches,
        "weirs": weirs,
    }


def _dist_to_line(xg: np.ndarray, yg: np.ndarray, line: np.ndarray) -> np.ndarray:
    (xa, ya), (xb, yb) = line
    if abs(xb - xa) < 1e-9:
        return np.abs(xg - xa)
    if abs(yb - ya) < 1e-9:
        return np.abs(yg - ya)
    return np.abs((yb - ya) * xg - (xb - xa) * yg + xb * ya - yb * xa) / math.hypot(
        xb - xa, yb - ya
    )


def run(out: Path, duration_s: float, arch: str, wetdry: str) -> None:
    import taichi as ti

    from celeris.domain import BoundaryConditions, Domain, Topodata
    from celeris.hydrograph import DischargeBoundary
    from celeris.runner import Evolve
    from celeris.solver import Solver
    from celeris.spillway import SpillwaySink

    g = build_grid()
    bed, eta0, pond = g["bed"], g["eta0"], g["pond"]
    nx, ny = bed.shape
    out.mkdir(parents=True, exist_ok=True)
    xg, yg = np.meshgrid(np.arange(nx) * DX_M, np.arange(ny) * DX_M, indexing="ij")
    np.savetxt(
        out / "bathy.xyz",
        np.column_stack([xg.ravel(), yg.ravel(), (DATUM_M - bed).ravel()]),
    )
    ti.init(arch=getattr(ti, arch), default_fp=ti.f32)
    topo = Topodata(filename="bathy.xyz", path=str(out), datatype="xyz")
    bc = BoundaryConditions(celeris=False, North=0, East=0, South=0, West=5)
    dom = Domain(
        topodata=topo,
        x1=0.0,
        x2=nx * DX_M,
        y1=0.0,
        y2=ny * DX_M,
        Nx=nx,
        Ny=ny,
        isManning=1,
        friction=MANNING_N,
        Courant=COURANT,
    )
    solver = Solver(
        domain=dom,
        boundary_conditions=bc,
        model="SWE",
        infiltrationRate=0.0,
        wetdry_scheme=wetdry,
    )
    evolve = Evolve(solver=solver, maxsteps=1)
    state = np.zeros((nx, ny, 4), dtype=np.float32)
    state[:, :, 0] = eta0 - DATUM_M
    solver.InitStates()
    for f in (
        solver.State,
        solver.stateUVstar,
        solver.NewState,
        solver.current_stateUVstar,
    ):
        f.from_numpy(state)
    solver.InitStates = lambda: None
    solver.inflows.append(DischargeBoundary(solver, "west", INFLOW_T_S, INFLOW_Q_M3S))
    sinks = []
    for (s1, s2, pa, pb), w in zip(g["patches"], g["weirs"], strict=True):
        rate = weir_rating(w["segments"])
        # the rating works in m a.s.l.; the sink hands over datum-relative levels
        sinks.append(
            SpillwaySink(
                solver,
                pa,
                lambda h1, h2, r=rate: r(h1 + DATUM_M, h2 + DATUM_M),
                outlet_mask=pb,
                min_depth_m=0.02,
                smoothing_s=SMOOTHING_S,
            )
        )
        solver.inflows.append(sinks[-1])
        print(
            f"weir {len(sinks)}: pond {s1 + 1} -> pond {s2 + 1}, {len(w['segments'])} "
            f"segments, sill {w['segments'][:, 2].min():.2f}-"
            f"{w['segments'][:, 2].max():.2f} m, patches {pa.sum()}/{pb.sum()} cells"
        )
    evolve.Evolve_0()

    outlet = ti.field(ti.i32, shape=(nx, ny))
    outlet.from_numpy(g["outlet"].astype(np.int32))
    eta_out = float(OUTLET_LEVEL_M - DATUM_M)

    @ti.kernel
    def impose_outlet():  # type: ignore[no-untyped-def]
        for i, j in outlet:
            if outlet[i, j] == 1:
                solver.State[i, j][0] = eta_out
                solver.stateUVstar[i, j][0] = eta_out

    dt = float(solver.dt)
    n_steps = int(duration_s / dt)
    every = max(1, int(SAMPLE_S / dt))
    pond_masks = [pond == k for k in range(len(PONDS))]
    cell = DX_M * DX_M
    t_s, levels, q_weirs, volume = [], [], [], []
    t0 = time.perf_counter()
    print(f"grid {nx}x{ny} | dt {dt:.4f} s | {n_steps} steps", flush=True)
    for step in range(n_steps + 1):
        t = step * dt
        if step % every == 0 or step == n_steps:
            st = solver.State.to_numpy()
            eta = st[:, :, 0] + DATUM_M
            if not np.isfinite(st).all() or np.abs(st[:, :, 0]).max() > 1e3:
                print(f"BLOW-UP at t={t:.1f}s", flush=True)
                break
            t_s.append(t)
            levels.append([float(eta[m].mean()) for m in pond_masks])
            q_weirs.append([s.q_now for s in sinks])
            volume.append(float(np.maximum(eta - bed, 0.0)[pond >= 0].sum() * cell))
            if step % (every * 12) == 0:
                rate = t / max(time.perf_counter() - t0, 1e-9)
                print(
                    f"t={t:6.0f}s ({rate:5.0f}x rt) ponds "
                    f"{np.round(levels[-1], 2).tolist()} Q "
                    f"{np.round(q_weirs[-1], 1).tolist()}",
                    flush=True,
                )
        if step == n_steps:
            break
        evolve.Evolve_Steps(step)
        impose_outlet()
    np.savez_compressed(
        out / "results.npz",
        t_s=np.array(t_s),
        levels=np.array(levels),
        q_weirs=np.array(q_weirs),
        volume_m3=np.array(volume),
        volume_out_m3=np.array([s.volume_out_m3 for s in sinks]),
        dt=dt,
        eta_final=(solver.State.to_numpy()[:, :, 0] + DATUM_M).astype(np.float32),
        bed=bed.astype(np.float32),
    )
    print(f"wall {time.perf_counter() - t0:.0f}s -> {out / 'results.npz'}", flush=True)


# ----------------------------------------------------------------------------
# Comparison
# ----------------------------------------------------------------------------
def compare(out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tel = dict(np.load(TELEMAC_OUT / "telemac_weirs2.npz"))
    cel = dict(np.load(out / "results.npz"))
    RESULTS.mkdir(exist_ok=True)
    t_h = cel["t_s"] / 3600.0
    tel_levels = np.stack(
        [np.interp(cel["t_s"], tel["t_s"], tel["levels"][:, k]) for k in range(6)],
        axis=1,
    )
    tel_q = np.stack(
        [np.interp(cel["t_s"], tel["q_t_s"], tel["q_weirs"][:, k]) for k in range(5)],
        axis=1,
    )
    rows = []
    for k in range(6):
        d = cel["levels"][:, k] - tel_levels[:, k]
        rows.append(
            {
                "item": f"pond {k + 1} level (m)",
                "telemac_final": f"{tel_levels[-1, k]:.3f}",
                "celeris_final": f"{cel['levels'][-1, k]:.3f}",
                "rmse": f"{np.sqrt(np.mean(d**2)):.3f}",
                "max_abs": f"{np.abs(d).max():.3f}",
            }
        )
    for k in range(5):
        d = np.abs(cel["q_weirs"][:, k]) - np.abs(tel_q[:, k])
        rows.append(
            {
                "item": f"weir {k + 1} |Q| (m3/s)",
                "telemac_final": f"{abs(tel_q[-1, k]):.1f}",
                "celeris_final": f"{abs(cel['q_weirs'][-1, k]):.1f}",
                "rmse": f"{np.sqrt(np.mean(d**2)):.1f}",
                "max_abs": f"{np.abs(d).max():.1f}",
            }
        )
    with open(RESULTS / "spillway_metrics.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    colors = plt.get_cmap("viridis")(np.linspace(0.0, 0.9, 6))
    ax = axes[0]
    for k in range(6):
        ax.plot(
            tel["t_s"] / 3600.0,
            tel["levels"][:, k],
            color=colors[k],
            lw=2.2,
            alpha=0.45,
            label=f"Pond {k + 1} (Bed {PONDS[k][4]:.0f} m)",
        )
        ax.plot(t_h, cel["levels"][:, k], color=colors[k], lw=1.2)
    ax.plot([], [], color="0.3", lw=2.2, alpha=0.45, label="TELEMAC-2D")
    ax.plot([], [], color="0.3", lw=1.2, label="Celeris")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Mean Pond Level (m)")
    ax.set_title("Pond Levels")
    ax.legend(fontsize=8, loc="center right")
    ax = axes[1]
    wc = plt.get_cmap("plasma")(np.linspace(0.0, 0.85, 5))
    for k in range(5):
        ax.plot(
            tel["q_t_s"] / 3600.0,
            np.abs(tel["q_weirs"][:, k]),
            color=wc[k],
            lw=2.2,
            alpha=0.45,
            label=f"Weir {k + 1}",
        )
        ax.plot(t_h, np.abs(cel["q_weirs"][:, k]), color=wc[k], lw=1.2)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Weir Discharge (m$^3$/s)")
    ax.set_title("Weir Discharges (TELEMAC Thick, Celeris Thin)")
    ax.legend(fontsize=8)
    ax = axes[2]
    for k in range(6):
        ax.plot(
            t_h,
            cel["levels"][:, k] - tel_levels[:, k],
            color=colors[k],
            lw=1.2,
            label=f"Pond {k + 1}",
        )
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Celeris Minus TELEMAC (m)")
    ax.set_title("Level Difference")
    ax.legend(fontsize=8)
    for ax, letter in zip(axes, "abc", strict=True):
        ax.text(
            0.02,
            0.96,
            f"({letter})",
            transform=ax.transAxes,
            va="top",
            fontweight="bold",
        )
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "spillway_weirs2.png", dpi=140)
    print(f"-> {RESULTS / 'spillway_metrics.csv'}, {RESULTS / 'spillway_weirs2.png'}")
    for r in rows:
        print(r)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["telemac", "run", "compare"])
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--duration", type=float, default=DURATION_S)
    ap.add_argument("--arch", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--wetdry", choices=["legacy", "conserving"], default="conserving")
    a = ap.parse_args()
    if a.stage == "telemac":
        telemac()
    elif a.stage == "run":
        run(a.out, a.duration, a.arch, a.wetdry)
    else:
        compare(a.out)


if __name__ == "__main__":
    main()
