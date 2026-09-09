"""TELEMAC-2D ``breach`` example, dyke intact: hydrograph routing in a channel.

Model-to-model check of :class:`celeris.hydrograph.HydrographSource`. The
TELEMAC reference (``t2d_breach.cas`` with ``BREACH = NO``, restart from
``ini_breach.slf``) routes an inflow rising from 50 to 406 m3/s over 2700 s
down a 5 km trapezoidal channel (slope 1e-3, Strickler 15) with a prescribed
outlet stage. Celeris gets the same bed, the same initial state, the same
inflow as a volume source on a wet strip just inside the inlet, and the same
outlet stage imposed on a strip at the outlet.

    uv run python examples/validation/breach/breach_validation.py prep
    uv run python examples/validation/breach/breach_validation.py run
    uv run python examples/validation/breach/breach_validation.py compare

Large products go to ``--out`` (default /mnt/d/Homathko/Validation/celeris/breach);
only the metrics CSV and the figure land in ``results/`` next to this file.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "malpasset"))
from read_selafin import read_selafin  # noqa: E402

MESH_DIR = Path.home() / "telemac-mascaret/examples/telemac2d/breach"
TELEMAC_REF = Path("/mnt/d/Homathko/Validation/telemac/breach_nobreach_restart")
OUT_DEFAULT = Path("/mnt/d/Homathko/Validation/celeris/breach")

# Celeris datum (m a.s.l.). It must sit BELOW every piece of terrain that has to
# stay dry: the wet/dry logic treats cells with bed <= datum as sea floor and lets
# films survive there, and with the dyke below the datum the solver manufactured
# ~10,000 m3 on the dyke in 60 s (see README). Lowest floodplain bed is 5.0 m.
DATUM_M = 4.9
DX_M = 2.5
WALL_M = 20.0  # bed elevation given to cells outside the TELEMAC mesh hull
DRY_M = 0.01
MANNING_N = 1.0 / 15.0  # Strickler K = 15
COURANT = 0.2
DURATION_S = 2700.0
SAMPLE_S = 10.0
FRAME_TIMES_S = (0.0, 1000.0, 2000.0, 2400.0, 2700.0)
PROBES_X_M = (500.0, 1000.0, 1500.0, 1900.0, 3100.0, 4000.0, 4500.0)
PROBE_Y_M = 13.0  # channel centreline (TELEMAC README)
INLET_X_M = (10.0, 30.0)
OUTLET_COLS = 5  # stage-controlled columns inside the 2-cell ghost rim
# t2d_breach.liq: inflow Q (boundary 2) and outlet free surface (boundary 1)
LIQ_T_S = np.array([0.0, 7200.0, 18800.0])
LIQ_Q_M3S = np.array([50.0, 1000.0, 50.0])
LIQ_SL_M = np.array([0.87, 5.19, 0.87])


def _interp_on_mesh(mesh, values, xg, yg):
    from matplotlib.tri import LinearTriInterpolator, Triangulation

    tri = Triangulation(mesh["x"], mesh["y"], mesh["ikle"])
    return LinearTriInterpolator(tri, values)(xg, yg)  # masked outside the hull


def prep(out: Path) -> None:
    geo = read_selafin(MESH_DIR / "geo_breach.slf")
    ini = read_selafin(MESH_DIR / "ini_breach.slf")
    assert ini["npoin"] == geo["npoin"]
    v = {n: ini["values"][0, k] for k, n in enumerate(ini["varnames"])}
    nx, ny = int(round(5000.0 / DX_M)), int(round(500.0 / DX_M))
    xs, ys = np.arange(nx) * DX_M, np.arange(ny) * DX_M
    xg, yg = np.meshgrid(xs, ys, indexing="ij")
    # Interpolate nodal depth and momentum, not the free surface: dry nodes carry
    # eta = bed, so interpolating eta across a wet/dry bank perches spurious water
    # on the slope above the real surface.
    depth_n = np.maximum(v["FREE SURFACE"] - v["BOTTOM"], 0.0)
    bed = _interp_on_mesh(ini, v["BOTTOM"], xg, yg)
    depth = _interp_on_mesh(ini, depth_n, xg, yg)
    hu = _interp_on_mesh(ini, depth_n * v["VELOCITY U"], xg, yg)
    hv = _interp_on_mesh(ini, depth_n * v["VELOCITY V"], xg, yg)
    inside = ~np.ma.getmaskarray(bed)
    bed = np.where(inside, bed.filled(WALL_M), WALL_M)
    depth = np.where(inside, depth.filled(0.0), 0.0)
    depth[depth <= DRY_M] = 0.0
    eta = bed + depth
    hu = np.where(depth > 0, hu.filled(0.0), 0.0)
    hv = np.where(depth > 0, hv.filled(0.0), 0.0)
    inlet = (xg >= INLET_X_M[0]) & (xg < INLET_X_M[1]) & (depth > 0.3)
    outlet = np.zeros_like(inlet)
    outlet[nx - 2 - OUTLET_COLS : nx - 2, :] = True
    outlet &= inside & (bed < 6.0)  # channel section only
    floodplain = inside & (yg > 42.0) & (xg > 2000.0) & (xg < 3000.0)
    probes = np.array(
        [[int(round(px / DX_M)), int(round(PROBE_Y_M / DX_M))] for px in PROBES_X_M]
    )
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "grid.npz",
        dx=DX_M,
        datum=DATUM_M,
        bed=bed.astype(np.float32),
        eta0=eta.astype(np.float32),
        hu0=hu.astype(np.float32),
        hv0=hv.astype(np.float32),
        inside=inside,
        inlet=inlet,
        outlet=outlet,
        floodplain=floodplain,
        probes_ij=probes,
    )
    print(
        f"grid {nx}x{ny} @ {DX_M} m | inside {inside.sum()} cells | inlet {inlet.sum()} "
        f"cells ({inlet.sum() * DX_M**2:.0f} m2) | outlet {outlet.sum()} | floodplain "
        f"{floodplain.sum()} | wet volume {depth.sum() * DX_M**2:.0f} m3 | "
        f"bed probes {[round(float(bed[i, j]), 2) for i, j in probes]}"
    )


def run(out: Path, duration_s: float) -> None:
    import taichi as ti
    from celeris.domain import BoundaryConditions, Domain, Topodata
    from celeris.hydrograph import HydrographSource
    from celeris.runner import Evolve
    from celeris.solver import Solver

    g = dict(np.load(out / "grid.npz"))
    dx = float(g["dx"])
    bed = g["bed"].astype(np.float64)
    nx, ny = bed.shape
    xg, yg = np.meshgrid(np.arange(nx) * dx, np.arange(ny) * dx, indexing="ij")
    np.savetxt(
        out / "bathy.xyz",
        np.column_stack([xg.ravel(), yg.ravel(), (DATUM_M - bed).ravel()]),
    )
    ti.init(arch=ti.cuda, default_fp=ti.f32)
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
        friction=MANNING_N,
        Courant=COURANT,
    )
    solver = Solver(domain=dom, boundary_conditions=bc, model="SWE", infiltrationRate=0.0)
    solver.landslide = HydrographSource(
        solver, g["inlet"], LIQ_T_S, LIQ_Q_M3S, min_depth_m=0.2
    )
    evolve = Evolve(solver=solver, maxsteps=1)
    # Initial state must be written before Evolve_0's wet check runs.
    state = np.zeros((nx, ny, 4), dtype=np.float32)
    state[:, :, 0] = g["eta0"] - DATUM_M
    state[:, :, 1] = g["hu0"]
    state[:, :, 2] = g["hv0"]
    solver.InitStates()
    for f in (solver.State, solver.stateUVstar, solver.NewState, solver.current_stateUVstar):
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
    n_steps = int(duration_s / dt)
    every = max(1, int(SAMPLE_S / dt))
    probes = [tuple(int(v) for v in ij) for ij in g["probes_ij"]]
    fp, inside = g["floodplain"], g["inside"]
    inner = (slice(2, -2), slice(2, -2))
    t_s, eta_p, fp_wet, volume = [], [], [], []
    frames = {}
    t0 = time.perf_counter()
    print(f"grid {nx}x{ny} | dt {dt:.4f} s | {n_steps} steps", flush=True)
    for step in range(n_steps + 1):
        t = step * dt
        if step % every == 0 or step == n_steps:
            st = solver.State.to_numpy()
            eta = st[:, :, 0] + DATUM_M
            depth = np.where(inside, eta - bed, 0.0)
            if not np.isfinite(st).all() or np.abs(st[:, :, 0]).max() > 1e3:
                print(f"BLOW-UP at t={t:.1f}s", flush=True)
                break
            t_s.append(t)
            eta_p.append([eta[i, j] for i, j in probes])
            fp_wet.append(float((depth[fp] > DRY_M).mean()))
            volume.append(float(np.maximum(depth[inner], 0.0).sum() * dx * dx))
            for ft in FRAME_TIMES_S:
                if abs(t - ft) < dt and ft not in frames:
                    frames[ft] = np.where(depth > DRY_M, eta, np.nan).astype(np.float32)
            if step % (every * 30) == 0:
                rate = t / max(time.perf_counter() - t0, 1e-9)
                print(
                    f"t={t:6.0f}s ({rate:5.1f}x rt) probes "
                    f"{np.round(eta_p[-1], 3).tolist()} fp_wet {fp_wet[-1]:.3f}",
                    flush=True,
                )
        if step == n_steps:
            break
        evolve.Evolve_Steps(step)
        impose_outlet_stage(float(np.interp(t + dt, LIQ_T_S, LIQ_SL_M)) - DATUM_M)
    np.savez_compressed(
        out / "results.npz",
        t_s=np.array(t_s),
        eta_probes=np.array(eta_p, dtype=np.float32),
        probes_x=np.array(PROBES_X_M),
        floodplain_wet_fraction=np.array(fp_wet, dtype=np.float32),
        volume_m3=np.array(volume),
        dt=dt,
        **{f"frame_{int(k):04d}s": v for k, v in frames.items()},
    )
    print(f"wall {time.perf_counter() - t0:.0f}s -> {out / 'results.npz'}", flush=True)


def compare(out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = dict(np.load(out / "results.npz"))
    ref = np.genfromtxt(TELEMAC_REF / "free_surface_centreline.csv", delimiter=",", names=True)
    t_ref = ref["time_s"]
    rows = []
    fig, axes = plt.subplots(4, 2, figsize=(11, 12), sharex=True)
    for k, (px, ax) in enumerate(zip(PROBES_X_M, axes.ravel())):
        tel = ref[f"fs_x{int(px)}"]
        cel = np.interp(t_ref, r["t_s"], r["eta_probes"][:, k])
        for label, sel in (("t<=2100s", t_ref <= 2100.0), ("full", np.ones_like(t_ref, bool))):
            d = cel[sel] - tel[sel]
            rows.append(
                dict(probe_x_m=px, window=label, rmse_m=float(np.sqrt(np.mean(d**2))),
                     max_abs_m=float(np.abs(d).max()), bias_m=float(d.mean()))
            )
        ax.plot(t_ref, tel, "k-", label="TELEMAC-2D")
        ax.plot(t_ref, cel, "C0--", label="Celeris")
        ax.set_title(f"Free Surface At X = {px:.0f} m")
        ax.set_ylabel("Elevation (m)")
        ax.grid(alpha=0.3)
    ax = axes.ravel()[-1]
    ax.plot(r["t_s"], r["floodplain_wet_fraction"], "C0--", label="Celeris")
    ax.plot([2110, 2400, 2700], [0.0, 0.28, 0.64], "ko", label="TELEMAC-2D (README)")
    ax.set_title("Floodplain Wet Fraction (h > 1 cm)")
    ax.set_ylabel("Fraction")
    ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    axes[0, 0].legend()
    fig.suptitle("TELEMAC Breach Example, Dyke Intact: Hydrograph Routing")
    fig.tight_layout()
    fig.savefig(HERE / "results" / "breach_probes.png", dpi=130)
    with open(HERE / "results" / "breach_metrics.csv", "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0]))
        wri.writeheader()
        wri.writerows(rows)
    wet = r["floodplain_wet_fraction"]
    first_wet = r["t_s"][np.argmax(wet > 0.0)] if (wet > 0).any() else np.nan
    print(f"floodplain first wet: Celeris {first_wet:.0f} s vs TELEMAC 2110 s")
    for tq, ref_frac in ((2400.0, 0.28), (2700.0, 0.64)):
        print(f"wet fraction at {tq:.0f} s: Celeris {np.interp(tq, r['t_s'], wet):.3f} vs TELEMAC {ref_frac}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["prep", "run", "compare"])
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--duration", type=float, default=DURATION_S)
    a = ap.parse_args()
    {"prep": lambda: prep(a.out), "run": lambda: run(a.out, a.duration), "compare": lambda: compare(a.out)}[a.stage]()
