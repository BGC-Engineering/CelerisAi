"""Malpasset dam break with Celeris (SWE). Reads <workdir>/grid.npz + bathy.xyz from prep_malpasset.py.

Usage: python run_malpasset.py [--workdir DIR] [--arch cuda|cpu] [--duration 4000] [--courant 0.2]
       [--steps N] (override number of steps, for smoke tests) [--sample-dt 1.0]
       [--seed-wetting H] (0 = plain Celeris; H > 0 seeds dry cells below a wet neighbour's surface, see README)
       [--tag NAME] (suffix for the output files)
Writes <workdir>/results<tag>.npz, gauges<tag>.csv, frame<tag>_t*.npy.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import taichi as ti
from celeris.domain import BoundaryConditions, Domain, Topodata
from celeris.runner import Evolve
from celeris.solver import Solver

WORKDIR = Path("/mnt/d/Homathko/Validation/celeris/malpasset")
MANNING_N = 1.0 / 30.0          # TELEMAC Strickler 30 -> n = 1/30 (FrictionCalc: isManning=1 uses friction as n)
FRAME_TIMES = (100.0, 500.0, 1000.0, 2000.0, 4000.0)
BLOWUP_ETA = 1e4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=Path, default=WORKDIR)
    ap.add_argument("--arch", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--duration", type=float, default=4000.0)
    ap.add_argument("--courant", type=float, default=0.2)
    ap.add_argument("--steps", type=int, default=0, help="override step count (smoke test)")
    ap.add_argument("--sample-dt", type=float, default=1.0, help="readback interval (s)")
    ap.add_argument("--seed-wetting", type=float, default=0.0, help="seed depth (m) for the wetting kernel, 0 = off")
    ap.add_argument("--tag", default="")
    ap.add_argument("--wetdry", default="legacy", choices=["legacy", "conserving"], help="Solver wet/dry scheme")
    args = ap.parse_args()
    wd = args.workdir

    g = dict(np.load(wd / "grid.npz"))
    bed, eta0, dx = g["bed"].astype(np.float64), g["eta0"].astype(np.float64), float(g["dx"])
    nx, ny = bed.shape
    gi, gj = g["gauge_i"], g["gauge_j"]
    hmax0 = float((eta0 - bed).max())

    ti.init(arch=ti.cuda if args.arch == "cuda" else ti.cpu, default_fp=ti.f32)
    topo = Topodata(filename="bathy.xyz", path=str(wd), datatype="xyz")
    bc = BoundaryConditions(celeris=False, North=0, East=0, South=0, West=0, BoundaryWidth=20)  # all solid walls
    dom = Domain(topodata=topo, x1=0.0, x2=nx * dx, y1=0.0, y2=ny * dx, Nx=nx, Ny=ny,
                 isManning=1, friction=MANNING_N, Courant=args.courant, base_depth=hmax0)
    solver = Solver(domain=dom, boundary_conditions=bc, model="SWE", useBreakingModel=True,
                    infiltrationRate=0.0, show_window=False, wetdry_scheme=args.wetdry)
    evolve = Evolve(solver=solver, maxsteps=1)
    evolve.Evolve_0()
    dt = float(solver.dt)
    bed_solver = solver.Bottom.to_numpy()[2].astype(np.float64)
    assert np.allclose(bed_solver, bed, atol=1e-3), "solver bed != grid bed (xyz round trip)"
    bed = bed_solver                       # f32 round trip of the xyz; use it so eta - bed is exactly 0 on dry cells
    eta0 = np.maximum(eta0, bed)
    eta_ref = np.maximum(bed, float(g["sea_level"]))   # rest state: dry land + sea at datum

    # initial free surface: reservoir at 100 m, everything else dry (eta = bed); zero momentum
    s0 = np.zeros((nx, ny, 4), dtype=np.float32)
    s0[:, :, 0] = eta0
    for f in (solver.State, solver.stateUVstar, solver.NewState, solver.current_stateUVstar):
        f.from_numpy(s0)

    # Celeris keeps dry cells at eta = bed and uses a one-sided surface gradient next to them, so a wet cell
    # whose surface is above a dry neighbour's bed does not flood it unless momentum already points that way
    # (the front stalls on flat floodplains). This kernel moves a thin seed layer from the wettest neighbour
    # onto such a dry cell (volume-conserving); the next step then sees a wet-wet face and a proper gradient.
    seed_h = float(args.seed_wetting)
    delta = float(solver.delta)

    @ti.kernel
    def seed_wetting():
        for i, j in solver.State:
            if i > 1 and j > 1 and i < nx - 2 and j < ny - 2:
                B = solver.Bottom[2, i, j]
                if solver.State[i, j][0] - B <= delta:
                    best = -1.0e9
                    bi, bj = -1, -1
                    for di, dj in ti.static(((1, 0), (-1, 0), (0, 1), (0, -1))):
                        e = solver.State[i + di, j + dj][0]
                        if e - solver.Bottom[2, i + di, j + dj] > 4.0 * seed_h and e > B + 2.0 * seed_h and e > best:
                            best, bi, bj = e, i + di, j + dj
                    if bi >= 0:
                        solver.State[i, j][0] = B + seed_h
                        solver.stateUVstar[i, j][0] = B + seed_h
                        solver.State[bi, bj][0] -= seed_h
                        solver.stateUVstar[bi, bj][0] -= seed_h

    nsteps = args.steps or int(np.ceil(args.duration / dt))
    every = max(1, int(round(args.sample_dt / dt)))
    print(f"grid {nx}x{ny} dx={dx} m | dt={dt:.4f} s | steps={nsteps} | base_depth={hmax0:.2f} m | "
          f"delta={delta:.4f} m | Manning n={MANNING_N:.4f} | seed_wetting={seed_h} m | readback every {every} steps", flush=True)

    # volume = sum(eta - eta_ref): water on land + sea-surface anomaly; conserved by an exact scheme (walls all round)
    ts, gauge_eta, volume, hmax = [0.0], [eta0[gi, gj]], [float((eta0 - eta_ref).sum() * dx * dx)], np.zeros_like(bed)
    frames_left = sorted(FRAME_TIMES)
    t0 = time.time()
    for n in range(1, nsteps + 1):
        evolve.Evolve_Steps(n - 1)
        if seed_h > 0:
            seed_wetting()
        if n % every and n != nsteps:
            continue
        t = n * dt
        eta = solver.State.to_numpy()[:, :, 0].astype(np.float64)
        if not np.isfinite(eta).all() or np.abs(eta).max() > BLOWUP_ETA:
            print(f"BLOW-UP at t={t:.1f} s step {n}: finite={np.isfinite(eta).all()} max|eta|={np.nanmax(np.abs(eta)):.3g}")
            sys.exit(1)
        depth = eta - bed
        np.maximum(hmax, depth, out=hmax)
        ts.append(t); gauge_eta.append(eta[gi, gj])
        volume.append(float((eta - eta_ref).sum() * dx * dx))
        while frames_left and t >= frames_left[0] - 0.5 * every * dt:
            np.save(wd / f"frame{args.tag}_t{int(frames_left.pop(0)):04d}.npy", eta.astype(np.float32))
        if n % (every * 200) == 0 or n == nsteps:
            el = time.time() - t0
            print(f"t={t:7.1f} s  wall={el:6.1f} s  speedup={t / el:5.1f}x  vol={volume[-1] / 1e6:.3f} Mm3 "
                  f"max_depth_now={depth.max():.2f} m", flush=True)
    wall = time.time() - t0
    hmax_solver = solver.Auxiliary.to_numpy()[:, :, 0]
    np.savez_compressed(wd / f"results{args.tag}.npz", t=np.array(ts), gauge_eta=np.array(gauge_eta), volume=np.array(volume),
                        hmax=hmax.astype(np.float32), hmax_solver=hmax_solver, bed=bed.astype(np.float32),
                        dt=dt, dx=dx, courant=args.courant, manning_n=MANNING_N, delta=float(solver.delta),
                        wall_s=wall, nsteps=nsteps, seed_wetting=seed_h, eta_ref=eta_ref.astype(np.float32), gauge_names=g["gauge_names"], gauge_i=gi, gauge_j=gj)
    with open(wd / f"gauges{args.tag}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t_s", *g["gauge_names"]])
        for t, row in zip(ts, gauge_eta):
            w.writerow([f"{t:.2f}", *[f"{v:.3f}" for v in row]])
    print(f"done: {nsteps} steps, simulated {ts[-1]:.1f} s in {wall:.1f} s wall (realtime factor {ts[-1] / wall:.1f}x); "
          f"volume t0 {volume[0] / 1e6:.3f} -> end {volume[-1] / 1e6:.3f} Mm3", flush=True)


if __name__ == "__main__":
    main()
