"""Malpasset: TELEMAC fine mesh (geo_malpasset-large.slf) -> regular grid.npz for Celeris.

Usage: python prep_malpasset.py [--dx 15] [--workdir /mnt/d/Homathko/Validation/celeris/malpasset]
Writes <workdir>/grid.npz (bed, eta0, masks, gauge ij) and <workdir>/bathy.xyz (Celeris input).
"""
import argparse
from pathlib import Path

import matplotlib.tri as mtri
import numpy as np
from scipy import ndimage

from read_selafin import read_selafin

# ---------------- config (TELEMAC example values) ----------------
TELEMAC_DIR = Path("/home/cstringari/telemac-mascaret/examples/telemac2d/malpasset")
MESH = TELEMAC_DIR / "geo_malpasset-large.slf"
WORKDIR = Path("/mnt/d/Homathko/Validation/celeris/malpasset")
DX = 15.0
RESERVOIR_LEVEL = 100.0                                   # m a.s.l. (user_condin_h.f)
SEA_LEVEL = 0.0                                           # user_condin_h.f sets H = -ZF before the reservoir loop,
                                                          # i.e. free surface 0 m where the bed is below 0 (the sea)
DAM_LINE = (4701.183, 4143.407, 4655.553, 4392.104)       # x1,y1,x2,y2 (user_condin_h.f)
DRY_CIRCLE = (4500.0, 5350.0, 200.0)                      # xc,yc,r: forced dry (user_condin_h.f)
LAND_ABOVE_MESH_MAX = 50.0                                # elevation added outside the mesh hull
# Celeris deviation: a wet cell at rest next to a dry cell uses a one-sided (wet-side) surface gradient,
# so a flat reservoir against a dry bed never starts moving. A thin film on the dry cells touching the
# dam face (downstream side of the dam line) starts the flow; added volume is ~1e-5 of the reservoir.
KICK_DEPTH = 0.1                                          # m
# 14 points of user_utimp_telemac2d.f: 1=A 2=B 3=C 4,5 duplicates of B,C, 6..14 = P6..P14
GAUGE_NAMES = ["A", "B", "C", "B_dup", "C_dup", "P6", "P7", "P8", "P9", "P10", "P11", "P12", "P13", "P14"]
GAUGE_X = [5550, 11900, 13000, 11900, 13000, 4947, 5717, 6775, 7128, 8585, 9674, 10939, 11724, 12723]
GAUGE_Y = [4400, 3250, 2700, 3250, 2700, 4289, 4407, 3869, 3162, 3443, 3085, 3044, 2810, 2485]


def distan(x1, y1, x2, y2, x, y):
    """Signed distance to the dam line, same sign convention as distan.f (>0 = reservoir side)."""
    a1, b1, c1 = y1 - y2, -x1 + x2, x1 * y2 - x2 * y1
    return (a1 * x + b1 * y + c1) / np.hypot(a1, b1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=DX)
    ap.add_argument("--workdir", type=Path, default=WORKDIR)
    args = ap.parse_args()
    dx = args.dx
    args.workdir.mkdir(parents=True, exist_ok=True)

    m = read_selafin(MESH)
    assert m["npoin"] == 53081 and m["varnames"][0] == "BOTTOM", (m["npoin"], m["varnames"])
    zb = m["values"][0, 0].astype(np.float64)
    x, y = m["x"], m["y"]

    # grid with a 2-cell land margin around the mesh extent; bed[i, j] at (x0 + i dx, y0 + j dx)
    x0, y0 = x.min() - 2 * dx, y.min() - 2 * dx
    nx = int(np.ceil((x.max() - x0) / dx)) + 3
    ny = int(np.ceil((y.max() - y0) / dx)) + 3
    xg, yg = np.meshgrid(x0 + dx * np.arange(nx), y0 + dx * np.arange(ny), indexing="ij")

    # linear interpolation on the TELEMAC triangulation (respects the non-convex hull)
    tri = mtri.Triangulation(x, y, m["ikle"])
    bed_m = mtri.LinearTriInterpolator(tri, zb)(xg, yg)
    inmesh = ~np.ma.getmaskarray(bed_m)
    land = zb.max() + LAND_ABOVE_MESH_MAX
    bed = np.where(inmesh, bed_m.filled(land), land)

    # initial condition as in user_condin_h.f
    hd = distan(*DAM_LINE, xg, yg)
    xc, yc, r = DRY_CIRCLE
    reservoir = inmesh & (hd > 0.001) & (bed < RESERVOIR_LEVEL) & ((xg - xc) ** 2 + (yg - yc) ** 2 >= r**2)
    eta0 = np.where(reservoir, RESERVOIR_LEVEL, bed)

    # gauges -> nearest cell
    gi = np.rint((np.array(GAUGE_X, float) - x0) / dx).astype(int)
    gj = np.rint((np.array(GAUGE_Y, float) - y0) / dx).astype(int)

    # keep only the main reservoir body (drops isolated below-100 m pockets on the reservoir side of the line)
    lab, ncomp = ndimage.label(reservoir)
    sizes = np.bincount(lab.ravel())[1:]
    reservoir = lab == (np.argmax(sizes) + 1)
    eta0 = np.where(reservoir, RESERVOIR_LEVEL, bed)
    kick = ndimage.binary_dilation(reservoir) & ~reservoir & inmesh & (hd <= 0.001)
    eta0[kick] = bed[kick] + KICK_DEPTH
    sea = inmesh & (bed < SEA_LEVEL) & ~reservoir
    eta0[sea] = SEA_LEVEL
    vol_grid = float(((eta0 - bed) * reservoir).sum() * dx * dx)
    h_node = np.where((distan(*DAM_LINE, x, y) > 0.001) & (zb < RESERVOIR_LEVEL)
                      & ((x - xc) ** 2 + (y - yc) ** 2 >= r**2), RESERVOIR_LEVEL - zb, 0.0)
    ik = m["ikle"]
    area = 0.5 * np.abs((x[ik[:, 1]] - x[ik[:, 0]]) * (y[ik[:, 2]] - y[ik[:, 0]])
                        - (x[ik[:, 2]] - x[ik[:, 0]]) * (y[ik[:, 1]] - y[ik[:, 0]]))
    vol_mesh = float((area * h_node[ik].mean(axis=1)).sum())
    print(f"grid {nx} x {ny} @ {dx} m, origin ({x0:.1f}, {y0:.1f}); in-mesh cells {inmesh.sum()}")
    print(f"bed in mesh: min {bed[inmesh].min():.2f} max {bed[inmesh].max():.2f} m; land fill {land:.1f} m")
    print(f"reservoir cells {reservoir.sum()}, components {ncomp} (sizes {sorted(sizes, reverse=True)[:5]})")
    print(f"dam-face kick cells {kick.sum()} (bed {bed[kick].min():.1f}..{bed[kick].max():.1f} m, "
          f"added volume {KICK_DEPTH * kick.sum() * dx * dx:.0f} m3)")
    print(f"max initial depth {float((eta0 - bed).max()):.2f} m; sea cells (bed<{SEA_LEVEL} m, set to eta={SEA_LEVEL}) {sea.sum()}, "
          f"sea volume {float((SEA_LEVEL - bed[sea]).sum() * dx * dx) / 1e6:.2f} Mm3, components {ndimage.label(sea)[1]}")
    print(f"reservoir volume: grid {vol_grid / 1e6:.3f} Mm3 vs mesh (P1 lumped) {vol_mesh / 1e6:.3f} Mm3")
    for n, i, j in zip(GAUGE_NAMES, gi, gj):
        print(f"  {n:6s} ij=({i},{j}) bed {bed[i, j]:.2f} m  mesh-interp "
              f"{float(mtri.LinearTriInterpolator(tri, zb)(GAUGE_X[GAUGE_NAMES.index(n)], GAUGE_Y[GAUGE_NAMES.index(n)])):.2f}")

    np.savez_compressed(args.workdir / "grid.npz", bed=bed.astype(np.float32), eta0=eta0.astype(np.float32),
                        reservoir=reservoir, kick=kick, sea=sea, inmesh=inmesh, dx=dx, x0=x0, y0=y0,
                        gauge_names=np.array(GAUGE_NAMES), gauge_x=GAUGE_X, gauge_y=GAUGE_Y, gauge_i=gi, gauge_j=gj,
                        reservoir_level=RESERVOIR_LEVEL, sea_level=SEA_LEVEL, land=land)
    # Celeris xyz: local coordinates, z = depth-positive (= -bed elevation)
    np.savetxt(args.workdir / "bathy.xyz",
               np.column_stack([(dx * np.arange(nx))[:, None].repeat(ny, 1).ravel(),
                                (dx * np.arange(ny))[None, :].repeat(nx, 0).ravel(), -bed.ravel()]), fmt="%.3f")
    print("wrote", args.workdir / "grid.npz", "and bathy.xyz")


if __name__ == "__main__":
    main()
