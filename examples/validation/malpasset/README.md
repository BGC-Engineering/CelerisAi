# Malpasset Dam-Break Validation (CelerisAi, SWE)

Real dam break (Fréjus, December 1959) run as in the TELEMAC-2D example
`telemac-mascaret/examples/telemac2d/malpasset`: reservoir at 100 m a.s.l. behind the dam line
(4701.183, 4143.407)-(4655.553, 4392.104), dry valley downstream, Mediterranean at 0 m, 4000 s.
Reference data (from `doc/malpasset.tex`): maximum water depth at the 1/400 physical-model gauges P6-P14
(LNHE, 1964) and the wave travel times between the electric transformers A (5550, 4400), B (11900, 3250)
and C (13000, 2700): A->B 1140 s, A->C 1320 s. TELEMAC-2D results (HLLC, small mesh, v8p5) are compared as well.

## Files

| file | role |
|---|---|
| `read_selafin.py` | minimal Selafin reader. `read_selafin(path) -> dict(title, varnames, nelem, npoin, ndp, ikle [0-based, nelem x 3], ipobo, x, y, times, values [nt, nvar, npoin], variables {name: (nt, npoin)})`. Checks the record markers and that the node count matches `ikle`. |
| `prep_malpasset.py` | fine mesh `geo_malpasset-large.slf` (53081 nodes) -> regular grid: bed, initial free surface, masks, gauge cells. Writes `grid.npz` and the Celeris `bathy.xyz`. Config constants at the top. |
| `run_malpasset.py` | GPU run (taichi/CUDA): gauge time series, max-depth envelope, frames at 100/500/1000/2000/4000 s, volume series, blow-up guard. |
| `compare_malpasset.py` | metrics vs lab table and TELEMAC npz; writes `results/*.csv` and `results/malpasset_comparison*.png`. |
| `results/` | small CSVs and figures only (two variants, see below). |

Large outputs go to `/mnt/d/Homathko/Validation/celeris/malpasset/` (`--workdir`, default set at the top of the
scripts): `grid.npz`, `bathy.xyz` (~20 MB), `results*.npz`, `gauges*.csv`, `frame*_t*.npy`, run logs.
TELEMAC reference npz: `/mnt/d/Homathko/Validation/telemac/malpasset_hllc/malpasset_hllc.npz` (`--telemac`).

## How to run

```bash
cd /home/cstringari/Celeris-Wave-Model/CelerisAi
uv run --no-sync python examples/validation/malpasset/prep_malpasset.py --dx 15
# plain Celeris
LD_LIBRARY_PATH=/usr/lib/wsl/lib uv run --no-sync python examples/validation/malpasset/run_malpasset.py --arch cuda
uv run --no-sync python examples/validation/malpasset/compare_malpasset.py
# with the seed-wetting kernel (see "Choices")
LD_LIBRARY_PATH=/usr/lib/wsl/lib uv run --no-sync python examples/validation/malpasset/run_malpasset.py --arch cuda --seed-wetting 0.02 --tag _seed
uv run --no-sync python examples/validation/malpasset/compare_malpasset.py --tag _seed
# CPU smoke test: prep with --dx 60 --workdir <dir>/dx60, then run with --arch cpu --steps 50 --workdir <dir>/dx60
```

## Choices

* **Grid**: dx = 15 m, 1154 x 618 cells, origin (506.5, -2373.5) = mesh extent plus a 2-cell margin.
  Bed = linear interpolation of `BOTTOM` on the TELEMAC triangulation (`matplotlib.tri.LinearTriInterpolator`
  with the `ikle` connectivity, so the non-convex hull is respected); cells outside the mesh get
  mesh max + 50 m = 150 m (dry wall). Bed range in the mesh: -20 .. 100 m.
* **Datum**: Celeris bed = TELEMAC elevation (xyz z = -bed), so eta is in m a.s.l.; Celeris' "still water level"
  is the sea at 0 m. `base_depth` is passed explicitly as the initial max depth (55.0 m), which sets
  dt = Courant dx / sqrt(g 55) and the wet/dry threshold delta = min(0.005, base_depth/5000) = 5 mm.
  Do not move the datum up (e.g. to the reservoir level): `BoundaryPass` treats cells with bed <= 0 as sea floor
  and refills them to the datum / keeps thin films there, so terrain that must stay dry needs bed > 0. Here the
  only below-datum cells are the sea basin (bed -20..0 m, one connected component, 19179 cells), which is
  supposed to be full; the volume series never grows and no water appeared away from the front.
* **Initial condition** (as in `user_condin_h.f`): free surface 100 m on the reservoir side of the dam line
  (`DISTAN > 0.001`, bed < 100 m), the 200 m disc around (4500, 5350) forced dry, sea (bed < 0) at 0 m
  (the Fortran sets H = -ZF before the reservoir loop, so TELEMAC also starts with the sea full; its initial
  volume 96.75 Mm3 = 48.5 reservoir + 47.9 sea in the small mesh). Only the main connected reservoir body is
  kept (isolated below-100 m pockets on the reservoir side of the infinite dam line are dropped; none on this
  grid). Reservoir volume on the grid 48.94 Mm3 vs 48.96 Mm3 on the fine mesh (P1 lumped).
* **Dam-face kick** (Celeris deviation): a wet cell at rest next to a dry cell uses a one-sided (wet-side)
  surface gradient in `Pass3`, so a flat reservoir against a dry bed never starts to move (verified: 4000 s
  with zero change). The 17 dry cells touching the reservoir on the downstream side of the dam line start
  with 0.1 m of water (383 m3, 8e-6 of the reservoir).
* **Boundaries**: `geo_malpasset-large.cli` has LIHBOR = 2 on all 2160 boundary nodes, i.e. solid walls
  everywhere (also stated in the doc). All four Celeris edges are walls (type 0); the sea area acts as the
  receiving basin.
* **Friction**: Strickler 30 -> Manning n = 1/30 = 0.0333, `Domain(isManning=1, friction=n)`.
  `FrictionCalc` in `celeris/utils.py` uses `g n^2 h^(-1/3)` when `isManning == 1`, so `friction` is n.
  `Solver(infiltrationRate=0.0)`: the Celeris default 0.001 m/s would remove 4 m of water per cell over 4000 s
  on every cell above the datum.
* **Model**: `model='SWE'`, `useBreakingModel=True`, Courant 0.2 (dt = 0.1292 s, 30966 steps), predictor-corrector
  (Celeris default `timeScheme=2`). Readback every 8 steps (about 1 s) for gauges, max-depth envelope and volume.
* **Thresholds**: arrival = first sample with depth > 0.05 m at the nearest cell (TELEMAC's `user_utimp` uses
  H > 1e-4 m at the nearest node). Max depth = envelope of cell depth eta - bed over the 1 s samples
  (`celeris_m`); the solver's own per-step envelope `Auxiliary[...,0]` is also written (`celeris_solver_envelope_m`).
* **TELEMAC values**: max depth per gauge = max over the 201 frames (20 s) of `free_surface - bottom` linearly
  interpolated on the small-mesh triangulation (nearest-node value also in the CSV). Travel times from the
  rfo file (`telemac_hllc_rfo`) and from the 20 s frames with the 0.05 m threshold (`telemac_hllc_frames20s`).
  The rfo "MAXIMUM WATER DEPTHS" (P6 = 82.07 m ...) are `HAUT + FON` = maximum free-surface elevation, not depth:
  82.07 - 44.0 (bed at P6) = 38.1 m, which is what the npz gives.
* **Seed-wetting kernel** (`--seed-wetting 0.02`, optional, off by default): the same one-sided-gradient rule
  stalls the lateral advance of the flood on flat ground; in the plain run the flood edge stopped 30 m south
  of transformer B with 3.4 m of water at eta 15.5 m next to dry cells with bed 12.3 m, so B was never
  reached (TELEMAC: 2.6 m at B). The kernel in `run_malpasset.py` runs after every step and, for each dry cell
  whose wettest 4-neighbour has its surface more than 2 seed depths above the dry bed (and at least 4 seed
  depths of its own), moves a 2 cm seed layer from that neighbour onto the dry cell (volume-conserving).
  The next step then sees a wet-wet face and a central gradient. This is a case-side workaround, not a
  solver change.

## First results (dx = 15 m, Courant 0.2, n = 0.0333, RTX A5000)

Maximum water depth (m). `Celeris` = plain run, `Celeris+seed` = with the seed-wetting kernel.

| gauge | bed (Celeris cell) | measured | TELEMAC HLLC | Celeris | Celeris+seed |
|---|---|---|---|---|---|
| P6  | 44.06 | 40.3 | 38.11 | 41.15 | 40.87 |
| P7  | 34.21 | 14.6 | 21.55 | 19.60 | 19.52 |
| P8  | 29.99 | 24.0 | 23.33 | 24.52 | 24.30 |
| P9  | 29.17 | 12.8 | 19.27 | 18.13 | 17.96 |
| P10 | 20.75 | 11.8 | 15.53 | 16.88 | 16.31 |
| P11 | 18.34 |  8.3 |  7.29 |  7.22 |  6.98 |
| P12 | 11.30 | 10.1 |  7.68 |  8.02 |  7.50 |
| P13 |  3.10 |  6.8 | 14.63 | 14.22 | 14.39 |
| P14 |  8.12 |  5.4 |  4.76 |  4.80 |  4.52 |

Travel times (s), depth > 0.05 m:

| quantity | observed | TELEMAC HLLC (rfo, 1e-4 m) | TELEMAC HLLC (20 s frames) | Celeris | Celeris+seed |
|---|---|---|---|---|---|
| arrival at A | - | 97.2 | 120 | 111.6 | 115.7 |
| A -> B | 1140 | 1127.8 | 1160 | not reached | 1216.3 |
| A -> C | 1320 | 1369.3 | 1360 | 1296.9 | 1310.4 |

Celeris follows TELEMAC closely at every gauge (within 0.1-2.3 m) and shares its known discrepancies with the
physical model at P7, P9, P10 and P13 (discussed in the TELEMAC doc / Hervouet 2007). Travel time A->C is within
2 % of the observation in both runs; A->B needs the seed-wetting kernel and is then 7 % slow.

Run cost: 30966 steps in 90 s wall (plain, 44x realtime) and 76 s (seed, 52x realtime); the difference is
disk I/O on `/mnt/d`, not the kernel. Smoke test: dx = 60 m, CPU, 50 steps in 2.5 s.

Files: `results/malpasset_max_depth[_seed].csv`, `results/malpasset_arrival[_seed].csv`,
`results/malpasset_comparison[_seed].png`.

## Mass conservation and known deviations

* **Volume leak at wet/dry faces**: the conserved quantity sum(eta - max(bed, 0)) dx^2 (water on land plus the
  sea-surface anomaly, walls all round) drops from 48.94 Mm3 to 41.00 Mm3 at 4000 s in the plain run (-16 %;
  -5.6 % at 1000 s, -10.7 % at 2000 s) and to 39.62 Mm3 with seeding (-19 %). TELEMAC conserves volume to 1e-12.
  Diagnosed mechanism (checked step by step with `solver.dU_by_dt`): a dry cell next to a wet cell receives a
  small mass flux every step (the reconstructed face state on the dry side has a phantom depth because the face
  bed is the average of the two beds), the resulting depth stays below delta, and `BoundaryPass` resets the
  cell to eta = bed, discarding that inflow. The discarded inflow summed over all cells that end a step dry
  matches the volume loss (0.179 vs 0.166 Mm3 after 800 steps). It scales with the wetted perimeter, not with
  delta (delta = 1 mm gives the same loss). Fixing it needs a solver change (zero the mass flux into a dry cell
  whose bed is above the wet neighbour's free surface, or keep the sub-threshold water), so the first results
  above are with the leak; downstream depths are therefore somewhat low and arrivals somewhat late.
* **Stalled fronts / dam start**: see the dam-face kick and the seed-wetting kernel above. Same root cause
  (one-sided gradient plus eta := bed on dry cells). Flooded land area (hmax > 0.05 m, bed >= 0): 15.6 km2
  plain, 25.6 km2 with seeding, TELEMAC HLLC 23.8-30.7 km2 (elements with all / any node flooded).
* **dx = 60 m blows up** (NaN at 27 s, GPU, Courant 0.2, with or without the breaking model): thin layers on
  the 30 m cell-to-cell bed steps next to the dam reach > 60 m/s. The `FrictionCalc` regularisation switches
  friction off below about 0.0032 base_depth = 0.17 m, so thin fast sheets are unresisted. The 15 m grid is stable.
* Gauge/bed values come from the fine mesh rasterised at 15 m; the TELEMAC comparison run uses the small
  (13541-node) mesh. Bed at the gauge cells differs from the fine-mesh interpolation by <= 0.9 m (P10).
