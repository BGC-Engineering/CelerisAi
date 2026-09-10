# TELEMAC-2D `breach` example, dyke intact: hydrograph routing

Model-to-model validation of `celeris.hydrograph.HydrographSource` against
TELEMAC-2D v8p5 (`examples/telemac2d/breach`, case `t2d_breach.cas` with
`BREACH = NO`, restart from `ini_breach.slf`). No measurements exist for this
case; TELEMAC is the reference.

## Case

5 km trapezoidal channel (bed 20 m wide, 1:1 banks, 26 m at bank level), slope
1e-3, Strickler K = 15. For 2000 < x < 3000 m a dyke (crest 2 m above the bank,
8.0 m at x = 2000) separates the channel from a 500 m wide dry floodplain.
Inflow rises linearly from 50 m3/s at t = 0 to 406 m3/s at t = 2700 s; the outlet
free surface is prescribed (0.87 to 2.49 m). Initial state: developed flow at
50 m3/s from `ini_breach.slf`. TELEMAC overtops the dyke from t = 2090 s and has
26 % of the floodplain area wet at 2700 s (8 % at 2400 s).

## Celeris setup (`breach_validation.py`)

| item | choice |
|---|---|
| grid | 2.5 m, 2004 x 204 cells: TELEMAC domain plus a 2-cell ghost rim that mirrors the adjacent interior cells (bed, depth, momentum), so the rim holds no physical cell and no artificial step; bed and initial depth/momentum linearly interpolated on the TELEMAC triangulation; outside the mesh hull = 20 m wall |
| datum | 4.9 m a.s.l., just below the lowest floodplain cell (see finding 1) |
| inflow | `--inflow source` (default): `HydrographSource` on the wet cells of the strip 10 <= x < 30 m; `--inflow boundary`: `DischargeBoundary` on the west edge (boundary type 5), the like-for-like twin of TELEMAC's prescribed flowrate. Q(t) from `t2d_breach.liq` in both |
| outlet | free surface from the `.liq` imposed on 5 columns inside the ghost rim (post-step kernel), momentum kept |
| friction | Manning n = 1/15 (`isManning=1`) |
| model | SWE, Courant 0.2, `infiltrationRate=0`, no breaking model |
| run | 2700 s = 37,400 steps, 45 s wall on an RTX A5000; `--wetdry legacy` (default) or `conserving` |

```bash
uv run python examples/validation/breach/breach_validation.py prep
uv run python examples/validation/breach/breach_validation.py run
uv run python examples/validation/breach/breach_validation.py compare
```

Reference: `/mnt/d/Homathko/Validation/telemac/breach_nobreach_restart/` (see the
README there). Large Celeris products go to `/mnt/d/Homathko/Validation/celeris/breach/`.

## Results (`results/breach_metrics.csv`, `results/breach_probes.png`)

Free surface at the channel centreline over the full 2700 s, Celeris minus
TELEMAC, for both wet/dry schemes of the fork (`Solver(wetdry_scheme=...)`):

| x (m) | legacy RMSE (m) | legacy bias (m) | conserving RMSE (m) | conserving bias (m) |
|---|---|---|---|---|
| 500 | 1.02 | +0.85 | 0.06 | +0.03 |
| 1000 | 0.90 | +0.69 | 0.05 | +0.02 |
| 1500 | 0.90 | +0.61 | 0.05 | +0.01 |
| 1900 | 1.06 | +0.67 | 0.06 | +0.00 |
| 3100 | 0.30 | -0.16 | 0.08 | -0.00 |
| 4000 | 0.23 | -0.03 | 0.09 | -0.02 |
| 4500 | 0.20 | +0.02 | 0.09 | -0.04 |

| | TELEMAC | legacy | conserving |
|---|---|---|---|
| dyke first overtopped | 2090 s | never | 2021 s |
| floodplain wet area at 2400 s | 8 % | 0 % | 16 % |
| floodplain wet area at 2700 s | 26 % | 0 % | 42 % |
| floodplain volume at 2400 s (weir law: 20,600 m3) | 9,000 m3 | 0 | 27,900 m3 |
| run | | blows up at 2628 s | stable |

With the conserving scheme the two models agree to within 5 to 9 cm RMSE at
every probe and the channel discharge Q(x) matches within 3 m3/s.

**Overtopping volume.** Celeris floods the plain earlier and wider than TELEMAC
(16 % vs 8 % of the area at 2400 s, 27,900 vs 9,000 m3). The crest elevations
of the two grids are identical to 1 cm and the channel level beside the dyke
agrees to 1 cm (8.26 vs 8.27 m at x = 2100 m, 2400 s), so the difference is the
discharge over the crest at the same head. Integrating the broad-crested weir
law q = (2/3)^1.5 sqrt(g) H^1.5 along the dyke with TELEMAC's own channel
levels gives 20,600 m3 by 2400 s and 82,000 m3 by 2700 s: Celeris is 35 % above
the weir value, TELEMAC 55 % below it (41,000 m3 at 2700 s). Neither is
validated against a measurement here; the ideal weir sits between them, nearer
Celeris. Scaling the first-wetting flux by the head-based wave speed changed
the Celeris volume by 0.3 %, so the excess comes from the shallow-water flow over
the wet crest on 2.5 m cells, not from the wet/dry rule.

The legacy scheme on this padded grid is worse than on the unpadded grid of
the first version of this case (bias 0.15 to 0.4 m): the bank cells are now
interior cells next to dry ones, and legacy drains momentum across every
wet/dry face (a numerical wall drag: bank velocity 0.2 m/s next to 1.8 m/s
mid-channel), backs the water up, never overtops, and blows up late in the run.

**Inlet hump (both schemes).** The injected water has no momentum and sits
against the x = 0 wall; the level in the first ~500 m rises to drive the flow.
Keep probes at least ~1 km from a channel inlet, or inject into a lake.

## Findings that apply to any Celeris run (all fixed by `wetdry_scheme="conserving"`)

1. **Dry terrain must lie above the datum.** With the datum above the dyke
   (bed <= 0 everywhere) the solver manufactured ~10,000 m3 of water on the dyke
   within 60 s (`BoundaryPass` treats bed <= 0 as sea floor and keeps
   sub-`delta` films and islands there). With the dyke above the datum this
   vanished. Still water stays dry on ridges either side of the datum; the
   artefact needs a sloping surface or flow nearby.
2. **Mass leak at wet/dry banks.** Same channel, outlet control off, 300 s of
   inflow: the domain gains 19,215 m3 of the 20,937 m3 injected (-8 %). The flat
   closed basin in `tests/test_hydrograph.py` conserves to < 1 %, so the loss
   sits on the sloping banks: sub-`delta` inflow into a dry cell is reset to
   `eta = bed` by `BoundaryPass` each step and discarded. Malpasset loses 16 %
   over 4000 s by the same route.
3. **Wetting needs momentum** (legacy). A rising lake will not wet its shore or
   spill over a crest unless a wave carries water there; the conserving scheme
   adds gravity-driven wetting.
5. **Run-to-run reproducibility.** Upstream `BoundaryPass` and `Pass_Breaking`
   read neighbours of the field they write; two identical GPU runs differed by
   centimetres to metres at wet/dry fronts. Both now read from a snapshot, in
   both schemes, and identical runs are bitwise equal.
4. `infiltrationRate` defaults to 0.001 m/s on every cell with bed above the
   datum. Set it to 0 for anything longer than a few minutes.

Interpolate nodal depth and momentum onto the grid, never the free surface:
dry nodes carry eta = bed and interpolating eta across a wet/dry bank perches
spurious water on the slope.

## Inflow types and the ghost rim (later addition)

The case now runs with either hydrograph mechanism. Conserving scheme, full
2700 s, Celeris minus TELEMAC at the probes:

| x (m) | source, 20 m wall pads (earlier) | source, mirrored pads | discharge boundary, mirrored pads |
|---|---|---|---|
| 500 | RMSE 0.06, bias +0.03 | 0.14, +0.12 | 0.09, +0.06 |
| 1000 | 0.05, +0.02 | 0.13, +0.09 | 0.08, +0.04 |
| 1900 | 0.06, +0.00 | 0.10, +0.04 | 0.08, +0.01 |
| 3100 | 0.08, -0.00 | 0.10, +0.00 | 0.10, -0.02 |
| 4500 | 0.09, -0.04 | 0.10, -0.05 | 0.12, -0.06 |
| floodplain wet area at 2700 s (TELEMAC 26 %) | 42 % | 48 % | 45 % |

The rim was changed from a 20 m wall to a mirror of the interior after the
`bump` case showed that a tall pad next to small cells triggers the solver's
steep-slope Froude cap in the adjacent rows (see `../bump/README.md`). On this
2.5 m grid the pad step was 5 cells high, so the cap did not bite, and the
mirrored rim costs 4 to 8 cm of extra upstream level for both inflow types; the
cause was not traced. The discharge boundary removes the inlet hump the source
made (bias at x = 500 m halves) and reproduces TELEMAC's own inflow condition.
Legacy blows up within 10 s on the mirrored-pad grid (it ran to 2628 s on the
wall-pad grid), so this case is a conserving-scheme case in the regression set.
