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
| grid | 2.5 m, 2004 x 204 cells (TELEMAC domain plus a 2-cell wall pad so the ghost rim holds no physical cell), bed and initial depth/momentum linearly interpolated on the TELEMAC triangulation; outside the mesh hull = 20 m wall |
| datum | 4.9 m a.s.l., just below the lowest floodplain cell (see finding 1) |
| inflow | `HydrographSource` on the wet cells of the strip 10 <= x < 30 m, Q(t) from `t2d_breach.liq` |
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
| run | | blows up at 2628 s | stable |

With the conserving scheme the two models agree to within 5 to 9 cm RMSE at
every probe and the channel discharge Q(x) matches within 3 m3/s. Celeris
floods the plain earlier and wider than TELEMAC once the dyke is overtopped;
the sill rule (a wet surface above a dry cell's bed drives a dam-break flux
capped at the wet depth) is more permissive than TELEMAC's finite-element
wetting. Not tuned.

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
