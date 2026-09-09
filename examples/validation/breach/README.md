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
50 m3/s from `ini_breach.slf`. TELEMAC overtops the dyke from t = 2110 s and has
64 % of the floodplain wet at 2700 s.

## Celeris setup (`breach_validation.py`)

| item | choice |
|---|---|
| grid | 2.5 m, 2000 x 200 cells, bed and initial depth/momentum linearly interpolated on the TELEMAC triangulation; outside the mesh hull = 20 m wall |
| datum | 4.9 m a.s.l., just below the lowest floodplain cell (see finding 1) |
| inflow | `HydrographSource` on the wet cells of the strip 10 <= x < 30 m, Q(t) from `t2d_breach.liq` |
| outlet | free surface from the `.liq` imposed on 5 columns inside the ghost rim (post-step kernel), momentum kept |
| friction | Manning n = 1/15 (`isManning=1`) |
| model | SWE, Courant 0.2, `infiltrationRate=0`, no breaking model |
| run | 2700 s = 37,400 steps, 45 s wall on an RTX A5000 |

```bash
uv run python examples/validation/breach/breach_validation.py prep
uv run python examples/validation/breach/breach_validation.py run
uv run python examples/validation/breach/breach_validation.py compare
```

Reference: `/mnt/d/Homathko/Validation/telemac/breach_nobreach_restart/` (see the
README there). Large Celeris products go to `/mnt/d/Homathko/Validation/celeris/breach/`.

## Results (`results/breach_metrics.csv`, `results/breach_probes.png`)

Free surface at the channel centreline, Celeris minus TELEMAC:

| x (m) | bias t <= 2100 s (m) | RMSE t <= 2100 s (m) | bias full (m) |
|---|---|---|---|
| 500 | +0.15 | 0.17 | +0.20 |
| 1000 | +0.06 | 0.11 | +0.14 |
| 1500 | +0.11 | 0.18 | +0.25 |
| 1900 | +0.20 | 0.31 | +0.42 |
| 3100 | +0.02 | 0.04 | +0.05 |
| 4000 | +0.05 | 0.05 | +0.06 |
| 4500 | +0.08 | 0.09 | +0.08 |

Downstream of the dyke the two models agree to a few centimetres for the whole
run. Discharge profiles Q(x) agree within 3 m3/s from x = 1000 m down (t = 1000 s:
132 vs 128 at x = 1000, 62 vs 63 at x = 2500, 42 vs 43 at x = 4900).

Two things do not agree, and neither is the hydrograph source:

**Inlet hump.** The injected water has no momentum and sits against the x = 0
wall, so the level in the first ~500 m rises to drive the flow: +0.3 m at
x = 100 m at t = 300 s, +0.6 m at t = 1000 s, decaying to +0.05 m by x = 1000 m.
Lengthening the inlet strip to 200 m changes nothing. Keep probes and results
at least ~1 km from a channel inlet, or inject into a lake, where a hump does
not form. TELEMAC imposes Q with a velocity profile and has no hump.

**No overtopping.** Celeris never wets the dyke or the floodplain even though
its own level at x = 1900 m exceeds the 8.0 m crest after ~2100 s; the excess
water instead backs up upstream (the bias growth after 2100 s). Cause: `Pass2`
zeroes the mass flux across a wet/dry interface (`if minH <= delta:
mass_diff = 0`), so water only enters a dry cell carried by momentum already
pointing at it. Cross-channel momentum is ~0 here, so a slowly rising level
cannot spill over a dry crest. The Malpasset case (`../malpasset`) hits the same
rule: a flat reservoir against a dry bed is frozen until a film is seeded, and a
flood front stalls on flat ground with 3.4 m of head 30 m away.

## Findings that apply to any Celeris run

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
3. **Wetting needs momentum** (above). A rising lake will not wet its shore or
   spill over a crest unless a wave carries water there. This matters for any
   level-rise problem and is a solver fix, not a case-setup fix.
4. `infiltrationRate` defaults to 0.001 m/s on every cell with bed above the
   datum. Set it to 0 for anything longer than a few minutes.

Interpolate nodal depth and momentum onto the grid, never the free surface:
dry nodes carry eta = bed and interpolating eta across a wet/dry bank perches
spurious water on the slope.
