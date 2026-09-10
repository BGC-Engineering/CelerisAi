# TELEMAC-2D `bump` example: steady flow over a bump, two inflow types

Exact-solution test of the flow over a crest (the control that sets the head
for a given discharge over a dyke or a dam crest) and, at the same time, the
like-for-like test of the two Celeris hydrograph inflows against their TELEMAC
twins:

| inflow | Celeris | TELEMAC-2D |
|---|---|---|
| river, at the domain edge | `DischargeBoundary` (boundary type 5): ghost cells carry `hu = h Q / A` | prescribed flowrate on the liquid boundary (`debimp.f`, velocity profile 1) |
| GLOF, inside the domain | `HydrographSource`: mass at ambient velocity | `SOURCES` without `VELOCITIES OF THE SOURCES` |

Frictionless 20 x 2 m channel, bed `0.25 exp(-(x-10)^2/2)`, exact solution from
TELEMAC's own `analytic_sol.py`. Transcritical: Q = 0.45 m3/s, outlet 0.35 m
(critical on the crest, hydraulic jump behind it). Subcritical: Q = 1.5 m3/s,
outlet 0.8 m. Celeris starts from a flat pool at the outlet level and runs
400 s; the outlet surface is imposed on two columns. Sources sit at x = 1 m in
both models.

```bash
uv run python examples/validation/bump/bump_validation.py --inflow boundary --wetdry conserving legacy
uv run python examples/validation/bump/bump_validation.py --inflow source --wetdry conserving
uv run python examples/validation/bump/bump_validation.py --inflow boundary --regime trans --dx 0.25
```

TELEMAC twins: `/mnt/d/Homathko/Validation/telemac/bump{trans,sub}_{bnd,src}/`
(400 s reruns, 5 s frames; `bnd` is the shipped case, `src` the walled-inlet
variant with 8 sources; both reach the same steady state, so TELEMAC's interior
source has no inlet hump).

## Results, discharge boundary (`results/bump_profiles_boundary*.png`)

| case | crest depth (m) | depth at x = 2 m | unit discharge upstream | L1 depth error |
|---|---|---|---|---|
| exact, transcritical | 0.1728 (critical) | 0.497 | 0.225 | 0 |
| TELEMAC HLLC, 0.25 m mesh | 0.180 | 0.497 | 0.225 | 0.0046 |
| Celeris, dx 0.05 m | 0.1727 | 0.499 | 0.225 | 0.0042 |
| Celeris, dx 0.25 m | 0.171 | 0.495 | 0.225 | 0.0044 |
| Celeris, dx 0.5 m (bump on 4 cells) | 0.166 | 0.485 | 0.225 | 0.014 |
| exact, subcritical | 0.471 | 0.809 | 0.75 | 0 |
| TELEMAC HLLC | 0.473 | 0.809 | 0.750 | 0.0074 |
| Celeris, dx 0.05 m | 0.471 | 0.806 | 0.750 | 0.0066 |

Legacy and conserving schemes give identical results here (no dry cell). The
crest passes the imposed discharge at the exact head down to a bump resolved by
four cells; the shock position is within one TELEMAC element.

## Results, interior source (`results/bump_profiles_source.png`)

| case | crest depth (m) | depth at x = 2 m | unit discharge upstream | L1 depth error |
|---|---|---|---|---|
| TELEMAC HLLC, 8 sources at x = 1 m, transcritical | 0.180 | 0.497 | 0.224 | 0.0048 |
| Celeris `HydrographSource`, strip at x = 1 m, dx 0.05 m | 0.1728 | 0.499 | 0.225 | 0.0045 |
| TELEMAC HLLC, subcritical | 0.473 | 0.807 | 0.745 | 0.0078 |
| Celeris `HydrographSource`, subcritical | 0.471 | 0.806 | 0.750 | 0.0067 |

The interior source reproduces the discharge-boundary result and the exact
solution to the same accuracy as TELEMAC's sources do, with no inlet hump in a
20 m channel once the strip sits clear of the wall column.

## Two setup rules this case taught

1. **Ghost rim cells continue the adjacent bed.** With a 20 m wall in the two
   pad rows next to 0.05 m cells, `Pass1`'s steep-slope Froude cap (3 / bed
   slope) froze the rows beside the pad and the centreline carried 30 % more
   than the mean discharge. The wall condition is the reflection itself; the
   pad must not be a step.
2. **Keep an interior source clear of the first interior column.** A strip
   touching the wall column leaked 25 % of its volume into the ghost rim.
