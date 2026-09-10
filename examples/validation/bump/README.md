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
| Celeris `HydrographSource`, strip at x = 1 m, dx 0.05 m | 0.176 | 0.503 | 0.230 (centreline) | 0.0054 |
| TELEMAC HLLC, subcritical | 0.473 | 0.807 | 0.745 | 0.0078 |
| Celeris `HydrographSource`, subcritical | 0.460 | 0.806 | 0.768 (centreline) | 0.0059 |

The interior source reproduces the exact solution to the same accuracy as
TELEMAC's sources do, with no inlet hump in a 20 m channel once the strip sits
clear of the wall column. The centreline unit discharge reads 2.4 % above the
imposed mean because Celeris' wall passes through the centre of the first
interior row (that row counts half) and the centreline carries slightly more
than the section mean; the section-integrated discharge is the imposed one.

## Two setup rules this case taught

1. **Ghost rim cells continue the adjacent bed.** With a 20 m wall in the two
   pad rows next to 0.05 m cells, `Pass1`'s steep-slope Froude cap (3 / bed
   slope) froze the rows beside the pad and the centreline carried 30 % more
   than the mean discharge. The wall condition is the reflection itself; the
   pad must not be a step.
2. **Keep an interior source off the three outer rows and columns.** Celeris'
   solid wall passes through the centre of the first interior cell (index 2 and
   n-3), so only half of that cell is inside the domain: a source there injects
   half its volume into the mirror image (a strip on the wall rows lost 3.6 %,
   one touching the wall column 25 %). `HydrographSource` now refuses such masks
   and `inlet_mask` never produces them. Volume accounting must weight those
   cells by one half; with that, the scheme conserves to 1e-4.

## Limiter parameter `theta` (`--theta`, default 2 as in Celeris)

Behind the hydraulic jump the default `theta = 2` (least dissipative end of the
generalized minmod limiter) is unsteady: the unit discharge at x = 15 m
oscillates in time with a standard deviation of 0.046 m2/s (20 % of q) while the
surface moves 2 mm. `theta = 1` (TVD minmod) removes the oscillation
(0.005 m2/s) but leaves q 8 % high behind the jump (TELEMAC HLLC: 1.5 % high)
and is visibly more diffusive on smooth flow: subcritical L1 0.021 m against
0.007 m with `theta = 2`, crest depth 0.497 m against the exact 0.471 m. Neither
value is right for both; the reservoir problem has no hydraulic jump, so the
default stays. Legacy and conserving are identical for every `theta` here.
