# TELEMAC-2D `weirs2` example: five weirs between six ponds, as `SpillwaySink`s

Model-to-model validation of `celeris.spillway.SpillwaySink`, the
level-dependent outlet (spillway, weir, rating-curve sink), against TELEMAC-2D
v8p5's generic weirs (`examples/telemac2d/weirs`, case `t2d_weirs2.cas`,
`TYPE OF WEIRS = 2`). No measurements exist for this case; TELEMAC is the
reference. It ships as a validation case of TELEMAC itself (validation
document test 6 since 1996 for option 1, the generic weirs since v6p3).

## What is tested

A TELEMAC weir is a line the mesh does not resolve: water is taken out of the
nodes on one side and put back on the other at the rate of Poleni's law on the
levels next to the line, `Q = mu sqrt(2g) (h_up - z_sill)^1.5` per unit width
with the drowned form when the tailwater rises (`loi_w_inc.f`, `mu = 0.4`
hard-coded, relaxation 0.5 per 10 s step). That is exactly the operation of a
`SpillwaySink` with an outlet patch: a level-dependent transfer of mass with
no momentum. The case checks that the Celeris sink removes and delivers the
right volume at the right time in a 2D pond cascade, i.e. the plumbing and its
coupling to the flow, not any particular spillway's hydraulics.

## Case

Six flat 1 km x 1 km ponds in an L, beds 100, 95, 90, 85, 80 and 75 m, 50 m
walls between them, Strickler 50. Five weirs of different shapes join them:
a straight sill at 102 m (1000 m), a zigzag sill at 97 m (1562 m of trace),
a ramp from 91 to 93 m, a 90 m crest with two 86 m slots, and a V from 84.5
down to 82.5 m. Inflow 0 to 500 m3/s in 200 s on the west edge of pond 1;
pond 6 held at 76 m (outlet); ponds 2 to 5 start 0.1 m deep; 12 h. TELEMAC:
977 nodes, 50 to 200 m triangles, 10 s steps, 10 s serial.

## Celeris setup (`spillway_validation.py`)

| item | choice |
|---|---|
| grid | 10 m, 315 x 210 cells (ponds plus a 2-cell ghost rim); walls 120 m |
| datum | 76 m (the outlet level) |
| inflow | `DischargeBoundary` on the west edge (type 5); pond 6's west columns are walled so the edge only wets pond 1 |
| outlet | pond 6 held at 76 m on a 50 m wide strip (post-step kernel) |
| weirs | one `SpillwaySink` per weir: intake strip 30 m deep along the upstream pond edge, outlet strip along the downstream edge; rating = TELEMAC's `LOI_W_INC` summed over the weir's segments from `weirs2.txt`, bidirectional; `smoothing_s = 14.4 s` (TELEMAC's relaxation e-folding time); `min_depth_m = 0.02` |
| friction | Manning n = 1/50 (`isManning=1`) |
| model | SWE, Courant 0.2, `--wetdry conserving`, dt 0.64 s, 67,641 steps, 450 s wall on an RTX A5000 |

```bash
uv run python examples/validation/spillway/spillway_validation.py telemac   # runs TELEMAC, ~10 s
uv run python examples/validation/spillway/spillway_validation.py run       # GPU (--arch cpu works)
uv run python examples/validation/spillway/spillway_validation.py compare
# or: uv run python examples/validation/cli.py run spillway
```

Reference: `<data root>/telemac/weirs2/` (result file, weir discharge CSV,
listing, `telemac_weirs2.npz`). Celeris products: `<data root>/celeris/spillway/`.

## Results (`results/spillway_metrics.csv`, `results/spillway_weirs2.png`)

| pond (bed) | TELEMAC final (m) | Celeris final (m) | RMSE over 12 h (m) | max diff (m) |
|---|---|---|---|---|
| 1 (100) | 102.432 | 102.431 | 0.001 | 0.003 |
| 2 (95) | 97.322 | 97.321 | 0.002 | 0.006 |
| 3 (90) | 92.390 | 92.382 | 0.007 | 0.009 |
| 4 (85) | 88.443 | 88.431 | 0.008 | 0.014 |
| 5 (80) | 83.469 | 83.467 | 0.004 | 0.010 |
| 6 (75, outlet) | 76.048 | 76.042 | 0.004 | 0.006 |

| weir | TELEMAC final Q (m3/s) | Celeris final Q | RMSE (m3/s) | max diff |
|---|---|---|---|---|
| 1, straight 102 m | 500.0 | 498.5 | 2.4 | 15.0 |
| 2, zigzag 97 m | 500.0 | 498.5 | 1.7 | 8.9 |
| 3, V 82.5 to 84.5 m | 500.0 | 498.4 | 1.3 | 5.2 |
| 4, slots 86 m | 500.0 | 498.5 | 1.1 | 2.2 |
| 5, ramp 91 to 93 m | 500.0 | 498.5 | 1.7 | 6.4 |

Every pond level stays within 1.4 cm of TELEMAC through the whole cascade
(the fill of pond 5 through the V weir starts at 6.5 h in both), and every
weir carries the same discharge to within 0.3 % at the end. The largest
transient difference (15 m3/s on weir 1 in the first hour) is Celeris
reaching the weir a few minutes earlier: the type-5 inflow is 500 m3/s from
t = 200 s in both models, but TELEMAC's 10 s steps and 0.5 relaxation delay
its weir response.

Two things the numbers include:

- the 1.5 m3/s (0.3 %) shortfall on every weir at the end is the discharge
  boundary passing 498.5 m3/s (all five weirs settle exactly on the inflow,
  which is the correct steady state); the sink itself is exact to float
  precision (`tests/test_spillway.py::test_tank_drawdown_matches_exact_poleni`,
  0.03 mm against the closed-form tank drawdown);
- TELEMAC loses 102,900 m3 over the run (0.7 % of the inflow, its own final
  volume balance), so a mass difference of that size against the reference is
  expected; the Celeris transfer conserves volume to 1e-3 (test
  `test_weir_transfers_between_two_basins`).

## What was learned

1. The weir sides come from the levee header of `weirs2.txt` (left line = side
   1, right line = side 2), and the flow order of the file is 1, 2, 5, 4, 3;
   the header is the only place that tells a weir which pond it drains.
2. Two weirs meeting in a pond corner (weirs 2 and 5 both feed pond 3, weirs
   4 and 5 both touch pond 4) must not share cells: the sink writes the
   continuity source per cell, so the corner cells belong to the first weir.
3. The head is read as the mean surface over the 30 m intake strip. In the
   tank test the strip sits at most 5 mm below the pool mean at 0.2 m3/s
   through 16 m2, and the exact solution is still met to 0.03 mm; a rating
   that must see the far-field level should use a wider or offset patch.
