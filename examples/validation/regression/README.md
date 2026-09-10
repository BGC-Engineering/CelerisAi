# Solver regression harness

Freezes the outputs of eight CelerisAi runs on a given code state and compares two
frozen sets cell by cell. Built to prove that a solver change behind a switch leaves
the switch-off path unchanged and to measure what the switch-on path changes.

Files: `regress.py` (CLI), `cases.py` (manifest + case builders), this README.
Baselines live outside the repo at `/mnt/d/Homathko/Validation/baseline/<tag>/<case>/`
(`fields.npz` + `metrics.json`; `compare.json` is written into the newer set).

## Commands

```bash
cd /home/cstringari/Celeris-Wave-Model/CelerisAi
export LD_LIBRARY_PATH=/usr/lib/wsl/lib          # CUDA on WSL2
R="uv run --no-sync python examples/validation/regression/regress.py"

$R freeze --tag legacy                            # all cases -> baseline/legacy/
$R freeze --tag legacy --only tracyarm_gen breach # subset
$R freeze --tag smoke --max-steps 30              # smoke test (seconds)
$R check  --tag legacy [--tol 1e-3]               # rerun everything into baseline/legacy__check/, compare, PASS/FAIL table
$R compare --a baseline/legacy --b baseline/wetdry_on   # compare two frozen sets, no runs
$R sanity                                          # tracyarm_gen vs calibration/TracyArm/data/gen_v1.npz
$R freeze --tag legacy_nobreak --no-breaking       # every case with the breaking model forced off (deterministic, see below)
```

Each case runs in its own subprocess (`regress.py _run <case> <dir>`) with
`ti.init(arch=ti.cuda, default_fp=ti.f32)`. Plain argparse + numpy + taichi, no new deps.

## Cases (`cases.py`)

| case | input on this machine | model / setup | duration | grid, dt, steps | wall (RTX A5000) |
|---|---|---|---|---|---|
| `tracyarm_gen` | `calibration/TracyArm/cases/Endicott_slide` (celeris loader) | `generate.py` defaults: SWE, breaking on, `MovingBodySlide` with Lynett's config (65 Mm3, expo 10, 1400 m, tau 12 s, shift 60 s, 85->115 deg, az-sign -1), `wet_dhdt_only`, emerge 1 m, delta 0.01 | 150 s | 801x701 @ 10 m, 0.0386 s, 3883 | 9 s |
| `balboa` | `examples/Balboa` | config.json: Bouss, breaking on, incoming waves S | 120 s | 1067x534 @ 1.5 m, 0.0263 s, 4559 | 47 s |
| `crescentcity` | `examples/CrescentCity` | config.json: `NLSW_or_Bous=0` -> SWE, no breaking key -> off, waves W+S | 120 s | 1257x580 @ 5 m, 0.0922 s, 1302 | 7 s |
| `mavericks` | `examples/Mavericks` | config.json: Bouss, breaking on, waves W | 120 s | 902x715 @ 2.5 m, 0.0271 s, 4429 | 55 s |
| `ventura` | `examples/Ventura` | config.json: Bouss, breaking on, waves W | 120 s | 856x1081 @ 3 m, 0.0469 s, 2558 | 62 s |
| `breach` | `/mnt/d/Homathko/Validation/celeris/breach` (prep from `~/telemac-mascaret/.../breach`) | `breach_validation.py` prep + run: SWE, Manning 1/15, `HydrographSource` inlet, outlet stage kernel, no breaking | 2700 s | 2000x200 @ 2.5 m, 0.0721 s, 37423 | 45 s |
| `malpasset` | `/mnt/d/Homathko/Validation/celeris/malpasset` (prep_malpasset.py --dx 15) | `run_malpasset.py` plain: SWE, breaking on, Manning 1/30, Courant 0.2 | 4000 s | 1154x618 @ 15 m, 0.1292 s, 30966 | 106 s |
| `malpasset_seed` | same | `run_malpasset.py --seed-wetting 0.02` (seed kernel copied) | 4000 s | same | 129 s |

Wall = time-stepping loop only (`wall_loop_s` in metrics.json; `wall_total_s` adds
Taichi init + case build). Full `freeze` of all eight: about 8.5 min including
subprocess start-up and the breach prep.

The coastal cases and `tracyarm_gen` are built exactly as `setrun_web.py` /
`generate.py` do (loader, config.json, `Evolve_0`, `Evolve_Steps` loop). `breach` and
`malpasset*` copy the setup code of the validation scripts (which are monolithic
`run()` functions) and import their constants and `prep`; the loop is the harness's
own so that frames at the requested times, envelopes and volume can be captured.
Sampling cadence and step counts use each script's own formula (`int(10/dt)` etc.).
Cross-check against the scripts' own outputs on disk: breach probe series agree to
5 mm and the floodplain fraction exactly; malpasset gauge series differ by up to 5.8 m
because the solver is not run-to-run deterministic (below), not because of the setup.

## What is saved per case

`fields.npz`
- `eta_tNNNN` (float32): free surface at t = NNNN s, NaN where depth <= 0.01 m. Times: tracyarm 30/60/100/150, coastal 30/60/90/120, breach 1000/2000/2700, malpasset 500/1000/2000/4000; plus `eta_final`. Captured after step `round(T/dt)`.
- `bed`: bed elevation (solver datum, positive up) at t = 0. For `tracyarm_gen` (moving bed) also `bed_tNNNN` per frame.
- `env_max`, `env_min`: per-cell max / min of eta over all steps while wet (depth > 0.01 m); updated every step by a GPU kernel; NaN where never wet.
- `t_s`, `volume_m3`: interior wet volume vs time (cells `[2:-2, 2:-2]`, depth > 0.01 m) at the sample cadence.
- tracyarm: `runup_m` (max eta over cells with depth > 0.5 m and bed0 > 0) and `trough_m` (min eta over depth > 0.5 m) per sample.
- breach: `probes_eta_masl` (7 probes, m a.s.l. = solver eta + 4.9 m datum), `floodplain_wet_fraction`, `probes_x_m`, `datum_m`.
- malpasset: `gauge_eta_m` (14 gauges, m a.s.l.), `gauge_names`, `gauge_i`, `gauge_j`.

`metrics.json` (scalars)
- all: `nx ny dx_m dt_s n_steps duration_s sample_every blowup_step max_eta_m min_eta_m wet_ever_cells volume_initial_m3 volume_final_m3 volume_change_frac wall_loop_s wall_total_s`.
- tracyarm: `max_runup_m` (max of `runup_m`), `min_trough_m`.
- breach: `floodplain_wet_fraction_final`, `floodplain_first_wet_s`, `eta_probe_x<X>_final_masl`.
- malpasset: `arrival_A_s`, `A_to_B_s`, `A_to_C_s` (first sample with depth > 0.05 m, as `compare_malpasset.py`), `hmax_P6_m` .. `hmax_P14_m` (max sampled depth at the gauge).

## Compare

For each frame: cells wet in either set; a dry cell contributes its bed (so a wet/dry
flip counts as eta - bed); reports max |delta eta|, RMSE and the number of cells whose
wet state differs. Envelopes: max |delta| over cells wet-ever in both plus the
wet-ever mismatch count. Volume: max |delta| over the common series. Scalars: b - a.
PASS if every frame and envelope max |delta| < `--tol` (default 1e-3 m).

## Sanity vs `gen_v1.npz` (tracyarm_gen)

`regress.py sanity` reruns the case, capturing frames at exactly the steps of the
golden file (gen_v1 frame k = state after 25k+1 steps; dt identical, 0.038629 s).

- Max wet runup (depth > 0.5 m, bed0 > 0): harness 420.50 m vs gen_v1 420.53 m (-0.01 %).
  RESULTS.md's 438 m is the *big-grid* run (`runup_big_v1`), not gen_v1; gen_v1 itself
  gives 420.5 m by the stated definition. gen_v1 also runs to 160 s (166 frames); the
  generate.py default (and the harness) is 150 s.
- Deepest trough: -243.67 m vs -244.03 m.
- Frames: t = 30 s bit-identical (RMSE 0). t = 60 s RMSE 0.095 m (max 21 m at one cell),
  t = 100 s RMSE 1.42 m, t = 150 s RMSE 2.22 m. **Not within a few cm**, and this is not
  a setup difference: two runs of the harness itself on the current code differ by the
  same amount (next section). The 60/100/150 s numbers are the solver's own
  run-to-run scatter, so gen_v1 cannot be reproduced closer than that by anyone.

## Determinism (sets the floor for `--tol`)

Two freezes of the same code (`legacy` vs `legacy_rerun`), GPU float32:

| case | breaking | worst frame max abs dEta (m) | worst frame RMSE (m) | wet-state mismatch cells (final frame) | env_max max abs (m) | volume max abs (m3) | scalar scatter | PASS at 1e-3 |
|---|---|---|---|---|---|---|---|---|
| tracyarm_gen | on | 4.5e+01 | 2.1e+00 | 504 | 1.3e+01 | 3.6e+05 | runup -1.24 m, trough -0.04 m | FAIL |
| balboa | on | 4.5e-01 | 6.9e-03 | 957 | 3.9e-01 | 2.6e+01 | max eta -2.4e-02 m | FAIL |
| crescentcity | off | 6.5e-05 | 1.3e-07 | 0 | 3.8e-05 | 0.0e+00 | max eta +0.0e+00 m | PASS |
| mavericks | on | 2.7e-05 | 5.3e-07 | 0 | 1.3e-05 | 8.0e+00 | max eta +0.0e+00 m | PASS |
| ventura | on | 7.6e-01 | 4.4e-03 | 32 | 6.6e-01 | 1.1e+02 | max eta -4.8e-07 m | FAIL |
| breach | off | 2.5e-02 | 9.6e-04 | 0 | 6.9e-01 | 7.9e+00 | probes <= 3.1e-04 m, floodplain fraction 0 -> 0 | FAIL |
| malpasset | on | 8.3e+00 | 2.0e+00 | 17684 | 4.7e+00 | 1.7e+05 | A->B +2.1 s, A->C -2.1 s, hmax P6-P14 <= 0.27 m | FAIL |
| malpasset_seed | on | 2.7e+00 | 1.3e-01 | 172 | 3.9e+00 | 4.0e+04 | A->B +1.0 s, A->C +2.1 s, hmax P6-P14 <= 0.23 m | FAIL |

Root cause (single-step tests, scratch scripts, not in the repo): from a bit-identical
snapshot of every Taichi field, each kernel of one `Evolve_Steps` was run three times
and its outputs compared. Two kernels are not reproducible; everything else
(`Pass1`, `Pass2`, `Pass3`/`Pass3Bous`, `Run_Tridiag_solver`, the landslide /
hydrograph sources) is bit-exact:

1. `Solver.BoundaryPass` (celeris/solver.py ~line 604) writes `txState[i, j]` while
   reading the neighbours `txState[leftIdx, j]`, `[rightIdx, j]` (and N/S in 2D) for the
   wet/dry sanitation (`dry_west` / `dry_east` / island removal) and reads
   `txState[BCShift - i, j]` etc. for the wall / sponge boundaries, all in the same
   parallel kernel. Whether a thread sees a neighbour's old or already-updated value
   depends on GPU scheduling. Found in every case: `breach` (SWE, breaking off, 4.4 m
   jump at a wet/dry cell in one step), `crescentcity` (SWE, 0.7 m in one step),
   `mavericks` (Bouss, 3e-5 m in one step). This is what makes wet/dry-front cases
   (`breach`, `malpasset*`, `tracyarm_gen`) scatter by metres.
2. `Solver.Pass_Breaking` (~line 2730) reads the neighbours' breaking time
   `t1 = self.Breaking[leftIdx, j].x` (`rightIdx`, `upIdx`, `downIdx`) and writes
   `self.Breaking[i, j]` in the same kernel. Only differs where breaking is triggered
   (`tracyarm_gen`: 304 cells per step at the impact; `mavericks`: 41 cells).

Both are upstream code; fixing them (double-buffer `txState` / `Breaking`, or read
the neighbours from the previous-step copy) is a solver change and out of scope here.

Consequences for the switch test:
- A 1e-3 m cell-wise check can only pass where wet/dry fronts are absent or the run
  is short: `crescentcity` and `mavericks` pass (max 6e-5 m); `balboa` / `ventura`
  scatter to 0.5-0.8 m at a few hundred cells after 120 s; `tracyarm_gen`,
  `malpasset*`, `breach` scatter by metres at wet/dry fronts.
- `--no-breaking` removes race 2 only. Floor with the breaking model forced off
  (`legacy_nobreak` vs `legacy_nobreak_rerun`):

| case (breaking off) | worst frame max abs dEta (m) | worst frame RMSE (m) | wet-state mismatch cells (final frame) | env_max max abs (m) | scalar scatter | PASS at 1e-3 |
|---|---|---|---|---|---|---|
| tracyarm_gen | 5.8e+01 | 1.8e+00 | 686 | 1.3e+01 | runup +4.06 m, trough -0.09 m | FAIL |
| balboa | 3.1e-01 | 6.8e-03 | 1729 | 4.1e-01 | max eta -7.5e-03 m | FAIL |
| crescentcity | 6.4e-05 | 1.3e-07 | 0 | 6.4e-05 | max eta +0.0e+00 m | PASS |
| mavericks | 2.7e-05 | 5.1e-07 | 0 | 5.2e-06 | max eta -2.9e-06 m | FAIL |
| ventura | 1.8e+00 | 7.4e-03 | 29 | 3.5e-01 | max eta +2.4e-06 m | FAIL |
| breach | 1.2e-01 | 1.2e-03 | 0 | 9.4e-02 | probes <= 2.5e-03 m | FAIL |
| malpasset | 7.8e+00 | 1.6e+00 | 13131 | 5.0e+00 | A->B +nan s, A->C +0.0 s, hmax P6-P14 <= 0.49 m | FAIL |
| malpasset_seed | 6.5e+00 | 1.5e-01 | 149 | 3.8e+00 | A->B -2.1 s, A->C -1.0 s, hmax P6-P14 <= 0.31 m | FAIL |

  (`mavericks` fails only on `env_min`: 0.82 m at one transient wet/dry cell.)
  Forcing breaking off does not help: race 1 dominates everywhere.

- Therefore: judge the switch-off path with the scalar metrics and RMSE against the
  scatter tables above (`compare` prints them), and expect frame max |delta| to fail
  at 1e-3 on the wet/dry cases whatever the code does. A bit-exact proof needs the two
  races fixed first (then `check --tol 1e-3` becomes meaningful for all eight cases).


## Update: races fixed, `--wetdry` added

`BoundaryPass` and `Pass_Breaking` now read from a snapshot (`StateScratch`,
`BreakingScratch`), so identical runs are bitwise equal and `check --tol 1e-3`
is meaningful again. The `legacy` baseline above was frozen with the racy code;
`legacy_det` is the deterministic legacy baseline and `conserving` the same
cases with `--wetdry conserving`. The breach case was padded by two wall cells
after `legacy` was frozen, so its frames do not align with that first tag.

## Update: v2 baselines (discharge boundary added, breach rim mirrored)

`legacy_v2` and `conserving_v2` are frozen on the code that adds boundary type 5
(`DischargeBoundary`); nothing in the eight cases uses it, so they must equal
`legacy_det` and `conserving3` bitwise except `breach`, whose grid changed (the
ghost rim now mirrors the interior instead of a 20 m wall). Legacy blows up at
step 138 on that grid; `breach` is compared for the conserving scheme only.
