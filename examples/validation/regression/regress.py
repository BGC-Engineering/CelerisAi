"""Regression harness for the CelerisAi solver: freeze, check, compare.

    uv run --no-sync python examples/validation/regression/regress.py freeze --tag legacy
    uv run --no-sync python examples/validation/regression/regress.py check --tag legacy
    uv run --no-sync python examples/validation/regression/regress.py compare --a DIR --b DIR
    uv run --no-sync python examples/validation/regression/regress.py sanity

Each case runs in its own subprocess (one ``ti.init`` per process) and writes
``<root>/<tag>/<case>/fields.npz`` + ``metrics.json``. See README.md.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cases

ROOT_DEFAULT = Path("/mnt/d/Homathko/Validation/baseline")
HERE = Path(__file__).resolve().parent
GHOST = 2  # ghost rim excluded from the volume


def _envelope(solver: object) -> object:
    """GPU max/min eta envelope over wet cells, updated every step."""
    import taichi as ti

    @ti.data_oriented
    class Envelope:
        def __init__(self) -> None:
            self.emax = ti.field(ti.f32, shape=(solver.nx, solver.ny))  # type: ignore[attr-defined]
            self.emin = ti.field(ti.f32, shape=(solver.nx, solver.ny))  # type: ignore[attr-defined]
            self.emax.fill(-1e30)
            self.emin.fill(1e30)

        @ti.kernel
        def update(self):  # type: ignore[no-untyped-def]
            for i, j in self.emax:
                eta = solver.State[i, j][0]  # type: ignore[attr-defined]
                if eta - solver.Bottom[2, i, j] > cases.DRY_M:  # type: ignore[attr-defined]
                    self.emax[i, j] = ti.max(self.emax[i, j], eta)
                    self.emin[i, j] = ti.min(self.emin[i, j], eta)

        def result(self) -> tuple[np.ndarray, np.ndarray]:
            mx, mn = self.emax.to_numpy(), self.emin.to_numpy()
            never = mx < -1e29
            mx[never] = np.nan
            mn[never] = np.nan
            return mx, mn

    return Envelope()


def run_case(
    *,
    name: str,
    out_dir: Path,
    max_steps: int | None = None,
    extra_frame_steps: dict[str, int] | None = None,
    no_breaking: bool = False,
    wetdry: str = "legacy",
) -> dict[str, object]:
    """Run one case and write ``fields.npz`` + ``metrics.json`` to ``out_dir``.

    Args:
        name: Case key in :data:`cases.CASES`.
        out_dir: Destination directory (created).
        max_steps: Step cap for smoke tests.
        extra_frame_steps: Additional ``{label: step}`` frames to capture.
        no_breaking: Force the wave-breaking model off after the build
            (``Pass_Breaking`` is not run-to-run deterministic, see README).
        wetdry: ``Solver`` wet/dry scheme, ``legacy`` or ``conserving``. Set
            after the build; the kernels read it as a constant when they compile
            on the first step.

    Returns:
        The metrics dict that was written.
    """
    import taichi as ti

    t_total = time.perf_counter()
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    cfg = cases.CASES[name]
    run = cases.build(name=name, max_steps=max_steps)
    solver, evolve, dt, n_steps = run.solver, run.evolve, run.dt, run.n_steps
    if no_breaking:
        solver.useBreakingModel = False
    if wetdry not in ("legacy", "conserving"):
        raise ValueError(wetdry)
    solver.wetdry_scheme = wetdry
    solver.wd_conserving = 1 if wetdry == "conserving" else 0
    env = _envelope(solver)
    bed0 = solver.Bottom.to_numpy()[2].astype(np.float32)
    dx, dy = float(solver.dx), float(solver.dy)
    inner = (slice(GHOST, -GHOST), slice(GHOST, -GHOST))
    frame_steps: dict[str, int] = {
        f"t{int(T):04d}": round(T / dt)
        for T in cfg["frames_s"]  # type: ignore[attr-defined]
    }
    frame_steps.update(extra_frame_steps or {})
    frame_steps = {k: s for k, s in frame_steps.items() if 0 < s <= n_steps}
    frame_steps["final"] = n_steps
    want = {}
    for k, s in frame_steps.items():
        want.setdefault(s, []).append(k)
    every = run.sample_every
    logger.info(
        "{}: {}x{} @ {} m, dt {:.5f} s, {} steps ({:.1f} s), sample every {} steps, "
        "frames {}",
        name,
        solver.nx,
        solver.ny,
        dx,
        dt,
        n_steps,
        n_steps * dt,
        every,
        {k: round(s * dt, 2) for k, s in frame_steps.items()},
    )

    def readback() -> tuple[np.ndarray, np.ndarray]:
        eta = solver.State.to_numpy()[:, :, 0]
        bed = solver.Bottom.to_numpy()[2] if run.moving_bed else bed0
        return eta, bed

    series: dict[str, list[object]] = {"t_s": [], "volume_m3": []}
    frames: dict[str, np.ndarray] = {}
    blowup: int | None = None

    def take_sample(n: int, eta: np.ndarray, bed: np.ndarray) -> None:
        depth = eta - bed
        wet = depth > cases.DRY_M
        series["t_s"].append(n * dt)
        series["volume_m3"].append(float(depth[inner][wet[inner]].sum() * dx * dy))
        if run.sample is not None:
            for k, v in run.sample(eta, bed).items():
                series.setdefault(k, []).append(v)

    eta, bed = readback()
    take_sample(0, eta, bed)
    t_loop = time.perf_counter()
    for step in range(n_steps):
        evolve.Evolve_Steps(step)
        if run.post_step is not None:
            run.post_step(step)
        env.update()
        n = step + 1
        is_sample = n % every == 0 or n == n_steps
        if not (is_sample or n in want):
            continue
        eta, bed = readback()
        if not np.isfinite(eta).all():
            logger.error(
                "{}: non-finite eta after step {} (t={:.2f} s)", name, n, n * dt
            )
            blowup = n
            break
        if is_sample:
            take_sample(n, eta, bed)
        for k in want.get(n, []):
            frames[f"eta_{k}"] = np.where(eta - bed > cases.DRY_M, eta, np.nan).astype(
                np.float32
            )
            if run.moving_bed:
                frames[f"bed_{k}"] = bed.astype(np.float32)
        if n % (every * 20) == 0:
            rate = n * dt / max(time.perf_counter() - t_loop, 1e-9)
            logger.info(
                "{}: t={:8.1f} s ({:5.1f}x rt) vol {:.4g} m3",
                name,
                n * dt,
                rate,
                series["volume_m3"][-1],
            )
    wall_loop = time.perf_counter() - t_loop
    ti.sync()

    emax, emin = env.result()
    ser = {k: np.asarray(v) for k, v in series.items()}
    vol = ser["volume_m3"]
    metrics: dict[str, object] = {
        "case": name,
        "nx": int(solver.nx),
        "ny": int(solver.ny),
        "dx_m": dx,
        "dt_s": dt,
        "n_steps": n_steps,
        "duration_s": n_steps * dt,
        "sample_every": every,
        "breaking_model": bool(solver.useBreakingModel),
        "wetdry_scheme": wetdry,
        "blowup_step": blowup,
        "max_eta_m": float(np.nanmax(emax)),
        "min_eta_m": float(np.nanmin(emin)),
        "wet_ever_cells": int(np.isfinite(emax).sum()),
        "volume_initial_m3": float(vol[0]),
        "volume_final_m3": float(vol[-1]),
        "volume_change_frac": float(vol[-1] / vol[0] - 1.0) if vol[0] else float("nan"),
    }
    if run.metrics is not None:
        metrics.update(run.metrics(ser))
    metrics["wall_loop_s"] = wall_loop
    metrics["wall_total_s"] = time.perf_counter() - t_total
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "fields.npz",
        bed=bed0,
        env_max=emax,
        env_min=emin,
        **frames,
        **ser,
        **run.arrays,
    )
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=1, default=float)
    logger.info(
        "{}: wall loop {:.1f} s, total {:.1f} s -> {}",
        name,
        wall_loop,
        metrics["wall_total_s"],
        out_dir,
    )
    return metrics


def freeze(
    *,
    root: Path,
    tag: str,
    only: list[str] | None,
    max_steps: int | None,
    no_breaking: bool = False,
    wetdry: str = "legacy",
) -> Path:
    """Run every (or the selected) case in a subprocess into ``root/tag``."""
    names = only or list(cases.CASES)
    dest = root / tag
    for name in names:
        cmd = [sys.executable, str(HERE / "regress.py"), "_run", name, str(dest / name)]
        if max_steps is not None:
            cmd += ["--max-steps", str(max_steps)]
        if no_breaking:
            cmd += ["--no-breaking"]
        cmd += ["--wetdry", wetdry]
        t0 = time.perf_counter()
        rc = subprocess.run(cmd, cwd=cases.CELERIS_DIR, check=False).returncode
        logger.info(
            "{}: exit {} after {:.0f} s wall", name, rc, time.perf_counter() - t0
        )
    return dest


def _load(d: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with open(d / "metrics.json") as f:
        return dict(np.load(d / "fields.npz")), json.load(f)


def compare_case(*, a: Path, b: Path, tol: float) -> dict[str, object]:
    """Compare one case's frozen outputs in ``a`` and ``b`` (b - a)."""
    fa, ma = _load(a)
    fb, mb = _load(b)
    rep: dict[str, object] = {"frames": {}, "envelopes": {}, "scalars": {}}
    worst = 0.0
    for k in sorted(set(fa) & set(fb)):
        if not k.startswith("eta_"):
            continue
        bed_key = k.replace("eta_", "bed_")
        ba = fa.get(bed_key, fa["bed"])
        bb = fb.get(bed_key, fb["bed"])
        ea, eb = fa[k], fb[k]
        wet_a, wet_b = np.isfinite(ea), np.isfinite(eb)
        either = wet_a | wet_b
        d = (np.where(wet_a, ea, ba) - np.where(wet_b, eb, bb))[either].astype(
            np.float64
        )
        mx = float(np.abs(d).max()) if d.size else 0.0
        worst = max(worst, mx)
        rep["frames"][k] = {  # type: ignore[index]
            "max_abs_m": mx,
            "rmse_m": float(np.sqrt(np.mean(d**2))) if d.size else 0.0,
            "wet_cells_either": int(either.sum()),
            "wet_mismatch_cells": int((wet_a != wet_b).sum()),
        }
    for k in ("env_max", "env_min"):
        ea, eb = fa[k], fb[k]
        both = np.isfinite(ea) & np.isfinite(eb)
        d = (ea - eb)[both].astype(np.float64)
        mx = float(np.abs(d).max()) if d.size else 0.0
        worst = max(worst, mx)
        rep["envelopes"][k] = {  # type: ignore[index]
            "max_abs_m": mx,
            "wet_mismatch_cells": int((np.isfinite(ea) != np.isfinite(eb)).sum()),
        }
    va, vb = fa["volume_m3"], fb["volume_m3"]
    n = min(len(va), len(vb))
    rep["volume_max_abs_m3"] = (
        float(np.abs(va[:n] - vb[:n]).max()) if n else float("nan")
    )
    rep["volume_len"] = [len(va), len(vb)]
    for k, v in ma.items():
        if isinstance(v, (int, float)) and k in mb and not k.startswith("wall"):
            rep["scalars"][k] = {"a": v, "b": mb[k], "delta": float(mb[k]) - float(v)}  # type: ignore[index]
    rep["wall_s"] = {"a": ma.get("wall_loop_s"), "b": mb.get("wall_loop_s")}
    rep["max_eta_delta_m"] = worst
    rep["status"] = "PASS" if worst < tol else "FAIL"
    return rep


def compare(*, a: Path, b: Path, tol: float) -> dict[str, object]:
    """Compare two frozen sets, print the table, write compare.json next to the newer."""
    names = sorted(
        {p.name for p in a.iterdir() if (p / "metrics.json").exists()}
        | {p.name for p in b.iterdir() if (p / "metrics.json").exists()}
    )
    out: dict[str, object] = {"a": str(a), "b": str(b), "tol_m": tol, "cases": {}}
    hdr = f"{'case':16s} {'frame':12s} {'max|d eta| m':>13s} {'RMSE m':>10s} {'wet mism':>9s} {'status':>7s}"
    logger.info(hdr)
    for name in names:
        if (
            not (a / name / "metrics.json").exists()
            or not (b / name / "metrics.json").exists()
        ):
            out["cases"][name] = {"status": "MISSING"}  # type: ignore[index]
            logger.info(
                "{:16s} {:12s} {:>13s} {:>10s} {:>9s} {:>7s}",
                name,
                "-",
                "-",
                "-",
                "-",
                "MISSING",
            )
            continue
        rep = compare_case(a=a / name, b=b / name, tol=tol)
        out["cases"][name] = rep  # type: ignore[index]
        for fk, fr in rep["frames"].items():  # type: ignore[attr-defined]
            logger.info(
                "{:16s} {:12s} {:13.3e} {:10.3e} {:9d}",
                name,
                fk[4:],
                fr["max_abs_m"],
                fr["rmse_m"],
                fr["wet_mismatch_cells"],
            )
        for ek, er in rep["envelopes"].items():  # type: ignore[attr-defined]
            logger.info(
                "{:16s} {:12s} {:13.3e} {:>10s} {:9d}",
                name,
                ek,
                er["max_abs_m"],
                "-",
                er["wet_mismatch_cells"],
            )
        logger.info(
            "{:16s} {:12s} {:13.3e} {:>10s} {:>9s} {:>7s}",
            name,
            "volume(m3)",
            rep["volume_max_abs_m3"],
            "-",
            "-",
            rep["status"],
        )
        for sk, sr in rep["scalars"].items():  # type: ignore[attr-defined]
            if abs(sr["delta"]) > 0:
                logger.info(
                    "{:16s}   {:34s} {:+.4g} ({} -> {})",
                    name,
                    sk,
                    sr["delta"],
                    sr["a"],
                    sr["b"],
                )
    fails = [n for n, r in out["cases"].items() if r["status"] != "PASS"]  # type: ignore[attr-defined]
    out["status"] = "PASS" if not fails else "FAIL"
    logger.info(
        "overall {} (tol {} m){}",
        out["status"],
        tol,
        f" failing: {fails}" if fails else "",
    )
    newer = max((a, b), key=lambda p: p.stat().st_mtime)
    with open(newer / "compare.json", "w") as f:
        json.dump(out, f, indent=1, default=float)
    logger.info("wrote {}", newer / "compare.json")
    return out


def sanity(*, root: Path) -> None:
    """Run tracyarm_gen and compare with calibration/TracyArm/data/gen_v1.npz.

    gen_v1 frame k is the state after ``k * every + 1`` steps (generate.py
    samples after ``Evolve_Steps``), so the harness captures frames at exactly
    those steps for the frames nearest 30/60/100/150 s.
    """
    gold = np.load(cases.TRACYARM_GOLDEN)
    t_gold = gold["t"]
    every = round((t_gold[1] - t_gold[0]) / (t_gold[1] - t_gold[0]) * 25)  # 25 steps
    dt_gold = (t_gold[1] - t_gold[0]) / every
    ks = {
        f"gold{int(T)}": int(np.argmin(np.abs(t_gold - T)))
        for T in (30.0, 60.0, 100.0, 150.0)
    }
    out = root / "sanity" / "tracyarm_gen"
    m = run_case(
        name="tracyarm_gen",
        out_dir=out,
        extra_frame_steps={lab: k * every + 1 for lab, k in ks.items()},
    )
    if abs(m["dt_s"] - dt_gold) > 1e-6:  # type: ignore[operator]
        logger.warning("dt differs: harness {} vs gen_v1 {}", m["dt_s"], dt_gold)
    f = dict(np.load(out / "fields.npz"))
    eta_g, bed_g, bed0 = gold["eta"], gold["bed"], gold["bed0"]
    run_g = []
    for k in range(eta_g.shape[0]):
        wet = (eta_g[k] - bed_g[k]) > cases.RUNUP_DEPTH_M
        on_land = wet & (bed0 > 0)
        run_g.append(eta_g[k][on_land].max() if on_land.any() else np.nan)
    runup_gold = float(np.nanmax(run_g))
    trough_gold = float(
        min(
            eta_g[k][(eta_g[k] - bed_g[k]) > cases.RUNUP_DEPTH_M].min()
            for k in range(eta_g.shape[0])
        )
    )
    logger.info(
        "gen_v1: {} frames to {:.1f} s (generate.py default duration is 150 s)",
        eta_g.shape[0],
        t_gold[-1],
    )
    logger.info(
        "runup  harness {:.2f} m vs gen_v1 {:.2f} m ({:+.2f} %)  [RESULTS.md quotes 438 m, "
        "which is the big-grid run runup_big_v1]",
        m["max_runup_m"],
        runup_gold,
        100 * (m["max_runup_m"] / runup_gold - 1),
    )  # type: ignore[operator]
    logger.info(
        "trough harness {:.2f} m vs gen_v1 {:.2f} m", m["min_trough_m"], trough_gold
    )
    res = {
        "runup_harness_m": m["max_runup_m"],
        "runup_gen_v1_m": runup_gold,
        "trough_harness_m": m["min_trough_m"],
        "trough_gen_v1_m": trough_gold,
        "frames": {},
    }
    for lab, k in ks.items():
        eh = f[f"eta_{lab}"]
        bh = f[f"bed_{lab}"]
        eg, bg = eta_g[k], bed_g[k]
        wet_g = (eg - bg) > cases.DRY_M
        either = np.isfinite(eh) | wet_g
        d = (np.where(np.isfinite(eh), eh, bh) - eg)[either].astype(np.float64)
        r = {
            "t_s": float(t_gold[k]),
            "step": k * every + 1,
            "rmse_m": float(np.sqrt(np.mean(d**2))),
            "max_abs_m": float(np.abs(d).max()),
            "wet_mismatch_cells": int((np.isfinite(eh) != wet_g).sum()),
        }
        res["frames"][lab] = r  # type: ignore[index]
        logger.info(
            "{}: t={:.2f} s step {}: RMSE {:.4f} m, max|d| {:.4f} m, wet mismatch {} cells",
            lab,
            r["t_s"],
            r["step"],
            r["rmse_m"],
            r["max_abs_m"],
            r["wet_mismatch_cells"],
        )
    with open(out / "sanity.json", "w") as fh:
        json.dump(res, fh, indent=1, default=float)


def main() -> None:
    """CLI entry point."""
    logger.remove()
    logger.add(sys.stderr, format="<level>{level: <5}</level> {message}")
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("freeze", "check"):
        p = sub.add_parser(c)
        p.add_argument("--tag", default="legacy")
        p.add_argument("--root", type=Path, default=ROOT_DEFAULT)
        p.add_argument("--only", nargs="+", choices=list(cases.CASES))
        p.add_argument("--max-steps", type=int, help="cap steps (smoke test)")
        p.add_argument(
            "--no-breaking",
            action="store_true",
            help="force useBreakingModel off (removes the Pass_Breaking race, see README)",
        )
        p.add_argument("--wetdry", default="legacy", choices=["legacy", "conserving"])
        if c == "check":
            p.add_argument("--tol", type=float, default=1e-3)
    p = sub.add_parser("compare")
    p.add_argument("--a", type=Path, required=True)
    p.add_argument("--b", type=Path, required=True)
    p.add_argument("--tol", type=float, default=1e-3)
    p = sub.add_parser("sanity")
    p.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    p = sub.add_parser("_run", help="internal: run one case in this process")
    p.add_argument("case", choices=list(cases.CASES))
    p.add_argument("out_dir", type=Path)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--no-breaking", action="store_true")
    p.add_argument("--wetdry", default="legacy", choices=["legacy", "conserving"])
    a = ap.parse_args()
    if a.cmd == "_run":
        run_case(
            name=a.case,
            out_dir=a.out_dir,
            max_steps=a.max_steps,
            no_breaking=a.no_breaking,
            wetdry=a.wetdry,
        )
    elif a.cmd == "freeze":
        freeze(
            root=a.root,
            tag=a.tag,
            only=a.only,
            max_steps=a.max_steps,
            no_breaking=a.no_breaking,
            wetdry=a.wetdry,
        )
    elif a.cmd == "check":
        new = freeze(
            root=a.root,
            tag=f"{a.tag}__check",
            only=a.only,
            max_steps=a.max_steps,
            no_breaking=a.no_breaking,
            wetdry=a.wetdry,
        )
        compare(a=a.root / a.tag, b=new, tol=a.tol)
    elif a.cmd == "compare":
        compare(a=a.a, b=a.b, tol=a.tol)
    elif a.cmd == "sanity":
        sanity(root=a.root)


if __name__ == "__main__":
    main()
