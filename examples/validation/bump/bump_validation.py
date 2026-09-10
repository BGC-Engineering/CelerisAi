"""TELEMAC-2D ``bump`` example: steady flow over a bump against the exact solution.

Frictionless 20 x 2 m channel, bed ``0.25 exp(-(x-10)^2/2)``. Transcritical
(Q = 0.45 m3/s, outlet surface 0.35 m: critical on the crest, hydraulic jump
downstream) and subcritical (Q = 1.5 m3/s, outlet 0.8 m). The crest acts as the
control that sets the head for a given discharge, the same physics as flow over
a dyke or dam crest. Celeris holds the upstream surface at the exact value and the outlet
surface at the prescribed one (two columns each), imposes no discharge, and
the steady discharge the crest passes is compared with q0, starts from a flat surface, and runs to steady
state; the profile is compared with TELEMAC's exact solution
(``analytic_sol.py``) and its HLLC result.

    uv run python examples/validation/bump/bump_validation.py [--regime trans|sub]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "malpasset"))
sys.path.insert(0, str(HERE.parent))
from paths import celeris_out, telemac_example, telemac_ref

TELEMAC_BUMP = telemac_example("bump")
sys.path.insert(0, str(TELEMAC_BUMP))
from analytic_sol import BumpAnalyticSol
from read_selafin import read_selafin

DATUM_M = 1.0
DX_M = 0.05  # overridden by --dx
PAD = 2
L_M, W_M = 20.0, 2.0
REGIMES = {  # flow, Q (m3/s), outlet surface (m), TELEMAC HLLC result file
    "trans": ("trans", 0.45, 0.35, "f2d_bumptrans-hllc.slf"),
    "sub": ("sub", 1.5, 0.8, "f2d_bumpsub-hllc.slf"),
}


def bed_fn(x: np.ndarray) -> np.ndarray:
    return 0.25 * np.exp(-0.5 * (x - 10.0) ** 2)


def run(
    regime: str,
    wetdry: str,
    duration_s: float,
    out: Path,
    inflow: str = "boundary",
    theta: float = 2.0,
) -> dict[str, np.ndarray]:
    import taichi as ti

    from celeris.domain import BoundaryConditions, Domain, Topodata
    from celeris.hydrograph import DischargeBoundary, HydrographSource
    from celeris.runner import Evolve
    from celeris.solver import Solver

    flow, q_total, z_out, _ = REGIMES[regime]
    sol = BumpAnalyticSol(flow=flow, Q=q_total, hl=z_out, bottom_function="exponential")
    sol()
    nx, ny = round(L_M / DX_M) + 2 * PAD, round(W_M / DX_M) + 2 * PAD
    xc = (np.arange(nx) - PAD + 0.5) * DX_M  # cell centres, physical x
    inside_x = (np.arange(nx) >= PAD) & (np.arange(nx) < nx - PAD)
    # The ghost rim continues the channel bed on every edge: the wall condition
    # is the reflection itself. A tall pad next to small cells would trigger the
    # solver's steep-slope Froude cap (3 / bed slope) in the adjacent rows and
    # freeze them, pushing the flow to the centreline.
    channel = np.ones((nx, ny), dtype=bool)
    bed = np.broadcast_to(bed_fn(xc)[:, None], (nx, ny)).copy()
    out.mkdir(parents=True, exist_ok=True)
    xg, yg = np.meshgrid(np.arange(nx) * DX_M, np.arange(ny) * DX_M, indexing="ij")
    np.savetxt(
        out / "bathy.xyz",
        np.column_stack([xg.ravel(), yg.ravel(), (DATUM_M - bed).ravel()]),
    )
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    dom = Domain(
        topodata=Topodata(filename="bathy.xyz", path=str(out), datatype="xyz"),
        x1=0.0,
        x2=nx * DX_M,
        y1=0.0,
        y2=ny * DX_M,
        Nx=nx,
        Ny=ny,
        isManning=0,
        friction=0.0,
        Courant=0.25,
    )
    solver = Solver(
        domain=dom,
        boundary_conditions=BoundaryConditions(
            celeris=False,
            North=0,
            East=0,
            South=0,
            West=5 if inflow == "boundary" else 0,
        ),
        model="SWE",
        infiltrationRate=0.0,
        wetdry_scheme=wetdry,
        theta=theta,  # 2 (Celeris default) oscillates behind a hydraulic jump; 1 is TVD
    )
    evolve = Evolve(solver=solver, maxsteps=1)
    state = np.zeros((nx, ny, 4), dtype=np.float32)
    eta0 = np.where(channel, z_out, bed)  # flat pool at the outlet level, at rest
    state[:, :, 0] = eta0 - DATUM_M
    solver.InitStates()
    for f in (
        solver.State,
        solver.stateUVstar,
        solver.NewState,
        solver.current_stateUVstar,
    ):
        f.from_numpy(state)
    solver.InitStates = lambda: None
    if inflow == "boundary":
        solver.inflows.append(
            DischargeBoundary(solver, "west", [0.0, 1.0e6], [q_total, q_total])
        )
    else:
        # Sources at x = 1 m as in the TELEMAC twin, clear of the wall: a strip
        # touching the first interior column leaks into the ghost rim.
        i_src = round(1.0 / DX_M) + PAD
        inlet = np.zeros((nx, ny), dtype=bool)
        inlet[i_src - 1 : i_src + 1, PAD : ny - PAD] = True
        solver.landslide = HydrographSource(
            solver, inlet, [0.0, 1.0e6], [q_total, q_total]
        )
    evolve.Evolve_0()

    eta_out = z_out - DATUM_M
    i_out0, i_out1 = nx - PAD - 2, nx - PAD
    j0, j1 = PAD, ny - PAD

    @ti.kernel
    def dirichlet():  # type: ignore[no-untyped-def]
        for i, j in ti.ndrange((i_out0, i_out1), (j0, j1)):
            solver.State[i, j][0] = eta_out
            solver.stateUVstar[i, j][0] = eta_out

    dt = float(solver.dt)
    n = int(duration_s / dt)
    for step in range(n):
        evolve.Evolve_Steps(step)
        dirichlet()
    st = solver.State.to_numpy()
    jm = ny // 2
    eta = st[:, jm, 0] + DATUM_M
    depth = eta - bed[:, jm]
    hu = st[:, jm, 1]
    return {
        "x": xc,
        "bed": bed[:, jm],
        "eta": eta,
        "depth": depth,
        "q": hu,
        "inside": inside_x,
        "n_steps": np.array(n),
        "dt": np.array(dt),
    }


TELEMAC_REF = telemac_ref()


def telemac_centreline(regime: str, inflow: str) -> dict[str, np.ndarray] | None:
    """Steady TELEMAC HLLC profile: the shipped result for the discharge-boundary
    twin, or the interior-source rerun (``bump<regime>_src``) if it exists."""
    if inflow == "boundary":
        m = read_selafin(TELEMAC_BUMP / REGIMES[regime][3])
        x, y = m["x"], m["y"]
        v = {n: m["values"][-1, k] for k, n in enumerate(m["varnames"])}
        h, fs, u = v["WATER DEPTH"], v["FREE SURFACE"], v["VELOCITY U"]
    else:
        f = TELEMAC_REF / f"bump{regime}_src" / f"bump{regime}_src.npz"
        if not f.exists():
            return None
        d = np.load(f)
        x, y = d["x"], d["y"]
        h, fs, u = d["water_depth"][-1], d["free_surface"][-1], d["velocity_u"][-1]
    sel = np.abs(y - 1.0) < 0.01  # centreline nodes
    order = np.argsort(x[sel])
    return {
        "x": x[sel][order],
        "depth": h[sel][order],
        "eta": fs[sel][order],
        "q": (h * u)[sel][order],
    }


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regime", choices=list(REGIMES), nargs="+", default=list(REGIMES))
    ap.add_argument(
        "--wetdry",
        choices=["legacy", "conserving"],
        nargs="+",
        default=["legacy", "conserving"],
    )
    ap.add_argument("--duration", type=float, default=400.0)
    ap.add_argument("--dx", type=float, default=0.05)
    ap.add_argument("--inflow", choices=["boundary", "source"], default="boundary")
    ap.add_argument(
        "--theta",
        type=float,
        nargs="+",
        default=[2.0],
        help="limiter parameter(s), one run per value",
    )
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", type=Path, default=celeris_out("bump"))
    a = ap.parse_args()
    global DX_M
    DX_M = a.dx
    rows = []
    fig, axes = plt.subplots(
        len(a.regime),
        2,
        figsize=(14, 4.6 * len(a.regime)),
        squeeze=False,
        constrained_layout=True,
    )
    for r, regime in enumerate(a.regime):
        flow, q_total, z_out, _ = REGIMES[regime]
        sol = BumpAnalyticSol(
            flow=flow, Q=q_total, hl=z_out, bottom_function="exponential"
        )
        sol()
        tel = telemac_centreline(regime, a.inflow)
        ax_h, ax_q = axes[r]
        ax_h.fill_between(sol.x, 0, sol.zb, color="0.6", label="Bed")
        ax_h.plot(sol.x, sol.E, "k-", lw=2, label="Exact")
        if tel is not None:
            ax_h.plot(tel["x"], tel["eta"], "k:", lw=1.5, label="TELEMAC-2D HLLC")
        ax_q.axhline(q_total / W_M, color="k", lw=2, label="Exact q")
        if tel is not None:
            ax_q.plot(tel["x"], tel["q"], "k:", lw=1.5, label="TELEMAC-2D HLLC")
        if tel is not None:
            tel_core = (tel["x"] > 1.0) & (tel["x"] < 19.0)
            rows.append(
                {
                    "regime": regime,
                    "model": "TELEMAC HLLC",
                    "L1_depth_m": float(
                        np.abs(np.interp(tel["x"], sol.x, sol.H) - tel["depth"])[
                            tel_core
                        ].mean()
                    ),
                    "crest_depth_m": float(np.interp(10.0, tel["x"], tel["depth"])),
                    "upstream_depth_x2_m": float(
                        np.interp(2.0, tel["x"], tel["depth"])
                    ),
                    "q_upstream": float(
                        np.mean(tel["q"][(tel["x"] > 2.0) & (tel["x"] < 8.0)])
                    ),
                    "q_downstream": float(
                        np.mean(tel["q"][(tel["x"] > 13.0) & (tel["x"] < 18.0)])
                    ),
                    "q_exact": q_total / W_M,
                }
            )
        styles = iter([("C3", "--"), ("C0", ":"), ("C2", "-."), ("C1", (0, (5, 1)))])
        for th, wd in [(th, wd) for th in a.theta for wd in a.wetdry]:
            c, ls = next(styles)
            res = run(
                regime,
                wd,
                a.duration,
                a.out / f"{regime}_{wd}_{a.inflow}_dx{a.dx:g}_th{th:g}",
                a.inflow,
                th,
            )
            x, ins = res["x"], res["inside"]
            h_ex = np.interp(x, sol.x, sol.H)
            l1 = float(np.abs(res["depth"] - h_ex)[ins & (x > 1.0) & (x < 19.0)].mean())
            qu = float(np.mean(res["q"][ins & (x > 2.0) & (x < 8.0)]))
            qd = float(np.mean(res["q"][ins & (x > 13.0) & (x < 18.0)]))
            rows.append(
                {
                    "regime": regime,
                    "model": f"Celeris {wd} theta={th:g}",
                    "L1_depth_m": l1,
                    "crest_depth_m": float(np.interp(10.0, x[ins], res["depth"][ins])),
                    "upstream_depth_x2_m": float(
                        np.interp(2.0, x[ins], res["depth"][ins])
                    ),
                    "q_upstream": qu,
                    "q_downstream": qd,
                    "q_exact": q_total / W_M,
                }
            )
            lab = f"Celeris {wd.capitalize()}, Theta = {th:g}"
            ax_h.plot(
                x[ins],
                res["eta"][ins],
                ls=ls,
                color=c,
                lw=1.6,
                label=f"{lab} (L1 {l1:.3f} m)",
            )
            ax_q.plot(x[ins], res["q"][ins], ls=ls, color=c, lw=1.6, label=lab)
        title = {
            "trans": "Transcritical (Q = 0.45 m3/s, Outlet 0.35 m)",
            "sub": "Subcritical (Q = 1.5 m3/s, Outlet 0.8 m)",
        }[regime]
        ax_h.set_title(f"{title}: Free Surface")
        ax_h.set_ylabel("Elevation (m)")
        ax_h.legend(fontsize=8)
        ax_h.grid(alpha=0.3)
        ax_q.set_title(f"{title}: Unit Discharge")
        ax_q.set_ylabel("q (m2/s)")
        ax_q.legend(fontsize=8)
        ax_q.grid(alpha=0.3)
        ax_q.set_ylim(0, 1.6 * q_total / W_M)
    for ax in axes[-1]:
        ax.set_xlabel("X (m)")
    fig.suptitle(
        f"TELEMAC Bump Example, Inflow = {a.inflow.capitalize()}: Exact Vs TELEMAC Vs Celeris (dx = {a.dx:g} m)"
    )
    fig.savefig(HERE / "results" / f"bump_profiles{a.tag}.png", dpi=130)
    import csv

    with open(HERE / "results" / f"bump_metrics{a.tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for row in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    hc = {r: (REGIMES[r][1] / W_M) ** 2 / 9.81 for r in a.regime}
    print("critical depth h_c:", {r: round(v ** (1 / 3), 4) for r, v in hc.items()})


if __name__ == "__main__":
    main()
