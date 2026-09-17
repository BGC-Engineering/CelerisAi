"""Analytic dam breaks on the TELEMAC ``dambreak`` geometry: Stoker and Ritter.

16 m channel, dam at x = 8 m, 1 m upstream depth; Stoker over 0.2 m of water
(t = 1.5 s), Ritter over a dry bed (t = 1.2 s, front at 15.5 m). Celeris runs on
the CPU at dx = 0.05 m for both wet/dry schemes and is compared with the exact
solution and, when the TELEMAC reference run exists under the data root, with
TELEMAC HLLC at the same instant.

    uv run python examples/validation/dambreak/dambreak_validation.py
"""

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paths import telemac_ref

G = 9.81
X0, HL, L_M = 8.0, 1.0, 16.0
CASES = {  # name: (h_right, t_end, TELEMAC npz)
    "Stoker (Wet Bed, h_r = 0.2 m)": (0.2, 1.5, "dambreak_stoker/dambreak_stoker.npz"),
    "Ritter (Dry Bed)": (0.0, 1.2, "dambreak_ritter/dambreak_ritter.npz"),
}


def stoker_depth(x, t, hr):
    cl = np.sqrt(G * HL)

    def f(cm):
        return -8 * G * hr * cm**2 * (cl - cm) ** 2 + (cm**2 - G * hr) ** 2 * (
            cm**2 + G * hr
        )

    cm = brentq(f, np.sqrt(G * hr) * (1 + 1e-9), cl * (1 - 1e-9))
    hm, um = cm**2 / G, 2 * (cl - cm)
    s = hm * um / (hm - hr)
    xi = (x - X0) / t
    h = np.where(xi <= -cl, HL, (2 * cl - xi) ** 2 / (9 * G))
    h = np.where(xi >= um - cm, hm, h)
    u = np.where(xi <= -cl, 0.0, 2 / 3 * (xi + cl))
    u = np.where(xi >= um - cm, um, u)
    return np.where(xi >= s, hr, h), np.where(xi >= s, 0.0, u)


def ritter_depth(x, t):
    cl = np.sqrt(G * HL)
    xi = (x - X0) / t
    h = np.where(xi <= -cl, HL, (2 * cl - xi) ** 2 / (9 * G))
    u = np.where(xi <= -cl, 0.0, 2 / 3 * (xi + cl))
    dry = xi >= 2 * cl
    return np.where(dry, 0.0, h), np.where(dry, 0.0, u)


def exact(x, t, hr):
    return stoker_depth(x, t, hr) if hr > 0 else ritter_depth(x, t)


def run_celeris(scheme, hr, t_end, nx=320, ny=20):
    import taichi as ti

    from celeris.domain import BoundaryConditions, Domain, Topodata
    from celeris.runner import Evolve
    from celeris.solver import Solver

    ti.init(arch=ti.cpu, default_fp=ti.f32)
    tmp = Path(tempfile.mkdtemp())
    xs, ys = np.linspace(0, L_M, nx // 2 + 1), np.linspace(0, 1, ny // 2 + 1)
    xg, yg = np.meshgrid(xs, ys)
    np.savetxt(
        tmp / "b.xyz", np.column_stack([xg.ravel(), yg.ravel(), np.full(xg.size, HL)])
    )
    dom = Domain(
        topodata=Topodata(filename="b.xyz", path=str(tmp), datatype="xyz"),
        x1=0.0,
        x2=L_M,
        y1=0.0,
        y2=1.0,
        Nx=nx,
        Ny=ny,
    )
    bc = BoundaryConditions(celeris=False, North=0, South=0, East=0, West=0)
    solver = Solver(
        domain=dom, boundary_conditions=bc, model="SWE", wetdry_scheme=scheme
    )
    run = Evolve(solver=solver, maxsteps=1)
    run.Evolve_0()
    x = (np.arange(nx) + 0.5) * solver.dx
    eta0 = np.where(x < X0, 0.0, hr - HL)[:, None] * np.ones((1, ny))
    state = np.zeros((nx, ny, 4), dtype=np.float32)
    state[:, :, 0] = eta0
    for f in (
        solver.State,
        solver.stateUVstar,
        solver.NewState,
        solver.current_stateUVstar,
    ):
        f.from_numpy(state)
    dt = float(solver.dt)
    n = round(t_end / dt)
    for i in range(n):
        run.Evolve_Steps(i)
    st = solver.State.to_numpy()
    depth = st[:, ny // 2, 0] - solver.Bottom.to_numpy()[2][:, ny // 2]
    u = np.where(depth > 1e-3, st[:, ny // 2, 1] / np.maximum(depth, 1e-3), 0.0)
    return x, depth, u, n * dt


def telemac_profile(rel, var, t_want):
    f = telemac_ref() / rel
    if not f.exists():
        return None
    d = np.load(f)
    k = int(np.argmin(np.abs(d["times"] - t_want)))
    sel = np.abs(d["y"] - 0.225) < 0.12
    order = np.argsort(d["x"][sel])
    return d["x"][sel][order], d[var][k][sel][order], float(d["times"][k])


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    rows = []
    xe = np.linspace(0, L_M, 1600)
    for r, (name, (hr, t_end, tel)) in enumerate(CASES.items()):
        ax_h, ax_u = axes[r]
        for scheme, c in (("legacy", "C3"), ("conserving", "C0")):
            x, h, u, t_act = run_celeris(scheme, hr, t_end)
            h_ex, _ = exact(x, t_act, hr)
            sl = slice(4, -4)
            l1 = float(np.abs(h[sl] - h_ex[sl]).mean())
            rows.append({"case": name, "model": f"Celeris {scheme}", "L1_depth_m": l1})
            ax_h.plot(
                x,
                h,
                "--",
                color=c,
                label=f"Celeris {scheme.capitalize()} (L1 {l1:.3f} m)",
            )
            ax_u.plot(x, u, "--", color=c, label=f"Celeris {scheme.capitalize()}")
        h_e, u_e = exact(xe, t_end, hr)
        ax_h.plot(xe, h_e, "k-", lw=2, label="Exact")
        ax_u.plot(xe, u_e, "k-", lw=2, label="Exact")
        tp = telemac_profile(tel, "water_depth", t_end)
        if tp is not None:
            xt, ht, tt = tp
            he_t, _ = exact(xt, tt, hr)
            l1_t = float(np.abs(he_t - ht)[(xt > 0.2) & (xt < 15.8)].mean())
            rows.append({"case": name, "model": "TELEMAC HLLC", "L1_depth_m": l1_t})
            ax_h.plot(xt, ht, "k:", lw=1.5, label=f"TELEMAC-2D HLLC (L1 {l1_t:.3f} m)")
            xu, ut, _ = telemac_profile(tel, "velocity_u", t_end)
            ax_u.plot(xu, ut, "k:", lw=1.5, label="TELEMAC-2D HLLC")
        ax_h.set_title(f"{name}: Depth At T = {t_end} s")
        ax_u.set_title(f"{name}: Velocity At T = {t_end} s")
        ax_h.set_ylabel("Depth (m)")
        ax_u.set_ylabel("Velocity (m/s)")
        for ax in (ax_h, ax_u):
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("X (m)")
    fig.suptitle(
        "Analytic Dam-Break Cases On The TELEMAC Geometry (16 m Channel, Dam At X = 8 m, Celeris dx = 0.05 m)"
    )
    fig.tight_layout()
    fig.savefig(HERE / "results" / "dambreak_analytic.png", dpi=130)
    with open(HERE / "results" / "dambreak_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
