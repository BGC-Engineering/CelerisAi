"""setrun_slide_netcdf.py -- idealized NetCDF-driven landslide tsunami.

End-to-end demonstration of the prescribed-bed-motion pathway:

1. Build a synthetic 300 x 100 m basin (15 m deep, plane beach, dry slope).
2. Sample an analytic subaerial slide (``MovingBodySlide`` mound) into
   frames and write them to ``slide.nc`` through :class:`BedMotion` -- the
   same intermediate format a DAN3D (or any other runout model) converter
   produces.
3. Run each case: (Bouss, SWE) x (analytic slide, NetCDF slide). The two
   sources must agree, proving the file pathway reproduces the reference
   mechanism.
4. Plot spatial fields per model and gauge time series per location.

Usage (cases are independent -- pin each to a GPU and run in parallel)::

    python setrun_slide_netcdf.py prep
    CUDA_VISIBLE_DEVICES=0 python setrun_slide_netcdf.py run Bouss analytic
    CUDA_VISIBLE_DEVICES=0 python setrun_slide_netcdf.py run Bouss netcdf
    CUDA_VISIBLE_DEVICES=1 python setrun_slide_netcdf.py run SWE analytic
    CUDA_VISIBLE_DEVICES=1 python setrun_slide_netcdf.py run SWE netcdf
    python setrun_slide_netcdf.py plot

Outputs land in ``examples/SlideBasinNC/``.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import taichi as ti

from celeris.domain import BoundaryConditions, Domain, Topodata
from celeris.landslide import (
    BedMotion,
    LandslideParams,
    MovingBodySlide,
    PrescribedBedSlide,
    peak_speed_timescale,
)
from celeris.runner import Evolve
from celeris.solver import Solver

CASE_DIR = "./examples/SlideBasinNC"
MODELS = ("Bouss", "SWE")
SOURCES = ("analytic", "netcdf")

# Basin: 15 m deep at x=0, plane beach to the waterline at x=220 m, dry
# slope beyond (xyz topo convention: positive z = depth below still water).
LX, LY = 300.0, 100.0
NX, NY = 600, 200
WATERLINE_X = 220.0
DEPTH0 = 15.0

# Subaerial slide: starts on the dry slope at x=260 m and runs 120 m
# downslope into the basin.
PARAMS = LandslideParams(
    thickness_m=3.0,
    length_m=15.0,
    width_m=20.0,
    x0_m=260.0,
    y0_m=50.0,
    azimuth_rad=np.pi,
    travel_distance_m=120.0,
    timescale_s=peak_speed_timescale(120.0, 12.0),
    emerge_min_depth_m=0.5,
)

FRAME_DT = 1.0  # NetCDF frame cadence (s) -- DAN3D-like coarseness
SIM_TIME = 60.0
GAUGES_X = (30.0, 100.0, 180.0)  # deep, mid-basin, near impact (m)
SNAP_TIMES = (10.0, 20.0, 40.0)


def build_basin() -> None:
    """Write the basin topo (xyz, positive down) into the case folder."""
    xs = np.linspace(0.0, LX, 151)
    ys = np.linspace(0.0, LY, 51)
    xg, yg = np.meshgrid(xs, ys)
    z = (WATERLINE_X - xg) * (DEPTH0 / WATERLINE_X)
    np.savetxt(
        os.path.join(CASE_DIR, "basin.xyz"),
        np.column_stack([xg.ravel(), yg.ravel(), z.ravel()]),
    )


def sample_motion() -> BedMotion:
    """Sample the analytic mound into BedMotion frames on a 2 m grid."""
    x = np.arange(0.0, LX + 1.0, 2.0)
    y = np.arange(0.0, LY + 1.0, 2.0)
    xg, yg = np.meshgrid(x, y)
    times = np.arange(0.0, PARAMS.end_time_s + FRAME_DT, FRAME_DT)
    dz = np.zeros((times.size, y.size, x.size), dtype=np.float32)
    for k, t in enumerate(times):
        disp = (
            PARAMS.travel_distance_m
            * (1.0 + np.tanh((t - PARAMS.time_shift_s) / PARAMS.timescale_s))
            / 2.0
        )
        cos_a, sin_a = np.cos(PARAMS.azimuth_rad), np.sin(PARAMS.azimuth_rad)
        xs = xg - PARAMS.x0_m - disp * cos_a
        ys = yg - PARAMS.y0_m - disp * sin_a
        along = xs * cos_a + ys * sin_a
        across = -xs * sin_a + ys * cos_a
        arg = (
            np.abs(along / PARAMS.length_m) ** PARAMS.expo
            + np.abs(across / PARAMS.width_m) ** PARAMS.expo
        ) / 2.0
        dz[k] = PARAMS.thickness_m * np.exp(-arg)
    return BedMotion(x=x, y=y, time=times, dz=dz, source="analytic sample")


def npz_path(model: str, source: str) -> str:
    """Result file for one (model, source) case."""
    return os.path.join(CASE_DIR, f"result_{model}_{source}.npz")


def run_case(model: str, source: str) -> None:
    """Run one case and save gauges, snapshots and the max-eta field."""
    ti.init(arch=ti.gpu, default_fp=ti.f32)  # falls back to CPU if no GPU
    topo = Topodata(filename="basin.xyz", path=CASE_DIR, datatype="xyz")
    bc = BoundaryConditions(celeris=False, North=0, South=0, East=0, West=0)
    # Breaking + Manning friction: a 3-5 m wave shoaling onto the beach is
    # physically breaking; without the breaking model the Boussinesq run
    # goes unstable near slide arrest (analytic and file-driven alike).
    domain = Domain(
        topodata=topo,
        x1=0.0,
        x2=LX,
        y1=0.0,
        y2=LY,
        Nx=NX,
        Ny=NY,
        isManning=1,
        friction=0.03,
    )
    solver = Solver(
        domain=domain, boundary_conditions=bc, model=model, useBreakingModel=True
    )
    if source == "analytic":
        solver.landslide = MovingBodySlide(solver, PARAMS)
    else:
        solver.landslide = PrescribedBedSlide(
            solver,
            BedMotion.from_netcdf(os.path.join(CASE_DIR, "slide.nc")),
            emerge_min_depth_m=PARAMS.emerge_min_depth_m,
        )
    run = Evolve(solver=solver, maxsteps=1)
    run.Evolve_0()

    dt = float(solver.dt)
    n_steps = int(SIM_TIME / dt)
    gi = [int(x / solver.dx) for x in GAUGES_X]
    gj = solver.ny // 2
    snap_steps = {int(t / dt): k for k, t in enumerate(SNAP_TIMES)}
    gauge_eta = np.zeros((len(gi), n_steps))
    snap_eta = np.zeros((len(SNAP_TIMES), solver.nx, solver.ny))
    bed0 = solver.Bottom.to_numpy()[2, :, :]
    wet0 = bed0 < 0.0
    max_eta = np.zeros((solver.nx, solver.ny))
    for i in range(n_steps):
        run.Evolve_Steps(i)
        state = solver.State.to_numpy()[:, :, 0]
        eta = np.where(wet0, state, np.nan)
        for g, ii in enumerate(gi):
            gauge_eta[g, i] = eta[ii, gj]
        max_eta = np.fmax(max_eta, eta)
        if i in snap_steps:
            snap_eta[snap_steps[i]] = eta
    if not np.isfinite(gauge_eta).all():
        raise RuntimeError(f"{model}/{source}: non-finite gauge values")
    np.savez_compressed(
        npz_path(model, source),
        gauge_t=np.arange(n_steps) * dt,
        gauge_eta=gauge_eta,
        snap_times=np.array(SNAP_TIMES),
        snap_eta=snap_eta,
        max_eta=max_eta,
        bed0=bed0,
    )
    print(f"{model}/{source}: peak gauge eta {np.abs(gauge_eta).max():.3f} m")


def spatial_figure(model: str) -> str:
    """Snapshots and max-eta maps for one model: analytic vs NetCDF."""
    ana = np.load(npz_path(model, "analytic"))
    net = np.load(npz_path(model, "netcdf"))
    extent = (0.0, LX, 0.0, LY)
    lim = float(np.nanmax(net["max_eta"]))
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), constrained_layout=True)
    for ax, t, eta in zip(axes[0], net["snap_times"], net["snap_eta"]):
        im = ax.imshow(
            eta.T,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-lim / 2,
            vmax=lim / 2,
        )
        ax.set_title(f"NetCDF-driven eta at t = {t:.0f} s")
    fig.colorbar(im, ax=axes[0], label="eta (m)", shrink=0.8)
    for ax, (name, case) in zip(
        axes[1], (("analytic slide", ana), ("NetCDF slide", net))
    ):
        im = ax.imshow(
            case["max_eta"].T,
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=0,
            vmax=lim,
        )
        ax.set_title(f"max eta -- {name}")
    diff = net["max_eta"] - ana["max_eta"]
    im2 = axes[1, 2].imshow(
        diff.T,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-0.1 * lim,
        vmax=0.1 * lim,
    )
    axes[1, 2].set_title("max eta difference (NetCDF - analytic)")
    fig.colorbar(im, ax=axes[1, :2], label="max eta (m)", shrink=0.8)
    fig.colorbar(im2, ax=axes[1, 2], label="difference (m)", shrink=0.8)
    for ax, t, eta_a, eta_n in zip(
        axes[2], ana["snap_times"], ana["snap_eta"], net["snap_eta"]
    ):
        im3 = ax.imshow(
            (eta_n - eta_a).T,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-0.1 * lim,
            vmax=0.1 * lim,
        )
        ax.set_title(f"snapshot difference at t = {t:.0f} s")
    fig.colorbar(im3, ax=axes[2], label="difference (m)", shrink=0.8)
    for ax in axes.ravel():
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
    fig.suptitle(f"{model}: analytic vs NetCDF-driven slide", fontsize=14)
    out = os.path.join(CASE_DIR, f"spatial_{model}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def gauge_figure(g: int) -> str:
    """Time series at gauge ``g``: all four (model, source) cases."""
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    styles = {"analytic": "-", "netcdf": "--"}
    colors = {"Bouss": "tab:blue", "SWE": "tab:orange"}
    for model in MODELS:
        for source in SOURCES:
            case = np.load(npz_path(model, source))
            ax.plot(
                case["gauge_t"],
                case["gauge_eta"][g],
                linestyle=styles[source],
                color=colors[model],
                label=f"{model} / {source}",
            )
    x = GAUGES_X[g]
    depth = (WATERLINE_X - x) * (DEPTH0 / WATERLINE_X)
    ax.set_title(f"Gauge at x = {x:.0f} m (still-water depth {depth:.1f} m)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("eta (m)")
    ax.legend()
    ax.grid(alpha=0.3)
    out = os.path.join(CASE_DIR, f"gauge_x{int(x):03d}m.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "prep"
    if mode == "prep":
        os.makedirs(CASE_DIR, exist_ok=True)
        build_basin()
        sample_motion().to_netcdf(os.path.join(CASE_DIR, "slide.nc"))
        print(f"Wrote {CASE_DIR}/basin.xyz and slide.nc")
    elif mode == "run":
        run_case(sys.argv[2], sys.argv[3])
    elif mode == "plot":
        for m in MODELS:
            print("Figure:", spatial_figure(m))
        for g in range(len(GAUGES_X)):
            print("Figure:", gauge_figure(g))
    else:
        raise SystemExit(f"unknown mode: {mode}")
