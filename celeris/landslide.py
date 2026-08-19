"""Moving-body landslide source (prescribed depth motion).

Port of the ``disturbanceType == 5`` mechanism from CelerisWebGPU
(``shaders/AddDisturbance.wgsl``, ``depth_motion``): a volume-normalized
super-Gaussian mound translates through the bed along a prescribed
trajectory with a tanh-smoothed displacement history. Each step the bed
update rate ``dhdt = (B_new - B_old) / dt`` is injected into the continuity
equation as a source term (read by ``Pass3`` / ``Pass3Bous``), so the moving
body forces the water at the equation level rather than by mutating state
between steps.
"""

from dataclasses import dataclass
from math import gamma
from typing import TYPE_CHECKING

import numpy as np
import taichi as ti

if TYPE_CHECKING:
    from celeris.solver import Solver


@dataclass(frozen=True)
class LandslideParams:
    """Prescribed-motion slide definition.

    The mound is ``thickness_m * exp(-((|along|/length_m)^expo
    + (|across|/width_m)^expo) / 2)``, centred at ``(x0_m, y0_m)`` at rest
    and translated ``displacement(t) = travel_distance_m
    * (1 + tanh((t - time_shift_s) / timescale_s)) / 2`` along the travel
    azimuth. Positions are in grid coordinates (``x = i * dx``); callers
    must subtract the domain origin when building the params.

    Attributes:
        thickness_m: Peak mound thickness above the initial bed (m). This is
            the calibration knob.
        length_m: Along-travel decay scale (m).
        width_m: Across-travel decay scale (m).
        x0_m: Mound centre x at rest, grid coordinates (m).
        y0_m: Mound centre y at rest, grid coordinates (m).
        azimuth_rad: Travel direction from the +x axis (rad).
        travel_distance_m: Total centre displacement (m).
        timescale_s: Tanh ramp timescale (s). Peak slide speed is
            ``travel_distance_m / (2 * timescale_s)``; see
            :func:`peak_speed_timescale`.
        time_shift_s: Time of peak slide speed (s). Defaults (when negative)
            to ``2.5 * timescale_s`` so the slide starts from rest.
        expo: Super-Gaussian exponent (2.0 = Gaussian).
        final_azimuth_rad: Travel direction at full displacement (rad). The
            azimuth interpolates linearly with displacement, matching the
            curving-trajectory option of the WebGPU shader. ``None`` keeps
            the trajectory straight.
        wet_dhdt_only: If True, suppress the continuity source in dry cells
            so the subaerial part of the slide cannot create water on land
            (the failure mode noted in the WebGPU shader comments).
        emerge_min_depth_m: Suppress the continuity source when the local
            water column is thinner than this (m). Without it, a mound
            thicker than the water it crosses pumps eta upward at bed-rise
            rate -- the cell stays marginally wet and a phantom water film
            rides the emerging slide, leaving spurious mass behind. Use a
            few metres at field scale; 0 preserves lab-scale behaviour.
        carve_scar: If True, also excavate the slide's rest-position footprint
            (the source scar) so the moving body conserves volume -- material
            is removed from the scar and deposited downslope, rather than
            only deposited. This restores the source-side drawdown (the wave
            trough) that a deposit-only mound cannot produce. Default False
            preserves the lab-scale deposit-only behaviour.
        blend_start_frac: Displacement fraction at which the mound starts
            morphing into the deposit surface (see
            :meth:`MovingBodySlide.set_deposit`). Negative (default)
            disables blending: the mound stays rigid, matching the WebGPU
            behaviour.
        blend_end_frac: Displacement fraction at which the morph completes
            (the bed is then ``initial + deposit``). Keep below the tanh
            asymptote actually reached at ``end_time_s`` (99.93 %) so the
            final bed is exactly the deposit.
    """

    thickness_m: float
    length_m: float
    width_m: float
    x0_m: float
    y0_m: float
    azimuth_rad: float
    travel_distance_m: float
    timescale_s: float
    time_shift_s: float = -1.0
    expo: float = 2.0
    final_azimuth_rad: float | None = None
    wet_dhdt_only: bool = True
    emerge_min_depth_m: float = 0.0
    carve_scar: bool = False
    blend_start_frac: float = -1.0
    blend_end_frac: float = 0.995

    def __post_init__(self) -> None:
        """Default the time shift so the slide starts from rest."""
        if self.time_shift_s < 0.0:
            object.__setattr__(self, "time_shift_s", 2.5 * self.timescale_s)

    @property
    def volume_m3(self) -> float:
        """Mound volume implied by the peak thickness and shape."""
        e = self.expo
        shape = 4.0 * 2.0 ** (2.0 / e) * gamma(1.0 / e) ** 2 / e**2
        return self.thickness_m * self.length_m * self.width_m * shape

    @property
    def end_time_s(self) -> float:
        """Time after which the slide is effectively at rest (99.9% tanh)."""
        return self.time_shift_s + 4.0 * self.timescale_s


def peak_speed_timescale(travel_distance_m: float, peak_speed_ms: float) -> float:
    """Tanh timescale giving a desired peak slide speed.

    The displacement history ``travel * (1 + tanh((t - t0)/tau)) / 2`` has
    peak speed ``travel / (2 * tau)``, so ``tau = travel / (2 * v_peak)``.

    Args:
        travel_distance_m: Total slide displacement (m).
        peak_speed_ms: Desired peak speed (m/s).

    Returns:
        Timescale ``tau`` (s).
    """
    return travel_distance_m / (2.0 * peak_speed_ms)


@ti.data_oriented
class MovingBodySlide:
    """Drives a solver's bed with a prescribed moving body.

    Owns a snapshot of the initial bed and, each step, rewrites the bed
    centre values and the ``dhdt`` continuity source on the solver. The
    caller (``Evolve_Steps``) must refresh the bed face values / near-dry
    flags and the Boussinesq tridiagonal coefficients afterwards.
    """

    def __init__(self, solver: "Solver", params: LandslideParams) -> None:
        """Bind the slide to a solver instance.

        Args:
            solver: A ``celeris.solver.Solver``; must expose ``Bottom``,
                ``State``, ``LandslideDhdt``, ``nx``, ``ny``, ``dx``, ``dy``,
                ``dt``, ``delta`` and ``precision``.
            params: The slide definition.
        """
        self.solver = solver
        self.params = params
        self.bottom_initial = ti.field(solver.precision, shape=(solver.nx, solver.ny))
        # Deposit blend target (zero thickness until set_deposit is called).
        self.deposit = ti.field(solver.precision, shape=(solver.nx, solver.ny))
        self._blend = 1 if params.blend_start_frac >= 0.0 else 0
        self._blend_f0 = float(params.blend_start_frac)
        self._blend_f1 = float(params.blend_end_frac)
        # Kernel-visible scalar copies (captured as compile-time constants).
        self._thickness = float(params.thickness_m)
        self._length = float(params.length_m)
        self._width = float(params.width_m)
        self._x0 = float(params.x0_m)
        self._y0 = float(params.y0_m)
        self._az0 = float(params.azimuth_rad)
        final_az = params.final_azimuth_rad
        self._az1 = float(self._az0 if final_az is None else final_az)
        self._travel = float(params.travel_distance_m)
        self._tau = float(params.timescale_s)
        self._tshift = float(params.time_shift_s)
        self._expo = float(params.expo)
        self._wet_only = 1 if params.wet_dhdt_only else 0
        self._carve = 1 if params.carve_scar else 0
        self._min_depth = max(float(solver.delta), params.emerge_min_depth_m)
        self._startup_guard_s = 5.0 * float(solver.dt)
        self._dx = float(solver.dx)
        self._dy = float(solver.dy)
        self._dt = float(solver.dt)
        self._delta = float(solver.delta)
        self._cleared = False
        self._deposit_set = False

    def set_deposit(self, thickness: "object") -> None:
        """Load the deposit-blend target surface.

        Args:
            thickness: ``(nx, ny)`` array of deposit thickness above the
                initial bed (m). The blend morphs the moving mound into this
                surface between ``blend_start_frac`` and ``blend_end_frac``
                of the displacement; for exact volume conservation its
                integral should equal the mound volume.

        Raises:
            ValueError: If blending is disabled or the shape mismatches.
        """
        if self._blend != 1:
            raise ValueError("set_deposit requires blend_start_frac >= 0")
        import numpy as np

        arr = np.asarray(thickness, dtype=np.float64)
        if arr.shape != (self.solver.nx, self.solver.ny):
            raise ValueError(
                f"deposit shape {arr.shape} != grid "
                f"({self.solver.nx}, {self.solver.ny})"
            )
        self.deposit.from_numpy(arr.astype(np.float32))
        self._deposit_set = True

    @ti.kernel
    def snapshot_initial_bed(self):
        """Copy the solver's current bed centres as the static reference."""
        for i, j in ti.ndrange(self.solver.nx, self.solver.ny):
            self.bottom_initial[i, j] = self.solver.Bottom[2, i, j]

    @ti.kernel
    def _apply(self, t: ti.f32):
        for i, j in ti.ndrange(self.solver.nx, self.solver.ny):
            disp = self._travel * (1.0 + ti.tanh((t - self._tshift) / self._tau)) / 2.0
            frac = disp / self._travel
            az = self._az0 + frac * (self._az1 - self._az0)
            cos_a = ti.cos(az)
            sin_a = ti.sin(az)
            xs = i * self._dx - self._x0 - disp * cos_a
            ys = j * self._dy - self._y0 - disp * sin_a
            along = xs * cos_a + ys * sin_a
            across = -xs * sin_a + ys * cos_a
            arg = (
                ti.pow(ti.abs(along) / self._length, self._expo)
                + ti.pow(ti.abs(across) / self._width, self._expo)
            ) / 2.0
            mound = 0.0
            if arg < 30.0:
                mound = self._thickness * ti.exp(-arg)
            # Volume-conserving carve: excavate the scar at the rest position
            # (mound shape fixed at displacement 0) so the slide moves mass
            # from source to deposit instead of only piling it on. Early on
            # (disp ~ 0) deposit and scar coincide and cancel -> no net change.
            scar = 0.0
            if self._carve == 1:
                xss = i * self._dx - self._x0
                yss = j * self._dy - self._y0
                along0 = xss * cos_a + yss * sin_a
                across0 = -xss * sin_a + yss * cos_a
                arg0 = (
                    ti.pow(ti.abs(along0) / self._length, self._expo)
                    + ti.pow(ti.abs(across0) / self._width, self._expo)
                ) / 2.0
                if arg0 < 30.0:
                    scar = self._thickness * ti.exp(-arg0)
            # Deposit blend: morph the rigid mound into the deposit surface
            # over [blend_f0, blend_f1] of the displacement (smoothstep), so
            # the slide decelerates INTO a real deposit instead of parking
            # its rigid shape. Both surfaces hold the same volume, so the
            # linear blend conserves volume at every instant.
            lam = 0.0
            if ti.static(self._blend == 1):
                xb = ti.min(
                    ti.max(
                        (frac - self._blend_f0) / (self._blend_f1 - self._blend_f0),
                        0.0,
                    ),
                    1.0,
                )
                lam = xb * xb * (3.0 - 2.0 * xb)
            b_old = self.solver.Bottom[2, i, j]
            b_new = (
                self.bottom_initial[i, j]
                + (1.0 - lam) * mound
                + lam * self.deposit[i, j]
                - scar
            )
            dhdt = (b_new - b_old) / self._dt
            if t < self._startup_guard_s:
                dhdt = 0.0
            if self._wet_only == 1:
                depth = self.solver.State[i, j][0] - b_old
                if depth <= self._min_depth:
                    dhdt = 0.0
                    # Keep dry cells dry under the moving bed (the solver's
                    # eta:=bed sanitation runs only after the flux passes, so
                    # a bed drop would otherwise read as a column of water).
                    if ti.abs(b_new - b_old) > 1e-6:
                        for f in ti.static(
                            [
                                self.solver.State,
                                self.solver.stateUVstar,
                                self.solver.NewState,
                                self.solver.current_stateUVstar,
                            ]
                        ):
                            f[i, j][0] = b_new
                            f[i, j][1] = 0.0
                            f[i, j][2] = 0.0
            self.solver.Bottom[2, i, j] = b_new
            self.solver.LandslideDhdt[i, j].x = dhdt

    @ti.kernel
    def clear_dhdt(self):
        """Zero the continuity source once the slide has stopped."""
        for i, j in ti.ndrange(self.solver.nx, self.solver.ny):
            self.solver.LandslideDhdt[i, j].x = 0.0

    def update(self, t: float) -> None:
        """Advance the bed to time ``t`` and refresh the continuity source.

        Args:
            t: Simulation time (s).

        Raises:
            RuntimeError: If blending is enabled but no deposit was loaded
                (the mound would silently fade to nothing).
        """
        if self._blend == 1 and not self._deposit_set:
            raise RuntimeError("blend enabled but set_deposit was never called")
        self._apply(t)

    def deactivate(self) -> None:
        """Zero the continuity source (idempotent; called after the slide stops)."""
        if not self._cleared:
            self.clear_dhdt()
            self._cleared = True

    def is_active(self, t: float) -> bool:
        """Whether the slide is still moving at time ``t``.

        Args:
            t: Simulation time (s).

        Returns:
            True while the bed must be updated each step.
        """
        return t <= self.params.end_time_s


@dataclass(frozen=True)
class BedMotion:
    """Prescribed bed motion on a rectilinear grid -- the intermediate format.

    This is the exchange format every bed-motion producer (a DAN3D converter,
    an analytic sampler, another runout model) writes and
    :class:`PrescribedBedSlide` consumes. :meth:`to_netcdf` /
    :meth:`from_netcdf` are the on-disk spec: a NetCDF file with dimensions
    ``(time, y, x)``, coordinate variables ``time`` (s, strictly increasing),
    ``x`` and ``y`` (m, in the same frame as the Celeris domain extents,
    e.g. UTM), and ``dz(time, y, x)`` (m), the bed-elevation change relative
    to the initial bed. Positive ``dz`` raises the bed (a slide deposit);
    negative carves it (a source scar).

    Attributes:
        x: ``(nx_f,)`` cell-centre x coordinates (m).
        y: ``(ny_f,)`` cell-centre y coordinates (m).
        time: ``(nt,)`` frame times (s), strictly increasing.
        dz: ``(nt, ny_f, nx_f)`` bed change per frame (m).
        source: Free-text provenance (e.g. ``"DAN3D 0.3_1000_2"``).
        crs: Coordinate reference of ``x``/``y`` (e.g. ``"EPSG:32609"``).
    """

    x: np.ndarray
    y: np.ndarray
    time: np.ndarray
    dz: np.ndarray
    source: str = ""
    crs: str = "local"

    def __post_init__(self) -> None:
        """Validate shapes and monotonicity."""
        if self.dz.shape != (self.time.size, self.y.size, self.x.size):
            raise ValueError(
                f"dz shape {self.dz.shape} != "
                f"({self.time.size}, {self.y.size}, {self.x.size})"
            )
        if self.time.size < 2 or np.any(np.diff(self.time) <= 0.0):
            raise ValueError("time must have >= 2 strictly increasing entries")
        if not np.isfinite(self.dz).all():
            raise ValueError("dz contains non-finite values")

    def to_netcdf(self, path: str) -> None:
        """Write the motion to ``path`` in the exchange format.

        Args:
            path: Output ``.nc`` file path.
        """
        import xarray as xr

        ds = xr.Dataset(
            data_vars={
                "dz": (
                    ("time", "y", "x"),
                    self.dz.astype(np.float32),
                    {
                        "units": "m",
                        "long_name": (
                            "bed elevation change relative to the initial bed"
                        ),
                    },
                )
            },
            coords={
                "time": ("time", self.time, {"units": "s"}),
                "y": ("y", self.y, {"units": "m"}),
                "x": ("x", self.x, {"units": "m"}),
            },
            attrs={"source": self.source, "crs": self.crs},
        )
        ds.to_netcdf(path, encoding={"dz": {"zlib": True}})

    @classmethod
    def from_netcdf(cls, path: str) -> "BedMotion":
        """Read a motion written by :meth:`to_netcdf`.

        Args:
            path: Input ``.nc`` file path.

        Returns:
            The loaded :class:`BedMotion`.
        """
        import xarray as xr

        # decode_timedelta=False: keep time as plain float seconds instead of
        # letting CF decoding turn units="s" into timedelta64 nanoseconds.
        with xr.open_dataset(path, decode_timedelta=False) as ds:
            return cls(
                x=ds["x"].values.astype(np.float64),
                y=ds["y"].values.astype(np.float64),
                time=ds["time"].values.astype(np.float64),
                dz=ds["dz"].values.astype(np.float32),
                source=str(ds.attrs.get("source", "")),
                crs=str(ds.attrs.get("crs", "local")),
            )


@ti.data_oriented
class PrescribedBedSlide:
    """Drives a solver's bed with frames read from a :class:`BedMotion`.

    File-driven counterpart of :class:`MovingBodySlide`: same solver hooks
    (bed rewrite plus the ``LandslideDhdt`` continuity source, with the same
    dry-cell guards) and the same four-method interface consumed by
    ``Evolve_Steps``, but the bed target comes from linear-in-time
    interpolation between two bracketing frames instead of an analytic
    mound. The frames are resampled onto the solver grid once at
    construction; only the two bracketing frames live on the device, and
    they are re-uploaded only when ``t`` crosses a frame boundary, so the
    per-step cost is one interpolation kernel.
    """

    def __init__(
        self,
        solver: "Solver",
        motion: BedMotion,
        wet_dhdt_only: bool = True,
        emerge_min_depth_m: float = 0.0,
    ) -> None:
        """Bind the motion to a solver, resampling frames onto its grid.

        Args:
            solver: A ``celeris.solver.Solver``.
            motion: The prescribed bed motion. Its ``x``/``y`` must be in
                the same frame as the domain extents (``domain.x1`` ...);
                cells outside the motion's coverage get ``dz = 0``.
            wet_dhdt_only: As :attr:`LandslideParams.wet_dhdt_only`.
            emerge_min_depth_m: As :attr:`LandslideParams.emerge_min_depth_m`.
        """
        from scipy.interpolate import RegularGridInterpolator

        self.solver = solver
        self.times = np.asarray(motion.time, dtype=np.float64)
        domain = solver.domain
        xg = float(domain.x1) + np.arange(solver.nx) * float(solver.dx)
        yg = float(domain.y1) + np.arange(solver.ny) * float(solver.dy)
        pts_x, pts_y = np.meshgrid(xg, yg, indexing="ij")
        pts = np.column_stack([pts_y.ravel(), pts_x.ravel()])
        nt = self.times.size
        self.frames = np.zeros((nt, solver.nx, solver.ny), dtype=np.float32)
        for k in range(nt):
            interp = RegularGridInterpolator(
                (motion.y, motion.x),
                motion.dz[k],
                bounds_error=False,
                fill_value=0.0,
            )
            self.frames[k] = (
                interp(pts).reshape(solver.nx, solver.ny).astype(np.float32)
            )
        self.bottom_initial = ti.field(solver.precision, shape=(solver.nx, solver.ny))
        self.frame_a = ti.field(solver.precision, shape=(solver.nx, solver.ny))
        self.frame_b = ti.field(solver.precision, shape=(solver.nx, solver.ny))
        self._loaded_k = -1
        self._wet_only = 1 if wet_dhdt_only else 0
        self._min_depth = max(float(solver.delta), emerge_min_depth_m)
        self._startup_guard_s = 5.0 * float(solver.dt)
        self._dt = float(solver.dt)
        self._cleared = False

    @ti.kernel
    def snapshot_initial_bed(self):
        """Copy the solver's current bed centres as the static reference."""
        for i, j in ti.ndrange(self.solver.nx, self.solver.ny):
            self.bottom_initial[i, j] = self.solver.Bottom[2, i, j]

    @ti.kernel
    def _apply(self, w: ti.f32, dhdt_on: ti.i32):
        for i, j in ti.ndrange(self.solver.nx, self.solver.ny):
            dz = (1.0 - w) * self.frame_a[i, j] + w * self.frame_b[i, j]
            b_old = self.solver.Bottom[2, i, j]
            b_new = self.bottom_initial[i, j] + dz
            dhdt = (b_new - b_old) / self._dt
            if dhdt_on == 0:
                dhdt = 0.0
            if self._wet_only == 1:
                depth = self.solver.State[i, j][0] - b_old
                if depth <= self._min_depth:
                    dhdt = 0.0
                    # Keep dry cells dry under the moving bed (see the same
                    # block in MovingBodySlide._apply).
                    if ti.abs(b_new - b_old) > 1e-6:
                        for f in ti.static(
                            [
                                self.solver.State,
                                self.solver.stateUVstar,
                                self.solver.NewState,
                                self.solver.current_stateUVstar,
                            ]
                        ):
                            f[i, j][0] = b_new
                            f[i, j][1] = 0.0
                            f[i, j][2] = 0.0
            self.solver.Bottom[2, i, j] = b_new
            self.solver.LandslideDhdt[i, j].x = dhdt

    @ti.kernel
    def clear_dhdt(self):
        """Zero the continuity source once the motion has ended."""
        for i, j in ti.ndrange(self.solver.nx, self.solver.ny):
            self.solver.LandslideDhdt[i, j].x = 0.0

    def _load_pair(self, k: int) -> None:
        """Upload frames ``k`` and ``k + 1`` to the device if not current."""
        if k != self._loaded_k:
            self.frame_a.from_numpy(self.frames[k])
            self.frame_b.from_numpy(self.frames[min(k + 1, len(self.frames) - 1)])
            self._loaded_k = k

    def update(self, t: float) -> None:
        """Advance the bed to time ``t`` and refresh the continuity source.

        Args:
            t: Simulation time (s). Clamped to the frame time range; after
                the last frame the bed holds the final ``dz``.
        """
        k = int(np.searchsorted(self.times, t, side="right") - 1)
        k = min(max(k, 0), len(self.times) - 2)
        self._load_pair(k)
        w = (t - self.times[k]) / (self.times[k + 1] - self.times[k])
        w = min(max(w, 0.0), 1.0)
        dhdt_on = 0 if t < self._startup_guard_s else 1
        self._apply(float(w), dhdt_on)

    def deactivate(self) -> None:
        """Zero the continuity source (idempotent; called after the motion ends)."""
        if not self._cleared:
            self.clear_dhdt()
            self._cleared = True

    def is_active(self, t: float) -> bool:
        """Whether the bed is still moving at time ``t``.

        Args:
            t: Simulation time (s).

        Returns:
            True while the bed must be updated each step. One step of grace
            past the last frame guarantees a final update lands exactly on
            the end state before the source is cleared.
        """
        return t <= float(self.times[-1]) + self._dt
