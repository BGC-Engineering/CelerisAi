"""Prescribed inflow hydrograph as a continuity source.

A discharge series ``Q(t)`` is injected over a patch of wet cells through the
same ``LandslideDhdt`` continuity source that the moving-body slide uses
(read by ``Pass3`` / ``Pass3Bous``), so the water is added at the equation
level and mass is conserved exactly: ``d(eta)/dt = Q(t) / A_patch`` on the
patch. No momentum is injected; the flow develops under gravity, which is the
intended behaviour for a river or outburst flood entering a lake.

The object speaks the protocol ``Evolve_Steps`` expects on ``solver.landslide``
(``snapshot_initial_bed`` / ``update`` / ``is_active`` / ``deactivate``), so
attaching it is one assignment and no solver or runner code changes.
"""

from typing import TYPE_CHECKING

import numpy as np
import taichi as ti

if TYPE_CHECKING:
    from celeris.solver import Solver

_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]  # numpy < 2


@ti.data_oriented
class HydrographSource:
    """Inject a discharge time series ``Q(t)`` over an inlet mask.

    Args:
        solver: A ``celeris.solver.Solver`` exposing ``LandslideDhdt``,
            ``State``, ``Bottom``, ``nx``, ``ny``, ``dx``, ``dy``, ``delta``.
        mask: ``(nx, ny)`` boolean array of inlet cells. Every cell must be
            wet when the run starts: ``Pass3`` zeroes the update in dry cells,
            so a source placed there silently vanishes.
        times_s: Strictly increasing sample times (s), length >= 2.
        discharge_m3s: Discharge at each sample time (m^3/s). Negative values
            act as a sink (e.g. a spillway). Linear interpolation between
            samples, zero outside the sampled interval.
        min_depth_m: Wet-cell threshold used by the start-up check (m).
            Defaults to the solver's ``delta``; use a few metres at field
            scale so the patch does not sit on a marginally wet fringe.
    """

    def __init__(
        self,
        solver: "Solver",
        mask: "np.ndarray",
        times_s: "np.ndarray",
        discharge_m3s: "np.ndarray",
        min_depth_m: float = 0.0,
    ) -> None:
        mask_np = np.asarray(mask, dtype=bool)
        if mask_np.shape != (solver.nx, solver.ny):
            raise ValueError(
                f"mask shape {mask_np.shape} != grid ({solver.nx}, {solver.ny})"
            )
        if not mask_np.any():
            raise ValueError("inlet mask has no cells")
        t = np.asarray(times_s, dtype=np.float64)
        q = np.asarray(discharge_m3s, dtype=np.float64)
        if t.ndim != 1 or t.shape != q.shape or t.size < 2:
            raise ValueError("times_s and discharge_m3s must be 1-D, equal length >= 2")
        if np.any(np.diff(t) <= 0.0):
            raise ValueError("times_s must be strictly increasing")
        if not (np.isfinite(t).all() and np.isfinite(q).all()):
            raise ValueError("times_s and discharge_m3s must be finite")
        self.solver = solver
        self.times_s = t
        self.discharge_m3s = q
        self.mask_np = mask_np
        self.area_m2 = float(mask_np.sum()) * float(solver.dx) * float(solver.dy)
        self.min_depth_m = max(float(solver.delta), float(min_depth_m))
        self.mask = ti.field(ti.i32, shape=mask_np.shape)
        self.mask.from_numpy(mask_np.astype(np.int32))
        self._cleared = False

    @property
    def volume_m3(self) -> float:
        """Total volume of the hydrograph (trapezoidal integral of Q)."""
        return float(_trapezoid(self.discharge_m3s, self.times_s))

    def rate_at(self, t: float) -> float:
        """Continuity source rate ``Q(t) / A_patch`` (m/s) at time ``t``."""
        q = float(np.interp(t, self.times_s, self.discharge_m3s, left=0.0, right=0.0))
        return q / self.area_m2

    def snapshot_initial_bed(self) -> None:
        """Protocol hook, called once after ``InitStates``: check the inlet is wet.

        The bed never moves, so nothing is snapshotted; this is the first
        moment the initial state exists, hence the wet check lives here.

        Raises:
            ValueError: If any inlet cell is dry (depth <= ``min_depth_m``).
        """
        eta = self.solver.State.to_numpy()[:, :, 0]
        bed = self.solver.Bottom.to_numpy()[2]
        dry = self.mask_np & ((eta - bed) <= self.min_depth_m)
        if dry.any():
            raise ValueError(
                f"{int(dry.sum())} of {int(self.mask_np.sum())} inlet cells are dry "
                f"(depth <= {self.min_depth_m:g} m); a source in a dry cell is lost"
            )

    @ti.kernel
    def _apply(self, rate: ti.f32):
        # Only the patch is written, so another source (e.g. a slide) can own
        # the remaining cells.
        for i, j in self.mask:
            if self.mask[i, j] == 1:
                self.solver.LandslideDhdt[i, j].x = rate

    def update(self, t: float) -> None:
        """Set the continuity source to ``Q(t) / A_patch`` on the inlet."""
        self._apply(self.rate_at(t))

    def is_active(self, t: float) -> bool:
        """True until the last hydrograph sample."""
        return t <= float(self.times_s[-1])

    def deactivate(self) -> None:
        """Zero the source on the inlet (idempotent)."""
        if not self._cleared:
            self._apply(0.0)
            self._cleared = True
