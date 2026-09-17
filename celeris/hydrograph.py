"""Prescribed inflow hydrographs: an interior source and a discharge boundary.

``HydrographSource`` (interior, GLOF-style, mass at ambient velocity) and
``DischargeBoundary`` (edge inflow, river-style, water enters with the velocity
of the imposed discharge) both take a ``Q(t)`` series, e.g. from
:func:`read_discharge_series`.

A discharge series ``Q(t)`` is injected over a patch of wet cells through the
same ``LandslideDhdt`` continuity source that the moving-body slide uses
(read by ``Pass3`` / ``Pass3Bous``), so the water is added at the equation
level and mass is conserved exactly: ``d(eta)/dt = Q(t) / A_patch`` on the
patch. No momentum is injected; the flow develops under gravity, which is the
intended behaviour for a river or outburst flood entering a lake.

Attach any number of sources with ``solver.inflows.append(src)`` (different
locations must not overlap; a slide can run alongside). The object also speaks
the ``solver.landslide`` protocol (``snapshot_initial_bed`` / ``update`` /
``is_active`` / ``deactivate``) for single-source scripts written before the
``inflows`` list existed.
"""

from typing import TYPE_CHECKING, ClassVar

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
        # Celeris puts a solid wall through the CENTRE of the first interior cell
        # (index 2 and n-3): only half of such a cell is inside the domain, so a
        # source there would inject half its volume into the mirror image.
        edge = np.zeros_like(mask_np)
        edge[:3, :] = edge[-3:, :] = True
        edge[:, :3] = edge[:, -3:] = True
        if (mask_np & edge).any():
            raise ValueError("inlet mask touches the three outer rows/columns (ghost rim and wall cell)")
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
        for other in getattr(solver, "inflows", []):
            if isinstance(other, HydrographSource) and (other.mask_np & mask_np).any():
                raise ValueError(
                    "inlet mask overlaps a HydrographSource already in solver.inflows"
                )
        self.mask = ti.field(ti.i32, shape=mask_np.shape)
        self.mask.from_numpy(mask_np.astype(np.int32))
        self._cleared = False
        self._checked = False

    @property
    def volume_m3(self) -> float:
        """Total volume of the hydrograph (trapezoidal integral of Q)."""
        return float(_trapezoid(self.discharge_m3s, self.times_s))

    def rate_at(self, t: float) -> float:
        """Continuity source rate ``Q(t) / A_patch`` (m/s) at time ``t``."""
        q = float(np.interp(t, self.times_s, self.discharge_m3s, left=0.0, right=0.0))
        return q / self.area_m2

    def snapshot_initial_bed(self) -> None:
        """Protocol hook (``solver.landslide`` slot): run the wet check."""
        self.check_wet()

    def check_wet(self) -> None:
        """Refuse an inlet with dry cells; runs once, on the initial state.

        Raises:
            ValueError: If any inlet cell is dry (depth <= ``min_depth_m``).
        """
        if self._checked:
            return
        self._checked = True
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
        self.check_wet()
        self._apply(self.rate_at(t))

    def is_active(self, t: float) -> bool:
        """True until the last hydrograph sample."""
        return t <= float(self.times_s[-1])

    def deactivate(self) -> None:
        """Zero the source on the inlet (idempotent)."""
        if not self._cleared:
            self._apply(0.0)
            self._cleared = True


def inlet_mask(
    solver: "Solver",
    x_m: float,
    y_m: float,
    radius_m: float,
    min_depth_m: float = 0.0,
) -> "np.ndarray":
    """Boolean ``(nx, ny)`` mask of the wet cells within ``radius_m`` of a point.

    Coordinates are grid coordinates (cell ``i, j`` sits at ``i * dx, j * dy``;
    subtract the domain origin first). Wetness is read from the current state,
    so call it after the initial condition is set (a still lake at the datum is
    the zero state). The ghost rim and the wall cells (three outer rows and
    columns) are excluded.

    Raises:
        ValueError: If no wet cell lies inside the disk.
    """
    import numpy as np

    nx, ny = solver.nx, solver.ny
    xg, yg = np.meshgrid(
        np.arange(nx) * float(solver.dx),
        np.arange(ny) * float(solver.dy),
        indexing="ij",
    )
    disk = (xg - x_m) ** 2 + (yg - y_m) ** 2 <= radius_m**2
    depth = solver.State.to_numpy()[:, :, 0] - solver.Bottom.to_numpy()[2]
    mask = disk & (depth > max(float(solver.delta), min_depth_m))
    mask[:3, :] = mask[-3:, :] = False  # ghost rim plus the wall cell (see HydrographSource)
    mask[:, :3] = mask[:, -3:] = False
    if not mask.any():
        raise ValueError(f"no wet cell within {radius_m} m of ({x_m}, {y_m})")
    return mask


def read_discharge_series(path: "str") -> "tuple[np.ndarray, np.ndarray]":
    """Read a two-column ``time_s  discharge_m3s`` text file (TELEMAC liquid-boundary
    style: ``#`` comments and any non-numeric header lines are skipped)."""
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    if len(rows) < 2:
        raise ValueError(f"{path}: need at least two numeric rows of time, discharge")
    arr = np.asarray(rows, dtype=np.float64)
    return arr[:, 0], arr[:, 1]


class DischargeBoundary:
    """Impose a discharge time series ``Q(t)`` through one domain edge (a river).

    Requires ``BoundaryConditions(<side>=5)``. Each step the runner calls
    :meth:`update`, which stores ``Q(t)`` and the wet cross-section area of the
    first interior row/column; the boundary kernel then gives the two ghost
    cells the interior surface and ``hu = h * Q / A`` (velocity uniform over the
    wet section, as TELEMAC's default profile). The ghost cells on that edge
    must carry the channel bed, not a wall.

    Args:
        solver: A ``celeris.solver.Solver`` built with the edge set to type 5.
        side: ``"west" | "east" | "south" | "north"``.
        times_s: Strictly increasing sample times (s), length >= 2.
        discharge_m3s: Total discharge through the edge (m^3/s); positive is inflow.
            Linear interpolation, zero outside the sampled interval.
    """

    SIDES: ClassVar[dict[str, int]] = {"west": 0, "east": 1, "south": 2, "north": 3}

    def __init__(
        self,
        solver: "Solver",
        side: str,
        times_s: "np.ndarray",
        discharge_m3s: "np.ndarray",
    ) -> None:
        if side not in self.SIDES:
            raise ValueError(f"side must be one of {list(self.SIDES)}, got {side!r}")
        bc_type = {
            "west": solver.bcWest,
            "east": solver.bcEast,
            "south": solver.bcSouth,
            "north": solver.bcNorth,
        }[side]
        if int(bc_type) != 5:
            raise ValueError(
                f"{side} boundary is type {bc_type}; DischargeBoundary needs type 5"
            )
        t = np.asarray(times_s, dtype=np.float64)
        q = np.asarray(discharge_m3s, dtype=np.float64)
        if t.ndim != 1 or t.shape != q.shape or t.size < 2 or np.any(np.diff(t) <= 0.0):
            raise ValueError(
                "times_s must be strictly increasing and match discharge_m3s"
            )
        self.solver = solver
        self.side = self.SIDES[side]
        self.times_s = t
        self.discharge_m3s = q

    @property
    def volume_m3(self) -> float:
        """Total volume of the series (trapezoidal integral of Q)."""
        return float(_trapezoid(self.discharge_m3s, self.times_s))

    def discharge_at(self, t: float) -> float:
        return float(
            np.interp(t, self.times_s, self.discharge_m3s, left=0.0, right=0.0)
        )

    def update(self, t: float) -> None:
        """Store ``Q(t)`` and the current wet section area for the boundary kernel."""
        self.solver.InflowQ[self.side] = self.discharge_at(t)
        self.solver.InflowArea[self.side] = float(
            self.solver.inflow_section_area(self.side)
        )
