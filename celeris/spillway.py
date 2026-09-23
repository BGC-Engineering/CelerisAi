"""Level-dependent outlets: a spillway or weir drawing water from a wet patch.

``SpillwaySink`` removes water from an interior patch of cells at a rate set
by a rating ``Q(h_up, h_down)`` of the water levels on the two sides, and can
re-inject the same discharge into an outlet patch (a weir between two basins,
TELEMAC-2D's "weir" singularity; ANUGA's ``Internal_boundary_operator``). With
no outlet the water leaves the model (a spillway discharging past the domain).

Two ratings are provided: :func:`poleni` (free and drowned weir law, the one
TELEMAC-2D uses for its weirs, ``LOIDEN`` / ``LOINOY``) and :func:`rating_table`
(a stage-discharge table, e.g. a design spillway curve). Any callable
``(h_up, h_down) -> Q`` works; positive ``Q`` flows from the intake to the
outlet.

The water is moved through the ``LandslideDhdt`` continuity source that
``HydrographSource`` and the moving-body slide use, so mass is exact and no
momentum is injected. Attach with ``solver.inflows.append(sink)``; the runner
calls :meth:`update` every step.
"""

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import taichi as ti

if TYPE_CHECKING:
    from celeris.solver import Solver

Rating = Callable[[float, float], float]

G = 9.81
DROWNED_FACTOR = 2.598  # (2/3 sqrt(1/3))^-1, TELEMAC LOINOY


def poleni(sill_m: float, width_m: float, mu: float = 0.4) -> Rating:
    """Poleni weir law, free and drowned, as TELEMAC-2D's weirs.

    ``Q = mu sqrt(2g) W (h_up - sill)^1.5`` while ``h_down`` stays below
    ``sill + 2/3 (h_up - sill)``; beyond that the drowned form
    ``Q = 2.598 mu sqrt(2g) W (h_down - sill) sqrt(h_up - h_down)``. Levels
    and the sill share the model datum. ``mu`` is 0.4 to 0.5 for a real crest.
    """
    if width_m <= 0.0 or mu <= 0.0:
        raise ValueError("width_m and mu must be positive")
    k = mu * math.sqrt(2.0 * G) * width_m

    def q(h_up: float, h_down: float) -> float:
        if h_up <= h_down:
            h_up, h_down, sign = h_down, h_up, -1.0
        else:
            sign = 1.0
        head = h_up - sill_m
        if head <= 0.0:
            return 0.0
        if h_down <= sill_m + 2.0 / 3.0 * head:
            return sign * k * head**1.5
        return sign * DROWNED_FACTOR * k * (h_down - sill_m) * math.sqrt(h_up - h_down)

    return q


def rating_table(levels_m: "np.ndarray", discharge_m3s: "np.ndarray") -> Rating:
    """Stage-discharge table ``Q(h_up)``: linear between samples, zero below the
    first level, held at the last value above the last. The tailwater is
    ignored (a free-flowing spillway)."""
    h = np.asarray(levels_m, dtype=np.float64)
    q = np.asarray(discharge_m3s, dtype=np.float64)
    if h.ndim != 1 or h.shape != q.shape or h.size < 2:
        raise ValueError("levels_m and discharge_m3s must be 1-D, equal length >= 2")
    if np.any(np.diff(h) <= 0.0):
        raise ValueError("levels_m must be strictly increasing")
    if np.any(q < 0.0) or np.any(np.diff(q) < 0.0):
        raise ValueError("discharge_m3s must be non-negative and non-decreasing")

    def rating(h_up: float, h_down: float) -> float:
        return float(np.interp(h_up, h, q, left=0.0, right=q[-1]))

    return rating


def read_rating_table(path: str) -> Rating:
    """:func:`rating_table` from a two-column ``level_m  discharge_m3s`` text file."""
    from celeris.hydrograph import read_discharge_series

    return rating_table(*read_discharge_series(path))


def _patch(solver: "Solver", mask: "np.ndarray", name: str) -> "np.ndarray":
    """Validate a patch mask: right shape, non-empty, off the ghost rim and the
    wall cells (the wall passes through the centre of cells 2 and n-3)."""
    m = np.asarray(mask, dtype=bool)
    if m.shape != (solver.nx, solver.ny):
        raise ValueError(f"{name} shape {m.shape} != grid ({solver.nx}, {solver.ny})")
    if not m.any():
        raise ValueError(f"{name} has no cells")
    edge = np.zeros_like(m)
    edge[:3, :] = edge[-3:, :] = True
    edge[:, :3] = edge[:, -3:] = True
    if (m & edge).any():
        raise ValueError(f"{name} touches the three outer rows/columns")
    return m


@ti.data_oriented
class SpillwaySink:
    """Draw ``Q(h_up, h_down)`` from an intake patch, optionally into an outlet.

    Args:
        solver: A ``celeris.solver.Solver`` (``LandslideDhdt``, ``State``,
            ``Bottom``, ``nx``, ``ny``, ``dx``, ``dy``, ``dt``, ``delta``).
        mask: ``(nx, ny)`` boolean intake patch; wet when the run starts.
        rating: ``(h_up, h_down) -> Q`` in m^3/s, positive intake -> outlet.
            ``h_up`` is the mean surface over the intake, ``h_down`` over the
            outlet (or ``tailwater_m`` with no outlet).
        outlet_mask: Optional ``(nx, ny)`` patch that receives the discharge.
            Without it the water leaves the model. Reverse flow (negative
            ``Q``) is allowed only with an outlet.
        tailwater_m: Downstream level passed to the rating when there is no
            outlet (default: far below any sill, i.e. free flow).
        min_depth_m: Depth the patches must keep (m). The discharge is
            clamped so no patch cell goes below it within one step.
        smoothing_s: Time scale for relaxing ``Q`` towards the rating
            (``0`` = none). TELEMAC and ANUGA both damp their weir discharge.
    """

    def __init__(
        self,
        solver: "Solver",
        mask: "np.ndarray",
        rating: Rating,
        outlet_mask: "np.ndarray | None" = None,
        tailwater_m: float = -1.0e9,
        min_depth_m: float = 0.0,
        smoothing_s: float = 0.0,
    ) -> None:
        self.solver = solver
        self.rating = rating
        self.mask_np = _patch(solver, mask, "intake mask")
        self.outlet_np = (
            None if outlet_mask is None else _patch(solver, outlet_mask, "outlet mask")
        )
        if self.outlet_np is not None and (self.outlet_np & self.mask_np).any():
            raise ValueError("intake and outlet masks overlap")
        for other in getattr(solver, "inflows", []):
            for m in (
                getattr(other, "mask_np", None),
                getattr(other, "outlet_np", None),
            ):
                if m is not None and (m & self.mask_np).any():
                    raise ValueError(
                        "intake mask overlaps a patch already in solver.inflows"
                    )
                if (
                    m is not None
                    and self.outlet_np is not None
                    and (m & self.outlet_np).any()
                ):
                    raise ValueError(
                        "outlet mask overlaps a patch already in solver.inflows"
                    )
        cell = float(solver.dx) * float(solver.dy)
        self.area_m2 = float(self.mask_np.sum()) * cell
        self.outlet_area_m2 = (
            0.0 if self.outlet_np is None else float(self.outlet_np.sum()) * cell
        )
        self.tailwater_m = float(tailwater_m)
        self.min_depth_m = max(float(solver.delta), float(min_depth_m))
        self.smoothing_s = float(smoothing_s)
        # patch field: 1 = intake, 2 = outlet
        patch_np = self.mask_np.astype(np.int32)
        if self.outlet_np is not None:
            patch_np[self.outlet_np] = 2
        self.patch = ti.field(ti.i32, shape=patch_np.shape)
        self.patch.from_numpy(patch_np)
        # per-patch reductions: [sum eta, min depth] for intake (0:2) and outlet (2:4)
        self.stats = ti.field(ti.f32, shape=4)
        self.q_now = 0.0
        self.volume_out_m3 = 0.0
        self._checked = False
        self._cleared = False

    @ti.kernel
    def _probe(self):
        for i, j in self.patch:
            p = self.patch[i, j]
            if p > 0:
                eta = self.solver.State[i, j][0]
                depth = eta - self.solver.Bottom[2, i, j]
                ti.atomic_add(self.stats[2 * (p - 1)], eta)
                ti.atomic_min(self.stats[2 * (p - 1) + 1], depth)

    @ti.kernel
    def _apply(self, rate_in: ti.f32, rate_out: ti.f32):
        for i, j in self.patch:
            if self.patch[i, j] == 1:
                self.solver.LandslideDhdt[i, j].x = rate_in
            elif self.patch[i, j] == 2:
                self.solver.LandslideDhdt[i, j].x = rate_out

    def levels(self) -> tuple[float, float, float, float]:
        """Mean surface and minimum depth over the intake and the outlet."""
        self.stats.from_numpy(np.array([0.0, 1e30, 0.0, 1e30], dtype=np.float32))
        self._probe()
        s = self.stats.to_numpy().astype(np.float64)
        h_up = s[0] / self.mask_np.sum()
        if self.outlet_np is None:
            return h_up, s[1], self.tailwater_m, np.inf
        return h_up, s[1], s[2] / self.outlet_np.sum(), s[3]

    def check_wet(self) -> None:
        """Refuse dry patch cells; runs once, on the initial state."""
        if self._checked:
            return
        self._checked = True
        _, d_in, _, d_out = self.levels()
        for name, d in (("intake", d_in), ("outlet", d_out)):
            if d <= self.min_depth_m:
                raise ValueError(
                    f"{name} patch has a cell with depth {d:.3g} m <= "
                    f"{self.min_depth_m:g} m; a source in a dry cell is lost"
                )

    def snapshot_initial_bed(self) -> None:
        """Protocol hook (``solver.landslide`` slot): run the wet check."""
        self.check_wet()

    def update(self, t: float) -> None:
        """Set this step's discharge from the current levels."""
        self.check_wet()
        dt = float(self.solver.dt)
        h_up, d_in, h_down, d_out = self.levels()
        q = float(self.rating(h_up, h_down))
        if self.outlet_np is None:
            q = max(q, 0.0)
        if self.smoothing_s > 0.0:
            q = self.q_now + (q - self.q_now) * min(1.0, dt / self.smoothing_s)
        # clamp: no patch cell may drop below min_depth_m within this step
        if q > 0.0:
            q = min(q, max(d_in - self.min_depth_m, 0.0) * self.area_m2 / dt)
        elif q < 0.0:
            q = -min(-q, max(d_out - self.min_depth_m, 0.0) * self.outlet_area_m2 / dt)
        self.q_now = q
        self.volume_out_m3 += q * dt
        rate_out = 0.0 if self.outlet_np is None else q / self.outlet_area_m2
        self._apply(-q / self.area_m2, rate_out)
        self._cleared = False

    def is_active(self, t: float) -> bool:
        """Always on: the rating decides when the discharge is zero."""
        return True

    def deactivate(self) -> None:
        """Zero the source on both patches (idempotent)."""
        if not self._cleared:
            self._apply(0.0, 0.0)
            self.q_now = 0.0
            self._cleared = True
