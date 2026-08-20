Prescribed bed motion: the NetCDF exchange format
=================================================

CelerisAi can drive the bed with motion computed outside the solver -- a
landslide runout model (DAN3D, r.avaflow, ...), another simulation, or an
analytic sampler. All producers write a single intermediate NetCDF format;
the solver side (``celeris.landslide.PrescribedBedSlide``) only ever reads
this format, so converters and solver stay decoupled.

The reference implementation of the format is
``celeris.landslide.BedMotion`` -- its ``to_netcdf`` / ``from_netcdf``
methods (xarray-backed) are the normative writer and reader.

File layout
-----------

Dimensions
    ``time``, ``y``, ``x``.

Coordinate variables
    ``time(time)``
        Frame times in seconds, ``float64``, at least two entries, strictly
        increasing. ``time[0]`` is the start of the bed motion in
        simulation time.
    ``x(x)``, ``y(y)``
        Cell-centre coordinates in metres, ``float64``, in the same
        coordinate frame as the Celeris domain extents (e.g. UTM easting /
        northing, or local metres for idealized cases).

Data variable
    ``dz(time, y, x)``
        Bed-elevation change relative to the initial bed, in metres,
        ``float32``. Positive ``dz`` raises the bed (a slide deposit);
        negative carves it (a source scar). Values must be finite.

Global attributes
    ``source``
        Free-text provenance, e.g. ``"DAN3D 0.3_1000_2"``.
    ``crs``
        Coordinate reference of ``x`` / ``y``, e.g. ``"EPSG:32609"``;
        ``"local"`` for idealized frames.

Semantics in the solver
-----------------------

* Frames are resampled bilinearly onto the solver grid once, when
  ``PrescribedBedSlide`` is constructed. Cells outside the file's spatial
  coverage get ``dz = 0``.
* Between frames the bed is interpolated linearly in time; each step the
  bed change rate enters the continuity equation as a source term, exactly
  like the analytic ``MovingBodySlide``.
* The file does not need to span the simulation. After the last frame the
  bed holds ``initial + dz[-1]`` (the deposit stays) and the continuity
  source is cleared; the simulation may run arbitrarily long past
  ``time[-1]``.

Choosing the frame cadence
--------------------------

The bed is interpolated *linearly* between frames, which cross-fades two
slide positions rather than translating one. Keep the per-frame slide
displacement small compared to the slide's own length scale: coarse
cadence mainly costs generated wave amplitude (about 3 percent of peak in
the ``setrun_slide_netcdf.py`` demo, where the slide moves 80 percent of
its length per 1 s frame). When the producer's cadence is fixed (DAN3D
writes every few seconds), quantify the effect by refining an analytic
test case, as in ``tests/test_bed_motion.py``.

Writing a file
--------------

.. code-block:: python

   import numpy as np
   from celeris.landslide import BedMotion

   motion = BedMotion(
       x=np.arange(0.0, 300.0, 2.0),        # metres, domain frame
       y=np.arange(0.0, 100.0, 2.0),
       time=np.arange(0.0, 33.0, 1.0),      # seconds
       dz=dz_frames,                        # (nt, ny, nx), metres
       source="DAN3D 0.3_1000_2",
       crs="EPSG:32609",
   )
   motion.to_netcdf("slide.nc")

Reading one back and driving a solver:

.. code-block:: python

   from celeris.landslide import BedMotion, PrescribedBedSlide

   solver.landslide = PrescribedBedSlide(
       solver, BedMotion.from_netcdf("slide.nc")
   )
