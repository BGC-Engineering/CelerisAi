"""Where the validation cases find their data.

Everything large lives outside the repo under one root, selected by the
``CELERIS_VALIDATION_DATA`` environment variable (default
``/mnt/d/Homathko/Validation``):

    <root>/telemac/           TELEMAC-2D reference runs (npz, csv, cas, logs)
    <root>/telemac_examples/  the TELEMAC example inputs the cases read (meshes,
                              cli, analytic_sol.py); used when no TELEMAC
                              install is present
    <root>/celeris/           Celeris case outputs (grids, results, frames)
    <root>/baseline/<tag>/    regression baselines (fields.npz, metrics.json)

The TELEMAC example folders are taken from a local install first
(``CELERIS_TELEMAC_EXAMPLES``, default ``~/telemac-mascaret/examples/telemac2d``)
and from ``<root>/telemac_examples`` otherwise.
"""

import os
from pathlib import Path

DEFAULT_ROOT = Path("/mnt/d/Homathko/Validation")
DEFAULT_TELEMAC_EXAMPLES = Path.home() / "telemac-mascaret/examples/telemac2d"


def data_root() -> Path:
    return Path(os.environ.get("CELERIS_VALIDATION_DATA", DEFAULT_ROOT))


def telemac_ref() -> Path:
    return data_root() / "telemac"


def celeris_out(case: str) -> Path:
    return data_root() / "celeris" / case


def baseline_root() -> Path:
    return data_root() / "baseline"


def telemac_example(name: str) -> Path:
    """Folder of one TELEMAC-2D example (``breach``, ``bump``, ``malpasset``, ``dambreak``)."""
    local = (
        Path(os.environ.get("CELERIS_TELEMAC_EXAMPLES", DEFAULT_TELEMAC_EXAMPLES))
        / name
    )
    if local.exists():
        return local
    packed = data_root() / "telemac_examples" / name
    if packed.exists():
        return packed
    raise FileNotFoundError(
        f"TELEMAC example {name!r} not found at {local} or {packed}; set "
        "CELERIS_TELEMAC_EXAMPLES or fetch the validation data (cli.py data fetch)"
    )
