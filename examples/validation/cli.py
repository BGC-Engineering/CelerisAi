"""Celeris validation CLI.

    uv run python examples/validation/cli.py --help
    uv run python examples/validation/cli.py run all
    uv run python examples/validation/cli.py regress check --tag conserving_v2 --wetdry conserving
    uv run python examples/validation/cli.py data fetch https://<blob>/celeris-validation

Every case reads and writes under one data root (``CELERIS_VALIDATION_DATA``,
default ``/mnt/d/Homathko/Validation``; see ``paths.py``). ``data pack`` builds
the archives and a sha256 manifest for upload; ``data fetch`` downloads and
verifies them on another machine.
"""

import hashlib
import json
import os
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

import click

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paths import data_root

PY = [sys.executable]
ENV = {
    **os.environ,
    "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH") or "/usr/lib/wsl/lib",
}


def _run(args: list[str], cwd: Path = HERE.parent.parent) -> None:
    click.secho("$ " + " ".join(str(a) for a in args), fg="cyan")
    subprocess.run([*PY, *args], cwd=cwd, env=ENV, check=True)


@click.group()
def cli() -> None:
    """Validation cases, regression baselines and data transfer for CelerisAi."""


# --------------------------------------------------------------------------- run
@cli.group()
def run() -> None:
    """Run validation cases (figures and metrics land in each case's results/)."""


@run.command()
@click.option(
    "--inflow", type=click.Choice(["boundary", "source", "both"]), default="both"
)
@click.option("--wetdry", multiple=True, default=("legacy", "conserving"))
@click.option("--theta", multiple=True, type=float, default=(2.0,))
@click.option("--dx", type=float, default=0.05)
def bump(inflow, wetdry, theta, dx) -> None:
    """TELEMAC bump: exact flow over a crest, discharge boundary and interior source."""
    for mode in ("boundary", "source") if inflow == "both" else (inflow,):
        _run(
            [
                HERE / "bump/bump_validation.py",
                "--inflow",
                mode,
                "--wetdry",
                *wetdry,
                "--theta",
                *map(str, theta),
                "--dx",
                str(dx),
                "--tag",
                f"_{mode}",
            ]
        )


@run.command()
@click.option("--inflow", type=click.Choice(["source", "boundary"]), default="source")
@click.option(
    "--wetdry", type=click.Choice(["legacy", "conserving"]), default="conserving"
)
def breach(inflow, wetdry) -> None:
    """TELEMAC breach channel, dyke intact: hydrograph routing and overtopping."""
    out = data_root() / "celeris" / f"breach_{inflow}_{wetdry}"
    s = HERE / "breach/breach_validation.py"
    _run([s, "prep", "--out", out])
    _run([s, "run", "--out", out, "--wetdry", wetdry, "--inflow", inflow])
    _run([s, "compare", "--out", out])


@run.command()
@click.option(
    "--wetdry", type=click.Choice(["legacy", "conserving"]), default="conserving"
)
@click.option("--seed-wetting", type=float, default=0.0)
def malpasset(wetdry, seed_wetting) -> None:
    """Malpasset 1959 dam break against lab gauges and TELEMAC."""
    d = HERE / "malpasset"
    tag = f"_{wetdry}"
    if not (data_root() / "celeris/malpasset/grid.npz").exists():
        _run([d / "prep_malpasset.py"])
    _run(
        [
            d / "run_malpasset.py",
            "--wetdry",
            wetdry,
            "--seed-wetting",
            str(seed_wetting),
            "--tag",
            tag,
        ]
    )
    _run([d / "compare_malpasset.py", "--tag", tag])


@run.command()
def dambreak() -> None:
    """Stoker and Ritter analytic dam breaks, both schemes (CPU)."""
    _run([HERE / "dambreak/dambreak_validation.py"])


@run.command(name="all")
@click.pass_context
def run_all(ctx) -> None:
    """Every case with default settings, then the regression check."""
    ctx.invoke(dambreak)
    ctx.invoke(bump)
    ctx.invoke(breach)
    ctx.invoke(malpasset)
    ctx.invoke(regress_check, tag="conserving_v2", wetdry="conserving", tol=1e-6)
    ctx.invoke(regress_check, tag="legacy_v2", wetdry="legacy", tol=1e-6)


# ----------------------------------------------------------------------- regress
@cli.group()
def regress() -> None:
    """Regression baselines: Tracy Arm, coastal examples, breach, Malpasset."""


REGRESS = HERE / "regression/regress.py"


@regress.command()
@click.option("--tag", required=True)
@click.option("--wetdry", type=click.Choice(["legacy", "conserving"]), default="legacy")
@click.option("--only", multiple=True)
def freeze(tag, wetdry, only) -> None:
    """Run every case and store fields + metrics under baseline/<tag>."""
    _run(
        [
            REGRESS,
            "freeze",
            "--tag",
            tag,
            "--wetdry",
            wetdry,
            *(["--only", *only] if only else []),
        ]
    )


@regress.command(name="check")
@click.option("--tag", required=True, help="baseline tag to compare against")
@click.option("--wetdry", type=click.Choice(["legacy", "conserving"]), default="legacy")
@click.option("--tol", type=float, default=1e-6)
def regress_check(tag, wetdry, tol) -> None:
    """Rerun every case and compare with baseline/<tag> (PASS/FAIL table)."""
    _run([REGRESS, "check", "--tag", tag, "--wetdry", wetdry, "--tol", str(tol)])


@regress.command()
@click.argument("tag_a")
@click.argument("tag_b")
@click.option("--tol", type=float, default=1e-3)
def compare(tag_a, tag_b, tol) -> None:
    """Compare two frozen baselines without running anything."""
    root = data_root() / "baseline"
    _run(
        [
            REGRESS,
            "compare",
            "--a",
            root / tag_a,
            "--b",
            root / tag_b,
            "--tol",
            str(tol),
        ]
    )


@regress.command()
@click.argument("tag_a")
@click.argument("tag_b")
def plot(tag_a, tag_b) -> None:
    """Per-case maps: max depth of A, of B, and B minus A."""
    _run([REGRESS, "plot", "--a", tag_a, "--b", tag_b])


# -------------------------------------------------------------------------- data
BUNDLE = "celeris_validation.tar"
PACKS = {  # archive name: paths under the data root
    "telemac_examples": ["telemac_examples"],
    "telemac_ref": ["telemac"],
    "baselines": [
        "baseline/legacy_det",
        "baseline/conserving3",
        "baseline/legacy_v2",
        "baseline/conserving_v2",
    ],
}
EXAMPLES_NEEDED = {  # TELEMAC example folder: files the cases read
    "breach": [
        "geo_breach.slf",
        "ini_breach.slf",
        "geo_breach.cli",
        "t2d_breach.liq",
        "doc/breach.tex",
    ],
    "bump": [
        "geo_bump.slf",
        "analytic_sol.py",
        "f2d_bumptrans-hllc.slf",
        "f2d_bumpsub-hllc.slf",
        "doc/bump.tex",
    ],
    "malpasset": [
        "geo_malpasset-large.slf",
        "geo_malpasset-large.cli",
        "user_fortran/user_condin_h.f",
        "user_fortran/user_utimp_telemac2d.f",
        "doc/malpasset.tex",
    ],
    "dambreak": [
        "geo_ritter.slf",
        "user_fortran/user_condin_h.f",
        "user_fortran2/user_condin_h.f",
        "doc/dambreak.tex",
    ],
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@cli.group()
def data() -> None:
    """Pack, fetch and verify the validation data (blob storage friendly)."""


@data.command()
@click.option(
    "--dest",
    type=click.Path(path_type=Path),
    default=None,
    help="where to write the archives (default <root>/pack)",
)
@click.option(
    "--telemac-examples",
    type=click.Path(path_type=Path),
    default=Path.home() / "telemac-mascaret/examples/telemac2d",
    show_default=True,
)
@click.option(
    "--bundle/--no-bundle",
    default=True,
    help="also write one celeris_validation.tar holding the archives and the manifest",
)
def pack(dest, telemac_examples, bundle) -> None:
    """Copy the needed TELEMAC example inputs into the root, then build .tar.gz archives + manifest.json (+ one bundle)."""
    root = data_root()
    dest = dest or root / "pack"
    dest.mkdir(parents=True, exist_ok=True)
    ex_root = root / "telemac_examples"
    for name, files in EXAMPLES_NEEDED.items():
        for rel in files:
            src, dst = telemac_examples / name / rel, ex_root / name / rel
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(src.read_bytes())
    manifest = {"root_layout": "see examples/validation/paths.py", "archives": {}}
    for name, members in PACKS.items():
        present = [m for m in members if (root / m).exists()]
        if not present:
            click.secho(f"skip {name}: nothing present", fg="yellow")
            continue
        arc = dest / f"{name}.tar.gz"
        with tarfile.open(arc, "w:gz") as tf:
            for m in present:
                tf.add(root / m, arcname=m)
        manifest["archives"][name] = {
            "file": arc.name,
            "sha256": _sha256(arc),
            "bytes": arc.stat().st_size,
            "members": present,
        }
        click.echo(f"{arc.name}: {arc.stat().st_size / 1e6:.1f} MB")
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
    click.secho(f"manifest: {dest / 'manifest.json'}", fg="green")
    if bundle:
        bundle_path = dest / BUNDLE
        with tarfile.open(bundle_path, "w") as tf:  # archives are already gzipped
            tf.add(dest / "manifest.json", arcname="manifest.json")
            for meta in manifest["archives"].values():
                tf.add(dest / meta["file"], arcname=meta["file"])
        (dest / (BUNDLE + ".sha256")).write_text(_sha256(bundle_path) + "\n")
        click.secho(
            f"bundle: {bundle_path} ({bundle_path.stat().st_size / 1e6:.0f} MB), sha256 in {BUNDLE}.sha256",
            fg="green",
        )


@data.command()
@click.argument("url")
@click.option(
    "--only",
    multiple=True,
    help="archive names to extract (default all in the manifest)",
)
def fetch(url, only) -> None:
    """Restore the validation data from URL.

    URL is either the single bundle (a URL with SAS query, or a local path, whose
    name contains ``celeris_validation.tar``) or a base URL/folder that holds
    ``manifest.json`` and the archives. Verifies sha256 and extracts into the root.
    """
    root = data_root()
    pack_dir = root / "pack"
    pack_dir.mkdir(parents=True, exist_ok=True)
    is_bundle = BUNDLE in url.split("?")[0]
    if is_bundle:
        local = Path(url) if Path(url).exists() else pack_dir / BUNDLE
        if local != Path(url):
            click.echo(f"fetching {BUNDLE}")
            urllib.request.urlretrieve(url, local)
        with tarfile.open(local, "r") as tf:
            tf.extractall(pack_dir, filter="data")
        manifest = json.loads((pack_dir / "manifest.json").read_text())
    else:
        base = url.rstrip("/")
        with urllib.request.urlopen(f"{base}/manifest.json") as r:
            manifest = json.loads(r.read())
    for name, meta in manifest["archives"].items():
        if only and name not in only:
            continue
        arc = pack_dir / meta["file"]
        if not is_bundle:
            click.echo(f"fetching {meta['file']} ({meta['bytes'] / 1e6:.1f} MB)")
            urllib.request.urlretrieve(f"{base}/{meta['file']}", arc)
        got = _sha256(arc)
        if got != meta["sha256"]:
            raise click.ClickException(
                f"{meta['file']}: sha256 mismatch ({got} != {meta['sha256']})"
            )
        with tarfile.open(arc, "r:gz") as tf:
            tf.extractall(root, filter="data")
        click.secho(f"{name}: verified and extracted", fg="green")


@data.command()
@click.option(
    "--manifest",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="default <root>/pack/manifest.json",
)
def verify(manifest) -> None:
    """Check that the archives under <root>/pack match manifest.json."""
    root = data_root()
    manifest = manifest or root / "pack/manifest.json"
    m = json.loads(Path(manifest).read_text())
    bad = 0
    for name, meta in m["archives"].items():
        arc = root / "pack" / meta["file"]
        ok = arc.exists() and _sha256(arc) == meta["sha256"]
        bad += not ok
        click.secho(
            f"{name}: {'ok' if ok else 'MISSING OR CHANGED'}",
            fg="green" if ok else "red",
        )
    if bad:
        raise click.ClickException(f"{bad} archive(s) failed")


if __name__ == "__main__":
    cli()
