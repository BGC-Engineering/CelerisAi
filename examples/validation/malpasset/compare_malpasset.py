"""Malpasset metrics: Celeris vs lab gauges (doc/malpasset.tex) and TELEMAC-2D reference runs.

Usage: python compare_malpasset.py [--workdir DIR] [--telemac /mnt/d/Homathko/Validation/telemac] [--tag NAME]
Reads <workdir>/results<tag>.npz; writes results/malpasset_max_depth<tag>.csv, malpasset_arrival<tag>.csv,
malpasset_comparison<tag>.png next to this script.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

HERE = Path(__file__).resolve().parent
WORKDIR = Path("/mnt/d/Homathko/Validation/celeris/malpasset")
TELEMAC = Path("/mnt/d/Homathko/Validation/telemac")
ARRIVAL_DEPTH = 0.05                       # m: arrival = first time depth exceeds this
GAUGES = ["P6", "P7", "P8", "P9", "P10", "P11", "P12", "P13", "P14"]
MEASURED = dict(zip(GAUGES, [40.3, 14.6, 24.0, 12.8, 11.8, 8.3, 10.1, 6.8, 5.4]))   # doc/malpasset.tex table
OBSERVED_AB, OBSERVED_AC = 1140.0, 1320.0                                          # transformer shutdowns (s)
# TELEMAC rfo_malpasset-hllc.txt (small mesh, HLLC), threshold H > 1e-4 m in user_utimp_telemac2d.f
TELEMAC_RFO = {"hllc": dict(A=97.2, AB=1127.8, AC=1369.3)}
COL = dict(measured="#444444", telemac="#2a78d6", celeris="#eb6834")


def arrival(t, depth, thr=ARRIVAL_DEPTH):
    k = np.nonzero(depth > thr)[0]
    return float(t[k[0]]) if len(k) else np.nan


def telemac_metrics(npz, names, xs, ys):
    """Max depth and arrival per point from a TELEMAC npz (linear interpolation on the triangulation)."""
    d = np.load(npz)
    tri = mtri.Triangulation(d["x"], d["y"], d["ikle"])
    finder = tri.get_trifinder()
    nearest = [int(np.argmin(np.hypot(d["x"] - x, d["y"] - y))) for x, y in zip(xs, ys)]
    depth = np.array([mtri.LinearTriInterpolator(tri, fs - d["bottom"], trifinder=finder)(xs, ys).filled(np.nan)
                      for fs in d["free_surface"]])                     # (nt, npts)
    depth_nn = d["water_depth"][:, nearest]
    t = d["times"]
    return dict(t=t, depth=depth, hmax=np.nanmax(depth, axis=0), hmax_nearest=depth_nn.max(axis=0),
                arrival={n: arrival(t, depth[:, k]) for k, n in enumerate(names)},
                bed=mtri.LinearTriInterpolator(tri, d["bottom"], trifinder=finder)(xs, ys).filled(np.nan))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=Path, default=WORKDIR)
    ap.add_argument("--telemac", type=Path, default=TELEMAC)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    out = HERE / "results"; out.mkdir(exist_ok=True)

    r = np.load(args.workdir / f"results{args.tag}.npz")
    g = np.load(args.workdir / "grid.npz")
    names = [str(n) for n in r["gauge_names"]]
    gi, gj = r["gauge_i"], r["gauge_j"]
    bed = r["bed"].astype(float)
    t, eta = r["t"], r["gauge_eta"]
    depth = eta - bed[gi, gj]
    xs, ys = g["gauge_x"].astype(float), g["gauge_y"].astype(float)
    idx = {n: k for k, n in enumerate(names)}

    tel = {}
    for tag in ("hllc", "fine"):
        f = args.telemac / f"malpasset_{tag}" / f"malpasset_{tag}.npz"
        if f.exists():
            tel[tag] = telemac_metrics(f, names, xs, ys)
            print(f"TELEMAC {tag}: {f}")

    # ---- max depth table
    rows = []
    for n in GAUGES:
        k = idx[n]
        row = dict(gauge=n, x=int(xs[k]), y=int(ys[k]), bed_celeris_m=round(bed[gi[k], gj[k]], 2),
                   measured_m=MEASURED[n], celeris_m=round(float(r["hmax"][gi[k], gj[k]]), 2),
                   celeris_solver_envelope_m=round(float(r["hmax_solver"][gi[k], gj[k]]), 2))
        for tag, m in tel.items():
            row[f"telemac_{tag}_m"] = round(float(m["hmax"][k]), 2)
            row[f"telemac_{tag}_nearest_node_m"] = round(float(m["hmax_nearest"][k]), 2)
        rows.append(row)
    with open(out / f"malpasset_max_depth{args.tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

    # ---- arrival times
    arr = {n: arrival(t, depth[:, idx[n]]) for n in ("A", "B", "C")}
    arows = [dict(quantity="arrival_A_s", observed="", celeris=round(arr["A"], 1)),
             dict(quantity="A_to_B_s", observed=OBSERVED_AB, celeris=round(arr["B"] - arr["A"], 1)),
             dict(quantity="A_to_C_s", observed=OBSERVED_AC, celeris=round(arr["C"] - arr["A"], 1))]
    for tag, m in tel.items():
        a = m["arrival"]
        vals = [a["A"], a["B"] - a["A"], a["C"] - a["A"]]
        for row, v in zip(arows, vals):
            row[f"telemac_{tag}_frames20s"] = round(float(v), 1)
        if tag in TELEMAC_RFO:
            for row, key in zip(arows, ("A", "AB", "AC")):
                row[f"telemac_{tag}_rfo"] = TELEMAC_RFO[tag][key]
    arows.append(dict(quantity="arrival_threshold_m", observed="", celeris=ARRIVAL_DEPTH))
    with open(out / f"malpasset_arrival{args.tag}.csv", "w", newline="") as f:
        keys = list(dict.fromkeys(k for a in arows for k in a))
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(arows)

    # ---- figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw=dict(width_ratios=[2.2, 1]))
    xg = np.arange(len(GAUGES)); wbar = 0.26
    series = [("Physical Model (1964)", [MEASURED[n] for n in GAUGES], COL["measured"])]
    if "hllc" in tel:
        series.append(("TELEMAC-2D HLLC", [tel["hllc"]["hmax"][idx[n]] for n in GAUGES], COL["telemac"]))
    series.append(("Celeris SWE", [rows[i]["celeris_m"] for i in range(len(GAUGES))], COL["celeris"]))
    off = (np.arange(len(series)) - (len(series) - 1) / 2) * wbar
    for o, (lab, v, c) in zip(off, series):
        ax1.bar(xg + o, v, wbar - 0.03, color=c, label=lab, linewidth=0)
    ax1.set_xticks(xg, GAUGES); ax1.set_ylabel("Maximum Water Depth (m)")
    ax1.set_title("Maximum Depth At Gauges P6-P14"); ax1.legend(frameon=False)
    ax1.spines[["top", "right"]].set_visible(False); ax1.grid(axis="y", color="#e5e5e5", linewidth=0.6); ax1.set_axisbelow(True)

    labels = ["A To B", "A To C"]; xa = np.arange(2)
    aser = [("Observed", [OBSERVED_AB, OBSERVED_AC], COL["measured"])]
    if "hllc" in tel:
        aser.append(("TELEMAC-2D HLLC", [TELEMAC_RFO["hllc"]["AB"], TELEMAC_RFO["hllc"]["AC"]], COL["telemac"]))
    aser.append(("Celeris SWE", [arr["B"] - arr["A"], arr["C"] - arr["A"]], COL["celeris"]))
    off = (np.arange(len(aser)) - (len(aser) - 1) / 2) * wbar
    for o, (lab, v, c) in zip(off, aser):
        b = ax2.bar(xa + o, v, wbar - 0.03, color=c, label=lab, linewidth=0)
        ax2.bar_label(b, fmt="%.0f", fontsize=8, color="#333333", padding=2)
    ax2.set_xticks(xa, labels); ax2.set_ylabel("Travel Time (s)")
    ax2.set_ylim(0, 1.35 * max(max(v) for _, v, _ in aser))
    ax2.set_title(f"Wave Travel Time (Depth > {ARRIVAL_DEPTH:g} m)"); ax2.legend(frameon=False, fontsize=8, loc="upper left")
    ax2.spines[["top", "right"]].set_visible(False); ax2.grid(axis="y", color="#e5e5e5", linewidth=0.6); ax2.set_axisbelow(True)
    fig.suptitle(f"Malpasset Dam Break: Celeris (dx = {float(r['dx']):g} m, Courant {float(r['courant']):g}, "
                 f"n = {float(r['manning_n']):.4f}, Seed Wetting {float(r['seed_wetting']) if 'seed_wetting' in r else 0:g} m) Vs Reference Data", fontsize=11)
    fig.tight_layout(); fig.savefig(out / f"malpasset_comparison{args.tag}.png", dpi=150); plt.close(fig)

    print(f"{'gauge':6s} {'bed':>6s} {'measured':>9s} {'celeris':>8s} " + " ".join(f"{'tel_' + k:>9s}" for k in tel))
    for row, n in zip(rows, GAUGES):
        print(f"{n:6s} {row['bed_celeris_m']:6.2f} {row['measured_m']:9.1f} {row['celeris_m']:8.2f} "
              + " ".join(f"{row[f'telemac_{k}_m']:9.2f}" for k in tel))
    for row in arows:
        print(row)
    print(f"volume: t0 {r['volume'][0] / 1e6:.3f} Mm3, end {r['volume'][-1] / 1e6:.3f} Mm3 "
          f"({100 * (r['volume'][-1] / r['volume'][0] - 1):+.1f} %); wall {float(r['wall_s']):.0f} s for {t[-1]:.0f} s "
          f"({t[-1] / float(r['wall_s']):.1f}x realtime), dt {float(r['dt']):.4f} s, {int(r['nsteps'])} steps")


if __name__ == "__main__":
    main()
