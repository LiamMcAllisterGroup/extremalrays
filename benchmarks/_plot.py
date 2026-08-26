# =============================================================================
#    Copyright (C) 2026  Nate MacFadden for the Liam McAllister Group
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description:  Plot the prior-art comparison written by
#               benchmark_prior_art.py: time to compute the extremal rays of
#               the torically inherited Mori cone against h11, one series
#               scale. A method that stops appearing hit the time limit or
#               failed; that point is marked rather than silently dropped.
#
#               Run:  python benchmarks/plot_prior_art.py
# -----------------------------------------------------------------------------
from __future__ import annotations

# stdlib imports
import argparse
import json
import os
import pathlib
import re

# external imports
import matplotlib.pyplot as plt
import numpy as np


# TeX-quality typography without requiring a LaTeX installation: Computer
# Modern for both maths and text, so the fitted laws in the legend render the
# way they would in the paper. text.usetex=True would be truer still, but it
# needs latex+dvipng on the machine, which is not assumed here
plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["cmr10", "DejaVu Serif"],
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
})

HERE = pathlib.Path(__file__).parent
DOCS = HERE.parent / "docs"

# power-law fits use only this part of the range (see the note at the fit)
FIT_FROM = 10

# one style per method; ours first and heaviest
STYLE = {
    "extremalrays": ("o", "steelblue", 1.0),
    "CYTools (per-ray LP)": ("D", "mediumpurple", 0.85),
    "lrs": ("s", "tomato", 0.85),
    "cddlib": ("^", "goldenrod", 0.85),
    "Normaliz": ("v", "seagreen", 0.85),
}


def _sci(value):
    """Coefficient to one significant figure, as mathtext."""
    if value <= 0:
        return "0"
    exp = int(np.floor(np.log10(value)))
    mant = value / 10.0 ** exp
    if round(mant) == 10:                    # 9.6 -> 1x10^{n+1}
        mant, exp = 1.0, exp + 1
    if -2 <= exp <= 2:
        return f"{float(f'{value:.1g}'):g}"
    return f"{mant:.0f}{{\\times}}10^{{{exp}}}"


def _pm(value, err):
    """Exponent and its error, the error to one significant figure and the
    value carried to the same decimal place."""
    if err <= 0:
        return f"{value:.2f}", "0"
    places = -int(np.floor(np.log10(err)))
    places = max(places, 0)
    return f"{value:.{places}f}", f"{err:.{places}f}"


def from_log(path):
    """
    Rebuild the plot data from a benchmark log.

    Lets a long run be plotted while it is still going: the log carries the
    medians (though not the per-h11 spread, so this view has no error bars).
    """
    times, ext = {}, {}
    line_re = re.compile(r"h11=\s*(\d+).*?ext~\s*(\d+)")
    for line in pathlib.Path(path).read_text().splitlines():
        m = line_re.search(line)
        if not m:
            continue
        h11, n_ext = int(m.group(1)), int(m.group(2))
        ext[str(h11)] = n_ext
        # fields are separated by two or more spaces; parse each on its own
        # so a "<resolution" marker cannot be swallowed into the next name
        for field in re.split(r"\s{2,}", line.strip()):
            m2 = re.fullmatch(r"([^:]+):\s*([\d.]+)s", field.strip())
            if not m2:
                continue
            name = m2.group(1).strip()
            if name in STYLE:
                times.setdefault(name, {})[str(h11)] = [float(m2.group(2))] * 3
    return {"extremal": ext, "times": times, "startup_subtracted": {}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(HERE.parent / "perf-work"
                                          / "prior_art.json"))
    ap.add_argument("--from-log", metavar="PATH",
                    help="read a running benchmark's log instead of its JSON")
    ap.add_argument("--out", default=str(DOCS / "benchmark_prior_art.png"))
    ap.add_argument("--title",
                    default="Extremal rays of the torically inherited "
                            "Mori cone")
    ap.add_argument("--samples", type=int, default=3,
                    help="polytopes sampled per h11 (h21 is not controlled)")
    ap.add_argument("--watch", type=float, metavar="SECONDS",
                    help="redraw every SECONDS until the source stops "
                         "changing; open the output once and let the viewer "
                         "refresh it")
    args = ap.parse_args()

    if args.watch:
        import time
        src = pathlib.Path(args.from_log or args.json)
        last, idle = None, 0
        while True:
            stamp = src.stat().st_mtime if src.exists() else None
            if stamp != last:
                last, idle = stamp, 0
                try:
                    draw(args)
                except Exception as exc:               # noqa: BLE001
                    print(f"  (skipped a redraw: {type(exc).__name__})")
            else:
                idle += 1
                if idle > 20:      # source quiet for 20 polls: run is over
                    print("source stopped changing; final redraw done")
                    return
            time.sleep(args.watch)
    draw(args)


def draw(args, rec=None):

    if rec is None:
        rec = (from_log(args.from_log) if args.from_log
               else json.loads(pathlib.Path(args.json).read_text()))
    plt.close("all")
    times = rec["times"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    # CYTools is measured in a subprocess, so its compute time is only
    # meaningful once it clears the interpreter+import cost subtracted from
    # it. Below that the points are noise pinned to a floor, and plotting
    # them invents a flat plateau and flattens the fit
    floor = rec.get("cytools_floor", 2e-3)
    if "CYTools (per-ray LP)" in times:
        times["CYTools (per-ray LP)"] = {
            k: v for k, v in times["CYTools (per-ray LP)"].items()
            if v[0] > floor}

    for name, series in times.items():
        if not series:
            continue
        marker, color, alpha = STYLE.get(name, ("x", "grey", 0.8))
        h = np.array(sorted(int(k) for k in series))
        med = np.array([series[str(k)][0] for k in h])
        lo = np.array([series[str(k)][1] for k in h])
        hi = np.array([series[str(k)][2] for k in h])
        # after subtracting process startup the fastest trial can land at or
        # below zero, which on a log axis draws a bar down to the floor and
        # reads as enormous uncertainty. Where that happens the downward
        # spread is simply not resolvable, so no lower bar is drawn
        lower = np.where(lo > 0, med - lo, 0.0)
        # On log-log axes these are close to straight, i.e. power laws
        # t ~ h^b, so fit that rather than joining consecutive points: a
        # connecting line implies structure between samples that is not
        # measured, and the exponent is the quantity worth reading off.
        # Fit only h11 >= FIT_FROM: below it the fastest methods sit near
        # the timer floor (~1 ms) and flatten out, which drags the exponent
        # well below the asymptotic slope the plot is actually about
        label = name.replace(" (per-ray LP)", "")
        # CYTools is given cones it can actually finish: any instance it
        # cannot complete inside the cap is dropped and replaced. That is a
        # thumb on the scale in its favour, so the legend says so
        dropped = rec.get("cytools_dropped", {}) if name.startswith("CYTools") \
            else {}
        # cones it could not finish were dropped and replaced, so its curve
        # is the instances it handles: flagged in the legend as "generous"
        generous = "generous; " if (dropped and sum(dropped.values())) else ""
        keep = h >= FIT_FROM
        if keep.sum() >= 3:
            # weight by each point's own spread, then rescale the covariance
            # by the reduced chi-square so the quoted error reflects the
            # scatter about the line, between-instance variation, which
            # dwarfs the repeat-timing noise
            span = 0.5 * (np.log10(np.maximum(hi[keep], 1e-12))
                          - np.log10(np.maximum(lo[keep], 1e-12)))
            have = span[span > 0]
            default = float(np.median(have)) if len(have) else 0.02
            sigma = np.where(span > 0, span, default)
            fit, cov = np.polyfit(np.log10(h[keep]), np.log10(med[keep]), 1,
                                  w=1.0 / sigma, cov=True)
            b, log_a = fit
            err = float(np.sqrt(cov[0, 0]))
            grid = np.logspace(np.log10(h[keep][0]), np.log10(h[keep][-1]), 64)
            ax.plot(grid, 10 ** log_a * grid ** b, color=color, lw=1.2,
                    alpha=0.65, zorder=1)
            bs, es = _pm(b, err)
            label = (f"{label}  ({generous}${_sci(10 ** log_a)}"
                     f"\\,h^{{{bs}\\pm{es}}}$)")
        ax.errorbar(h, med, yerr=[lower, np.maximum(hi - med, 0.0)],
                    fmt=marker, color=color, ms=5, capsize=2, alpha=alpha,
                    linestyle="none", label=label, zorder=3)

    ax.set_xlabel(r"$h^{1,1}$")
    ax.set_ylabel("time to fully prune rays [s]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", alpha=0.25)
    plural = "s" if args.samples != 1 else ""
    ax.set_title(f"{args.title}\n"
                 f"(Delaunay triangulation of {args.samples} "
                 f"arbitrary-$h^{{2,1}}$ polytope{plural})\n"
                 f"{os.cpu_count()} cores allowed to every method "
                 "(some stay serial by design)",
                 fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    # mathtext for the emphasis: matplotlib does not read markdown
    txt = (r"avg fork timing is subtracted from $\bf{other\ tools}$ "
           "(generous to them)")
    ax.text(0.98, 0.02, txt, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="dimgrey")
    plt.tight_layout()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
