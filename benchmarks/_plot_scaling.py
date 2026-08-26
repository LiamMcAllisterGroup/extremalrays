# =============================================================================
#    Copyright (C) 2026  Nate MacFadden for the Liam McAllister Group
#    GPL-3.0-or-later; see LICENSE.
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description:  Mcap scaling against the size of the problem rather than
#               against h11. Cap ray counts are not monotonic in h11 (the
#               h11=90 cap has 126,363 rays against 115,678 at h11=100), so
#               an h11 axis mixes two effects. Time versus ray count gives the
#               exponent that actually characterises a method.
#
#               Run:  python benchmarks/plot_cap_scaling.py
# -----------------------------------------------------------------------------
from __future__ import annotations

# stdlib imports
import argparse
import json
import pathlib

# external imports
import matplotlib.pyplot as plt
import numpy as np


# TeX-quality typography without requiring a LaTeX installation: Computer
# Modern for both maths and text, so the fitted laws in the legend render the
# way they would in the paper. text.usetex=True would be truer still, but it
# needs latex+dvipng on the machine, which is not assumed here.
plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["cmr10", "DejaVu Serif"],
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
})

HERE = pathlib.Path(__file__).parent
STYLE = {
    "extremal-rays": ("o", "steelblue"),
    "CYTools (per-ray LP)": ("D", "mediumpurple"),
    "lrs": ("s", "tomato"),
    "cddlib": ("^", "goldenrod"),
    "Normaliz": ("v", "seagreen"),
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(HERE.parent / "perf-work" / "caps.json"))
    ap.add_argument("--cones", default=str(HERE / "data" / "mori_caps_by_h11.npz"))
    ap.add_argument("--out", default=str(HERE.parent / "docs" / "benchmark_cap_scaling.png"))
    args = ap.parse_args()

    rec = json.loads(pathlib.Path(args.json).read_text())
    z = np.load(args.cones)
    size = {int(k.split("_")[2]): z[k].shape[0] for k in z.files
            if k.startswith("cap_h11_")}

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    summary = []
    for name, series in rec["times"].items():
        pts = sorted((size[int(h)], v[0], v[1], v[2])
                     for h, v in series.items() if int(h) in size)
        if not pts:
            continue
        n = np.array([p[0] for p in pts], dtype=float)
        t = np.array([p[1] for p in pts], dtype=float)
        lo = np.array([p[2] for p in pts], dtype=float)
        hi = np.array([p[3] for p in pts], dtype=float)
        marker, color = STYLE.get(name, ("x", "grey"))
        label = name.replace(" (per-ray LP)", "")
        if len(n) >= 3:
            # Uncertainty on each point, in the log space the fit lives in:
            # half the observed min-max spread of its repeated timings. Points
            # measured only once carry no spread of their own, so they are
            # given the median relative spread of the points that do, an
            # assumption, and the reason the quoted errors are indicative
            # rather than rigorous.
            span = 0.5 * (np.log10(np.maximum(hi, 1e-12))
                          - np.log10(np.maximum(lo, 1e-12)))
            have = span[span > 0]
            default = float(np.median(have)) if len(have) else 0.02
            sigma = np.where(span > 0, span, default)
            wfit, cov_meas = np.polyfit(np.log10(n), np.log10(t), 1,
                                        w=1.0 / sigma, cov="unscaled")
            b, log_a = wfit
            # Two error estimates, and they disagree by an order of
            # magnitude. cov="unscaled" propagates only the timing spread of
            # each point, which is tiny; but the points scatter about the
            # line far more than that, because different cones of the same
            # size genuinely differ. cov=True rescales by the reduced
            # chi-square, so the quoted error reflects that scatter. The
            # larger one is the honest number to publish.
            err_meas = float(np.sqrt(cov_meas[0, 0]))
            _, cov_scat = np.polyfit(np.log10(n), np.log10(t), 1,
                                     w=1.0 / sigma, cov=True)
            err = float(np.sqrt(cov_scat[0, 0]))
            grid = np.logspace(np.log10(n[0]), np.log10(n[-1]), 64)
            ax.plot(grid, 10 ** log_a * grid ** b, color=color, lw=1.2,
                    alpha=0.65, zorder=1)
            bs, es = _pm(b, err)
            label = (f"{name}  (${_sci(10 ** log_a)}"
                     f"\\,n^{{{bs}\\pm{es}}}$)")
            summary.append((name, b, err, err_meas, n[-1], t[-1], len(n)))
        ax.errorbar(n, t, yerr=[np.maximum(t - lo, 0), np.maximum(hi - t, 0)],
                    fmt=marker, color=color, ms=5, capsize=2,
                    linestyle="none", label=label, zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("rays in the Mcap")
    ax.set_ylabel("time to fully prune rays [s]")
    ax.grid(True, which="major", alpha=0.25)
    ax.set_title("Mcap scaling in the size of the problem", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    print("exponents (error from scatter about the fit; the "
          "measurement-only error is quoted second and is far smaller)")
    for name, b, err, err_meas, n_last, t_last, npts in summary:
        print(f"  {name:24s} t ~ n^({b:.2f} +- {err:.2f})   "
              f"[meas-only +-{err_meas:.3f}, {npts} points]"
              f"   last: {int(n_last):7d} rays in {t_last:8.1f}s")


if __name__ == "__main__":
    main()
