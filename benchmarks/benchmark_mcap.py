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
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description:  Mori-cone CAP benchmark against the prior art, and the two
#               figures it produces: time against h11, and time against the
#               number of rays. The second is the more honest view: cap size
#               is not monotonic in h11 (h11=90 gives 126,363 rays against
#               115,678 at h11=100), so an h11 axis mixes size with dimension.
#
#               Caps come from make_caps.py; everything else is measured here.
#
#               Run:  python benchmarks/benchmark_mcap.py
#                     python benchmarks/benchmark_mcap.py --plot-only
# -----------------------------------------------------------------------------
from __future__ import annotations

# stdlib imports
import argparse
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).parent
CAPS = HERE / "data" / "mori_caps_by_h11.npz"
JSON = HERE.parent / "perf-work" / "caps.json"
DOCS = HERE.parent / "docs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--plot-only", action="store_true",
                    help="redraw the figures from the existing results")
    args = ap.parse_args()

    if not args.plot_only:
        subprocess.run(
            [sys.executable, str(HERE / "benchmark_mori_cone.py"),
             "--cones", str(CAPS), "--timeout", str(args.timeout),
             "--json", str(JSON)], check=True)

    # time against h11
    subprocess.run(
        [sys.executable, str(HERE / "_plot.py"), "--json", str(JSON),
         "--out", str(DOCS / "benchmark_cap.png"),
         "--title", "Extremal rays of the Mcap", "--samples", "1"], check=True)
    # time against problem size
    subprocess.run(
        [sys.executable, str(HERE / "_plot_scaling.py"), "--json", str(JSON),
         "--cones", str(CAPS),
         "--out", str(DOCS / "benchmark_cap_scaling.png")], check=True)


if __name__ == "__main__":
    main()
