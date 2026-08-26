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
# Description:  Shared timing helper for the benchmark scripts: warmup +
#               repeated trials, returning the median and min/max so the
#               plots can show error bars, not a single representative run.
# -----------------------------------------------------------------------------

# stdlib imports
import time

WARMUP = 1
REPEATS = 5


def timed_median(fn, *args, warmup=WARMUP, repeats=REPEATS, max_total=6.0,
                 **kwargs):
    """
    Time ``fn(*args, **kwargs)`` over repeated trials and return statistics.

    Runs ``warmup`` untimed calls, then up to ``repeats`` timed calls,
    stopping early once the cumulative timed cost exceeds ``max_total``
    seconds so expensive points are not re-run many times.

    Returns
    -------
    median, lo, hi : float
        Median wall time and the fastest/slowest trial, for error bars.
    """
    for _ in range(warmup):
        fn(*args, **kwargs)
    ts, total = [], 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        dt = time.perf_counter() - t0
        ts.append(dt)
        total += dt
        if total >= max_total:
            break
    ts.sort()
    n = len(ts)
    median = ts[n // 2] if n % 2 else 0.5 * (ts[n // 2 - 1] + ts[n // 2])
    return median, ts[0], ts[-1]
