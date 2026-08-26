# =============================================================================
#    Copyright (C) 2026  Nate MacFadden for the Liam McAllister Group
#    GPL-3.0-or-later; see LICENSE.
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description:  Run CYTools' extremalrays on a saved ray matrix, in a fresh
#               interpreter. Spawned rather than forked: CYTools parallelises
#               with joblib, and forking a process that then starts joblib
#               workers deadlocks. A separate process is also the only way to
#               impose a wall-clock limit on a call that has none of its own.
#
#               Run:  python _cytools_driver.py rays.npy      # -> "count N"
#                     python _cytools_driver.py --import-only # startup cost
# -----------------------------------------------------------------------------
import sys


def main():
    import numpy as np
    from cytools import config
    from cytools.cone import Cone            # noqa: F401  (import is the cost)
    for arg in sys.argv:
        if arg.startswith("--threads="):     # 1 isolates joblib as a cause
            config.n_threads = int(arg.split("=")[1])
    if "--import-only" in sys.argv:
        print("count -1")
        return
    rays = np.load([a for a in sys.argv[1:] if not a.startswith("--")][0])
    print(f"count {len(Cone(rays.tolist()).extremal_rays())}")


if __name__ == "__main__":
    main()
