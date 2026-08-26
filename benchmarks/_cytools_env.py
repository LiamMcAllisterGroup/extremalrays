# =============================================================================
#    Copyright (C) 2026  Nate MacFadden for the Liam McAllister Group
#    GPL-3.0-or-later; see LICENSE.
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description:  A pinned CYTools for the benchmarks, in its own virtualenv.
#               The comparison is against a RELEASED version, so it cannot be
#               left to whatever cytools happens to be importable: a working
#               checkout drifts, and an editable install reports the version
#               it was installed at rather than the version of its source
#               (measured here: metadata said 1.4.9 for a tree at 1.4.12+27).
#
#               The env is built once and cached. Nothing is installed into
#               the caller's environment.
# -----------------------------------------------------------------------------
# stdlib imports
import pathlib
import subprocess
import sys
import venv

PINNED = "1.4.12"

HERE = pathlib.Path(__file__).parent
ENV = HERE / f".venv-cytools-{PINNED}"


def _python(env: pathlib.Path) -> pathlib.Path:
    sub = "Scripts" if sys.platform == "win32" else "bin"
    return env / sub / ("python.exe" if sys.platform == "win32" else "python")


def interpreter(quiet: bool = False) -> str:
    """Path to a python with cytools==PINNED, building the env if needed."""
    py = _python(ENV)
    if py.exists() and installed_version(str(py)) == PINNED:
        return str(py)
    if not quiet:
        print(f"building the pinned CYTools env (cytools=={PINNED}) in "
              f"{ENV.name}; this happens once")
    venv.create(ENV, with_pip=True, clear=True)
    subprocess.run([str(py), "-m", "pip", "install", "-q", "--upgrade", "pip"],
                   check=True)
    subprocess.run([str(py), "-m", "pip", "install", "-q",
                    f"cytools=={PINNED}"], check=True)
    got = installed_version(str(py))
    if got != PINNED:
        raise RuntimeError(f"asked for cytools=={PINNED}, env reports {got}")
    return str(py)


def installed_version(py: str) -> "str | None":
    """Version of cytools in the interpreter `py`, or None if absent."""
    out = subprocess.run(
        [py, "-c", "from importlib.metadata import version;"
                   "print(version('cytools'))"],
        capture_output=True, text=True)
    return out.stdout.strip() or None


if __name__ == "__main__":
    p = interpreter()
    print(f"{p}\ncytools {installed_version(p)}")
