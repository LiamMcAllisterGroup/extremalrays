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
# Description:  The README usage snippet, run as documented on a small cone
#               so the public API it shows keeps working.
# -----------------------------------------------------------------------------

# stdlib imports
import pathlib
import re

# external imports

# local imports
import extremal_rays
from extremal_rays import exhaustive, sample, verify

README = pathlib.Path(__file__).resolve().parents[1] / "README.md"


def test_readme_usage_snippet():
    """The README's example must run exactly as printed.

    It previously loaded a `rays.npy` the repo does not ship, and this test
    manufactured that file before exec'ing the block -- so the test passed
    while the documented snippet failed for every reader. The snippet is now
    self-contained, and nothing is set up for it here on purpose.
    """
    block = re.search(r"```python\n(.*?)```", README.read_text(), re.S).group(1)
    assert "np.load" not in block, "the example must not depend on absent files"
    ns = {}
    exec(block, ns)                      # no fixtures, no chdir, no setup
    assert ns["idx"].tolist() == [0, 3]  # the values the README prints
    assert ns["ext"].tolist() == [[1, 0], [0, 1]]
    assert ns["ok"] is True, ns["report"]["failures"]
    assert set(ns["some"].tolist()) <= set(ns["idx"].tolist())
    assert ns["curve"].shape[1] == 2


def test_readme_documents_the_extras_it_tells_you_to_use():
    # the README told readers to run pytest after an install that does not
    # provide it; the install section must name the extra that does
    text = README.read_text()
    assert 'pip install -e ".[test]"' in text
    assert "[tool.setuptools.packages.find]" not in text


def test_public_api_and_version():
    assert set(extremal_rays.__all__) == {"exhaustive", "positive_functional",
                                          "sample", "verify"}
    # the literal lives in pyproject.toml only (see the sync test below)
    assert re.fullmatch(r"\d+\.\d+\.\d+", extremal_rays.__version__)
    assert callable(exhaustive) and callable(sample) and callable(verify)


def test_readme_version_matches_pyproject():
    pyproject = (README.parent / "pyproject.toml").read_text()
    assert f'version = "{extremal_rays.__version__}"' in pyproject
