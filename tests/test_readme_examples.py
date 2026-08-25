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
import numpy as np

# local imports
import extremal_rays
from extremal_rays import exhaustive, sample, verify
from conftest import random_pointed_rays

README = pathlib.Path(__file__).resolve().parents[1] / "README.md"


def test_readme_usage_snippet(tmp_path, monkeypatch):
    src = README.read_text()
    block = re.search(r"```python\n(.*?)```", src, re.S).group(1)
    assert "exhaustive(R)" in block and "verify(R, idx)" in block
    R = random_pointed_rays(0, n=40, d=4)
    np.save(tmp_path / "rays.npy", R)
    monkeypatch.chdir(tmp_path)
    ns = {}
    exec(block, ns)
    assert ns["ok"], ns["report"]["failures"]
    assert np.array_equal(ns["ext"], R[ns["idx"]])
    assert set(ns["some"].tolist()) <= set(ns["idx"].tolist())
    assert ns["curve"].shape[1] == 2


def test_public_api_and_version():
    assert set(extremal_rays.__all__) == {"exhaustive", "positive_functional",
                                          "sample", "verify"}
    assert extremal_rays.__version__ == "0.3.0"
    assert callable(exhaustive) and callable(sample) and callable(verify)


def test_readme_version_matches_pyproject():
    pyproject = (README.parent / "pyproject.toml").read_text()
    assert f'version = "{extremal_rays.__version__}"' in pyproject
