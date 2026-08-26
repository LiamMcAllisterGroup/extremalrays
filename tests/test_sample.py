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
# Description:  Tests of sample(): the certified-subset guarantee, discovery
#               curve shape, determinism, option coverage, and agreement
#               between sparse and dense input.
# -----------------------------------------------------------------------------

# external imports
import numpy as np
import pytest
from scipy.sparse import csr_matrix

# local imports
from extremalrays import exhaustive, sample
from extremalrays import core
from conftest import grid_face, random_pointed_rays


def test_sample_is_certified_subset():
    R = random_pointed_rays(4, n=60, d=5)
    true = set(exhaustive(R).tolist())
    idx, curve = sample(R, work=500)
    assert set(idx.tolist()) <= true
    assert len(idx) > 0
    assert (np.diff(idx) > 0).all()
    assert (np.diff(curve[:, 1]) >= 0).all()  # monotone discovery
    assert (np.diff(curve[:, 0]) > 0).all()   # work strictly increases
    assert curve[0].tolist() == [0, 0] and curve[-1, 0] >= 500


@pytest.mark.parametrize("trial", range(4))
def test_sample_subset_random(trial):
    R = random_pointed_rays(40 + trial, n=50 + 10 * trial, d=4 + trial % 2)
    true = set(exhaustive(R).tolist())
    idx, _ = sample(R, work=400, rng_seed=trial)
    assert set(idx.tolist()) <= true


def test_sample_saturates_on_small_cone():
    # a simplicial cone with a few redundant rays: enough work finds all of it
    R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                  [1, 1, 1], [2, 1, 3], [1, 2, 0]])
    idx, _ = sample(R, work=400)
    assert idx.tolist() == [0, 1, 2]


def test_sample_tie_grid_certifies_only_corners():
    R, corners = grid_face(5)
    idx, _ = sample(R, work=600)
    assert set(idx.tolist()) <= set(corners)


def test_sample_sparse_matches_dense():
    R = random_pointed_rays(8, n=50, d=5)
    a, _ = sample(R, work=300)
    b, _ = sample(csr_matrix(R), work=300)
    assert a.tolist() == b.tolist()


def test_sample_deterministic():
    R = random_pointed_rays(2, n=50, d=4)
    a, ca = sample(R, work=300, rng_seed=3)
    b, cb = sample(R, work=300, rng_seed=3)
    assert a.tolist() == b.tolist() and (ca == cb).all()


def test_sample_duplicates_map_to_first_occurrence():
    R = random_pointed_rays(2, n=30, d=4)
    R2 = np.vstack([R, 3 * R])  # positive multiples appended
    idx, _ = sample(R2, work=300)
    assert idx.max() < len(R)


@pytest.mark.parametrize("kwargs", [
    dict(center="functional"),
    dict(targeted=False),
    dict(jitter=0.0),
    dict(n_walkers=8, stall=3),
])
def test_sample_options_stay_certified(kwargs):
    R = random_pointed_rays(6, n=60, d=5)
    true = set(exhaustive(R).tolist())
    idx, _ = sample(R, work=300, **kwargs)
    assert set(idx.tolist()) <= true


def test_sample_unknown_center_raises():
    with pytest.raises(ValueError, match="unknown center"):
        sample(random_pointed_rays(1), center="nope")


def test_sample_verbose(capsys):
    sample(random_pointed_rays(1, n=20), work=100, verbosity=1)
    assert "certified" in capsys.readouterr().out


def test_sample_margin_center_cg(monkeypatch):
    monkeypatch.setattr(core, "_CG_THRESHOLD", 20)
    R = random_pointed_rays(13, n=80, d=4)
    idx, _ = sample(R, work=300)
    assert set(idx.tolist()) <= set(exhaustive(R).tolist())


def test_sample_seeds_exhaustive():
    R = random_pointed_rays(6, n=60, d=5)
    full = exhaustive(R)
    seeds, _ = sample(R, work=200)
    assert len(seeds) > 0
    assert exhaustive(R, known=seeds).tolist() == full.tolist()
