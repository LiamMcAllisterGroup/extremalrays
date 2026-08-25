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
# Description:  Tests of verify(): it accepts correct answers and rejects
#               incomplete, non-minimal, and malformed claims, for dense and
#               sparse input, with an inspectable report.
# -----------------------------------------------------------------------------

# external imports
import numpy as np
import pytest
from scipy.sparse import csr_matrix

# local imports
from extremal_rays import exhaustive, verify
from conftest import random_float_rays, random_pointed_rays


def test_verify_catches_wrong_answers():
    R = np.array([[1, 0], [1, 1], [0, 1], [2, 1]])
    ok, report = verify(R, [0])  # missing an extremal ray
    assert not ok and any("membership" in f for f in report["failures"])
    ok, report = verify(R, [0, 1, 2])  # including a redundant ray
    assert not ok and any("separation" in f for f in report["failures"])
    ok, report = verify(R, [0, 2])  # correct answer
    assert ok and report["failures"] == []
    assert report["worst_membership_residual"] < 1e-9
    assert report["worst_separation_margin"] > 0


def test_verify_empty_claim_fails():
    ok, report = verify(np.array([[1, 0], [0, 1]]), [])
    assert not ok and report["failures"]


def test_verify_non_representative_index_raises():
    R = np.array([[1, 0], [2, 0], [0, 1], [0, 0]])
    with pytest.raises(ValueError, match="representative"):
        verify(R, [1, 2])  # 1 duplicates 0
    with pytest.raises(ValueError, match="representative"):
        verify(R, [0, 3])  # 3 is the zero ray
    assert verify(R, [0, 2])[0]


def test_verify_accepts_repeated_indices():
    R = np.array([[1, 0], [1, 1], [0, 1]])
    assert verify(R, [0, 2, 0])[0]


@pytest.mark.parametrize("trial", range(3))
def test_verify_sparse_matches_dense(trial):
    R = random_pointed_rays(50 + trial, n=50, d=5)
    idx = exhaustive(R)
    ok_d, rep_d = verify(R, idx)
    ok_s, rep_s = verify(csr_matrix(R), idx)
    assert ok_d and ok_s
    assert rep_d["failures"] == rep_s["failures"] == []
    # a corrupted claim fails identically
    assert not verify(csr_matrix(R), idx[1:])[0]


def test_verify_float_rays():
    R = random_float_rays(9, n=40, d=4)
    idx = exhaustive(R)
    assert verify(R, idx)[0]
    redundant = [k for k in range(len(R)) if k not in idx][0]
    assert not verify(R, np.append(idx, redundant))[0]


def test_verify_verbose(capsys):
    R = np.array([[1, 0], [1, 1], [0, 1]])
    verify(R, [0, 1, 2], verbosity=1)
    out = capsys.readouterr().out
    assert "worst" in out and "FAIL" in out


def test_verify_residual_tolerance():
    # an absurdly tight tolerance rejects a correct float answer;
    # a loose one accepts it
    R = random_float_rays(3, n=40, d=4)
    idx = exhaustive(R)
    assert verify(R, idx, tol=1e-3)[0]
    assert not verify(R, idx, tol=0.0)[0]


def test_verify_time_limit_option():
    R = random_pointed_rays(5, n=40, d=4)
    idx = exhaustive(R)
    assert verify(R, idx, time_limit=30.0)[0]
