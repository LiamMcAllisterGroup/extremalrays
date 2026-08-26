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
from extremalrays import exhaustive, verify
from conftest import random_float_rays, random_pointed_rays


def test_verify_catches_wrong_answers():
    R = np.array([[1, 0], [1, 1], [0, 1], [2, 1]])
    ok, report = verify(R, [0])  # missing an extremal ray
    assert not ok and any("membership" in f for f in report["failures"])
    ok, report = verify(R, [0, 1, 2])  # including a redundant ray
    assert not ok and any("redundant" in f for f in report["failures"])
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


# --- near-tolerance geometry -------------------------------------------------

# a cone whose coefficients span 1e6: ray 9 is genuinely extremal but sits so
# close to cone(the others) that its separation margin is 3.75e-08, below the
# default sep_tol. verify must still accept it (a nearly-redundant ray is a
# ray) and must still reject the answer that omits it
ILL_CONDITIONED = np.array([
    [2, -1000, 200000], [2, 30000, -200], [2, 2000, 2000], [2, -10000, -4],
    [3, 10000, -50], [2, 4000, 400], [2, 400000, -50000], [2, 50, 5000],
    [2, 1, 4], [3, 300, 300000], [1, -40, -5], [2, -300000, -300]])
ILL_TRUTH = [0, 6, 9, 11]


def test_verify_accepts_narrow_margin_extremal_ray():
    ok, report = verify(ILL_CONDITIONED, ILL_TRUTH)
    assert ok, report["failures"]
    # the margin really is below sep_tol: the escalation is what saved it
    assert report["worst_separation_margin"] < 1e-7


def test_verify_rejects_answer_missing_a_narrow_margin_ray():
    ok, report = verify(ILL_CONDITIONED, [0, 6, 11])
    assert not ok and any("membership" in f for f in report["failures"])


def test_verify_sep_tol_escalation_is_reachable():
    # with escalation disabled (sep_tol=0) the narrow ray passes on margin
    # alone; with the default it passes via the membership check. Either way
    # the verdict is the same; the parameter only chooses the evidence
    assert verify(ILL_CONDITIONED, ILL_TRUTH, sep_tol=0.0)[0]
    assert verify(ILL_CONDITIONED, ILL_TRUTH)[0]


def test_verify_accepts_supplied_w():
    R = random_pointed_rays(5, n=40, d=4)
    idx = exhaustive(R)
    assert verify(R, idx, w=[1, 0, 0, 0])[0]
    with pytest.raises(ValueError, match="not positive"):
        verify(R, idx, w=[0, 1, 0, 0])


# --- parallel audit ----------------------------------------------------------
#
# Once the claimed result is fixed, every candidate is an independent
# question, so the audit parallelises cleanly. The verdicts must be
# byte-identical to the serial ones; a faster audit that disagrees with the
# slow one is worthless

@pytest.mark.parametrize("trial", range(2))
def test_parallel_audit_matches_serial(trial):
    R = random_pointed_rays(60 + trial, n=70, d=5)
    idx = exhaustive(R)
    ok_s, rep_s = verify(R, idx)
    ok_p, rep_p = verify(R, idx, n_workers=2)
    assert ok_s and ok_p
    assert rep_s["failures"] == rep_p["failures"] == []
    assert rep_s["worst_membership_residual"] == pytest.approx(
        rep_p["worst_membership_residual"], abs=1e-12)


def test_parallel_audit_still_rejects_wrong_answers():
    R = random_pointed_rays(62, n=70, d=5)
    idx = exhaustive(R)
    assert not verify(R, idx[1:], n_workers=2)[0]          # a ray missing
    extra = [k for k in range(len(R)) if k not in idx][0]
    assert not verify(R, np.append(idx, extra), n_workers=2)[0]  # one too many


def test_parallel_audit_catches_the_near_tolerance_case():
    # the exact escalation must survive the round trip through workers
    assert verify(ILL_CONDITIONED, ILL_TRUTH, n_workers=2)[0]
    assert not verify(ILL_CONDITIONED, [0, 6, 11], n_workers=2)[0]


def test_parallel_audit_falls_back_for_sparse_input():
    # the shared-memory path is dense-only; sparse input must still work
    R = random_pointed_rays(63, n=50, d=4)
    idx = exhaustive(R)
    assert verify(csr_matrix(R), idx, n_workers=2)[0]
