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
# Description:  Tests of exhaustive(): known cones, degenerate and invalid
#               input, invariances (order, scaling, unimodular change of
#               basis, sparse vs dense), option coverage, and randomized
#               cross-checks against brute-force references.
# -----------------------------------------------------------------------------

# stdlib imports
import os

# external imports
import numpy as np
import pytest
from scipy.sparse import csr_matrix

# local imports
from extremal_rays import exhaustive, verify
from extremal_rays import core
from conftest import (brute_force_indices, cytools_reference, grid_face,
                      random_float_rays, random_pointed_rays,
                      random_unimodular)


# --- known cones -------------------------------------------------------------

def test_quadrant_2d():
    R = [[0, 1], [1, 3], [1, 1], [3, 1], [1, 0]]
    assert sorted(exhaustive(R)) == [0, 4]


def test_duplicates_and_scalings():
    # duplicated directions (exact copies and positive multiples) collapse
    # to the first occurrence
    R = [[1, 0], [2, 0], [1, 1], [0, 3], [0, 1], [1, 0]]
    assert sorted(exhaustive(R)) == [0, 3]


def test_zero_ray_dropped():
    R = [[0, 0], [1, 0], [0, 1], [1, 1]]
    assert sorted(exhaustive(R)) == [1, 2]


def test_single_ray():
    assert exhaustive([[2, 3, 5]]).tolist() == [0]
    assert exhaustive([[2, 3, 5], [4, 6, 10]]).tolist() == [0]
    assert exhaustive([[0, 0, 0], [2, 3, 5]]).tolist() == [1]


def test_simplicial_3d():
    R = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [2, 1, 3]]
    assert sorted(exhaustive(R)) == [0, 1, 2]


def test_all_extremal_cube():
    # cube vertices lifted to the slice x0 = 1: all 8 are extremal
    cube = [(1, a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
    assert sorted(exhaustive(cube)) == list(range(8))


def test_moment_curve_all_extremal():
    # points on the moment curve are in convex position: every ray extremal
    t = np.arange(-6, 7)
    R = np.column_stack([np.ones_like(t), t, t**2, t**3])
    assert exhaustive(R).tolist() == list(range(len(t)))


def test_2d_fan_is_angular_extremes():
    # in the plane, exactly the two angularly extreme rays are extremal
    rng = np.random.default_rng(0)
    ang = rng.uniform(-1.2, 1.2, 30)
    R = np.column_stack([np.cos(ang), np.sin(ang)])
    expected = sorted([int(ang.argmin()), int(ang.argmax())])
    assert sorted(exhaustive(R)) == expected


def test_tie_stress_grid_face():
    R, corners = grid_face(5)
    assert sorted(exhaustive(R)) == corners


def test_tie_stress_grid_face_no_seeding():
    R, corners = grid_face(4)
    assert sorted(exhaustive(R, seed_shots=0)) == corners


def test_cleanup_off_is_generating_superset():
    # without cleanup the answer may be non-minimal but still generates
    R, corners = grid_face(6)
    idx = exhaustive(R, cleanup=False)
    assert set(corners) <= set(idx.tolist())
    # every dropped ray still has a membership certificate in cone(R[idx])
    _, report = verify(R, idx)
    assert not [f for f in report["failures"] if "membership" in f]


def test_lower_dimensional_cone():
    # rays spanning only a 2-plane inside R^3 (cone not full-dimensional)
    R = np.array([[1, 0, 0], [1, 1, -1], [1, 2, -2], [1, -1, 1], [1, 3, -3]])
    idx = exhaustive(R)
    assert sorted(idx) == [3, 4]
    assert verify(R, idx)[0]


@pytest.mark.parametrize("M", [1e2, 1e9, 1e14])
def test_squashed_extremal_ray_not_dropped(M):
    # regression: the separation value must be scale-free. With the slice
    # functional w = (M, 1-M, 0), rays with x = y sit at |p| ~ 1 while the
    # extremal ray (1, 0, 0) is squashed to |p| = 1/M; unnormalized, its
    # separation value fell below tol at M ~ 1e9 and it was dropped (the
    # same mechanism lost an extremal ray of the h11=491 Mori cone, where
    # the LP's own w is that anisotropic)
    R = np.array([[1, 1, 0], [1, 1, 1], [1, 1, -1], [3, 3, 1], [2, 2, -1],
                  [5, 5, 0], [1, 0, 0]])
    w = np.array([M, 1 - M, 0])
    assert exhaustive(R, w=w, seed_shots=0).tolist() == [1, 2, 6]
    assert exhaustive(R, seed_shots=0).tolist() == [1, 2, 6]


def test_negative_orthant():
    # pointedness must not assume a positive coordinate
    R = -np.array([[1, 0], [1, 1], [0, 1], [2, 3]])
    assert sorted(exhaustive(R)) == [0, 2]


def test_large_integers_exact():
    b = 10**9
    R = np.array([[b, 0], [b, b], [0, b], [b + 1, b], [b, b + 1]])
    assert sorted(exhaustive(R)) == [0, 2]


def test_mixed_scale_floats():
    R = random_float_rays(5, n=40, d=4)
    R *= 10.0 ** np.random.default_rng(1).uniform(-3, 3, len(R))[:, None]
    assert exhaustive(R).tolist() == brute_force_indices(R)


# --- invalid input -----------------------------------------------------------

def test_non_pointed_raises():
    with pytest.raises(ValueError, match="not pointed"):
        exhaustive([[1, 0], [-1, 0], [0, 1]])


def test_full_space_raises():
    with pytest.raises(ValueError, match="not pointed"):
        exhaustive(np.vstack([np.eye(3, dtype=int), -np.eye(3, dtype=int)]))


@pytest.mark.parametrize("bad", [np.zeros((3, 2), int), np.zeros((0, 2))])
def test_no_nonzero_rays_raises(bad):
    with pytest.raises(ValueError, match="nonzero"):
        exhaustive(bad)


@pytest.mark.parametrize("bad", [[1, 2, 3], [[]], np.zeros((2, 2, 2))])
def test_malformed_input_raises(bad):
    with pytest.raises(ValueError, match="2d"):
        exhaustive(bad)


def test_known_non_representative_raises():
    R = np.array([[1, 0], [2, 0], [0, 1]])
    with pytest.raises(ValueError, match="representative"):
        exhaustive(R, known=[1])  # index 1 duplicates index 0


# --- randomized cross-checks -------------------------------------------------

@pytest.mark.parametrize("trial", range(10))
def test_random_vs_cytools_reference(trial):
    rng = np.random.default_rng(trial)
    d = int(rng.integers(3, 6))
    n = int(rng.integers(10, 50))
    R = random_pointed_rays(trial, n=n, d=d)
    idx = exhaustive(R)
    assert idx.dtype.kind == "i" and (np.diff(idx) > 0).all()
    assert {tuple(r) for r in R[idx]} == cytools_reference(R)
    ok, report = verify(R, idx)
    assert ok, report["failures"]


@pytest.mark.parametrize("trial", range(6))
def test_random_floats_vs_brute_force(trial):
    R = random_float_rays(trial, n=35, d=3 + trial % 3)
    idx = exhaustive(R)
    assert idx.tolist() == brute_force_indices(R)
    assert verify(R, idx)[0]


@pytest.mark.parametrize("trial", range(4))
def test_random_high_redundancy(trial):
    # many interior rays: generate rays as positive combinations of a few
    rng = np.random.default_rng(100 + trial)
    d = 5
    gens = random_pointed_rays(trial, n=12, d=d)
    lam = rng.integers(0, 4, (60, len(gens)))
    lam = lam[lam.sum(axis=1) > 0]
    R = np.vstack([gens, lam @ gens])
    g = np.gcd.reduce(np.abs(R), axis=1)
    R = np.unique(R // g[:, None], axis=0)
    assert {tuple(r) for r in R[exhaustive(R)]} == cytools_reference(R)


# --- invariances -------------------------------------------------------------

def test_seeding_disabled_matches():
    R = random_pointed_rays(7, n=30, d=4, lo=-4, hi=5)
    assert exhaustive(R, seed_shots=0).tolist() == exhaustive(R).tolist()
    assert exhaustive(R, seed_shots=3).tolist() == exhaustive(R).tolist()


def test_rng_seed_independent():
    R = random_pointed_rays(21, n=60, d=5)
    a = exhaustive(R, rng_seed=1)
    assert a.tolist() == exhaustive(R, rng_seed=2).tolist()


def test_sorted_and_unsorted_agree():
    R = random_pointed_rays(11, n=60, d=5)
    rng = np.random.default_rng(0)
    shuffled = R[rng.permutation(len(R))]
    a = {tuple(r) for r in R[exhaustive(R)]}
    b = {tuple(r) for r in shuffled[exhaustive(shuffled)]}
    c = {tuple(r) for r in
         shuffled[exhaustive(shuffled, sort_candidates=True)]}
    assert a == b == c


def test_positive_row_scaling_invariant():
    R = random_pointed_rays(3, n=50, d=4).astype(float)
    scaled = R * np.random.default_rng(0).uniform(0.1, 10, len(R))[:, None]
    assert exhaustive(scaled).tolist() == exhaustive(R).tolist()


@pytest.mark.parametrize("trial", range(3))
def test_unimodular_change_of_basis_invariant(trial):
    R = random_pointed_rays(30 + trial, n=50, d=5)
    M = random_unimodular(trial, 5)
    assert abs(round(np.linalg.det(M))) == 1
    assert exhaustive(R @ M.T).tolist() == exhaustive(R).tolist()


def test_float_input_integral_values():
    R = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    assert sorted(exhaustive(R)) == [0, 2]


def test_float_input_matches_integer_input():
    R = random_pointed_rays(2, n=50, d=5)
    assert exhaustive(R.astype(float)).tolist() == exhaustive(R).tolist()


@pytest.mark.parametrize("trial", range(5))
def test_sparse_matches_dense(trial):
    R = random_pointed_rays(trial, n=50, d=5)
    assert exhaustive(csr_matrix(R)).tolist() == exhaustive(R).tolist()


def test_sparse_float_matches_dense():
    rng = np.random.default_rng(3)
    R = np.column_stack([rng.uniform(1, 2, 40), rng.standard_normal((40, 3))])
    R[np.abs(R) < 1.2] = 0.0  # genuine sparsity, non-integral values
    R[:, 0] = np.abs(R[:, 0]) + 1
    assert exhaustive(csr_matrix(R)).tolist() == exhaustive(R).tolist()


def test_sparse_duplicates_and_zero_rows():
    R = np.array([[1, 0], [2, 0], [0, 0], [1, 1], [0, 1], [0, 3]])
    assert sorted(exhaustive(csr_matrix(R)).tolist()) == [0, 4]


def test_sparse_non_pointed_raises():
    with pytest.raises(ValueError, match="not pointed"):
        exhaustive(csr_matrix(np.array([[1, 0], [-1, 0], [0, 1]])))


def test_sparse_sort_candidates():
    R = random_pointed_rays(14, n=60, d=5)
    shuffled = csr_matrix(R[np.random.default_rng(0).permutation(len(R))])
    a = {tuple(r) for r in R[exhaustive(R)]}
    idx = exhaustive(shuffled, sort_candidates=True)
    b = {tuple(r) for r in shuffled[idx].toarray()}
    assert a == b


# --- options -----------------------------------------------------------------

def test_supplied_w_matches_lp_path():
    R = random_pointed_rays(5, n=50, d=4)
    assert exhaustive(R, w=[1, 0, 0, 0]).tolist() == exhaustive(R).tolist()
    idx_sparse = exhaustive(csr_matrix(R), w=[1, 0, 0, 0])
    assert idx_sparse.tolist() == exhaustive(R).tolist()
    with pytest.raises(ValueError, match="not positive"):
        exhaustive(R, w=[0, 1, 0, 0])


def test_known_preload_reproduces_answer():
    R = random_pointed_rays(6, n=60, d=5)
    full = exhaustive(R)
    assert exhaustive(R, known=full[:3]).tolist() == full.tolist()
    assert exhaustive(R, known=full).tolist() == full.tolist()


def test_constraint_generation_matches_direct(monkeypatch):
    R = random_pointed_rays(12, n=400, d=5)
    monkeypatch.setattr(core, "_CG_THRESHOLD", 100)
    w = core.positive_functional(R)
    assert (R @ w).min() > 0.5
    assert exhaustive(R).tolist() == exhaustive(csr_matrix(R)).tolist()


def test_constraint_generation_not_pointed(monkeypatch):
    monkeypatch.setattr(core, "_CG_THRESHOLD", 10)
    R = np.vstack([random_pointed_rays(1, n=30, d=4),
                   -random_pointed_rays(1, n=30, d=4)])
    with pytest.raises(ValueError, match="not pointed"):
        exhaustive(R)


def test_checkpoint_hint_warns(monkeypatch):
    monkeypatch.setattr(core, "_CKPT_HINT_DELAY", 0.0)
    monkeypatch.setattr(core, "_CKPT_HINT_REMAINING", 0.0)
    with pytest.warns(UserWarning, match="checkpoint"):
        exhaustive(random_pointed_rays(3, n=40, d=4))


def test_profile_populated():
    R = random_pointed_rays(4, n=50, d=4)
    idx = exhaustive(R)
    prof = core.LAST_PROFILE
    assert prof["total"] > 0
    # every redundant ray costs at least one LP
    assert prof["n_lp_main"] >= len(R) - len(idx)
    assert prof["n_shoot"] + prof["n_suspects"] >= 0
    assert set(prof) >= {"preprocess", "positive_functional", "seeding",
                         "main_separation_lp", "cleanup_membership_lp"}


def test_verbose_output(capsys):
    R = random_pointed_rays(4, n=30, d=4)
    exhaustive(R, verbosity=1)
    out = capsys.readouterr().out
    assert "seeding:" in out and "done:" in out


# --- regression fixtures -----------------------------------------------------

DATA = os.path.join(os.path.dirname(__file__), "data",
                    "mori_cap_crosscheck.npz")


@pytest.mark.parametrize("h11", [15, 20, 25])
def test_mori_cap_matches_cytools(h11):
    # Mori-cone caps: bundled ray matrices with the extremal sets a real
    # CYTools run produced (16/16 agreement across h11 = 5..25 when
    # frozen); the vendored reference must reproduce the stored answer,
    # and exhaustive must match both
    data = np.load(DATA)
    R = data[f"rays_h11_{h11}"]
    stored = {tuple(r) for r in data[f"extremal_h11_{h11}"]}
    assert cytools_reference(R) == stored
    idx = exhaustive(csr_matrix(R))
    assert {tuple(r) for r in R[idx]} == stored
    assert verify(R, idx)[0]


@pytest.mark.skipif(not os.environ.get("EXTREMAL_RAYS_SLOW"),
                    reason="~20 s; set EXTREMAL_RAYS_SLOW=1 to run")
def test_mori_h11_491_benchmark():
    path = os.path.join(os.path.dirname(__file__), os.pardir, "benchmarks",
                        "data", "mori_rays_h11_491.npz")
    rays = np.load(path)["rays"].astype(np.int64)
    assert len(exhaustive(rays)) == 884


# --- conditioning and scale (regressions) ------------------------------------

# coefficients spanning 1e6: ray 9 is genuinely extremal but its separation
# value is 3.75e-08, i.e. 0.375x the default tol, so it was silently dropped
# in the main loop; the returned set then did not generate the cone.
ILL_CONDITIONED = np.array([
    [2, -1000, 200000], [2, 30000, -200], [2, 2000, 2000], [2, -10000, -4],
    [3, 10000, -50], [2, 4000, 400], [2, 400000, -50000], [2, 50, 5000],
    [2, 1, 4], [3, 300, 300000], [1, -40, -5], [2, -300000, -300]])


def test_near_tol_ray_is_not_dropped():
    idx = exhaustive(ILL_CONDITIONED)
    assert idx.tolist() == [0, 6, 9, 11]
    assert verify(ILL_CONDITIONED, idx)[0]
    # the exact escalation is what saves it, and it is recorded
    assert core.LAST_PROFILE["n_near_tol_rescued"] >= 1
    # a tighter tolerance reaches the same answer without escalating
    assert exhaustive(ILL_CONDITIONED, tol=1e-9).tolist() == [0, 6, 9, 11]


def test_near_tol_float_input_warns():
    # float rays cannot be re-decided exactly, so the run must at least say
    # that the tolerance, not the geometry, made the call
    R = ILL_CONDITIONED * (1.0 + 1e-10)
    with pytest.warns(UserWarning, match="tol"):
        exhaustive(R)


@pytest.mark.parametrize("scale", [1e-20, 1e-12, 1e-10, 1e-8, 1e8, 1e20])
def test_float_answer_is_scale_invariant(scale):
    # scaling every ray leaves the cone unchanged, so the answer must not
    # move. An absolute integer-snapping tolerance used to collapse small
    # float rays onto integers (and raise "no nonzero ray" below ~1e-9).
    rng = np.random.default_rng(0)
    R = np.column_stack([rng.uniform(1, 2, 12), rng.standard_normal((12, 3))])
    assert exhaustive(R * scale).tolist() == exhaustive(R).tolist()


def test_mixed_scale_rows_do_not_change_the_answer():
    rng = np.random.default_rng(0)
    R = np.column_stack([rng.uniform(1, 2, 12), rng.standard_normal((12, 3))])
    mixed = R.copy()
    mixed[[1, 3, 5, 7]] *= 1e-10  # same directions, wildly different norms
    assert exhaustive(mixed).tolist() == exhaustive(R).tolist()


# --- the cleanup removal path ------------------------------------------------
#
# Floating-point tie-breaking can admit a redundant ray into E; cleanup is the
# safety net that takes it back out, and it is the justification for the whole
# tie-breaking design. On well-conditioned cones it essentially never fires
# (searching grids, cubes and 60 random cones found no case where cleanup=True
# and cleanup=False differ), so the branch is forced here rather than waited
# for: _shoot is patched to admit a ray that is provably redundant.

REDUNDANT_CONE = np.array([[1, 0], [0, 1], [1, 1]])  # ray 2 = ray 0 + ray 1


def _force_admitting(monkeypatch, victim, tied=True):
    """Make the first ray-shoot admit `victim`, flagged as tie-admitted."""
    real = core._shoot
    state = {"done": False}

    def fake(P, c, cand, rel_tol=1e-9, all_vals=None):
        if not state["done"] and victim in cand:
            state["done"] = True
            return victim, tied
        return real(P, c, cand, rel_tol, all_vals)

    monkeypatch.setattr(core, "_shoot", fake)


def test_cleanup_removes_a_tie_admitted_redundant_ray(monkeypatch, capsys):
    _force_admitting(monkeypatch, victim=2)
    idx = core.exhaustive(REDUNDANT_CONE, seed_shots=0, verbosity=1)
    out = capsys.readouterr().out
    assert idx.tolist() == [0, 1], "the redundant ray must not survive cleanup"
    assert "removed redundant ray" in out
    assert core.LAST_PROFILE["n_lp_cleanup"] >= 1
    assert core.LAST_PROFILE["cleanup_membership_lp"] > 0.0
    assert verify(REDUNDANT_CONE, idx)[0]


def test_cleanup_disabled_keeps_the_impostor(monkeypatch):
    # the same forced admission, with cleanup off: the ray stays, so the
    # result still generates the cone but is no longer minimal
    _force_admitting(monkeypatch, victim=2)
    idx = core.exhaustive(REDUNDANT_CONE, seed_shots=0, cleanup=False)
    assert idx.tolist() == [0, 1, 2]
    ok, report = verify(REDUNDANT_CONE, idx)
    assert not ok and any("redundant" in f for f in report["failures"])


def test_cleanup_keeps_an_uncertifiable_suspect_and_warns(monkeypatch):
    # if no membership certificate can be obtained, removing the ray would be
    # unsound, so cleanup must keep it and say so; the conservative branch
    _force_admitting(monkeypatch, victim=2)

    class NoCertificate(core._MembershipOracle):
        def residual(self, p):
            return np.inf, None            # solver cannot certify anything

    monkeypatch.setattr(core, "_MembershipOracle", NoCertificate)
    with pytest.warns(UserWarning, match="borderline"):
        idx = core.exhaustive(REDUNDANT_CONE, seed_shots=0)
    assert idx.tolist() == [0, 1, 2], "an uncertified suspect must be kept"


def test_cleanup_skips_rays_admitted_as_unique_maximizers(monkeypatch):
    # a ray admitted as the UNIQUE maximizer of some functional is provably
    # extremal, so it must never reach the cleanup retest
    _force_admitting(monkeypatch, victim=2, tied=False)
    core.exhaustive(REDUNDANT_CONE, seed_shots=0)
    assert core.LAST_PROFILE["n_suspects"] == 0
