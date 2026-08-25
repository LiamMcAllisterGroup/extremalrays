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
# Description:  Tests of the parallel sweep (n_workers) and of checkpointing:
#               save/resume round trips, resuming from a genuinely partial
#               state, the input fingerprint guard, and the .bak rotation.
# -----------------------------------------------------------------------------

# external imports
import numpy as np
import pytest
from scipy.sparse import csr_matrix

# local imports
from extremal_rays import exhaustive
from extremal_rays import core
from conftest import grid_face, random_pointed_rays


# --- parallel ----------------------------------------------------------------

@pytest.mark.parametrize("trial", range(3))
def test_parallel_matches_serial(trial):
    R = random_pointed_rays(trial)
    assert exhaustive(R, n_workers=2).tolist() == exhaustive(R).tolist()


def test_parallel_tie_stress():
    R, corners = grid_face(5)
    assert sorted(exhaustive(R, n_workers=2)) == corners


def test_parallel_no_seeding_forces_failures():
    # with an empty initial E every worker verdict is a separation failure,
    # so the serial re-resolution path carries the whole computation
    R = random_pointed_rays(17, n=80, d=5)
    idx = exhaustive(R, n_workers=2, seed_shots=0)
    assert idx.tolist() == exhaustive(R).tolist()


def test_sparse_parallel_matches_serial():
    R = csr_matrix(random_pointed_rays(5, n=60, d=5))
    assert exhaustive(R, n_workers=2).tolist() == exhaustive(R).tolist()


def test_parallel_with_checkpoint(tmp_path):
    R = random_pointed_rays(8, n=60, d=5)
    ck = str(tmp_path / "state.npz")
    idx = exhaustive(R, n_workers=2, checkpoint=ck)
    assert idx.tolist() == exhaustive(R).tolist()
    assert (tmp_path / "state.npz").exists()


# --- checkpointing -----------------------------------------------------------

def test_checkpoint_resume(tmp_path):
    R = random_pointed_rays(7)
    ck = str(tmp_path / "state.npz")
    a = exhaustive(R, checkpoint=ck)
    assert (tmp_path / "state.npz").exists()
    # rerun resumes the completed state and reproduces the result
    b = exhaustive(R, checkpoint=ck, verbosity=1)
    assert a.tolist() == b.tolist()


def test_sparse_checkpoint_resume(tmp_path):
    R = csr_matrix(random_pointed_rays(9))
    ck = str(tmp_path / "state.npz")
    a = exhaustive(R, checkpoint=ck)
    assert a.tolist() == exhaustive(R, checkpoint=ck).tolist()


def test_checkpoint_fingerprint_guard(tmp_path):
    ck = str(tmp_path / "state.npz")
    R1 = random_pointed_rays(1)
    exhaustive(R1, checkpoint=ck)
    # different rays with the same path must not resume the stale state
    R2 = random_pointed_rays(2)
    assert exhaustive(R2, checkpoint=ck).tolist() == exhaustive(R2).tolist()


def _partial_checkpoint(R, path, n_resolved):
    """Write a checkpoint in which the first n_resolved candidates carry
    their true verdicts and E holds the extremal ones among them (flagged
    suspect, so cleanup retests them)."""
    U, rep = core._reduce(R)
    full = set(exhaustive(R).tolist())
    status = np.zeros(len(U), dtype=np.int8)
    E = []
    for j in range(n_resolved):
        if int(rep[j]) in full:
            status[j] = 1
            E.append(j)
        else:
            status[j] = -1
    core._ckpt_save(path, status=status, E=np.array(E, dtype=np.int64),
                    suspects=np.ones(len(E), dtype=bool),
                    fingerprint=np.array(core._fingerprint(U)))


def test_resume_from_partial_state(tmp_path, capsys):
    R = random_pointed_rays(19, n=70, d=5)
    ck = str(tmp_path / "state.npz")
    _partial_checkpoint(R, ck, n_resolved=30)
    idx = exhaustive(R, checkpoint=ck, verbosity=1)
    assert "resumed:" in capsys.readouterr().out
    assert idx.tolist() == exhaustive(R).tolist()


def test_resume_ignores_known_and_seeding(tmp_path):
    # on resume the saved state is authoritative: known= must not be applied
    # (it would be redundant at best) and the answer must still be right
    R = random_pointed_rays(23, n=60, d=5)
    ck = str(tmp_path / "state.npz")
    _partial_checkpoint(R, ck, n_resolved=10)
    full = exhaustive(R)
    idx = exhaustive(R, checkpoint=ck, known=full[:2])
    assert idx.tolist() == full.tolist()


def test_checkpoint_bak_rotation(tmp_path):
    R = random_pointed_rays(4, n=50, d=4)
    ck = tmp_path / "state.npz"
    exhaustive(R, checkpoint=str(ck))
    exhaustive(R, checkpoint=str(ck))  # second save rotates the first to .bak
    assert (tmp_path / "state.npz.bak").exists()
    ck.write_bytes(b"garbage")  # corrupt the primary
    fp = core._fingerprint(core._reduce(R)[0])
    assert core._ckpt_load(str(ck), fp) is not None
    assert exhaustive(R, checkpoint=str(ck)).tolist() == exhaustive(R).tolist()


def test_ckpt_load_missing_or_mismatched(tmp_path):
    ck = str(tmp_path / "none.npz")
    assert core._ckpt_load(ck, "abc") is None
    core._ckpt_save(ck, status=np.zeros(2, np.int8), E=np.zeros(0, np.int64),
                    suspects=np.zeros(0, bool), fingerprint=np.array("abc"))
    assert core._ckpt_load(ck, "abc") is not None
    assert core._ckpt_load(ck, "xyz") is None
