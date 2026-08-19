# extremal-rays
*[Nate MacFadden](https://github.com/natemacfadden), Liam McAllister Group, Cornell*

Fast extremal rays of pointed polyhedral cones via [Clarkson's output-sensitive algorithm](https://doi.org/10.1109/SFCS.1994.365723). Built for cones that defeat the classical per-ray LP: many generators, high dimension, mostly-redundant rays -- e.g., toric Mori cones of Calabi-Yau hypersurfaces at large $h^{1,1}$. On the Mori cone of the $h^{1,1}=491$ CY (3509 generators in 491 dimensions, 884 extremal), `extremal-rays` finishes in ~20s single-threaded on an Apple M1 Pro (32 GB RAM, macOS 26) where the classical method does not terminate; reproduce this with [`benchmarks/bench_mori.py`](benchmarks/bench_mori.py) (data bundled).

## Description

Given $R\in\mathbb{Z}^{n\times d}$ (or floats) whose rows generate a pointed cone

$$ \mathcal{C} = \\{\textstyle\sum_i \lambda_i R_i : \lambda_i \geq 0\\}, $$

`extremal-rays` provides three methods: `exhaustive` returns the indices of the unique minimal generating subset -- the extremal rays; `sample` cheaply certifies a subset of them (an inner bound, no completeness claim); `verify` independently audits an answer via explicit certificates.

The standard method (as in [CYTools](https://cy.tools/)' `Cone.extremal_rays`) asks, for each ray, "is it a non-negative combination of all $n-1$ others?". Redundant rays answer quickly (the LP is feasible), but each *extremal* ray requires an **infeasibility proof** for a large degenerate system -- on the $h^{1,1}=491$ Mori cone, HiGHS exceeds 15 minutes on a single one. `extremal-rays` never asks that question: candidates are tested only against the small confirmed-extremal set, and extremality is only ever established constructively (see [Algorithm Notes](#algorithm-notes)).

## Limitations

- The cone must be pointed (strongly convex); non-pointed input raises `ValueError`. Decompose into lineality space + pointed quotient first (as CYTools already does).
- Parallel sweeps (`n_workers`) only pay off on long jobs: worker startup and snapshot refreshes cost a few seconds, so at the ~20 s benchmark scale `n_workers=8` is marginally *slower* (22.2 s vs 20.6 s serial); on a 10M-candidate job it gave ~1.9x.

## Installation

```
pip install -e .
```

Dependencies: numpy, scipy, highspy. Run the tests (includes brute-force cross-checks) with `pytest`.

## Usage

```python
import numpy as np
from extremal_rays import exhaustive, sample, verify

R = np.load("rays.npy")          # (n, d) generators, integer or float
idx = exhaustive(R)              # indices of the minimal generating subset
ext = R[idx]

ok, report = verify(R, idx)      # optional certificate audit

some, curve = sample(R, work=5000)   # cheap certified subset (no completeness)
```

Integer input enables exact primitive-vector deduplication and an exact rational fallback in cleanup. Duplicate directions collapse to their first occurrence. A wall-time breakdown of the last call is stored in `extremal_rays.core.LAST_PROFILE`.

For long jobs, `n_workers=8` sweeps candidates in parallel against frozen snapshots of the confirmed set (verdicts stay exact; rare separation failures are re-resolved serially), and `checkpoint="state.npz"` saves state atomically every minute -- rerunning the same call resumes from the last checkpoint, guarded by a fingerprint of the input rays. Candidate *order* matters for speed: the separation oracle warm-starts between consecutive LPs, so orderings that keep similar rays adjacent run far faster than shuffled input. Structured generator order (the common case) is typically best and is kept by default; for unstructured input pass `sort_candidates=True` to lexsort internally -- on the benchmark cone, shuffled input ran > 78 min without it vs 20.7 s with it.

## Algorithm Notes

An LP finds $w$ with $w\cdot s\geq 1$ on all rays (doubling as the pointedness check); scaling onto the slice $w\cdot x=1$ turns conic redundancy into point-hull redundancy. Each candidate $p$ is then tested against the confirmed-extremal set $E$ with the separation LP

$$ \max\ c\cdot p \quad \text{s.t.} \quad c\cdot e \leq 0\ \forall e\in E, \quad -1\leq c_i\leq 1, $$

which is always feasible and bounded. Value $0$ means $p\in\text{cone}(E)$ (Farkas): redundant, valid however incomplete $E$ is. A positive value proves nothing about $p$ -- it proves $E$ is missing an extremal ray, and the optimizer $c$ locates it by *ray shooting*: the tie-broken maximizer of $c\cdot s$ over the remaining candidates is provably a vertex, joins $E$, and $p$ is retested. Total LP count is $\leq n+|E|$, all small, solved on one persistent warm-started HiGHS model (objective swap per test, one row appended per confirmed ray).

Floating-point tie-breaking can rarely admit a redundant ray into $E$; a cleanup pass retests exactly the tie-admitted rays (unique maximizers are provably vertices and skip it), removing a ray only on a positive certificate of redundancy -- escalating to exact rational arithmetic for integer input -- so the result always generates the cone. See the module docstrings in [`core.py`](src/extremal_rays/core.py) for details; I encourage you to read it.

## Benchmarks

Toric Mori cone (in a basis) of the CY hypersurface with $h^{1,1}=491$: 3509 generating rays in 491 dimensions, 884 extremal.

| method | time |
| --- | --- |
| per-ray LP vs all others (CYTools `extremal_rays`, `method="lp"`) | does not terminate (> 15 min per extremal ray) |
| per-ray NNLS (CYTools `method="nnls"`) | ~15 CPU-hours, verdicts at ~10x the tolerance |
| this package (single-threaded, incl. cleanup) | **~20 s** |
| full certificate audit (optional) | ~2 min |

The largest job run to date: the Mori-cone *cap* of the same CY, 10,026,843 candidate rays in 491 dimensions (1,218 extremal), in **79.6 min end-to-end** with `n_workers=8`, sparse input, generation order kept (`sort_candidates=False`), and the slice functional supplied via `w=` (the cap's dual cone has a compact description; solving the pointedness LP over 10M rows instead fails).

~94% of the runtime is HiGHS solve time (~3500 separation LPs at ~5 ms), so the Python layer is not the bottleneck. Results agree exactly with CYTools on Mori cones small enough for CYTools to finish ($h^{1,1}\in\\{10,25,50,100\\}$); the $h^{1,1}=491$ answer passes the full certificate audit and was cross-checked against NNLS on a sample.

## License

GPL-3.0-or-later (matching [CYTools](https://github.com/LiamMcAllisterGroup/cytools)).
