# extremal-rays
*[Nate MacFadden](https://github.com/natemacfadden), Liam McAllister Group, Cornell*

Fast computation of the extremal rays of pointed polyhedral cones, using [Clarkson's output-sensitive algorithm](https://doi.org/10.1109/SFCS.1994.365723). Built for the cones that defeat the classical approach: many generators, high ambient dimension, and a large fraction of redundant rays -- e.g., toric Mori cones of Calabi-Yau hypersurfaces at large $h^{1,1}$. On the Mori cone of the $h^{1,1}=491$ CY (3509 generators in 491 dimensions, 884 extremal), `extremal-rays` finishes in ~20s single-threaded where the classical per-ray LP does not terminate; reproduce this with [`benchmarks/bench_mori.py`](benchmarks/bench_mori.py) (data bundled).

## Description

Given a matrix $R\in\mathbb{Z}^{n\times d}$ whose rows generate a pointed cone

$$ \mathcal{C} = \\{\textstyle\sum_i \lambda_i R_i : \lambda_i \geq 0\\}, $$

`extremal-rays` returns the indices of the unique minimal generating subset (the extremal rays). Float input is also accepted.

The standard method (as in [CYTools](https://cy.tools/)' `Cone.extremal_rays`) tests each ray $r$ against all $n-1$ others with the feasibility LP "does $\lambda\geq 0$ exist with $(R\setminus r)^T\lambda = r$?". When $r$ is redundant the LP is feasible and solves quickly. When $r$ is **extremal**, the solver must *prove infeasibility* of a large, typically degenerate system, which can be catastrophically slow: on the $h^{1,1}=491$ Mori cone, HiGHS spends 15+ minutes on a single such proof without terminating. Since every extremal ray requires one, the computation never finishes.

This package arranges things so the expensive question is never asked:

1. **Normalize.** One LP finds a functional $w$ with $w\cdot s\geq 1$ for all rays (this doubles as a pointedness check). Rays are scaled onto the slice $w\cdot x = 1$, turning conic redundancy into point-hull redundancy.

2. **Test candidates only against confirmed extremal rays** $E$, via the separation LP

$$ \max\ c\cdot p \quad \text{s.t.} \quad c\cdot e \leq 0\ \forall e\in E, \quad -1\leq c_i\leq 1. $$

   This LP is always feasible ($c=0$) and bounded, so the solver always terminates at an optimum. Value $0$ means $p\in\text{cone}(E)$ (Farkas), hence redundant -- a verdict that is valid no matter how incomplete $E$ is. A positive value proves nothing about $p$; it proves $E$ is missing an extremal ray, and hands us the certificate $c$ to go find it.

3. **Ray shooting.** The maximizer of $c\cdot s$ over the remaining candidates (lexicographically tie-broken) is guaranteed to be a vertex of the hull. It joins $E$, and the candidate is retested. Extremality is thus never established by an infeasibility proof, only *constructively*, by being the maximizer of an explicit functional. The total LP count is $\leq n + |E|$, every LP small. A single persistent HiGHS model is reused across all tests (objective swap per candidate, one row appended per confirmed ray), so warm starts carry over.

4. **Cleanup.** Floating-point tie-breaking in ray shooting can rarely admit a redundant ray into $E$. Only tie-broken shots are at risk -- a ray shot as the *unique* maximizer of a functional is provably a vertex -- so only tie-admitted rays are retested (typically a small fraction). Removal requires a positive certificate of redundancy (an explicit non-negative combination with checked residual, escalating to exact rational arithmetic for integer input); ambiguous rays are kept with a warning, so the result always generates the cone.

5. **Optional audit.** `verify_extremal_rays` re-derives every classification from explicit witnesses: a non-negative combination for every discarded ray, a separating functional for every kept ray.

## Limitations

- The cone must be pointed (strongly convex); non-pointed input raises `ValueError`. Decompose into lineality space + pointed quotient first (as CYTools already does).
- Single-threaded. At the sizes tested this has not been the bottleneck (see Benchmarks); the candidate loop is parallelizable in principle with a shared $E$.

## Installation

```
pip install -e .
```

Dependencies: numpy, scipy, highspy. Run the tests with

```
pytest
```

## Usage

```python
import numpy as np
from extremal_rays import extremal_rays, verify_extremal_rays

R = np.load("rays.npy")          # (n, d) generators, integer or float
idx = extremal_rays(R)           # indices of the minimal generating subset
ext = R[idx]

ok, report = verify_extremal_rays(R, idx)   # optional certificate audit
```

Integer input enables exact primitive-vector deduplication and an exact rational fallback in cleanup. Duplicate directions are collapsed to their first occurrence. A wall-time breakdown of the last call is stored in `extremal_rays.core.LAST_PROFILE`.

## Benchmarks

Toric Mori cone (in a basis) of the CY hypersurface with $h^{1,1}=491$: 3509 generating rays in 491 dimensions, of which 884 are extremal ([`benchmarks/bench_mori.py`](benchmarks/bench_mori.py), data bundled).

| method | time |
| --- | --- |
| per-ray LP vs all others (CYTools `extremal_rays`, `method="lp"`) | does not terminate (> 15 min per extremal ray) |
| per-ray NNLS (CYTools `method="nnls"`) | ~15 CPU-hours, verdicts at ~10x the tolerance |
| this package (single-threaded, incl. cleanup) | **~20 s** |
| full certificate audit (optional) | ~2 min |

Of the ~20s, ~94% is HiGHS solve time (~3500 separation LPs at ~5 ms), so the Python layer is not the bottleneck. Results agree exactly with CYTools on Mori cones small enough for CYTools to finish ($h^{1,1} \in \\{10, 25, 50, 100\\}$), and the $h^{1,1}=491$ answer passes the full certificate audit and was cross-checked against the NNLS method on a sample.

## Reference

K. L. Clarkson, [*More output-sensitive geometric algorithms*](https://doi.org/10.1109/SFCS.1994.365723), FOCS 1994 -- the output-sensitive redundancy-removal scheme adapted here from halfspaces to cone generators.

## License

GPL-3.0-or-later (matching [CYTools](https://github.com/LiamMcAllisterGroup/cytools)).
