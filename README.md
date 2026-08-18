# extremal-rays

Fast computation of the extremal rays of a pointed polyhedral cone, using
Clarkson's output-sensitive algorithm. Built to handle the cones that defeat
the classical approach: many generators, high ambient dimension, and a large
fraction of redundant rays — e.g. toric Mori cones of Calabi-Yau hypersurfaces
at large h¹¹.

## The problem with the classical method

The standard way to extract extremal rays from a generating set R (as in
CYTools' `Cone.extremal_rays`) tests each ray r against all n−1 others with
the feasibility LP "does λ ≥ 0 exist with (R∖r)ᵀλ = r?". When r is redundant
the LP is feasible and solves quickly. When r is **extremal**, the solver must
*prove infeasibility* of a huge, highly degenerate system — and this is
catastrophically slow: on the Mori cone of the h¹¹ = 491 CY (3509 rays in 491
dimensions), HiGHS spends **15+ minutes on a single such proof** without
terminating. Since every extremal ray requires one, the computation never
finishes.

## What this package does instead

1. **Normalize.** One LP finds a functional w with w·s ≥ 1 for all rays
   (this doubles as a pointedness check). Rays are scaled onto the slice
   w·x = 1, turning conic redundancy into point-hull redundancy.

2. **Test candidates only against confirmed extremal rays** E, via the
   separation LP

   ```
   max c·p   s.t.   c·e ≤ 0 ∀ e ∈ E,   −1 ≤ c ≤ 1
   ```

   This LP is always feasible (c = 0) and bounded, so the solver always
   terminates at an optimum. Value 0 means p ∈ cone(E) (Farkas) — redundant,
   done, regardless of how incomplete E is. A positive value proves nothing
   about p; it proves E is missing an extremal ray, and hands us the
   certificate c to go find it.

3. **Ray shooting.** The maximizer of c·s over the remaining candidates
   (lexicographically tie-broken) is guaranteed to be a vertex of the hull —
   it joins E, and the candidate is retested. Extremality is thus never
   established by an infeasibility proof, only *constructively*, by being the
   maximizer of an explicit functional. Total LP count ≤ n + |E|, every LP
   small. A single persistent HiGHS model is reused across all tests
   (objective swap per candidate, one row appended per confirmed ray), so
   warm starts carry over.

4. **Cleanup.** Floating-point tie-breaking in ray shooting can rarely admit
   a redundant ray into E. Each confirmed ray is retested against the others
   (cheap — E is small); removal requires a positive certificate of
   redundancy (an explicit non-negative combination with checked residual,
   escalating to exact rational arithmetic for integer input). Ambiguous rays
   are kept with a warning — the result then still generates the cone.

5. **Optional audit.** `verify_extremal_rays` re-derives every classification
   from explicit witnesses: a non-negative combination for every discarded
   ray, a separating functional for every kept ray.

## Benchmark

Toric Mori cone (in a basis) of the CY hypersurface with h¹¹ = 491:
3509 generating rays in 491 dimensions, of which 884 are extremal
(`benchmarks/bench_mori.py`, data bundled).

| method | time |
| --- | --- |
| per-ray LP vs all others (CYTools `extremal_rays`, `method="lp"`) | does not terminate (> 15 min per extremal ray) |
| per-ray NNLS (`method="nnls"`) | ~15 CPU-hours, verdicts at ~10× the tolerance |
| this package (single-threaded, incl. cleanup) | **33 s** |
| full certificate audit (optional) | ~2 min |

## Usage

```python
import numpy as np
from extremal_rays import extremal_rays, verify_extremal_rays

R = np.load("rays.npy")          # (n, d) generators, integer or float
idx = extremal_rays(R)           # indices of the minimal generating subset
ext = R[idx]

ok, report = verify_extremal_rays(R, idx)   # optional certificate audit
```

Input rays may be integer (enables exact primitive-vector deduplication and
an exact rational fallback in cleanup) or float. Duplicate directions are
collapsed to their first occurrence. Non-pointed cones raise `ValueError`;
decompose into lineality space + pointed quotient first (as CYTools already
does).

## Install

```
pip install -e .
pytest            # 21 tests, includes brute-force cross-checks
python benchmarks/bench_mori.py [--verify]
```

Dependencies: numpy, scipy, highspy.

## Reference

K. L. Clarkson, *More output-sensitive geometric algorithms*, FOCS 1994 —
the output-sensitive redundancy-removal scheme adapted here from halfspaces
to cone generators.

## License

GPL-3.0 (matching [CYTools](https://github.com/LiamMcAllisterGroup/cytools)).
