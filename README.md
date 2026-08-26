# extremal-rays
*[Nate MacFadden](https://github.com/natemacfadden), Liam McAllister Group, Cornell*

Fast extremal rays of pointed polyhedral cones via [Clarkson's output-sensitive algorithm](https://doi.org/10.1109/SFCS.1994.365723). Built for cones that defeat the classical per-ray LP: many generators, high dimension, mostly-redundant rays -- e.g., toric Mori cones of Calabi-Yau hypersurfaces at large $h^{1,1}$. On the Mori cone of the $h^{1,1}=491$ CY (3509 generators in 491 dimensions, 884 extremal), `extremal-rays` finishes in ~20s single-threaded on an Apple M1 Pro (32 GB RAM, macOS 26) where the classical method does not terminate; reproduce this with [`benchmarks/bench_mori.py`](benchmarks/bench_mori.py) (data bundled).

## Description

Given $R\in\mathbb{Z}^{n\times d}$ (or floats) whose rows generate a pointed cone

$$ \mathcal{C} = \\{\textstyle\sum_i \lambda_i R_i : \lambda_i \geq 0\\}, $$

`extremal-rays` provides three methods: `exhaustive` returns the indices of the unique minimal generating subset -- the extremal rays; `sample` cheaply certifies a subset of them (an inner bound, no completeness claim); `verify` audits an answer via explicit certificates, each one re-derived and re-checked rather than taken from a solver status code.

`verify`'s independence is real but bounded, and worth stating precisely: the certificates are constructed independently of `exhaustive` -- a different formulation, the opposite LP direction, and exact rational arithmetic for integer rays whose float certificate is borderline -- but both share this package's preprocessing (primitive deduplication and the slice scaling). An error inside that shared step is invisible to the audit. What `verify` checks is the extremal-ray decision, not the deduplication.

### Prior art

Removing redundant generators is a classical problem with mature implementations: the double description method of Motzkin et al., realized in Fukuda's [cddlib](https://people.inf.ethz.ch/fukuda/cdd_home/) (which has shipped a Clarkson-style redundancy removal since v0.94), Avis's [lrslib](http://cgm.cs.mcgill.ca/~avis/C/lrs.html) (`redund`), [Normaliz](https://www.normaliz.uni-osnabrueck.de/), and [polymake](https://polymake.org/). Those are the right tools for most cones, and this package does not replace them. It targets one regime they handle poorly: many generators, high dimension, and a large mostly-redundant majority, where per-ray infeasibility proofs dominate. For calibration, on the bundled fixtures cddlib agrees with this package exactly, and on the $h^{1,1}=491$ Mori cone below it did not finish within 300 s.

The standard method (as in [CYTools](https://cy.tools/)' `Cone.extremal_rays`) asks, for each ray, "is it a non-negative combination of all $n-1$ others?". Redundant rays answer quickly (the LP is feasible), but each *extremal* ray requires an **infeasibility proof** for a large degenerate system -- on the $h^{1,1}=491$ Mori cone, HiGHS exceeds 15 minutes on a single one. `extremal-rays` never asks that question: candidates are tested only against the small confirmed-extremal set, and extremality is only ever established constructively (see [Algorithm Notes](#algorithm-notes)).

## Limitations

- The cone must be pointed (strongly convex); non-pointed input raises `ValueError`. Decompose into lineality space + pointed quotient first (as CYTools already does).
- Parallel sweeps (`n_workers`) only pay off on long jobs: worker startup and snapshot refreshes cost a few seconds, so at benchmark scale `n_workers=8` is marginally *slower* in wall time AND costs roughly 2x the CPU-seconds; on a 10M-candidate job it gave ~1.9x. Workers are spawned, so a script passing `n_workers > 0` must guard its entry point with `if __name__ == "__main__":`; without it, `exhaustive` detects the nested call and falls back to a serial sweep with a warning rather than recursing.
- Sparse (CSR) input is for *feasibility at scale*, not for speed: it is what makes a 10M-ray cone possible at all (dense would be tens of GB), but at benchmark scale it saves no time and costs ~1.8x peak RSS.
- Tolerance is not a free parameter on badly conditioned input: see the note on missing rays in [Algorithm Notes](#algorithm-notes). Integer rays get exact escalation; float rays only get a warning.

## Installation

```
pip install -e .                 # runtime only
pip install -e ".[test]"         # + pytest, to run the test suite
pip install -e ".[exact]"        # + python-flint, faster exact arithmetic
```

Dependencies: numpy, scipy, highspy, with version floors that CI installs and tests (see the `Dependency floors` job). `python-flint` is optional: it accelerates the exact rational fallback, and without it a pure-Python implementation is used instead. Run the tests -- which include brute-force cross-checks against a vendored copy of CYTools' own method -- with `pytest`.

## Usage

```python
import numpy as np
from extremal_rays import exhaustive, sample, verify

# four rays in the plane; only the outer two generate the cone
R = np.array([[1, 0], [2, 1], [1, 1], [0, 1]])

idx = exhaustive(R)              # -> array([0, 3]), indices into R
ext = R[idx]                     # -> [[1, 0], [0, 1]]

ok, report = verify(R, idx)      # -> True, with the certificates checked

some, curve = sample(R, work=200)   # cheap certified subset (no completeness)
```

Rays usually arrive from a file or another library rather than a literal, and any `(n, d)` array-like works: `exhaustive(np.load("rays.npy"))`, or a `scipy.sparse` CSR matrix, which is kept sparse end to end.

Integer input enables exact primitive-vector deduplication and an exact rational fallback in cleanup. Duplicate directions collapse to their first occurrence. A wall-time breakdown of the last call is stored in `extremal_rays.core.LAST_PROFILE`.

For long jobs, `n_workers=8` sweeps candidates in parallel against frozen snapshots of the confirmed set (verdicts stay exact; rare separation failures are re-resolved serially), and `checkpoint="state.npz"` saves state atomically every minute -- rerunning the same call resumes from the last checkpoint, guarded by a fingerprint of the input rays. Candidate *order* matters for speed: the separation oracle warm-starts between consecutive LPs, so orderings that keep similar rays adjacent run far faster than shuffled input. Structured generator order (the common case) is typically best and is kept by default; for unstructured input pass `sort_candidates=True` to lexsort internally -- on the benchmark cone, shuffled input ran > 78 min without it vs 20.7 s with it.

## Algorithm Notes

An LP finds $w$ with $w\cdot s\geq 1$ on all rays (doubling as the pointedness check); scaling onto the slice $w\cdot x=1$ turns conic redundancy into point-hull redundancy. Each candidate $p$ is then tested against the confirmed-extremal set $E$ with the separation LP

$$ \max\ c\cdot p \quad \text{s.t.} \quad c\cdot e \leq 0\ \forall e\in E, \quad -1\leq c_i\leq 1, $$

which is always feasible and bounded. Value $0$ means $p\in\text{cone}(E)$ (Farkas): redundant, valid however incomplete $E$ is. The oracle unit-normalizes $p$ and every $e$ (membership is conic, so this changes nothing mathematically) so that the value is scale-free and a fixed tolerance means the same thing for every candidate; on the $h^{1,1}=491$ cone, redundant rays then score $\lesssim 10^{-12}$ and the closest extremal ray $\sim 4\times10^{-5}$ (the two closest calls of a run are reported in `LAST_PROFILE`). Without normalization, slice coordinates under the LP's anisotropic $w$ squash some rays by $10^{3}$ and an extremal ray of that cone scored $8\times10^{-8}$ -- under the tolerance -- and was dropped. Normalization *mitigates* that failure rather than removing it: on cones whose integer coefficients span $10^{6}$, extremal rays have still been observed scoring below `tol` with normalization on. That is why near-tolerance verdicts are now escalated to exact arithmetic rather than trusted. A positive value proves nothing about $p$ -- it proves $E$ is missing an extremal ray, and the optimizer $c$ locates it by *ray shooting*: the tie-broken maximizer of $c\cdot s$ over the remaining candidates is provably a vertex, joins $E$, and $p$ is retested. Total LP count is $\leq n+|E|$, all small, solved on one persistent warm-started HiGHS model (objective swap per test, one row appended per confirmed ray).

Floating-point tie-breaking can rarely admit a redundant ray into $E$; a cleanup pass retests exactly the tie-admitted rays (unique maximizers are provably vertices and skip it), removing a ray only on a positive certificate of redundancy -- escalating to exact rational arithmetic for integer input. So cleanup never costs you a ray you had.

The opposite direction -- a ray *missing* from the result -- is the failure mode to understand, because it is the one that breaks generation. A candidate is called redundant when its separation value falls below `tol`, and on badly conditioned cones a genuinely extremal ray can score just under it. Three things now guard that: any verdict landing between the solver's noise floor and `tol` is re-decided in exact rational arithmetic when the rays are integral; a run that could not do so (float rays) warns; and `verify` re-checks the whole answer, escalating borderline rays to exact arithmetic in both directions. What remains is a genuine limit rather than an oversight: for float input at extreme conditioning, no float LP can separate "inside the cone" from "$10^{-15}$ outside it", and there is no exact path to fall back on. Pass integer rays when you have them.

See the module docstrings in [`core.py`](src/extremal_rays/core.py) for details -- particularly `_SeparationOracle` and `_MembershipOracle`, which document the numerical failure modes that shaped them. I encourage you to read them.

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

## Citation

If you use `extremal-rays` in your research, please cite it:

```bibtex
@software{extremal_rays,
  author  = {MacFadden, Nate},
  title   = {extremal-rays},
  url     = {https://github.com/LiamMcAllisterGroup/extremal_rays},
  orcid   = {0000-0002-8481-3724},
}
```

## Organization

```
extremal_rays/
├── src/extremal_rays/
│   ├── core.py                     # exhaustive(): Clarkson sweep, separation + membership oracles, cleanup, checkpoints, workers
│   ├── inner.py                    # sample(): cheap certified subset via dual-cone facet walks
│   └── verify.py                   # verify(): independent certificate audit, parallelisable
├── tests/
│   ├── conftest.py                 # shared helpers: random pointed cones, vendored CYTools per-ray LP reference
│   ├── test_exhaustive.py          # known cones, degenerate input, invariances, brute-force cross-checks
│   ├── test_parallel_checkpoint.py # n_workers sweeps, checkpoint save/resume/fingerprint
│   ├── test_sample.py              # certified-subset property, determinism, options
│   ├── test_verify.py              # accepts right answers, rejects wrong ones, parallel audit
│   ├── test_internals.py           # unit tests of the oracles, dedup, exact arithmetic
│   ├── test_readme_examples.py     # the README snippet runs as documented
│   └── data/                       # Mori-cone caps with CYTools-produced extremal sets (regression fixtures)
├── benchmarks/                     # perf benchmarks; double as usage examples + make the README figures
│   ├── benchmark_h11_491.py        # the flagship cone: repeats, dispersion, machine capture, opt-in classical baseline
│   ├── benchmark_mori_cone.py      # runtime vs h11 against CYTools, lrs, cddlib, Normaliz
│   ├── benchmark_mcap.py           # the same against the Mori-cone cap, vs h11 and vs problem size
│   ├── benchmark_parallel.py       # where parallelism pays: the sweep vs the audit
│   ├── make_cones.py               # build the Mori-cone family from the Kreuzer-Skarke polytopes
│   ├── make_caps.py                # build the Mori-cone cap family (needs the mori-cap tooling)
│   ├── _bench.py                   # shared timing helper: warmup, repeats, median with spread
│   ├── _plot.py                    # the runtime-vs-h11 figure, with fitted power laws
│   ├── _plot_scaling.py            # the runtime-vs-ray-count figure
│   ├── _cytools_driver.py          # runs CYTools in a fresh interpreter so it can be time-limited
│   └── data/                       # cone and cap ray matrices (polytope tables are fetched on demand)
├── docs/                           # README figures (benchmark_*.png)
├── CHANGELOG.md
└── pyproject.toml
```

### Reproducing the figures

```
python benchmarks/make_cones.py --per-h11 3 --h11 3 5 10 20 50 100 200 491
python benchmarks/benchmark_mori_cone.py            # -> docs/benchmark_prior_art.png
python benchmarks/make_caps.py --h11 10 20 30 40 50
python benchmarks/benchmark_mcap.py                 # -> docs/benchmark_cap*.png
```

Comparisons are skipped for tools that are not installed, and every method's
answer is checked against this package's before it is timed. Add `--plot-only`
to redraw a figure from results already measured.

## License

GPL-3.0-or-later (matching [CYTools](https://github.com/LiamMcAllisterGroup/cytools)).
