# extremalrays
*[Nate MacFadden](https://github.com/natemacfadden), Liam McAllister Group, Cornell*

Fast extremal rays of pointed polyhedral cones via [Clarkson's output-sensitive algorithm](https://doi.org/10.1109/SFCS.1994.365723). Alternative to a per-ray feasibility LP that should be quicker for hard cases.

## Description

Given $R\in\mathbb{Z}^{n\times d}$ (or floats) whose rows generate a pointed cone

$$ \mathcal{C} = \\{\textstyle\sum_i \lambda_i R_i : \lambda_i \geq 0\\}, $$

`exhaustive` returns the indices of the unique minimal generating subset, i.e. the extremal rays. `sample` is a cheaper cousin that certifies a subset of them and makes no completeness claim. `verify` audits an answer from explicit certificates, checking each one rather than trusting a solver status code.

The usual approach (CYTools v1.4.12's `Cone.extremal_rays`, and `redund` in cddlib or lrs) asks, for each ray, whether it is a non-negative combination of the other $n-1$. Redundant rays answer quickly. Extremal ones need an infeasibility proof for a big degenerate system, and that is where the time goes. `extremalrays` never asks that question: a candidate is only ever tested against the small set of rays already confirmed extremal.

One caveat on `verify`. Its certificates are built independently of `exhaustive` (different formulation, opposite LP direction, exact arithmetic when the rays are integral), but the two share this package's preprocessing. A bug in the deduplication would be invisible to it.

### Prior art

Removing redundant generators is a classical problem with good tools: the double description method of Motzkin et al., as implemented in Fukuda's [cddlib](https://people.inf.ethz.ch/fukuda/cdd_home/); reverse search in Avis's [lrslib](http://cgm.cs.mcgill.ca/~avis/C/lrs.html); plus [Normaliz](https://www.normaliz.uni-osnabrueck.de/) and [polymake](https://polymake.org/). See [benchmarks](#benchmarks) for how we compare. Our benchmarks are focused to our cases of interest (cones arising in CYTools/string theory).

## Limitations

- The cone must be pointed (strongly convex); non-pointed input raises `ValueError`. Decompose into lineality space + pointed quotient first (as CYTools already does).
- Parallel sweeps (`n_workers`) only pay off on long jobs: worker startup and snapshot refreshes cost a few seconds, so at benchmark scale `n_workers=8` is marginally *slower* in wall time AND costs roughly 2x the CPU-seconds; on a 10M-candidate job it gave ~1.9x. Workers are spawned, so a script passing `n_workers > 0` must guard its entry point with `if __name__ == "__main__":`; without it, `exhaustive` detects the nested call and falls back to a serial sweep with a warning rather than recursing.
- Sparse (CSR) input is for *feasibility at scale*, not for speed: it is what makes a 10M-ray cone possible at all (dense would be tens of GB), but at benchmark scale it saves no time and costs ~1.8x peak RSS.
- Tolerance is not a free parameter on badly conditioned input: see the note on missing rays in [Algorithm Notes](#algorithm-notes). Integer rays get exact escalation; float rays only get a warning.

## Installation

```
pip install -e .                 # runtime only
pip install -e ".[test]"         # + pytest
pip install -e ".[exact]"        # + python-flint, faster exact arithmetic
```

Dependencies: numpy, scipy, highspy, with version floors that CI installs and tests. `python-flint` is optional; without it a pure-Python fallback is used.

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

Rays can also arrive from a file or another library rather than a literal, and any `(n, d)` array-like works: `exhaustive(np.load("rays.npy"))`, or a `scipy.sparse` CSR matrix, which is kept sparse end to end.

Integer input enables exact primitive-vector deduplication and an exact rational fallback in cleanup. Duplicate directions collapse to their first occurrence. A wall-time breakdown of the last call is stored in `extremal_rays.core.LAST_PROFILE`.

For long jobs, `n_workers=8` sweeps candidates in parallel against frozen snapshots of the confirmed set (verdicts stay exact; rare separation failures are re-resolved serially), and `checkpoint="state.npz"` saves state atomically every minute. Rerunning the same call resumes from the last checkpoint, guarded by a fingerprint of the input rays.

Candidate *order* matters for speed. The separation oracle warm-starts between consecutive LPs, so an order that keeps similar rays adjacent runs faster. On the benchmark cone, generation order takes 13.8 s and a shuffle of it takes 21 to 25 s. Structured order is the common case, so it is the default; for unstructured input, `sort_candidates=True` lexsorts internally and gets the shuffle back to 13.7 s.

## Algorithm Notes

Write $R_1,\dots,R_n\in\mathbb{R}^d$ for the rows of $R$, the generators of $\mathcal{C}$. An LP finds a functional $w\in\mathbb{R}^d$ with $w\cdot R_i\geq 1$ for every $i$. Such a $w$ exists precisely when $\mathcal{C}$ is pointed, so this doubles as the pointedness check. Rescaling each generator to $p_i=R_i/(w\cdot R_i)$ moves it onto the slice $w\cdot x=1$, which turns conic redundancy into point-hull redundancy: $R_i$ is extremal in $\mathcal{C}$ exactly when $p_i$ is a vertex of $\mathrm{conv}\\{p_1,\dots,p_n\\}$.

Let $E$ be the rays confirmed extremal so far, which starts empty and only grows. A candidate $p$ is tested against it by searching for a linear functional $c\in\mathbb{R}^d$ that separates $p$ from $\mathrm{cone}(E)$:

$$ \max_{c\in\mathbb{R}^d}\ c\cdot p \quad \text{s.t.} \quad c\cdot e \leq 0\ \ \forall e\in E, \quad -1\leq c_i\leq 1, $$

always feasible ($c=0$) and bounded (the box on $c_i$). Value $0$ means $p\in\mathrm{cone}(E)$ by Farkas, so $p$ is redundant, and that verdict holds no matter how incomplete $E$ still is. A positive value says nothing about $p$ itself; it says $E$ is missing an extremal ray. The optimizer $c$ then points at one: the tie-broken maximizer of $c\cdot p_i$ over the remaining candidates is provably a vertex, joins $E$, and $p$ gets retested. That bounds the LP count by $n+|E|$, all of them small, on one persistent warm-started HiGHS model.

Both oracles unit-normalize their inputs first. Membership in a cone is scale-free, so this changes nothing mathematically, but it means a fixed tolerance means the same thing for every candidate. Skip it and the slice coordinates $p_i$ under an anisotropic $w$ span orders of magnitude, at which point rays start falling under `tol` for no geometric reason.

The failure mode to watch is a ray going *missing*, since that is the one that breaks generation. A candidate is called redundant when its separation value lands below `tol`, and on badly conditioned cones a genuinely extremal ray can score just under. Three things guard it now. Verdicts between the solver's noise floor and `tol` get re-decided in exact rational arithmetic when the rays are integral; float input cannot do that, so it warns instead; and `verify` rechecks the whole answer, escalating borderline rays in both directions. What is left is a real limit rather than an oversight, since no floating-point LP can tell "inside the cone" from "$10^{-15}$ outside it". Pass integer rays when you have them.

The `_SeparationOracle` and `_MembershipOracle` docstrings in [`core.py`](src/extremal_rays/core.py) record the numerical failures that shaped both, and I encourage you to read them.

## Benchmarks

Compared against CYTools, lrs, cddlib and Normaliz on cone families built from the Kreuzer-Skarke polytopes, on an Apple M1 Pro (32 GB RAM, macOS 26). Each method gets all 10 cores, though several stay serial by design. Every method's answer is checked against this package's before it is timed, so a fast wrong answer cannot win, and the fork cost of the command-line tools is measured and subtracted. Points are medians over three polytopes per $h^{1,1}$; error bars are the spread between them. Recreate with the [`benchmarks/`](benchmarks) scripts.

**Toric Mori cone**, $h^{1,1} = 3$ to $491$:

<p align="center">
  <img src="docs/benchmark_prior_art.png" alt="Runtime vs h11 for toric Mori cones: extremalrays alone reaches h11=491"/>
</p>

**Mori-cone cap**, which is the same cone family capped, and much larger: 20,899 rays at
$h^{1,1}=50$ against 333 for the cone itself. Plotted against ray count, since cap size is not monotonic in $h^{1,1}$:

<p align="center">
  <img src="docs/benchmark_cap_scaling.png" alt="Runtime vs cap size: extremalrays and CYTools share an exponent, cddlib and lrs do not"/>
</p>

Against CYTools the exponents agree and the gap is a constant of roughly an order of magnitude. Against cddlib and lrs it is the exponent that differs ($n^{2.6}$ and $n^{3.6}$ against $n^{1.34}$), which is why they stop around 1,500 rays.

On the $h^{1,1}=491$ Mori cone itself (3509 rays in 491 dimensions, 884 extremal):

| method | time |
| --- | --- |
| this package | **~13 s** |
| CYTools `extremal_rays` | does not finish |
| full certificate audit (optional) | ~11 s on 8 workers |

The largest run so far is the cap of that same CY: 10,026,843 rays in 491 dimensions, 1,218 extremal, in 79.6 min with `n_workers=8` and the slice functional handed in via `w=`.

## Citation

If you use `extremalrays` in your research, please cite it:

```bibtex
@software{extremal_rays,
  author  = {MacFadden, Nate},
  title   = {extremalrays},
  url     = {https://github.com/LiamMcAllisterGroup/extremalrays},
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

Comparisons are skipped for tools that are not installed, and every method's answer is checked against this package's before it is timed. Add `--plot-only` to redraw a figure from results already measured.

## License

GPL-3.0-or-later (matching [CYTools](https://github.com/LiamMcAllisterGroup/cytools)).
