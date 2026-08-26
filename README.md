# extremal-rays
*[Nate MacFadden](https://github.com/natemacfadden), Liam McAllister Group, Cornell*

Fast extremal rays of pointed polyhedral cones via [Clarkson's output-sensitive algorithm](https://doi.org/10.1109/SFCS.1994.365723). Built for cones that defeat the classical per-ray LP: many generators, high dimension, mostly-redundant rays -- e.g., toric Mori cones of Calabi-Yau hypersurfaces at large $h^{1,1}$. On the Mori cone of the $h^{1,1}=491$ CY (3509 generators in 491 dimensions, 884 extremal), `extremal-rays` finishes in ~20s single-threaded on an Apple M1 Pro (32 GB RAM, macOS 26) where the classical method does not terminate; reproduce with [`benchmarks/benchmark_h11_491.py`](benchmarks/benchmark_h11_491.py) (data bundled).

## Description

Given $R\in\mathbb{Z}^{n\times d}$ (or floats) whose rows generate a pointed cone
$\mathcal{C} = \{\sum_i \lambda_i R_i : \lambda_i \geq 0\}$, this package provides:

- **`exhaustive`** -- the indices of the unique minimal generating subset, i.e. the extremal rays.
- **`sample`** -- a cheaply certified subset of them (an inner bound; no completeness claim).
- **`verify`** -- an audit of an answer from explicit certificates, re-derived and re-checked rather than read off solver status codes.

The standard method (CYTools' `Cone.extremal_rays`, cddlib, lrs) asks per ray: "is it a
non-negative combination of the other $n-1$?" Redundant rays answer fast; each *extremal*
ray needs an **infeasibility proof** for a large degenerate system. This package never asks
that: candidates are tested only against the small confirmed-extremal set, and extremality
is established constructively.

`verify`'s independence has a limit worth stating: its certificates are built independently
of `exhaustive`, but both share this package's preprocessing, so an error there is invisible
to it.

### Prior art

Redundancy removal is classical: the double description method (Motzkin et al.) in Fukuda's
[cddlib](https://people.inf.ethz.ch/fukuda/cdd_home/), reverse search in Avis's
[lrslib](http://cgm.cs.mcgill.ca/~avis/C/lrs.html), plus
[Normaliz](https://www.normaliz.uni-osnabrueck.de/) and [polymake](https://polymake.org/).
Those are the right tools for most cones. This one targets the regime they handle poorly:
many generators, high dimension, a mostly-redundant majority. See [Benchmarks](#benchmarks).

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

Dependencies: numpy, scipy, highspy, with version floors that CI installs and tests.
`python-flint` is optional; without it a pure-Python fallback is used.

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

An LP finds $w$ with $w\cdot s\geq 1$ on all rays (this is also the pointedness check);
scaling onto the slice $w\cdot x=1$ turns conic redundancy into point-hull redundancy.
Each candidate $p$ is tested against the confirmed set $E$ with

$$ \max\ c\cdot p \quad \text{s.t.} \quad c\cdot e \leq 0\ \forall e\in E, \quad -1\leq c_i\leq 1, $$

always feasible and bounded. Value $0$ means $p\in\text{cone}(E)$ (Farkas) -- redundant,
however incomplete $E$ is. A positive value instead proves $E$ is *missing* a ray, and the
optimizer $c$ locates it by ray shooting: the tie-broken maximizer of $c\cdot s$ is provably
a vertex, joins $E$, and $p$ is retested. Total LP count is $\leq n+|E|$, all small, on one
persistent warm-started HiGHS model.

Both oracles unit-normalize their inputs, so a fixed tolerance means the same thing for
every candidate. Without that, slice coordinates under an anisotropic $w$ vary by orders of
magnitude and rays get dropped below `tol`.

**The failure mode to know about** is a ray *missing* from the result, since that breaks
generation. Guards: near-tolerance verdicts are re-decided in exact rational arithmetic for
integer rays; float input warns instead; `verify` re-checks both directions. What remains is
a real limit -- no float LP separates "inside the cone" from "$10^{-15}$ outside" -- so pass
integer rays when you have them.

Details, including the numerical failures that shaped the design, are in the `_SeparationOracle`
and `_MembershipOracle` docstrings in [`core.py`](src/extremal_rays/core.py).

## Benchmarks

Against CYTools, lrs, cddlib and Normaliz on families built from the Kreuzer-Skarke
polytopes. Every method's answer is checked against this package's before it is timed, and
process start-up is subtracted from the external tools.

![Toric Mori cone](docs/benchmark_prior_art.png)

![Mori-cone cap](docs/benchmark_cap_scaling.png)

Toric Mori cone of the $h^{1,1}=491$ CY -- 3509 rays in 491 dimensions, 884 extremal:

| method | time |
| --- | --- |
| this package | **~13 s** |
| CYTools `extremal_rays` | does not finish |
| full certificate audit (optional) | ~11 s on 8 workers |

Largest run to date: the Mori-cone *cap* of the same CY, 10,026,843 rays in 491 dimensions
(1,218 extremal), in 79.6 min with `n_workers=8` and the slice functional supplied via `w=`.

## Citation

If you use `extremal-rays` in your research, please cite it:

```bibtex
@software{extremal_rays,
  author  = {MacFadden, Nate},
  title   = {extremal-rays},
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

Comparisons are skipped for tools that are not installed, and every method's
answer is checked against this package's before it is timed. Add `--plot-only`
to redraw a figure from results already measured.

## License

GPL-3.0-or-later (matching [CYTools](https://github.com/LiamMcAllisterGroup/cytools)).
