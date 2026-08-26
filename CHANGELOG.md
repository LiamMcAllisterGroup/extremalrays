# Changelog

All notable changes to this project are documented here. Versions follow
[semantic versioning](https://semver.org/); a release that can change which
rays are returned is a minor bump at minimum, and says so here.

## [0.4.0] -- unreleased

### Changed -- results

- **Answers can differ from 0.3.0.** Two independent tolerance defects were
  fixed (below). On the bundled `h11=491` Mori cone the extremal set is
  unchanged at 884 rays; on ill-conditioned cones the result changes, always
  by gaining rays that 0.3.0 dropped.
- Results produced by **0.3.0 or earlier may omit extremal rays** without any
  warning. Re-run anything load-bearing. `verify()` from this version detects
  the omission; `LAST_PROFILE["closest_member"]` flags runs at risk.

### Fixed

- **Extremal rays silently dropped near the tolerance.** A separation value
  just under `tol` was taken as proof of redundancy, so a genuinely extremal
  ray scoring, say, 0.375x `tol` was discarded in the main loop and the result
  no longer generated the cone. Incidence tracked conditioning: 4 of 20 random
  cones with coefficients spanning 1e6, none at 1e3. Now a verdict within
  `tol/100` is re-decided in exact rational arithmetic for integer input, and
  any run that produced one warns.
- **Float input silently reinterpreted as integers.** The integrality test
  used an absolute tolerance, so small float data was snapped onto the integer
  path: `[[1, 1e-10], [1, -1e-10]]` collapsed two distinct rays into one,
  `[[1e-10, 2e-10]]` became the zero ray, and a valid cone scaled by 1e-10
  raised "no nonzero ray". The test is now relative, so a nonzero entry can
  never round to zero while genuinely near-integral floats still qualify.
- **Float rays with a small norm were dropped.** Deduplication discarded rows
  with norm below 1e-12; a direction is scale-free, so only an exactly zero
  row is not a ray. Answers are now invariant from scale 1e-20 to 1e20.
- **`verify()` rejected correct answers.** A claimed extremal ray whose
  separation margin fell below a hardcoded 1e-7 was reported as "possibly
  redundant" -- but a nearly-redundant ray is still a ray. Such a ray is now
  re-decided by the opposite question (is it in the cone of the others?),
  where an infeasible LP proves extremality however narrow the margin.
- **`verify()` accepted some wrong answers.** Its membership tolerance was
  absolute on slice coordinates, which span orders of magnitude under an
  anisotropic `w`. The question is now asked about the unit-normalized ray,
  making the tolerance mean the same thing for every candidate.

### Added

- `verify(w=...)`, mirroring `exhaustive`, so a result can be audited without
  re-solving a pointedness LP that may be impractical at scale.
- `verify(sep_tol=...)` to control the separation-margin escalation.
- `LAST_PROFILE` gains `n_exact_checks`, `n_near_tol_rescued`.
- `py.typed`: the package ships its inline annotations.

### Performance

- Separation LPs run primal simplex (`simplex_strategy=4`). Only the objective
  changes between candidates, which leaves the basis primal feasible, so the
  default dual simplex was re-establishing dual feasibility every solve.
  Measured 22.2 s -> 13.0 s on the `h11=491` Mori cone (1.7x) with a
  byte-identical index set, and 1.22x on the 10M-ray Mcap.
- Membership LPs (`verify`, and the cleanup pass) reuse one persistent model
  instead of rebuilding it per candidate: 24.8 -> 15.7 ms per LP. The basis is
  deliberately not carried over -- see `_MembershipOracle` for why a
  warm-started verdict can depend on query order.

### Documentation

- The `n_workers` docstring states the `if __name__ == "__main__":`
  requirement; passing `n_workers > 0` from a spawned child now degrades to a
  serial sweep with a warning instead of recursing.
- Corrected the shuffled-input figure, which was measured before the 0.3.0
  oracle change and was two orders of magnitude stale.

## [0.3.0]

- Scale-free separation oracle; restructured tests; house-style alignment.
