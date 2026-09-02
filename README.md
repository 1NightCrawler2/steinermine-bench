# SteinerMineBench

[![Data licence: CC BY 4.0](https://img.shields.io/badge/data-CC--BY--4.0-blue.svg)](LICENSE)
[![Code licence: MIT](https://img.shields.io/badge/code-MIT-green.svg)](LICENSE-CODE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22239846.svg)](https://doi.org/10.5281/zenodo.22239846)

A citable benchmark suite for **geotechnically-weighted Steiner trees on 3-D voxel grids** — the
underground mine ramp network design problem.

28 frozen instances. Each ships a cost grid, complete metadata, a reference solution, and
**provably valid lower bounds — one per cost track**, so you can run your own solver on identical
inputs and report an optimality gap that means something. The two instances with no reference
solution still carry a bound.

> ### Why `constrained` is the normative track
>
> The obvious objective — pure voxel Steiner cost — is the optimum of a *relaxation* that admits
> geometry no ramp can be built to. Measured on solutions optimal for it, **100 % of descending
> steps exceed a 20 % grade limit and the median step grade is 100 % (45°)**, and re-ranking the
> solved instances on buildable cost flips the winning family on **15 of 23**.
>
> So **`constrained`** is normative here: cost integrated along a centreline that satisfies a stated
> geometric standard, re-checked by a **verifier shipped with the suite**
> (`steinerbench/buildable_check.py`, numpy only — no solver required). The relaxation is retained
> as the **`raw`** diagnostic track, because it is exactly reproducible and because the gap between
> it and a buildable answer is itself a result.
>
> A third **`total`** track adds life-of-mine haulage, and **Group E** varies tonnage across the
> point where the cheapest topology changes — a capex-only objective cannot express that trade-off.

> ### Why each track carries its own bound
>
> Bounding every track with the relaxation bound reads **+200 % to +340 %** on the normative
> `constrained` track, and nearly all of that is the relaxation being irrelevant rather than any
> search being poor. Each track is therefore bounded by something valid *for that track*: `raw`
> keeps the relaxation bound, while `constrained` and `total` add a **geometric floor** derived from
> the grade limit itself, which needs no search. Mean constrained gap under the per-track bounds:
> **113 %**, against 245 % when the relaxation bound was used throughout.
>
> Three defects in the `total` track were found and fixed while building this: operating cost was
> measured on the **raw** voxel network while the capex charged was the **constrained** one; the
> published `opex_model` rates were not actually applied by the solver; and a $1.5 M per-portal
> establishment charge was included but never declared. All three are corrected in the figures
> published here.

> ### All data here is synthetic
>
> Every instance is **procedurally generated** from a recorded seed. The rock mass quality fields,
> fault geometry, topography, portal locations and production-zone geometry were constructed to be
> *geotechnically representative* of a hard-rock underground operation, using the published NGI
> Barton Q-system support classes.
>
> **They are not derived from, measured at, or descriptive of any real or operating mine.** The
> coordinate frame is a neutral local origin at (0, 0, 0) with no geographic meaning. This statement
> is repeated in every `metadata.json` and every per-instance `README.md`.

---

## Quick start (under 10 minutes)

```bash
git clone https://github.com/1NightCrawler2/steinermine-bench.git
cd steinermine-bench
pip install -r requirements.txt        # numpy, scipy, jsonschema

python score.py                        # see the frozen references
python examples/minimal_solver.py --only scale-130k zones-04 > sub.json
python score.py --submission sub.json  # score it
```

The whole repository is about 10 MB. You do **not** need the reference solver, a GPU, or any
mining software — only numpy and scipy.

### Loading an instance

```python
from loader import load_instance

inst = load_instance("xgrid-f2x-qpoor")

inst["cost_grid"]      # float32 (126, 82, 100), support cost in $/m; 1e9 = not excavatable
inst["passable"]       # bool, True where the voxel can be excavated
inst["fault_count"]    # uint8, distinct faults intersecting each voxel
inst["portal_voxels"]  # list of (i, j, k) — the root terminal
inst["zone_voxels"]    # list of lists of (i, j, k) — one per production zone
inst["metadata"]       # grid geometry, cost model, faults, zones, generation seed
inst["reference"]      # reference_type, reference_cost, lower_bound, per-family results
```

`list_instances()` enumerates them; `world_from_voxel(ijk, metadata)` converts to metres.

---

## The problem

Find a minimum-cost network that connects a surface **portal** to every underground **production
zone**, on a 26-connected voxel graph, subject to **monotone descent** — a ramp may never go up and
back down — **and to a geometric standard the network must actually be buildable to**.

### The geometric standard (normative track)

| constraint | value | why |
|---|---|---|
| max grade | 20 % (1:5) | limit for a rubber-tyred haulage fleet |
| min turn radius | 25 m | outside kerb radius of an articulated truck |
| swept volume | 5.0 m span passable | the **bore**, not the centreline: a centreline sampled one voxel per step can thread a 5 m opening through a fault block or through air |
| continuity | network connected | segment ends meet; a terminal is an excavation with extent, so two openings reaching the same portal or stope are joined through it |
| pillar separation | ≥ 5 m | distinct openings leave rock between them, except inside a shared junction |

Two of these carry judgement calls, and both are stated rather than buried because they change
verdicts. **`min_pillar_m = 5 m` is a placeholder** — a site sets it from ground conditions and
span. And two drives leaving one junction at R_min need roughly `sqrt(R·(pillar+span)) ≈ 16 m`
before a full pillar between them is geometrically possible at all; inside that they are one
flared excavation. Breaches are therefore reported **with their distance from the shared
junction**, so a junction-flare finding is not confused with two independent drives sharing rock.

Turn radius is measured as the **circumradius** of consecutive vertex triples, not `L/dθ`. The
latter under-reads by `sinc(dθ/2)`: on a known 25 m circle it gives 24.9844 at four samples against
the circumradius's exact 25.0000. A verifier that under-reads condemns correct geometry by
millimetres, and a planner and a checker using different estimators cannot be reconciled by tuning
either one.

Level drives (crosscuts) are exempt from the **ramp** turning circle — they are unsmoothed voxel
staircases and this model has no crosscut design step — but are checked for grade, passability,
continuity and separation.

**Objective.** Total cost of the network, with edge weight

```
w(u, v) = (cost_grid[v] + excavation_rate_per_m) · ‖v − u‖ · cell_size_m
```

`excavation_rate_per_m` is $1,000/m and is deliberately **not** baked into the grid, so the shipped
array is pure geotechnical support cost. The exact formula is repeated in
`metadata["cost_model"]` on every instance — read it from there rather than hard-coding.

**Cost model.** Support cost is a step function of the Barton Q rock mass quality index, using the
six-tier NGI schedule:

| Q | $/m | Support class |
|---|---|---|
| ≤ 0.04 | 7,585.4 | Exceptionally poor — steel sets + shotcrete + spiling |
| ≤ 0.10 | 5,679.5 | Extremely poor — heavy systematic bolt + mesh + shotcrete |
| ≤ 0.34 | 4,460.9 | Very poor — systematic bolt + mesh + shotcrete |
| ≤ 1.181 | 3,059.0 | Poor — systematic bolting + shotcrete |
| ≤ 6.0 | 1,806.9 | Fair — spot bolting |
| > 6.0 | 1,111.9 | Good — unsupported / spot bolting |

Fault intercepts impose a **floor**, not a multiplier, because a fault destabilises the ground
independently of the host rock: `cost = max(tier_cost, floor)`, with the floor $3,059.0/m for one
distinct fault and $4,460.9/m for two or more.

**Conventions.** Axis order is `(EAST, NORTH, RL)`; **RL is positive upward**;
`world_m = min_coords_m + (ijk + 0.5) · cell_size_m`; impassable voxels (above ground surface) carry
`1e9`; test passability with `cost < 0.9e9` or just use `inst["passable"]`.

---

## The 28 instances

Every group holds all other parameters fixed and varies **exactly one axis**, so a difference in
solver behaviour is attributable. This is enforced structurally in `steinerbench/spec.py`: each
instance is built as `{**group_base, axis_key: value}`.

| Group | n | Varied axis | Instances |
|---|---|---|---|
| **A** Crossing grid | 12 | fault system × Q regime | `xgrid-{f0,f1,f2x,fcj}-{qcomp,qmix,qpoor}` |
| **B** Portal sweep | 4 | portal location | `portal-{south,north,east,west}` |
| **C** Zone count | 4 | number of production zones | `zones-{02,04,06,08}` |
| **D** Scale ladder | 4 | voxel resolution | `scale-{130k,1m,8m,129m}` |
| **E** Haulage ratio | 4 | tonnage moved (capex:opex) | `haul-{t05,t10,t20,t40}` |

### Group E and the threshold for branching

Group E exists because a capex-only objective cannot see the trade-off that decides whether a free
Steiner branch point is worth building. Its four members share **one cost grid** — the shipped
`cost_grid.npz` files are byte-identical — and differ only in `orebody.tonnage_scale`. Nothing in
the search reads tonnage, so geometry, `raw` cost, `constrained` cost and every buildability
verdict are identical across the group; only the `total` track moves, and it moves **linearly**.
That makes the group a clean instrument: any change of winner along it is attributable to the
capex:opex ratio and to nothing else.

The layout is not the Group A–D orebody. It is two plan-separated clusters at **interleaved**
depths — cluster A at RL 320 and 260, cluster B at RL 290 and 230, 390 m apart in plan — with the
portal far south, perpendicular to the cluster axis. The reason is a threshold condition worth
stating on its own, because it explains why **no free Steiner point pays on any Group A–D
instance**:

> A grade limit forces a ramp to travel at least `Δz/g` horizontally per `Δz` of descent. Any
> lateral repositioning **shorter** than that forced run is free — it is absorbed into distance the
> ramp had to cover anyway. Interleaving two clusters therefore only punishes a depth-ordered chain
> when
>
>     lateral separation  >  Δz / g

The Group A–D orebody sits ~140 m apart with 60 m between sublevels; at 20 % grade a 60 m drop
already forces 300 m of run, so the crossing is free and a chain is efficient. That is a real
result about those geometries, not a defect. Group E is 390 m apart with 30 m between levels, so
each crossing wastes ~240 m of ramp and a chain visiting four interleaved levels wastes ~720 m.
A tree branches once.

Fault systems are `none`, one through-going fault, two crossing faults (producing a multi-intercept
core where the higher floor applies), and a conjugate pair converging into a wedge with depth.
Q regimes are competent (median Q ≈ 8), mixed (bimodal), and poor-dominated (median Q ≈ 0.15).

Shared geometry across all instances: a 630 × 410 × 500 m domain and an analytic ridge topography,
with two structures that make routing non-trivial.

**A breached barrier.** A low-Q slab spans the whole plan at RL 340–400, between the surface and
the orebody, so every decline must cross it. Two **competent windows** breach it — at EAST
235–275 m and 350–395 m, confined to the slab's RL band. Driving through the intact lid costs
2.0–3.1× more per metre than going through a window (measured on the shipped grids), so a solver
must weigh a lateral detour against a more expensive descent. The windows straddle the north/south
portal at roughly equal offset, so which one wins is decided by the orebody layout and the faults;
the east and west portals sit close enough to one window to settle it outright.

> **Not the solver's `--corridor`.** These windows are a geological feature of the synthetic rock
> mass. They are unrelated to the MineOptimizer solver's `--corridor` search mask — a tube of
> `--corridor-radius` metres around WP2 A\* paths, used to shrink the Dijkstra graph. The benchmark
> never enables that mask; reference runs search the full grid.

**Fault damage.** Fault halos raise the cost floor to $3,059/m for one intercept and $4,461/m where
two cross. This is the strongest single driver of route shape: on the two-crossing instances,
14.3% of the passable domain is faulted but only ~1% of network length runs on a fault.

### The scale ladder is a genuine controlled experiment

The Q field is drawn once on a **fixed 10 m world-space lattice** and trilinearly interpolated to
voxel centres; all structural features are closed-form predicates with metre-valued parameters. So
all four rungs sample *the same geology* and differ only in discretisation.

| Instance | Cell | Dims | Voxels |
|---|---|---|---|
| `scale-130k` | 10 m | 63 × 41 × 50 | 129,150 |
| `scale-1m` | 5 m | 126 × 82 × 100 | 1,033,200 |
| `scale-8m` | 2.5 m | 252 × 164 × 200 | 8,265,600 |
| `scale-129m` | 1 m | 630 × 410 × 500 | 129,150,000 |

**The coarse rung cannot carry a constrained reference.** At a 10 m cell the turning radius is only
2.5 cells, and the rendered R_min arc cannot be held inside the corridor: measured per-leg pass
rates are 4/4 at `R_min/cell` of 10 and of 5, but 3/4 at 2.5. A coarse grid does not make the
network cheaper, it makes the answer wrong. `scale-130k` is therefore shipped and scored on the
`raw` track only, flagged `constrained_admissible: false`, rather than allowed to emit infeasible
results that read as findings. The rule is stated in
`metadata.tracks.constrained_admissible_rule`.

Two consequences worth knowing:

- A per-voxel RNG, or a fault halo specified in *voxels* rather than metres, would silently change
  the underlying problem between rungs. This benchmark therefore **departs from the reference
  solver's production pipeline**, which widens the fault cost floor with a 3×3×3 voxel dilation
  (30 m wide at a 10 m cell, 3 m at a 1 m cell). Here the damage halo is specified in metres and no
  voxel dilation is applied. `metadata["cost_model"]["fault_floor_rule"]` states this verbatim.
- **`scale-129m` ships with no reference solution.** A multi-source Dijkstra over 129 M voxels needs
  roughly 22 GB for the CSR graph alone, beyond the machine that built this suite. The grid,
  terminals and cost model are fully specified, so it stands as an open scaling frontier — the first
  valid solution submitted becomes its best-known bound.

### Topology family gating

The reference solver ranks seven topology families, four of which need a minimum number of
sublevels: `two_branch` and `chained_fan` need ≥ 2, `three_branch` and `hybrid_chained_fan_branch`
need ≥ 3. `zones-02` therefore exercises only 5 of the 7. This is a consequence of the varied axis,
not a defect; `metadata["topology_families"]["applicable"]` records it per instance, and `score.py`
only compares families present in both reference and submission.

---

## Reference solutions and what `exact` means

`reference_type` is now reported **per track**, because the labels mean different things on each.
`reference_type_raw` keeps the relaxation semantics below. `reference_type_constrained` is **always
`best_known`** — the constrained geometry is planned by a heading-augmented A\* with junction
iteration, which is heuristic, so nothing on that track is provably optimal and claiming otherwise
would be an overclaim. Where a rung is too coarse to carry constrained geometry it reads
`not_admissible` (see the scale ladder note below).

> ### References are solved on the CPU, and that is deliberate
>
> `solve_reference.py` runs `MINEOPT_FLOOD_ENGINE=scipy` with `MINEOPT_FORCE_CPU=1`: true Dijkstra,
> no relaxation tolerance, no CuPy reduction-order variance. A frozen reference has to be
> reproducible by whoever downloads it, which is what lets an instance be labelled `exact` at all.
> Every `reference.json` records `environment.bit_reproducible` so you can check.
>
> A GPU path exists (`MINEOPT_BENCH_GPU=1`) and is about **1.9× faster**, but it is *not* used for
> references, and the reason is measured rather than assumed. On `zones-04` it moves the **raw**
> track by +0.4 % — but the **constrained** track by **+5.9 %**, and it changes that track's winning
> family from `sequential_ramp` to `chained_fan`. The constrained track is the normative one, so a
> ~2× speed-up is not worth reordering it. The cause is not float noise: the constrained track runs
> through `lattice_gpu.plan_leg`, a different planner implementation, seeded by a flood that relaxes
> to 1e-3.
>
> The GPU path *is* used for `make_3d_views.py`, whose solves only supply geometry for the viewer
> and whose costs nobody cites. A viewer page may therefore differ slightly from the
> `reference.json` for the same instance; **the reference is authoritative.**
>
> Anything solved under `MINEOPT_BENCH_GPU=1` is forced to `best_known`, records
> `bit_reproducible: false` with the GPU, driver and CuPy version, and `validate.py` **fails** a
> bundle that claims `exact` on such a solve or that omits the disclosure entirely.
>
> To parallelise the CPU run — 26 of the 28 instances are ≤1 M voxels and each solve is its own
> subprocess, so this scales close to linearly and changes no result:
>
> ```bash
> python solve_reference.py --all --jobs 6
> ```

Each `reference.json` carries a `reference_type`:

**`exact`** — the cheapest family on this instance uses a closed-form argmin over *all* passable
voxels, and that argmin was **independently recomputed from scratch**. The verification
(`steinerbench/verify_exact.py`) shares no code and no distance fields with the reference solver: it
builds its own CSR graph, its own direction filters, its own `scipy.sparse.csgraph` Dijkstra and its
own argmin in float64, then asserts the junction voxels and the family cost agree. A mismatch aborts
the run rather than quietly downgrading the label. Re-running the solver's own argmin would prove
nothing — it would just agree with itself — which is why the check is built this way.

**`best_known`** — the cheapest family uses a greedy or coordinate-descent search
(`sequential_ramp`, `chained_fan`, `hybrid_chained_fan_branch`), so the cost is an upper bound that
may be improvable. Exact-search families on the same instance are still verified; see
`per_family[].exactness_check`.

**`unsolved`** — no reference solution computed: `scale-129m` (a 22 GB memory limit) and
`scale-8m` (a search-budget limit — the same geology at a 5 m cell plans every leg, so this is not
the discretisation floor). Both still carry valid constrained and total lower bounds, because the
geometric floor needs no solution. Neither is a claim that no network exists.

### The lower bounds, per track

Every instance carries **provably valid** lower bounds in `reference.lower_bounds`, one per cost
track, so each `gaps_to_lower_bound[track]` is a genuine upper bound on how far that track's
reference could be from optimal. Two different arguments are used, because the raw track and the
constrained track are different problems:

| track | bound | why that one |
|---|---|---|
| `raw` | pairwise divergence on the relaxation | the raw objective *is* the relaxation |
| `constrained` | `max(` pairwise divergence, geometric grade floor `)` | keeps the constraint that dominates the cost |
| `total` | the same, plus floors on haulage, ventilation and portal establishment | bounds every term the track charges |

Bounding each track with something valid for that track, rather than reporting the relaxation
bound against all three, cuts the mean constrained gap from **245 % to 113 %**.

#### The relaxation bound (raw track)

The method is the **pairwise-divergence bound**. In any arborescence the paths to two
terminals `z_i` and `z_j` share a prefix and separate at some node `v`, beyond which the two
branches are arc-disjoint. Hence for *every* pair

```
OPT  ≥  min_v [ d(portal, v) + d(v, z_i) + d(v, z_j) ]
```

and the maximum over pairs is valid. It costs one Dijkstra from the portal plus one reverse Dijkstra
per zone.

Three points that matter if you are reading the associated paper:

- The analogous all-terminals expression `min_v [d(portal,v) + Σ_k d(v,z_k)]` is **not** a lower
  bound. It forces every branch to diverge at a single node, which is the `single_junction`
  *feasible solution* and therefore an **upper** bound. Pairs are the largest subset for which the
  divergence argument is forced. Likewise, the sum of shortest descending paths is an upper bound
  (the union of K independent paths is feasible), not a lower one.
- **Validity across all seven families.** Each family produces a feasible portal-rooted subgraph over
  the descent arc set, so `reported_cost ≥ subgraph_cost ≥ OPT_arborescence ≥ bound`. The first step
  is an equality for `chained_fan`, which counts shared ramp voxels once, and strict for
  `sublevel_fan`/`two_branch`/`three_branch`, which charge a shared upper ramp once per branch — so
  the bound holds a fortiori for the double-counting families.
- Arcs are priced at `(min(cost[u], cost[v]) + excavation_rate) · length`, a pointwise lower bound on
  both traversal conventions the reference solver uses, so the bound survives regardless of which
  direction a segment was charged in.

#### The geometric grade floor (constrained and total tracks)

A relaxation bound cannot be tight against a constrained cost, because the relaxation throws away
the very constraint that dominates the answer: a raw network descends at 45°, a buildable one at
20 %, and the buildable one is therefore about three times longer. So the constrained track is
bounded from the constraint itself instead.

Partition any centreline into segments. The grade limit says `|dz_i| ≤ len_i · sin(θ_max)` with
`θ_max = atan(g)`, and the vertical changes must sum to the net drop, so

```
L  ≥  Δz / sin(atan(g))
```

for **any** grade-feasible path of any shape — spirals and switchbacks included; nothing about the
path enters beyond its net drop. The network must reach every zone, so its length is at least the
largest per-zone requirement, and every metre costs at least the cheapest support tariff *present in
that instance* plus the excavation rate. On a 250 m drop at 20 % that is 1,262 m of ramp before a
single dollar of routing is considered.

The floor needs **no search and no grid** — only the portal RL, the zone RLs and the tier
histogram, all of which live in `metadata.json`. Three consequences:

- Every bound in the suite recomputes in **under a second**
  (`solve_reference.py --all --recompute-bounds-only`), and `--check` makes bound drift a CI failure.
- `lower_bounds.constrained.audit` echoes every input, so you can re-derive the number from the
  record alone with a calculator. `validate.py` does exactly that, from `metadata.json`, and fails
  on any mismatch — the bound is verified, not trusted.
- The two **unsolved** instances (`scale-129m`, `scale-8m`) still ship a real constrained bound.
  An open frontier with a proven floor is a better open problem than one with nothing.

#### The floor must never be applied to the raw track

It exceeds the raw reference cost on **19 of the 25** solved instances — and that is correct, not a
bug. A raw network is not grade-feasible, so it is legitimately cheaper than any ramp anyone could
drive; quoting the floor against it would publish a "lower bound" above a network we actually
built. `geometric_bound.track_bound` raises if you try, and `validate.py` fails if a bundle ever
shows it. That the relaxation optimum sits *below* a valid floor for the real problem is the
sharpest single statement of why the relaxation is the wrong objective.

#### Not offered: Wong dual ascent

An earlier build offered `--dual-ascent` (the classical LP-dual bound for directed Steiner
arborescence). It is gone. Dual ascent grows its root component one zero-reduced-cost arc at a
time, and on a 26-connected lattice with millions of near-parallel arcs each step is worth a few
dollars: on `portal-north`, 30,000 iterations over ~85 minutes reached $1.61 M against the *free*
trivial bound's $1.68 M and pairwise divergence's $2.31 M — having saturated 0.45 % of the arcs
with no terminal yet connected to the root. It was the strongest bound on exactly one instance of
twenty-five, `scale-130k`, the coarsest rung at a 10 m cell, which is consistent with the method's
known preference for sparse graphs. `validate.py` now fails on any reference still carrying it.

---

## Scoring your solver

```
optimality_gap = (solver_cost − reference_cost) / reference_cost
```

Write a submission JSON (schema: `schemas/submission.schema.json`):

```json
{
  "solver": "MyDirectedSteiner v0.3",
  "authors": "A. Researcher",
  "url": "https://github.com/…",
  "hardware": "32-core EPYC 7543, 256 GB RAM",
  "results": {
    "xgrid-f2x-qpoor": {
      "cost": 2874112.5,
      "topology": "chained_fan",
      "junctions_voxel": [[61, 34, 63]],
      "runtime_s": 42.1
    }
  }
}
```

then

```bash
python score.py --submission mysolver.json                # constrained track (normative)
python score.py --submission mysolver.json --track raw    # diagnostic relaxation
python score.py --submission mysolver.json --track total  # capex + life-of-mine opex
```

Omitted instances are reported as *unattempted*, not counted as failures — partial submissions are
fine.

### Three tracks

**`constrained`** (default, **normative**) is the cost integrated along a centreline that satisfies
the geometric standard above. Submit `paths_world_m` and it is re-checked with the shipped
verifier against the shipped grid, so the track does **not** depend on your post-processor — or on
ours. A network that fails any of the five checks has no constrained cost; it is not scored at a
higher number, it is recorded as infeasible.

**`raw`** (diagnostic) is the relaxation objective, bit-for-bit reproducible: pure voxel
Steiner cost under monotone descent and nothing else. It is no longer normative for the reason in
the banner at the top — it is the optimum of a relaxation. It is kept because it is exactly
reproducible without any geometry code, because it is what most of the literature optimises, and
because the ratio between it and the constrained cost is a headline number in its own right.

**`total`** is constrained capex plus life-of-mine operating cost under
`metadata.tracks.opex_model` — haulage as `tonnes × distance along the network`, plus ventilation
and pumping per metre-year. Note that haulage is a life-of-mine **total** and does not scale with
mine life; only ventilation and pumping do. The two therefore push in opposite directions: a longer
mine life penalises a *longer* network, while more tonnes reward a *shorter route*. Only the second
can make a branch point pay.

A cost is only reported on the constrained or total track if the geometry was planned under the
standard **and** passed the verifier. Several families produce a `cost_buildable` figure from a
geometric post-processor without ever being shown constructible. Treating those as comparable
would conflate a planned network with an estimated one, and separating them is much of what this
suite is for.

### Negative gaps

- On a **`best_known`** instance, a negative gap is a **new best-known bound** — exactly what this
  benchmark is for. See below.
- On an **`exact`** instance, a negative gap is an **error**: the reference is provably optimal for
  its family, so either the submission is infeasible (check monotone descent, 26-connectivity, and
  that every zone is actually reached) or the reference is wrong. `score.py` flags both loudly and
  never averages them silently into the summary.

---

## Submitting an improved bound

Improvements to `best_known` instances are welcome and expected — that is why the label exists.

Open a pull request containing your submission JSON with **`paths_voxel` included** for each
improved instance, so the network can be re-costed straight off the grid and verified. Please also
give the solver name/version and a link to source or a preprint. Verified improvements are merged
into the frozen references with attribution, and the benchmark version is bumped.

If you believe you have beaten an `exact` reference, please open an issue rather than a pull
request — a genuine counterexample is a benchmark bug and we want to fix it.

---

## Repository layout

```
instances/<id>/  cost_grid.npz  metadata.json  reference.json
                 reference_paths.npz  README.md
loader.py                what you import
score.py                 gap, leaderboard, results.csv
validate.py              bundle completeness and self-consistency
generate.py              rebuilds every instance from its seed
solve_reference.py       recomputes references (needs the reference solver)
make_tables.py           paper-ready CSV + LaTeX tables
schemas/                 JSON Schema for metadata, reference, submission
tables/instance_manifest.csv   machine-readable manifest of all 28 instances
examples/minimal_solver.py
steinerbench/            the package (tiers, spec, geology, grid, loader,
                         verify_exact, lower_bound, mineopt_adapter)
```

### Scope

The MineOptimizer pipeline has three stages, and this benchmark
**exercises stage 3 only**: it synthesises the stage-1 and stage-2 artefacts
from each bundle and calls the stage-3 optimiser, bypassing the ordinary
kriging of stage 1 and the A\* pathfinding of stage 2. That is deliberate — it
removes geostatistical and search variability so a reference run is exactly
reproducible — but it means the suite validates the **network optimiser**, not
the grade-shell extraction or the interpolation.

Note that stage 3 does **not** consume stage 2's A\* paths: those are read only
under the solver's `--corridor` search mask, which reference runs never enable.
What stage 3 takes from stage 2 is the terminal geometry alone — the portal and
production-zone voxel sets — and that is fully specified in every
`metadata.json` and reconstructed by the loader.

### `cost_grid.npz`

Stores the **post-fault-floor tier index** as uint8 rather than float costs, plus the tier schedule
and the surface RL. This is lossless — both fault floors are themselves tier costs, so applying the
floor maps a tier index onto another tier index — and it compresses a 129 M-voxel grid to about
5 MB, which is why no Git LFS or out-of-band download is needed anywhere in the suite. The loader
reconstructs float32 $/m for you.

---

## Reproducing and verifying

```bash
python validate.py            # completeness, checksums, schemas, re-costing, exactness
python validate.py --fast     # skips the two largest instances
python generate.py --check    # regenerate every grid, diff SHA-256 against the shipped set
python make_tables.py         # rebuild tables/
```

`validate.py` re-costs every stored reference path straight off the grid using the published edge
weight and asserts it matches `reference.json` — an independent check that the shipped numbers
correspond to real, connected networks on the shipped grids.

---

## Citing

Please cite the dataset:

> Hasözdemir, K. (2026). *SteinerMineBench: a benchmark suite for
> geotechnically-weighted Steiner trees on voxel grids* (v1.0.0) [Data set].
> Zenodo. https://doi.org/10.5281/zenodo.22239846

```bibtex
@dataset{hasozdemir_steinerminebench_2026,
  author    = {Hasözdemir, Kürşat},
  title     = {{SteinerMineBench}: a benchmark suite for
               geotechnically-weighted {Steiner} trees on voxel grids},
  year      = {2026},
  publisher = {Zenodo},
  version   = {1.0.0},
  doi       = {10.5281/zenodo.22239846},
  url       = {https://doi.org/10.5281/zenodo.22239846}
}
```

The DOI above is the **concept DOI** — it always resolves to the newest
version of this dataset, so it's the one to cite rather than a version-specific
DOI. It's also recorded in `CITATION.cff` and `.zenodo.json`.

Machine-readable metadata is in `CITATION.cff`, so GitHub's **Cite this
repository** button and most reference managers pick it up directly.

The accompanying paper — *Geotechnically Weighted Steiner Trees on Voxel Grids:
Rock-Mass-Constrained Optimisation of Underground Ramp Networks for Minimum
Excavation and Support Cost* — is in preparation. Once it is published, please
cite both.

## Licence

- **Data** (`instances/`, `tables/`) — CC-BY-4.0, see `LICENSE`
- **Code** (everything else) — MIT, see `LICENSE-CODE`
