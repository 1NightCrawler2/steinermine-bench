"""
SteinerMineBench instance specifications -- the single source of truth.
======================================================================

SPDX-License-Identifier: MIT

Every one of the 24 benchmark instances is defined here as a plain dict built
from a group base plus **exactly one varied axis**.  Instances are constructed
as ``{**BASE, axis_key: value}`` so that unintended drift between the members
of a sweep is structurally impossible -- that is what makes each group a
controlled experiment rather than a bag of unrelated problems.

All geometry is expressed in a **local, neutral coordinate frame** with origin
at (0, 0, 0).  No real-world survey coordinates appear anywhere in this
benchmark; see PROVENANCE_STATEMENT.

Axis order is (EAST, NORTH, RL) = (x, y, z) with **RL positive upward**, matching
the reference solver's convention.  World position of voxel (i, j, k) is
``min_coords + (ijk + 0.5) * cell_size``.

Units
-----
All lengths in metres (m); all costs in US dollars per metre ($/m);
``tonnage_mt`` in millions of tonnes; ``mean_grade`` in g/t.
"""

from __future__ import annotations

BENCHMARK_VERSION = "2.1.0"
SCHEMA_VERSION = "2.1"
#: Unchanged at 2.0.0 on purpose: v2.1.0 alters no grid, no terminal set and no
#: cost grid, so every instance still reproduces bit-for-bit under
#: ``generate.py --check`` against the 2.0.0 checksums.
GENERATOR_VERSION = "2.0.0"

# ---------------------------------------------------------------------------
# What changed in 2.0.0, and why
# ---------------------------------------------------------------------------
# v1 scored the pure voxel Steiner cost: minimise support + excavation over a
# 26-connected descent graph.  Measured on v1's own reference solutions, 100 %
# of descending steps exceed a 20 % grade limit and the median step grade is
# 100 % (45 deg).  The v1 objective is therefore the optimum of a RELAXATION
# that admits geometry no ramp can be built to, and its ranking is an artefact
# of that: re-ranking the 23 solved instances on buildable cost flips the
# winning family on 15 of them.
#
# v2 keeps the v1 objective exactly as it was -- it is reproducible, it is what
# the literature optimises, and the gap between it and a buildable answer is
# itself a headline -- but demotes it to a diagnostic track and makes
# `constrained` normative.  A third `total` track adds life-of-mine haulage,
# because capex alone cannot express a result v1 structurally could not see:
# which topology wins depends on the capex:opex ratio, and a free Steiner
# branch point that loses on capex can win on total cost once enough tonnes
# move over it.  Group E varies exactly that ratio.
NOTABLE_CHANGES_2_0_0 = (
    "Normative track is now `constrained`, not `raw`; `raw` is retained "
    "unchanged as a diagnostic. New `total` track adds life-of-mine haulage. "
    "New Group E (haulage ratio) varies tonnage across the capex/opex "
    "breakeven on an interleaved-cluster layout. All `exact` labels are "
    "withdrawn on the constrained track -- a constrained search with junction "
    "iteration is heuristic, so every constrained reference is `best_known`."
)

# ---------------------------------------------------------------------------
# What changed in 2.1.0, and why
# ---------------------------------------------------------------------------
# v2.0.0 reported ONE lower bound -- pairwise divergence on the raw relaxation --
# and quoted it against every track.  Against the normative constrained track it
# read +200 % to +340 %, and essentially all of that was the relaxation being
# irrelevant rather than the search being poor: the relaxation discards the grade
# limit, which is precisely the constraint that triples ramp length.  A gap that
# large tells a reader nothing about whether a network is good.
#
# v2.1.0 bounds each track with something valid FOR THAT TRACK.  The raw track
# keeps pairwise divergence unchanged.  The constrained and total tracks add a
# geometric floor: a drive cannot descend faster than its grade limit, so
# reaching a zone `drop` metres down needs at least `drop / sin(atan(g))` metres
# of centreline, and every metre costs at least the cheapest support tariff in
# the model plus the excavation rate.  That floor is trigonometry over
# metadata.json -- no search, no grid -- so it can be re-derived and audited in
# seconds, and it even gives the unsolved 129 M-voxel rung a real bound.
#
# The floor must NEVER be applied to the raw track: an unconstrained network may
# descend at 45 degrees and is legitimately cheaper than any buildable ramp -- it
# exceeds the raw reference cost on 19 of the 25 solved instances.
# `geometric_bound.track_bound` refuses the combination outright.
#
# Three defects in the `total` track were found and fixed while doing this, all
# of which made v2.0.0 total figures unreproducible from the bundle:
#   * the operating cost was measured on the RAW voxel network while the capex
#     charged was the CONSTRAINED one -- two halves of one number measured on
#     different objects;
#   * `tracks.opex_model` was published in every metadata.json but never applied
#     to the solver, which used its own upstream rates (~2.7x higher);
#   * a $1.5 M per-portal establishment charge was included but never declared.
NOTABLE_CHANGES_2_1_0 = (
    "Per-track lower bounds: `raw` keeps pairwise divergence, `constrained` and "
    "`total` gain a geometric floor derived from the grade limit, which cuts "
    "the mean constrained gap from 245 % to 113 %. Wong dual ascent withdrawn "
    "(never competitive on a dense voxel lattice). The `total` track is fixed "
    "and now reproducible from the published OPEX_MODEL: operating cost is "
    "measured on the constrained centreline, the published rates are actually "
    "applied, and the per-portal establishment charge is declared. Grids, "
    "terminals and cost grids are untouched -- every instance still reproduces "
    "bit-for-bit against the 2.0.0 checksums."
)

# ---------------------------------------------------------------------------
# The geometric standard -- what makes a network BUILDABLE
# ---------------------------------------------------------------------------
#: These are hard constraints on the constrained and total tracks, checked by
#: the shipped verifier over the submitted centrelines.  They are properties of
#: the haulage fleet and the ground, not solver parameters, so they are frozen
#: here and repeated in every metadata.json.
GEOMETRIC_STANDARD = {
    "max_grade": 0.20,              # 1:5, the limit for a rubber-tyred fleet
    "min_turn_radius_m": 25.0,      # outside kerb radius of an artic truck
    "tunnel_span_m": 5.0,           # swept cross-section, not the centreline
    "min_pillar_m": 5.0,            # PLACEHOLDER: set to the site standard
    "portal_reach_m": 100.0,        # daylighting is the design intent here
}

#: A discretisation guard, not a preference.  The alignment is rendered as
#: R_min arcs and filleted joints; when the cell is a large fraction of R_min
#: the rendered curve cannot be held inside the corridor and legs fail for a
#: reason about the grid rather than about the rock.  Measured per-leg pass
#: rates: 4/4 at R_min/cell of 10 and of 5, 3/4 at 2.5.  Rungs below the floor
#: are shipped and scored on `raw` only.
MIN_RADIUS_CELLS_FOR_CONSTRAINED = 5.0

#: Scoring tracks.  `raw` is bit-for-bit the v1 objective.
TRACKS = {
    "raw": "Voxel Steiner cost over the descent graph; no geometric "
           "constraint. Reproducible from the bundle alone. DIAGNOSTIC.",
    "constrained": "Buildable capex: the same cost integrated along a "
                   "centreline that satisfies GEOMETRIC_STANDARD, which the "
                   "shipped verifier re-checks. NORMATIVE.",
    "total": "Constrained capex plus life-of-mine operating cost under "
             "OPEX_MODEL. The only track on which Group E varies.",
}
NORMATIVE_TRACK = "constrained"

# ---------------------------------------------------------------------------
# Axis E -- the operating-cost model
# ---------------------------------------------------------------------------
#: Haulage is tonnes x distance along the network, so it is a LIFE-OF-MINE
#: total and does NOT scale with mine life; only ventilation and pumping do.
#: That distinction matters: a longer mine life penalises a longer network,
#: whereas more tonnes reward a shorter route -- opposite pressures, and only
#: the second one can make a Steiner branch point pay.
OPEX_MODEL = {
    "haul_cost_per_t_km": 0.35,
    "haul_vert_cost_per_t_m": 0.012,
    "vent_cost_per_m_yr": 12.0,
    "pump_cost_per_m_yr_m_depth": 0.03,
    "mine_life_years": 10.0,
    "portal_establishment_cost": 1_500_000.0,
    "note": "Haulage = tonnes x (haul_m/1000 * haul_cost_per_t_km + lift_m * "
            "haul_vert_cost_per_t_m), summed over zones, where haul_m is the "
            "shortest path from the portal to the zone ALONG THE NETWORK. "
            "Ventilation and pumping are per metre of network per year. A "
            "fixed portal_establishment_cost is charged per distinct surface "
            "exit. Every one of these terms is measured on the SAME geometry "
            "the track's capex is measured on: the constrained centreline for "
            "the constrained and total tracks. v2.0.0 shipped this model "
            "declared but not applied -- the solver used its own upstream "
            "rates and measured the operating terms on the raw voxel network "
            "-- so v2.0.0 total-track figures are not reproducible from this "
            "block and are superseded.",
}

#: Multiplies every zone tonnage.  Nothing in the search reads tonnage, so this
#: leaves the geometry, the capex and the buildability verdicts untouched and
#: scales the haulage term linearly -- a clean instrument for the capex:opex
#: trade-off.  The rungs straddle the measured breakeven (~1.43x on the
#: interleaved layout) so the axis brackets a genuine change of winner rather
#: than reporting one tuned point.
TONNAGE_SCALES = [0.5, 1.0, 2.0, 4.0]
TONNAGE_TAG = {0.5: "t05", 1.0: "t10", 2.0: "t20", 4.0: "t40"}

PROVENANCE_STATEMENT = (
    "SYNTHETIC DATA. This instance is procedurally generated from the seed and "
    "parameters recorded in metadata.generation. The rock mass quality field, "
    "fault geometry, topography, portal and production-zone locations were "
    "constructed to be geotechnically representative of a hard-rock underground "
    "operation, using the published NGI Barton Q-system support classes. They are "
    "NOT derived from, measured at, or descriptive of any real or operating mine, "
    "and the coordinate frame is a neutral local origin with no geographic meaning."
)

# ---------------------------------------------------------------------------
# Domain -- identical for every instance in the suite
# ---------------------------------------------------------------------------
DOMAIN_SIZE_M = (630.0, 410.0, 500.0)   # EAST, NORTH, RL extent
MIN_COORDS_M = (0.0, 0.0, 0.0)

#: Random Q field is drawn on this fixed world-space lattice and trilinearly
#: interpolated to voxel centres, so every cell size samples the SAME geology.
#: This is what makes the scale ladder a controlled experiment.
Q_REFERENCE_LATTICE_M = 10.0

#: Voxels within this distance of the ground surface are portal-eligible.
SURFACE_BAND_M = 12.0

# ---------------------------------------------------------------------------
# Structural features -- fixed geometry, shared by every instance
# ---------------------------------------------------------------------------
#: Low-Q "lid" spanning the whole plan between the surface and the orebody, so
#: every decline must cross it somewhere.
BARRIER_SLAB = {"rl_range_m": (340.0, 400.0), "q_multiplier": 0.05}

#: Two competent breaches through the barrier -- the cheap ways down.
#:
#: NOTE ON THE NAME: these are a *geological* feature of the synthetic rock
#: mass. They are unrelated to the MineOptimizer solver's ``--corridor`` search
#: mask (a tube of ``--corridor-radius`` metres around WP2 A* paths used to
#: shrink the Dijkstra graph). The benchmark never enables that mask. An earlier
#: version of this file called these "corridors", which collided confusingly
#: with the solver's meaning.
#:
#: GEOMETRY. They are confined to the barrier's RL range plus a small margin,
#: rather than running the full depth, so the trade-off they pose is purely
#: "detour laterally to breach the barrier cheaply, or drive straight through
#: it". They straddle the southern/northern portal at EAST 315 m at roughly
#: equal offset, so which one wins is decided by the orebody layout and the
#: fault geometry rather than by the portal alone -- except for the east and
#: west portals, which are close enough to one breach to settle it.
COMPETENT_WINDOWS = [
    {"name": "window_west", "east_range_m": (235.0, 275.0),
     "rl_range_m": (330.0, 410.0), "q_multiplier": 4.0},
    {"name": "window_east", "east_range_m": (350.0, 395.0),
     "rl_range_m": (330.0, 410.0), "q_multiplier": 4.0},
]

# ---------------------------------------------------------------------------
# Axis A1 -- fault systems
# ---------------------------------------------------------------------------
# Each fault is a plan-view trace polyline extruded along its dip vector, with a
# damage halo of `damage_half_width_m` either side of the fault plane.  Faults
# are rasterised to a per-voxel count of DISTINCT intersecting faults, exactly
# as wp1_voxel/pipeline.py:step2_voxelize does for the upstream block model.
FAULT_SYSTEMS: dict[str, list[dict]] = {
    "none": [],
    "single": [
        {
            "id": "F1", "name": "Fault_A",
            "trace_plan_m": [[60.0, 120.0], [310.0, 205.0], [570.0, 300.0]],
            "dip_deg": 75.0, "dip_direction_deg": 340.0,
            "damage_half_width_m": 18.0, "rl_range_m": (0.0, 500.0),
            "q_multiplier": 0.10,
        },
    ],
    "two_crossing": [
        {
            "id": "F1", "name": "Fault_A",
            "trace_plan_m": [[60.0, 120.0], [310.0, 205.0], [570.0, 300.0]],
            "dip_deg": 75.0, "dip_direction_deg": 340.0,
            "damage_half_width_m": 18.0, "rl_range_m": (0.0, 500.0),
            "q_multiplier": 0.10,
        },
        {
            "id": "F2", "name": "Fault_B",
            "trace_plan_m": [[100.0, 350.0], [320.0, 220.0], [540.0, 90.0]],
            "dip_deg": 80.0, "dip_direction_deg": 60.0,
            "damage_half_width_m": 15.0, "rl_range_m": (0.0, 500.0),
            "q_multiplier": 0.10,
        },
    ],
    # Same strike, opposing dips: the two planes converge with depth and
    # intersect below the orebody, forming a classic conjugate wedge.
    "conjugate_pair": [
        {
            "id": "C1", "name": "Fault_Conj_N",
            "trace_plan_m": [[40.0, 265.0], [590.0, 305.0]],
            "dip_deg": 62.0, "dip_direction_deg": 184.0,
            "damage_half_width_m": 16.0, "rl_range_m": (0.0, 500.0),
            "q_multiplier": 0.10,
        },
        {
            "id": "C2", "name": "Fault_Conj_S",
            "trace_plan_m": [[40.0, 145.0], [590.0, 185.0]],
            "dip_deg": 62.0, "dip_direction_deg": 4.0,
            "damage_half_width_m": 16.0, "rl_range_m": (0.0, 500.0),
            "q_multiplier": 0.10,
        },
    ],
}
FAULT_SYSTEM_ORDER = ["none", "single", "two_crossing", "conjugate_pair"]
FAULT_SYSTEM_TAG = {"none": "f0", "single": "f1",
                    "two_crossing": "f2x", "conjugate_pair": "fcj"}

# ---------------------------------------------------------------------------
# Axis A2 -- Q regimes (background rock mass quality)
# ---------------------------------------------------------------------------
# `lobes` is a list of (weight, median_q, sigma_lognormal).  Weights are
# normalised; the lobe assignment itself is drawn on the reference lattice so it
# is also resolution-independent.
Q_REGIMES: dict[str, dict] = {
    "competent":      {"lobes": [(1.00, 8.00, 0.60)]},
    "mixed":          {"lobes": [(0.55, 4.00, 0.50), (0.45, 0.50, 0.70)]},
    "poor_dominated": {"lobes": [(1.00, 0.15, 0.70)]},
}
Q_REGIME_ORDER = ["competent", "mixed", "poor_dominated"]
Q_REGIME_TAG = {"competent": "qcomp", "mixed": "qmix", "poor_dominated": "qpoor"}

# ---------------------------------------------------------------------------
# Axis B -- portal locations (snapped to the ground surface by the generator)
# ---------------------------------------------------------------------------
PORTAL_SITES: dict[str, dict] = {
    "south": {"label": "P_South", "plan_m": [315.0, 40.0]},
    "north": {"label": "P_North", "plan_m": [315.0, 375.0]},
    "east":  {"label": "P_East",  "plan_m": [595.0, 205.0]},
    "west":  {"label": "P_West",  "plan_m": [35.0, 205.0]},
}
PORTAL_ORDER = ["south", "north", "east", "west"]

#: Portal terminal is a sphere of this radius about the snapped surface voxel.
PORTAL_RADIUS_M = 10.0

# ---------------------------------------------------------------------------
# Axis C -- production zones
# ---------------------------------------------------------------------------
#: The four sublevels used by every instance except zones-02 (see ZONE_LAYOUTS).
SUBLEVEL_RL_M = [320.0, 260.0, 200.0, 140.0]
ZONE_RL_THICKNESS_M = 10.0
ZONE_PLAN_HALF_M = (35.0, 30.0)   # half-extent EAST, NORTH of each zone rectangle

#: (plan centre east, plan centre north, sublevel RL, tonnage_mt, mean_grade_g_t)
#: Layouts share their first entries so the sweep nests cleanly.
_Z = [
    (250.0, 250.0, 320.0, 1.40, 2.9),   # Z1
    (390.0, 195.0, 260.0, 1.10, 3.4),   # Z2
    (245.0, 155.0, 200.0, 0.95, 2.6),   # Z3
    (385.0, 290.0, 140.0, 1.25, 3.1),   # Z4
    (150.0, 215.0, 320.0, 0.70, 2.4),   # Z5  (2nd zone on sublevel 1)
    (470.0, 245.0, 260.0, 0.80, 3.7),   # Z6  (2nd zone on sublevel 2)
    (155.0, 300.0, 200.0, 0.65, 2.2),   # Z7  (2nd zone on sublevel 3)
    (480.0, 130.0, 140.0, 0.90, 3.3),   # Z8  (2nd zone on sublevel 4)
]

#: Zone-count sweep.  n=2 uses the top and bottom sublevels so the depth extent
#: matches the larger layouts; n>=4 holds the sublevel set fixed at four and
#: varies only how many terminals hang off them.
ZONE_LAYOUTS: dict[int, list[int]] = {
    2: [0, 3],
    4: [0, 1, 2, 3],
    6: [0, 1, 2, 3, 4, 5],
    8: [0, 1, 2, 3, 4, 5, 6, 7],
}

#: The default zone layout for every instance that does not vary zone count.
DEFAULT_N_ZONES = 4

# ---------------------------------------------------------------------------
# Axis E -- the interleaved-cluster layout
# ---------------------------------------------------------------------------
#: THE THRESHOLD CONDITION for branching.
#:
#: A grade limit forces a ramp to travel at least dz/g horizontally per dz of
#: descent.  Any lateral repositioning SHORTER than that forced run is free --
#: it is absorbed into distance the ramp had to cover anyway.  So splitting an
#: orebody into two plan-separated clusters only punishes a depth-ordered chain
#: when
#:
#:      lateral separation  >  dz / g
#:
#: The `_Z` layout above does not satisfy this: its clusters sit ~140 m apart
#: with 60 m between sublevels, and at 20 % grade a 60 m drop already forces
#: 300 m of run, so the crossing is free and a chain is efficient.  That is why
#: no free Steiner point pays on any Group A-D instance -- a real result about
#: those geometries, not a defect.
#:
#: Here: 390 m apart, 30 m between levels.  A 30 m drop forces 150 m of run, so
#: each crossing wastes ~240 m of ramp and a chain visiting four interleaved
#: levels wastes ~720 m.  A tree branches once.
CLUSTER_A_PLAN_M = (120.0, 330.0)
CLUSTER_B_PLAN_M = (510.0, 330.0)

#: (plan east, plan north, sublevel RL, tonnage_mt, mean_grade_g_t)
_ZI = [
    (*CLUSTER_A_PLAN_M, 320.0, 1.20, 3.0),   # ZI1  cluster A, upper
    (*CLUSTER_B_PLAN_M, 290.0, 1.10, 3.2),   # ZI2  cluster B, upper
    (*CLUSTER_A_PLAN_M, 260.0, 1.00, 2.8),   # ZI3  cluster A, lower
    (*CLUSTER_B_PLAN_M, 230.0, 1.15, 3.1),   # ZI4  cluster B, lower
]

#: Named zone tables.  `nested` is the v1 layout, unchanged.
ZONE_TABLES = {"nested": _Z, "interleaved": _ZI}
INTERLEAVED_INDICES = [0, 1, 2, 3]


def zone_entries(instance_spec: dict) -> list[tuple]:
    """The (east, north, rl, tonnage_mt, grade) rows this instance uses.

    Tonnage carries `tonnage_scale`; nothing else does, which is what makes
    Group E a controlled experiment on the capex:opex ratio alone.
    """
    layout = instance_spec.get("zone_layout", "nested")
    table = ZONE_TABLES[layout]
    idx = (INTERLEAVED_INDICES if layout == "interleaved"
           else ZONE_LAYOUTS[instance_spec["n_zones"]])
    k = float(instance_spec.get("tonnage_scale", 1.0))
    return [(table[i][0], table[i][1], table[i][2], table[i][3] * k,
             table[i][4]) for i in idx]

# ---------------------------------------------------------------------------
# Axis D -- scale ladder
# ---------------------------------------------------------------------------
#: (instance id, cell_size_m, shipped in git?)
#: Every rung ships: the uint8 tier-index format compresses a 129 M-voxel grid
#: to ~5 MB, so no Git LFS and no out-of-band download is needed anywhere in the
#: suite.  The 129 M rung is still shipped WITHOUT a reference solution -- see
#: UNSOLVED_INSTANCES.
SCALE_RUNGS = [
    ("scale-130k", 10.0, True),
    ("scale-1m", 5.0, True),
    ("scale-8m", 2.5, True),
    ("scale-129m", 1.0, True),
]

#: Instances shipped with ``reference_type: "unsolved"``.  Both stand as open
#: frontiers with the grid, terminals and cost model fully specified, so the
#: first valid solution submitted becomes the best-known reference.  Since
#: v2.1.0 both still ship a real, valid CONSTRAINED and TOTAL lower bound: the
#: geometric floor follows from the grade limit and the cheapest rock in the
#: model, so it needs neither a solution nor a grid.  An open problem that
#: ships with a proven floor is a better open problem.
#:
#: ``scale-129m``  a multi-source Dijkstra over 129 M voxels needs roughly 22 GB
#:                 for the CSR graph alone, well beyond the machine used to
#:                 build this suite -- a MEMORY limit.
#: ``scale-8m``    8.27 M voxels at a 2.5 m cell.  The grid is small enough to
#:                 solve, but the constrained planner reaches no goal on any of
#:                 the twelve families, while the SAME geology at 5 m
#:                 (``scale-1m``) plans every leg.  The radius-to-cell ratio
#:                 here is 10, the finest in the suite, so this is not the
#:                 discretisation floor -- it is a SEARCH-BUDGET limit, believed
#:                 to be ``LATTICE_MAX_EXPAND`` binding on a heading-augmented
#:                 state space of that size.  Recorded as an absence of a
#:                 reference, never as infeasibility: nothing here says no
#:                 network exists, only that this solver did not find one.
UNSOLVED_INSTANCES = {"scale-129m", "scale-8m"}

DEFAULT_CELL_SIZE_M = 5.0

# ---------------------------------------------------------------------------
# Topology family gating (mirrors run_steiner_poly.py:2118/2133/2177/2195)
# ---------------------------------------------------------------------------
#: ``(name, min_sublevels, search_kind)``.  ``search_kind`` describes the RAW
#: track only: an "exact" family locates its junctions by a closed-form argmin
#: over all passable voxels, so its raw cost is optimal FOR THAT FAMILY.  No
#: family is exact on the constrained track -- the geometry is planned by a
#: heading-augmented A* with junction iteration, which is heuristic.
FAMILIES = [
    ("sublevel_fan", 1, "exact"),
    ("two_branch", 2, "exact"),
    ("three_branch", 3, "exact"),
    ("single_junction", 1, "exact"),
    ("sequential_ramp", 1, "heuristic"),
    ("chained_fan", 2, "heuristic"),
    ("hybrid_chained_fan_branch", 3, "heuristic"),
    # v2 -- see mineopt_adapter._FAMILY_PREFIXES for what each one is.
    ("spiral_decline", 1, "heuristic"),
    ("switchback_decline", 1, "heuristic"),
    # 3, not 2: the solver needs a non-trivial split between the upper and
    # lower decline, so a two-level orebody has nowhere to put one
    # (run_steiner_poly.evaluate_twin_decline returns None at N < 3).
    ("twin_decline", 3, "heuristic"),
    ("conventional_decline", 1, "rule"),
    ("steiner_insertion", 2, "heuristic"),
]
EXACT_FAMILIES = {name for name, _, kind in FAMILIES if kind == "exact"}


def applicable_families(n_sublevels: int) -> list[str]:
    """Families the reference solver will actually evaluate at this level count."""
    return [name for name, min_lev, _ in FAMILIES if n_sublevels >= min_lev]


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------
def geology_seed(q_regime: str, fault_system: str) -> int:
    """
    Deterministic seed for the random Q field.

    Keyed on the *geology* (regime + fault system), NOT on the instance id, so
    that every instance sharing a geology -- the whole portal sweep, the whole
    zone-count sweep, and all four scale rungs -- samples an identical rock mass.
    Without this the sweeps would not be controlled experiments.
    """
    return (42_000
            + 100 * Q_REGIME_ORDER.index(q_regime)
            + FAULT_SYSTEM_ORDER.index(fault_system))


# ---------------------------------------------------------------------------
# Instance construction
# ---------------------------------------------------------------------------
#: Every field a fully-resolved instance spec carries.  Group bases below set
#: all of them; each instance then overrides exactly one axis key.
BASE: dict = {
    "cell_size_m": DEFAULT_CELL_SIZE_M,
    "q_regime": "poor_dominated",
    "fault_system": "two_crossing",
    "portal_site": "south",
    "n_zones": DEFAULT_N_ZONES,
    "zone_layout": "nested",
    "tonnage_scale": 1.0,
    "shipped_in_git": True,
}


def _instance(instance_id: str, group: str, axis: str | list[str],
              overrides: dict, held_fixed_ref: str | None) -> dict:
    spec = {**BASE, **overrides}
    axes = [axis] if isinstance(axis, str) else list(axis)
    spec["instance_id"] = instance_id
    spec["group"] = group
    spec["varied_axis"] = {
        "group": group,
        "axis": axes,
        "value": {a: spec[a] for a in axes},
        "held_fixed_ref": held_fixed_ref,
    }
    spec["q_seed"] = geology_seed(spec["q_regime"], spec["fault_system"])
    return spec


def _build_all() -> dict[str, dict]:
    out: dict[str, dict] = {}

    # ── Group A: crossing grid, 4 fault systems x 3 Q regimes = 12 ───────────
    for fs in FAULT_SYSTEM_ORDER:
        for qr in Q_REGIME_ORDER:
            iid = f"xgrid-{FAULT_SYSTEM_TAG[fs]}-{Q_REGIME_TAG[qr]}"
            out[iid] = _instance(
                iid, "crossing_grid", ["fault_system", "q_regime"],
                {"fault_system": fs, "q_regime": qr},
                held_fixed_ref="xgrid-f0-qcomp",
            )

    # ── Group B: portal sweep on the hardest crossing-grid instance = 4 ──────
    for site in PORTAL_ORDER:
        iid = f"portal-{site}"
        out[iid] = _instance(
            iid, "portal_sweep", "portal_site",
            {"portal_site": site, "fault_system": "two_crossing",
             "q_regime": "poor_dominated"},
            held_fixed_ref="xgrid-f2x-qpoor",
        )

    # ── Group C: zone-count sweep = 4 ────────────────────────────────────────
    for n in sorted(ZONE_LAYOUTS):
        iid = f"zones-{n:02d}"
        out[iid] = _instance(
            iid, "zone_count", "n_zones",
            {"n_zones": n, "fault_system": "two_crossing",
             "q_regime": "poor_dominated"},
            held_fixed_ref="xgrid-f2x-qpoor",
        )

    # ── Group D: scale ladder = 4 ────────────────────────────────────────────
    for iid, cs, shipped in SCALE_RUNGS:
        out[iid] = _instance(
            iid, "scale_ladder", "cell_size_m",
            {"cell_size_m": cs, "shipped_in_git": shipped,
             "fault_system": "two_crossing", "q_regime": "poor_dominated"},
            held_fixed_ref="xgrid-f2x-qpoor",
        )
        # The 5 m rung reproduces xgrid-f2x-qpoor exactly; bundles stay
        # self-contained but the relationship is recorded.
        if cs == DEFAULT_CELL_SIZE_M:
            out[iid]["varied_axis"]["duplicate_of"] = "xgrid-f2x-qpoor"

    # ── Group E: haulage ratio on the interleaved layout = 4 ─────────────────
    # Competent rock throughout and no faults, so that a change of winning
    # topology across this axis is attributable to the capex:opex ratio and to
    # nothing about the geology.  The portal sits far south, perpendicular to
    # the cluster axis, so the trunk reaching either cluster is long and shared
    # -- which is what punishes a fan and gives a branch point something to win.
    for k in TONNAGE_SCALES:
        iid = f"haul-{TONNAGE_TAG[k]}"
        out[iid] = _instance(
            iid, "haulage_ratio", "tonnage_scale",
            {"tonnage_scale": k, "zone_layout": "interleaved",
             "n_zones": 4, "portal_site": "south",
             "fault_system": "none", "q_regime": "competent"},
            held_fixed_ref="haul-t10",
        )

    return out


INSTANCES: dict[str, dict] = _build_all()
INSTANCE_IDS: list[str] = list(INSTANCES)

GROUPS = {
    "A": "crossing_grid",
    "B": "portal_sweep",
    "C": "zone_count",
    "D": "scale_ladder",
    "E": "haulage_ratio",
}


def constrained_admissible(cell_size_m: float) -> bool:
    """Is this discretisation fine enough to carry a constrained reference?

    See MIN_RADIUS_CELLS_FOR_CONSTRAINED.  A coarse grid does not make the
    network cheaper, it makes the answer wrong, so the coarse rung is scored on
    `raw` only rather than allowed to emit infeasible results that read as
    findings.
    """
    return (GEOMETRIC_STANDARD["min_turn_radius_m"] / float(cell_size_m)
            >= MIN_RADIUS_CELLS_FOR_CONSTRAINED)


def dims_for(cell_size_m: float) -> tuple[int, int, int]:
    """Grid dimensions (nx, ny, nz) for a cell size, from the fixed domain."""
    return tuple(int(round(d / cell_size_m)) for d in DOMAIN_SIZE_M)  # type: ignore[return-value]


def get(instance_id: str) -> dict:
    """Look up one instance spec, with a helpful error listing valid ids."""
    try:
        return INSTANCES[instance_id]
    except KeyError:
        raise KeyError(
            f"unknown instance_id {instance_id!r}; "
            f"valid ids: {', '.join(INSTANCE_IDS)}"
        ) from None


def select(only: list[str] | None = None,
           group: str | None = None) -> list[str]:
    """Resolve --only / --group CLI filters to a list of instance ids."""
    if only:
        for iid in only:
            get(iid)
        return list(only)
    if group:
        key = GROUPS.get(group.upper(), group)
        ids = [i for i, s in INSTANCES.items() if s["group"] == key]
        if not ids:
            raise KeyError(f"unknown group {group!r}; valid: "
                           f"{', '.join(sorted(GROUPS) + sorted(set(GROUPS.values())))}")
        return ids
    return list(INSTANCE_IDS)


if __name__ == "__main__":  # pragma: no cover - quick sanity dump
    print(f"{len(INSTANCES)} instances\n")
    hdr = f"{'instance_id':<22} {'group':<14} {'cell':>5} {'dims':>16} {'voxels':>13} {'git':>4}"
    print(hdr)
    print("-" * len(hdr))
    for iid, s in INSTANCES.items():
        d = dims_for(s["cell_size_m"])
        print(f"{iid:<22} {s['group']:<14} {s['cell_size_m']:>5.1f} "
              f"{str(d):>16} {d[0]*d[1]*d[2]:>13,} "
              f"{'yes' if s['shipped_in_git'] else 'no':>4}")
