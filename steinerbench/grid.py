"""
Instance bundle assembly and the cost_grid.npz storage format.
==============================================================

SPDX-License-Identifier: MIT

Storage format
--------------
``cost_grid.npz`` is a compressed NumPy archive holding the POST-fault-floor
tier index rather than float costs:

    tier_index   uint8   (nx, ny, nz)  0-5 = cost tier, 255 = sentinel
    fault_count  uint8   (nx, ny, nz)  distinct faults intersecting the voxel
    tier_costs   float64 (6,)          the $/m schedule, worst rock first
    surface_rl   float32 (nx, ny)      ground surface RL per plan column

This is lossless.  Both fault cost floors ($3,059.0/m and $4,460.9/m) are
themselves tier costs, so ``cost = max(tier_cost, floor)`` maps a tier index
onto another tier index; nothing is rounded away.  It also cuts the raw size
4x versus float32 and compresses far better, since the payload has only seven
distinct values.

``loader.load_instance`` reconstructs the float32 cost grid in $/m with
``1e9`` at sentinel voxels.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from steinerbench import geology, spec, tiers


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, streamed so large grids do not land in memory."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_cost_grid_npz(path: Path, tier_index: np.ndarray,
                        fault_count: np.ndarray, surface_rl: np.ndarray) -> None:
    """Write the compressed instance grid archive."""
    np.savez_compressed(
        path,
        tier_index=tier_index,
        fault_count=fault_count,
        tier_costs=tiers.TIER_COSTS,
        surface_rl=surface_rl,
    )


def read_cost_grid_npz(path: Path) -> dict:
    """
    Read ``cost_grid.npz`` and expand it to a usable cost field.

    Returns a dict with ``cost_grid`` (float32, $/m), ``tier_index`` (uint8),
    ``fault_count`` (uint8), ``surface_rl`` (float32) and ``tier_costs``.
    """
    with np.load(path) as z:
        tier_index = z["tier_index"]
        fault_count = z["fault_count"]
        tier_costs = z["tier_costs"]
        surface_rl = z["surface_rl"]

    if not np.array_equal(tier_costs, tiers.TIER_COSTS):
        raise ValueError(
            f"{path} carries a tier schedule that differs from the frozen one; "
            f"this bundle was not produced by this version of SteinerMineBench")

    return {
        "cost_grid": tiers.tier_index_to_cost(tier_index),
        "tier_index": tier_index,
        "fault_count": fault_count,
        "surface_rl": surface_rl,
        "tier_costs": tier_costs,
    }


# ---------------------------------------------------------------------------
# Bundle construction
# ---------------------------------------------------------------------------
def build_instance(instance_id: str, want_q: bool = False) -> dict:
    """
    Build every array and descriptor for one instance, in memory.

    Deterministic: identical output for identical ``instance_id`` on any
    machine, because all randomness is drawn from ``spec.geology_seed`` on a
    fixed world-space lattice.

    Returns a dict with ``tier_index``, ``fault_count``, ``surface_rl``,
    ``portal``, ``zones``, ``metadata`` and optionally ``q``.
    """
    s = spec.get(instance_id)
    cell = float(s["cell_size_m"])
    dims = spec.dims_for(cell)
    min_coords = spec.MIN_COORDS_M

    fields = geology.build_fields(s, dims, cell, min_coords, want_q=want_q)

    tier_index = tiers.apply_fault_floor(fields["tier_index"],
                                         fields["fault_count"])
    passable = tier_index != tiers.SENTINEL_TIER

    portal = geology.portal_terminal(s, dims, cell, min_coords, passable)
    zones = geology.zone_terminals(s, dims, cell, min_coords, passable)
    n_lev = geology.n_sublevels(zones)

    n_vox = int(dims[0]) * int(dims[1]) * int(dims[2])
    n_passable = int(passable.sum())

    tier_hist = np.bincount(tier_index[passable].ravel(),
                            minlength=tiers.N_TIERS)[:tiers.N_TIERS]

    metadata = {
        "instance_id": instance_id,
        "schema_version": spec.SCHEMA_VERSION,
        "benchmark_version": spec.BENCHMARK_VERSION,
        "synthetic": True,
        "provenance_statement": spec.PROVENANCE_STATEMENT,

        "grid": {
            "dims": list(dims),
            "cell_size_m": cell,
            "min_coords_m": list(min_coords),
            "domain_size_m": list(spec.DOMAIN_SIZE_M),
            "axis_order": ["EAST", "NORTH", "RL"],
            "z_positive": "up",
            "world_from_voxel": "world_m = min_coords_m + (ijk + 0.5) * cell_size_m",
            "sentinel_cost_per_m": tiers.SENTINEL_COST,
            "sentinel_tier_index": int(tiers.SENTINEL_TIER),
            "passable_test": "cost_per_m < 0.9 * sentinel_cost_per_m",
            "n_voxels": n_vox,
            "n_passable": n_passable,
            "passable_fraction": n_passable / n_vox,
        },

        "cost_model": {
            "objective": (
                "Minimise the total cost of a portal-rooted, monotonically "
                "descending network connecting the portal to every production "
                "zone, on a 26-connected voxel graph."),
            "tier_schedule": tiers.tier_schedule_records(),
            "fault_single_floor_per_m": tiers.FAULT_SINGLE_MIN_COST,
            "fault_multi_floor_per_m": tiers.FAULT_MULTI_MIN_COST,
            "fault_floor_rule": (
                "cost_per_m = max(tier_cost, floor), where floor is "
                "fault_single_floor_per_m for voxels intersecting exactly one "
                "distinct fault and fault_multi_floor_per_m for two or more. "
                "The fault damage halo is specified in METRES "
                "(faults[].damage_half_width_m) and no voxel dilation is "
                "applied, so the fault footprint is identical at every cell "
                "size. This differs deliberately from the MineOptimizer "
                "production pipeline, which widens the floor with a 3x3x3 "
                "VOXEL dilation and is therefore resolution dependent."),
            "excavation_rate_per_m": tiers.EXCAVATION_RATE,
            "excavation_rate_in_grid": False,
            "edge_weight_formula": (
                "w(u,v) = (cost_grid[v] + excavation_rate_per_m) * "
                "effective_length_m(u,v)"),
            "segment_direction_convention": (
                "The edge weight charges the voxel being ENTERED, so a "
                "segment's cost depends on which way it is traversed. The "
                "convention, which every reference cost in this benchmark "
                "follows, is: each segment is costed as the shortest-path "
                "distance FROM ITS OWN TERMINAL TO THE JUNCTION. The portal "
                "ramp is therefore costed portal -> junction over descent arcs "
                "(dz <= 0); a horizontal crosscut is costed zone -> junction "
                "over dz == 0 arcs; and a 3-D drive is costed zone -> junction "
                "over ascent arcs (dz >= 0). Costing a zone leg in the "
                "opposite (junction -> zone) direction is a defensible reading "
                "of the same physical drive but yields costs differing by "
                "roughly 1 percent, which would swamp the optimality gaps this "
                "benchmark measures. Reproduce the stated convention exactly. "
                "The polylines in reference_paths.npz are stored already "
                "oriented terminal -> junction, so summing "
                "(cost_grid[p[i+1]] + excavation_rate_per_m) * "
                "||p[i+1] - p[i]|| * cell_size_m over each stored path "
                "reproduces the reference cost; validate.py asserts this."),
            "effective_length_plain_m": (
                "Euclidean distance between voxel centres, i.e. "
                "cell_size_m * sqrt(dx^2 + dy^2 + dz^2) for the 26-neighbour "
                "offset (dx,dy,dz)."),
            "neighbourhood": 26,
            "descent_constraint": (
                "Ramp arcs are restricted to dz <= 0 (monotone descent); "
                "junction-to-zone crosscut arcs are restricted to dz == 0. "
                "NOTE: monotone descent is the ONLY geometric constraint on "
                "the `raw` track. There is no grade limit and no turn-radius "
                "limit -- see tracks.raw."),
            "tier_histogram_passable": [int(v) for v in tier_hist],
        },

        # ── Scoring tracks (v2) ──────────────────────────────────────────────
        "tracks": {
            "normative": spec.NORMATIVE_TRACK,
            "definitions": dict(spec.TRACKS),
            "geometric_standard": dict(spec.GEOMETRIC_STANDARD),
            "constrained_admissible": spec.constrained_admissible(cell),
            "constrained_admissible_rule": (
                "min_turn_radius_m / cell_size_m >= "
                f"{spec.MIN_RADIUS_CELLS_FOR_CONSTRAINED:g}. A cell that is a "
                "large fraction of the turning radius cannot carry the "
                "rendered arc, so legs fail for a reason about the grid rather "
                "than about the rock. Rungs below the floor are shipped and "
                "scored on `raw` only."),
            "verifier": (
                "steinerbench/buildable_check.py -- numpy only, no solver "
                "required. Submissions on the constrained and total tracks are "
                "re-checked with it against the shipped grid."),
            "opex_model": dict(spec.OPEX_MODEL),
            "why_raw_is_not_normative": (
                "Measured on the v1 reference solutions, 100 % of descending "
                "steps exceed a 20 % grade limit and the median step grade is "
                "100 % (45 deg). The raw objective is the optimum of a "
                "relaxation that admits geometry no ramp can be built to; "
                "re-ranking on buildable cost flips the winning family on 15 "
                "of the 23 solved instances. `raw` is retained unchanged "
                "because it is exactly reproducible and because the gap "
                "between it and a buildable answer is itself a result."),
        },

        "geology": {
            "q_regime": s["q_regime"],
            "q_regime_lobes": [
                {"weight": w, "median_q": m, "sigma_lognormal": sg}
                for w, m, sg in spec.Q_REGIMES[s["q_regime"]]["lobes"]],
            "q_reference_lattice_m": spec.Q_REFERENCE_LATTICE_M,
            "q_lattice_smoothing_nodes": geology._LATTICE_SMOOTH_NODES,
            "q_clip": list(geology.Q_CLIP),
            "resolution_independence": (
                "The Q field is drawn on a fixed 10 m world-space lattice and "
                "trilinearly interpolated to voxel centres, so every cell size "
                "samples the same geology."),
            "barrier_slab": spec.BARRIER_SLAB,
            "competent_windows": spec.COMPETENT_WINDOWS,
            "competent_windows_note": (
                "A geological feature of the synthetic rock mass: competent "
                "breaches through the barrier slab. Unrelated to the "
                "MineOptimizer solver's --corridor search mask, which the "
                "benchmark never enables."),
            "surface_model": (
                "RL = 415 + 55*exp(-((N-230)/130)^2) + 18*sin(2*pi*E/420) "
                "- 12*(E/630); voxels above this RL are sentinel."),
        },

        "fault_system": s["fault_system"],
        "faults": [
            {k: (list(v) if isinstance(v, tuple) else v)
             for k, v in f.items()} for f in
            spec.FAULT_SYSTEMS[s["fault_system"]]],

        "portals": [{k: v for k, v in portal.items() if not k.startswith("_")}],
        "zones": [{k: v for k, v in z.items() if not k.startswith("_")}
                  for z in zones],
        "orebody": {
            "zone_layout": s.get("zone_layout", "nested"),
            "tonnage_scale": float(s.get("tonnage_scale", 1.0)),
            "total_tonnage_mt": round(
                sum(float(z["tonnage_mt"]) for z in zones), 4),
            "tonnage_note": (
                "Tonnage enters the `total` track only, through haulage. "
                "Nothing in the search reads it, so instances differing only "
                "in tonnage_scale have identical geometry, identical raw and "
                "constrained costs and identical buildability verdicts."),
        },

        "topology_families": {
            "n_zones": len(zones),
            "n_sublevels": n_lev,
            "level_grouping_tolerance_m": 15.0,
            "applicable": spec.applicable_families(n_lev),
            "gating_note": (
                "two_branch and chained_fan require >= 2 sublevels; "
                "three_branch and hybrid_chained_fan_branch require >= 3. "
                "Families outside 'applicable' are not evaluated on this "
                "instance and must not be scored against it."),
            "search_kind": {name: kind for name, _, kind in spec.FAMILIES},
        },

        "generation": {
            "seed": s["q_seed"],
            "seed_note": (
                "The seed is keyed on (q_regime, fault_system), not on the "
                "instance id, so every instance sharing a geology -- the whole "
                "portal sweep, the whole zone-count sweep and all four scale "
                "rungs -- samples an identical rock mass."),
            "generator_version": spec.GENERATOR_VERSION,
            "generator_command": f"python generate.py --only {instance_id}",
            "shipped_in_git": bool(s["shipped_in_git"]),
        },

        "varied_axis": s["varied_axis"],
    }

    return {
        "tier_index": tier_index,
        "fault_count": fields["fault_count"],
        "surface_rl": fields["surface_rl"],
        "passable": passable,
        "portal": portal,
        "zones": zones,
        "metadata": metadata,
        **({"q": fields["q"]} if want_q else {}),
    }


def write_bundle(built: dict, out_dir: Path) -> dict:
    """
    Write ``cost_grid.npz``, ``metadata.json`` and ``README.md`` for one
    instance.  Returns the finalised metadata (including the grid checksum).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "cost_grid.npz"

    write_cost_grid_npz(npz_path, built["tier_index"], built["fault_count"],
                        built["surface_rl"])

    metadata = built["metadata"]
    metadata["generation"]["created_utc"] = (
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    metadata["checksums"] = {
        "cost_grid.npz": f"sha256:{sha256_file(npz_path)}",
        "size_bytes": npz_path.stat().st_size,
    }

    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    (out_dir / "README.md").write_text(instance_readme(metadata),
                                       encoding="utf-8")
    return metadata


# ---------------------------------------------------------------------------
# Per-instance README
# ---------------------------------------------------------------------------
_GROUP_BLURB = {
    "crossing_grid": (
        "the core 4x3 crossing grid, which isolates how fault architecture and "
        "background rock mass quality interact"),
    "portal_sweep": (
        "the portal-location sweep, which holds the geology and the orebody "
        "completely fixed and moves only the surface access point"),
    "zone_count": (
        "the zone-count sweep, which holds the geology and portal fixed and "
        "varies only how many production terminals the network must reach"),
    "scale_ladder": (
        "the scale ladder, which holds the geology fixed in world coordinates "
        "and varies only the voxel resolution"),
    "haulage_ratio": (
        "the haulage-ratio axis, which holds the rock mass, the portal and the "
        "orebody geometry completely fixed and varies only the tonnage moved "
        "over the network -- and therefore only the ratio of construction cost "
        "to operating cost. Nothing in the search reads tonnage, so every "
        "member of this group has identical geometry, identical construction "
        "cost and identical buildability verdicts; only the `total` track "
        "differs. The layout is two plan-separated clusters at interleaved "
        "depths, the geometry in which a depth-ordered chain must cross "
        "between clusters at every level while a tree branches once"),
}

_FAULT_BLURB = {
    "none": "no faulting at all, giving a clean baseline",
    "single": "a single through-going fault",
    "two_crossing": "two faults that cross near the centre of the domain, "
                    "producing a multi-intercept core where the higher "
                    "$4,460.9/m floor applies",
    "conjugate_pair": "a conjugate pair with a common strike and opposing "
                      "dips, converging into a wedge with depth",
}

_Q_BLURB = {
    "competent": "competent ground (median Q around 8), where support cost is "
                 "dominated by the cheapest tier and the faults and barrier "
                 "supply nearly all the cost contrast",
    "mixed": "a bimodal mixed regime (competent and poor lobes) spanning most "
             "of the tier schedule",
    "poor_dominated": "poor-dominated ground (median Q around 0.15), where "
                      "most of the rock mass sits in the expensive tiers and "
                      "the competent windows become decisive",
}


def instance_readme(metadata: dict) -> str:
    """One-paragraph description of what a single instance tests."""
    m = metadata
    g, cm = m["grid"], m["cost_model"]
    tf, va = m["topology_families"], m["varied_axis"]
    dims = g["dims"]

    varied = ", ".join(f"`{k}` = `{v}`" for k, v in va["value"].items())
    dup = va.get("duplicate_of")

    lines = [
        f"# Instance `{m['instance_id']}`",
        "",
        f"**SYNTHETIC -- not derived from any operating mine.** "
        f"See `metadata.json` for the full provenance statement.",
        "",
        f"This instance belongs to {_GROUP_BLURB[va['group']]}. Within that "
        f"group it varies {varied}; every other parameter is held identical to "
        f"the rest of the group"
        + (f", with `{va['held_fixed_ref']}` as the reference member"
           if va.get("held_fixed_ref") else "") + ".",
        "",
        f"The rock mass is {_Q_BLURB[m['geology']['q_regime']]}, cut by "
        f"{_FAULT_BLURB[m['fault_system']]}. A low-Q barrier slab spans "
        f"RL {m['geology']['barrier_slab']['rl_range_m'][0]:.0f}-"
        f"{m['geology']['barrier_slab']['rl_range_m'][1]:.0f} m and must be "
        f"crossed by any decline; two competent windows breach it, posing a "
        f"choice between a lateral detour to a breach and driving straight "
        f"through the lid. The grid is "
        f"{dims[0]} x {dims[1]} x {dims[2]} = {g['n_voxels']:,} voxels at a "
        f"{g['cell_size_m']:g} m cell size, of which "
        f"{g['passable_fraction']*100:.1f}% lie below the ground surface and "
        f"are excavatable.",
        "",
        f"The network must connect one portal ({m['portals'][0]['label']}, "
        f"{m['portals'][0]['n_voxels']} candidate voxels) to "
        f"{tf['n_zones']} production zone(s) distributed over "
        f"{tf['n_sublevels']} sublevel(s), which makes "
        f"{len(tf['applicable'])} of the 7 topology families applicable: "
        f"{', '.join('`' + f + '`' for f in tf['applicable'])}.",
    ]

    if len(tf["applicable"]) < 7:
        missing = [f for f, _, _ in spec.FAMILIES if f not in tf["applicable"]]
        lines += [
            "",
            f"> **Family gating.** {', '.join('`' + f + '`' for f in missing)} "
            f"require more sublevels than this instance provides and are not "
            f"evaluated here. This is a consequence of the varied axis, not a "
            f"defect; do not score those families against this instance.",
        ]

    if dup:
        lines += [
            "",
            f"> This instance reproduces the geology and terminals of "
            f"`{dup}` exactly. Both bundles are shipped in full so each "
            f"directory is self-contained.",
        ]

    if m["instance_id"] in spec.UNSOLVED_INSTANCES:
        lines += [
            "",
            f"> **No reference solution -- open scaling frontier.** At "
            f"{g['n_voxels']:,} voxels, a multi-source Dijkstra over this grid "
            f"needs roughly 22 GB for the CSR graph alone, beyond the machine "
            f"used to build this suite. The grid, terminals and cost model are "
            f"fully specified and `reference.json` carries "
            f"`reference_type: \"unsolved\"`, so nothing here is graded against "
            f"a reference. The first valid solution submitted becomes the "
            f"best-known bound for this instance.",
        ]

    lines += [
        "",
        "## Cost model",
        "",
        f"Edge weight is `{cm['edge_weight_formula']}` with "
        f"`excavation_rate_per_m` = ${cm['excavation_rate_per_m']:,.0f}/m, "
        f"which is **not** baked into `cost_grid.npz`. Support cost per metre "
        f"follows the six-tier NGI Barton Q schedule; fault intercepts impose "
        f"a floor of ${cm['fault_single_floor_per_m']:,.1f}/m (single) or "
        f"${cm['fault_multi_floor_per_m']:,.1f}/m (multiple).",
        "",
        "```python",
        "from loader import load_instance",
        f"inst = load_instance({m['instance_id']!r})",
        "inst['cost_grid']   # float32 $/m, 1e9 = not excavatable",
        "inst['reference']   # reference_type, reference_cost, lower_bound",
        "```",
        "",
    ]
    return "\n".join(lines)
