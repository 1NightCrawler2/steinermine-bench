"""
Valid lower bounds for the geotechnically-weighted directed Steiner problem.
============================================================================

SPDX-License-Identifier: MIT

What is being bounded
---------------------
A ramp network is a portal-rooted, monotonically descending subgraph that
reaches every production zone.  Formally that is a **directed Steiner
arborescence**: root ``r`` = the portal terminal set, terminals ``T`` = the
production zones, arc set ``A`` = the descent arcs (dz <= 0), which already
contains the horizontal crosscut arcs (dz == 0) as a subset.

Why the bound is valid for all seven families
---------------------------------------------
Every one of the seven topology families produces a feasible portal-rooted
subgraph over exactly that arc set.  Therefore

    reported_cost(family)  >=  subgraph_cost(family)  >=  OPT_arborescence
                           >=  LP_relaxation  >=  the bounds below

The first inequality is an equality for ``chained_fan``, which counts shared
ramp voxels once, and is strict for ``sublevel_fan``, ``two_branch`` and
``three_branch``, which charge a shared upper ramp once per branch.  The bound
therefore holds a fortiori for the double-counting families.

Direction-safe arc cost
-----------------------
The reference solver charges a traversal ``u -> v`` as
``(cost[v] + excavation_rate) * length``, but it obtains junction-to-zone drive
costs from an *ascent* search run in the opposite direction, which charges
``(cost[u] + excavation_rate) * length`` for the same physical segment.  To
stay valid against whichever direction was charged, this module prices every
arc at

    c_LB(u, v) = (min(cost[u], cost[v]) + excavation_rate) * length

which is a pointwise lower bound on both conventions.  Any path's true reported
cost is therefore at least its ``c_LB`` cost, and the bound survives.

The bounds implemented here
---------------------------
Two valid bounds are computed; the reported value is the stronger, and both are
sound, differing only in tightness.

1. ``max_terminal_shortest_path`` -- ``max_k d(r, z_k)``.  Every arborescence
   contains a root-to-terminal path.  Free, very weak.

2. ``pairwise_divergence`` (**default**) -- for any two terminals ``i`` and
   ``j``, the paths ``r -> z_i`` and ``r -> z_j`` inside an arborescence share a
   prefix and separate at some node ``v``; after ``v`` the two branches are
   arc-disjoint, because a tree has no other way to reach both.  Hence

       OPT >= min_v [ d(r, v) + d(v, z_i) + d(v, z_j) ]

   for *every* pair, so the maximum over all pairs is valid.  It costs one
   descent Dijkstra from the portal plus one reverse Dijkstra per zone, then a
   vectorised argmin per pair.  It strictly dominates bound 1, since taking the
   best pair already implies ``max(d(r,z_i), d(r,z_j))``.

   Note the analogous all-terminals expression ``min_v [d(r,v) + sum_k
   d(v,z_k)]`` is **not** a lower bound -- it forces every branch to diverge at
   one node, which is the single-junction *feasible solution* and therefore an
   upper bound.  Pairs are the largest subset for which the divergence argument
   is forced.

These bounds are all computed on the RELAXATION.  They are the right bounds for
the ``raw`` track and the wrong ones for every other track: see
``geometric_bound.py``, which bounds the constrained problem directly from the
grade limit, and ``compute_track_bounds`` below, which routes each track to the
bounds that are valid for it.

Removed in v2.1.0: Wong dual ascent
-----------------------------------
v2.0.0 also offered ``--dual-ascent``, Wong (1984), "A dual ascent approach for
Steiner tree problems on a directed graph", Math. Programming 28:271-287.  It
was withdrawn as a reported component because it is not competitive on this
graph class and cost ~85 minutes per instance to establish that.  Dual ascent
grows a root component one zero-reduced-cost arc at a time; on a 26-connected
voxel lattice with millions of near-parallel arcs each step is worth a few
dollars.  On ``portal-north`` 30,000 iterations reached $1.61 M against the free
trivial bound's $1.68 M and pairwise divergence's $2.31 M, having saturated
0.45 % of the arcs with no terminal yet connected to the root.

It was the argmax on exactly one instance of twenty-five -- ``scale-130k``, the
coarsest rung at a 10 m cell -- which is consistent with the method's known
preference for sparse graphs, and is recorded here because it is the only
positive evidence the suite produced for it.
"""

from __future__ import annotations

import math
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order

_SENTINEL_TEST = 0.9e9

#: Bound components withdrawn in v2.1.0.  A reference carrying one of these is
#: a stale bundle: ``validate.py`` fails on it and ``--recompute-bounds-only``
#: strips it.
FORBIDDEN_BOUND_METHODS = frozenset({"wong_dual_ascent"})

#: Method names the suite currently produces.  Checked as a WARNING, not an
#: error -- a closed enum in a published schema makes every future bound a
#: breaking schema change, and ``components`` has always been open.
KNOWN_BOUND_METHODS = frozenset({
    "pairwise_divergence", "max_terminal_shortest_path", "raw_relaxation",
    "geometric_grade_floor", "geometric_grade_floor+opex_floor", "none",
})



def _descent_arcs(cost_grid: np.ndarray, cell_size_m: float,
                  excavation_rate: float):
    """
    Build the direction-safe descent arc list over passable voxels.

    Returns ``(tail, head, cost, n_nodes, flat_to_node, node_flat)`` where arcs
    point downhill (dz <= 0) and are priced with the min-endpoint rule above.
    """
    nx, ny, nz = cost_grid.shape
    flat_cost = cost_grid.ravel().astype(np.float64)
    passable = flat_cost < _SENTINEL_TEST

    node_flat = np.flatnonzero(passable).astype(np.int64)
    n_nodes = node_flat.size
    flat_to_node = np.full(nx * ny * nz, -1, dtype=np.int64)
    flat_to_node[node_flat] = np.arange(n_nodes)

    ijk = np.empty((n_nodes, 3), dtype=np.int64)
    ijk[:, 0] = node_flat // (ny * nz)
    ijk[:, 1] = (node_flat // nz) % ny
    ijk[:, 2] = node_flat % nz

    tails, heads, costs = [], [], []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0):          # descent only; dz == 0 covers crosscuts
                if dx == dy == dz == 0:
                    continue
                length = math.sqrt(dx * dx + dy * dy + dz * dz) * cell_size_m
                nc = ijk + (dx, dy, dz)
                ok = ((nc[:, 0] >= 0) & (nc[:, 0] < nx)
                      & (nc[:, 1] >= 0) & (nc[:, 1] < ny)
                      & (nc[:, 2] >= 0) & (nc[:, 2] < nz))
                nc_safe = np.clip(nc, 0, (nx - 1, ny - 1, nz - 1))
                n_flat = (nc_safe[:, 0] * ny + nc_safe[:, 1]) * nz + nc_safe[:, 2]
                dst = flat_to_node[n_flat]
                valid = ok & (dst >= 0)
                if not valid.any():
                    continue
                src = np.flatnonzero(valid)
                cu = flat_cost[node_flat[src]]
                cv = flat_cost[n_flat[valid]]
                tails.append(src)
                heads.append(dst[valid])
                costs.append((np.minimum(cu, cv) + excavation_rate) * length)

    return (np.concatenate(tails), np.concatenate(heads),
            np.concatenate(costs), n_nodes, flat_to_node, node_flat)


def _add_virtual_terminals(tail, head, cost, n_nodes, portal_nodes,
                           zone_node_sets):
    """
    Add a virtual root with zero-cost arcs to the portal, and one virtual sink
    per zone with zero-cost arcs from that zone's voxels.

    Node ``n_nodes`` is the root; ``n_nodes + 1 + k`` is zone ``k``'s sink.
    """
    root = n_nodes
    sinks = [n_nodes + 1 + k for k in range(len(zone_node_sets))]
    total = n_nodes + 1 + len(zone_node_sets)

    extra_tail = [np.full(portal_nodes.size, root, dtype=np.int64)]
    extra_head = [portal_nodes]
    extra_cost = [np.zeros(portal_nodes.size)]

    for sink, zn in zip(sinks, zone_node_sets):
        extra_tail.append(zn)
        extra_head.append(np.full(zn.size, sink, dtype=np.int64))
        extra_cost.append(np.zeros(zn.size))

    return (np.concatenate([tail] + extra_tail),
            np.concatenate([head] + extra_head),
            np.concatenate([cost] + extra_cost),
            total, root, sinks)




def trivial_bound(tail, head, cost, n_total, root, sinks) -> float:
    """
    ``max_k d(root, sink_k)``: every arborescence contains a root-to-terminal
    path, so the longest such shortest path is a valid (if weak) lower bound.

    Free to compute; used as a floor under the other two bounds.
    """
    from scipy.sparse.csgraph import dijkstra
    g = csr_matrix((cost, (tail, head)), shape=(n_total, n_total))
    d = dijkstra(g, directed=True, indices=root)
    finite = [d[s] for s in sinks if np.isfinite(d[s])]
    if len(finite) != len(sinks):
        raise ValueError("some terminal is unreachable from the portal over "
                         "the descent arc set; the instance is infeasible")
    return float(max(finite))


def pairwise_divergence_bound(tail, head, cost, n_total, root, sinks,
                              verbose: bool = False) -> dict:
    """
    The pairwise-divergence lower bound (see the module docstring).

    In any arborescence the paths to two terminals ``z_i`` and ``z_j`` share a
    prefix and then separate at a node ``v``, beyond which the two branches are
    arc-disjoint.  So for every pair

        OPT >= min_v [ d(r, v) + d(v, z_i) + d(v, z_j) ]

    and the maximum over pairs is valid.  ``v`` ranges over all nodes, including
    ``r`` itself and the terminals, so no divergence structure is excluded.

    Costs one forward Dijkstra from the root plus one reverse Dijkstra per
    terminal, then an O(n) vectorised minimum per pair.
    """
    from itertools import combinations

    from scipy.sparse.csgraph import dijkstra

    t0 = time.time()
    fwd = csr_matrix((cost, (tail, head)), shape=(n_total, n_total))
    rev = csr_matrix((cost, (head, tail)), shape=(n_total, n_total))

    d_root = dijkstra(fwd, directed=True, indices=root)
    # dijkstra on the reversed graph from sink k gives d(v, sink_k) for all v.
    d_to = [dijkstra(rev, directed=True, indices=s) for s in sinks]

    for k, s in enumerate(sinks):
        if not np.isfinite(d_root[s]):
            raise ValueError(
                f"terminal {k} is unreachable from the portal over the descent "
                f"arc set; the instance is infeasible")

    best = 0.0
    best_pair = None
    for i, j in combinations(range(len(sinks)), 2):
        total = d_root + d_to[i] + d_to[j]
        v = float(np.nanmin(np.where(np.isfinite(total), total, np.inf)))
        if v > best:
            best, best_pair = v, (i, j)

    if len(sinks) == 1:
        best = float(d_root[sinks[0]])
        best_pair = (0, 0)

    if verbose:
        print(f"        pairwise divergence: best pair {best_pair} -> "
              f"{best:,.0f}  ({time.time() - t0:.1f} s)")

    return {
        "value": best,
        "best_pair": list(best_pair) if best_pair else None,
        "runtime_s": round(time.time() - t0, 2),
    }


def compute_lower_bound(instance: dict, verbose: bool = True) -> dict:
    """
    The strongest valid lower bound on the RELAXATION -- i.e. the ``raw`` track.

    Runs the trivial and pairwise-divergence bounds and reports the larger,
    since the maximum of valid bounds is valid.

    Returns a ``lower_bound`` record ready to embed in ``reference.json``:
    ``value``, ``method``, ``valid``, ``track``, ``note``, ``runtime_s``, plus
    the individual component values under ``components``.

    Every method reported is provably valid against all seven topology
    families; they differ in tightness, not soundness.  For the constrained and
    total tracks see :func:`compute_track_bounds`.
    """
    md = instance["metadata"]
    g = md["grid"]
    exc = md["cost_model"]["excavation_rate_per_m"]

    t0 = time.time()
    tail, head, cost, n_nodes, flat_to_node, _ = _descent_arcs(
        instance["cost_grid"], float(g["cell_size_m"]), exc)

    nx, ny, nz = instance["cost_grid"].shape

    def to_nodes(voxels):
        flat = np.array([(v[0] * ny + v[1]) * nz + v[2] for v in voxels],
                        dtype=np.int64)
        n = flat_to_node[flat]
        return np.unique(n[n >= 0])

    portal_nodes = to_nodes(instance["portal_voxels"])
    zone_nodes = [to_nodes(zv) for zv in instance["zone_voxels"]]

    tail, head, cost, n_total, root, sinks = _add_virtual_terminals(
        tail, head, cost, n_nodes, portal_nodes, zone_nodes)

    if verbose:
        print(f"      lower bound: {n_nodes:,} passable voxels, "
              f"{cost.size:,} descent arcs, {len(sinks)} terminals")

    components: dict[str, float] = {}

    components["max_terminal_shortest_path"] = trivial_bound(
        tail, head, cost, n_total, root, sinks)

    pair = pairwise_divergence_bound(tail, head, cost, n_total, root, sinks,
                                     verbose=verbose)
    components["pairwise_divergence"] = pair["value"]

    method = max(components, key=components.get)
    value = components[method]

    notes = {
        "pairwise_divergence": (
            "Pairwise-divergence bound: for the binding terminal pair "
            f"{pair['best_pair']}, min over all nodes v of "
            "d(portal,v) + d(v,zone_i) + d(v,zone_j). Valid because the two "
            "root-to-terminal paths in any arborescence separate at some v and "
            "are arc-disjoint thereafter."),
        "max_terminal_shortest_path": (
            "Max over terminals of the shortest descending path from the "
            "portal. Valid but weak; reported because it exceeded the other "
            "bounds computed."),
    }
    note = notes.get(method, "Lower bound on the relaxation.") + (
        " Arcs are priced at (min(cost[u], cost[v]) + excavation_rate) * length "
        "so the bound stays valid regardless of which traversal direction the "
        "reference solver charged for a segment. The bound applies to all seven "
        "topology families: each produces a feasible portal-rooted subgraph over "
        "this arc set, and the families that charge a shared ramp once per "
        "branch report at least their subgraph cost.")

    return {
        "value": value,
        "method": method,
        "valid": True,
        "track": "raw",
        "note": note,
        "components": {k: round(v, 4) for k, v in components.items()},
        "runtime_s": round(time.time() - t0, 2),
    }


def compute_track_bounds(metadata: dict, raw_record: dict,
                         reference_cost_raw: float | None = None) -> dict:
    """
    Per-track lower bounds for one instance: ``{"raw": ..., "constrained": ...,
    "total": ...}``, each a record in ``geometric_bound.track_bound`` form.

    Needs no grid, no solver and no search -- the relaxation bound is passed in
    (it was computed once, expensively, by :func:`compute_lower_bound`) and the
    geometric floor is trigonometry over ``metadata.json``.  That is what makes
    ``solve_reference.py --recompute-bounds-only`` a seconds-long operation over
    the whole suite.

    ``raw`` gets the relaxation bound alone.  The grade floor is **not** valid
    there: an unconstrained network may descend at 45 degrees, so it is
    legitimately cheaper than any grade-feasible ramp, and on this suite the
    floor exceeds the raw reference cost on most instances.
    ``geometric_bound.track_bound`` refuses the combination outright.

    ``reference_cost_raw``, when given, is recorded on the constrained and total
    tracks as a *conditional* component: the per-family domination argument
    (each family's raw cost is a min over junction placements of shortest paths,
    so it cannot exceed that family's constrained cost) is believed but rests on
    the centreline-versus-arc costing convention being exact, which is not
    proved.  It never contributes to ``value``.
    """
    from steinerbench import bound_inputs, geometric_bound as gb

    inp = bound_inputs.floor_inputs_from_metadata(metadata)

    capex_floor = gb.geometric_grade_floor(
        portal_rl_m=inp["portal_rl_m"], zones=inp["zones"],
        max_grade=inp["max_grade"], unit_cost_per_m=inp["unit_cost_per_m"],
        excavation_rate_per_m=inp["excavation_rate_per_m"])

    opex = gb.haulage_floor(inp["zones"], portal_rl_m=inp["portal_rl_m"],
                            max_grade=inp["max_grade"],
                            opex_model=inp["opex_model"])

    total_floor = gb.geometric_grade_floor(
        portal_rl_m=inp["portal_rl_m"], zones=inp["zones"],
        max_grade=inp["max_grade"], unit_cost_per_m=inp["unit_cost_per_m"],
        excavation_rate_per_m=inp["excavation_rate_per_m"],
        n_portals=inp["n_portals"],
        portal_establishment_cost=inp["portal_establishment_cost"],
        opex_floor=opex["value"])
    total_floor["opex_audit"] = opex

    conditional = ({"raw_best_known_conditional": float(reference_cost_raw)}
                   if reference_cost_raw else None)

    # An unsolved instance has no relaxation bound -- that one needs the grid.
    # The floors do not, so it still ships a valid constrained/total bound.
    pairwise = raw_record if raw_record.get("value") else None

    return {
        "raw": dict(raw_record, track="raw"),
        "constrained": gb.track_bound("constrained", pairwise=pairwise,
                                      floor=capex_floor,
                                      conditional=conditional),
        "total": gb.track_bound("total", pairwise=pairwise, floor=total_floor,
                                conditional=conditional),
    }
