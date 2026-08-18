"""
Independent verification of the exact-labelled topology families.
=================================================================

SPDX-License-Identifier: MIT

Why this module exists
----------------------
A benchmark instance may only be labelled ``reference_type: "exact"`` if its
reference cost is genuinely optimal for the family that produced it.  The
reference solver's ``find_junction`` (run_steiner_poly.py:298) already *is* a
full-grid ``np.argmin`` over all passable voxels, so re-running it would prove
nothing -- it would just agree with itself.

This module therefore rebuilds everything from scratch, sharing **no code and
no distance fields** with the reference solver:

  * its own CSR graph, built here from ``cost_grid.npz``
  * its own direction filters (descent, ascent, horizontal)
  * its own multi-source Dijkstra via ``scipy.sparse.csgraph``
  * its own argmin over the summed distance fields

and then asserts that the junction voxels and the family cost agree with what
the solver reported.  That agreement is what earns the ``exact`` label.

Numerical note
--------------
The reference solver accumulates edge weights in float32; this verifier uses
float64 throughout, which is strictly more accurate.  Costs are therefore
compared with a relative tolerance (``COST_RTOL``) rather than for equality.
Junction voxels are compared exactly, but a differing voxel is not
automatically a failure: ties are common on a discrete grid, so a mismatch is
accepted only when the objective at the solver's junction equals the objective
at this module's argmin to within tolerance -- i.e. it is a genuine tie and not
a suboptimality.  Anything else fails the run.

Verified families
-----------------
``single_junction``  one argmin over the ascent fields of all zones
``sublevel_fan``     one RL-restricted argmin per sublevel, over crosscut fields
``two_branch``       exhaustive over consecutive-level partitions, K = 2
``three_branch``     the same, K = 3

``sequential_ramp``, ``chained_fan`` and ``hybrid_chained_fan_branch`` use
greedy or coordinate-descent search and are labelled ``best_known``; there is
nothing to verify as exact and they are skipped.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as sp_dijkstra

#: Relative tolerance when comparing a float64 recomputation against the
#: solver's float32 accumulation.
COST_RTOL = 1e-4

#: Level grouping tolerance, matching run_steiner_poly.py:group_zones_by_level.
LEVEL_TOL_M = 15.0

#: RL-slice half-width for the crosscut junction search, matching
#: run_steiner_poly.py:find_junction_at_rl.
IZ_TOL = 2

_SENTINEL_TEST = 0.9e9


# ---------------------------------------------------------------------------
# Graph construction -- independent reimplementation
# ---------------------------------------------------------------------------
def _neighbour_offsets():
    """The 26-neighbourhood with Euclidean step lengths in cell units."""
    out = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                out.append((dx, dy, dz, math.sqrt(dx * dx + dy * dy + dz * dz)))
    return tuple(out)


class VoxelGraph:
    """
    A compact directed graph over the passable voxels of one instance.

    Edge weight into voxel ``v`` from a 26-neighbour ``u`` is

        w(u, v) = (cost_grid[v] + excavation_rate) * ||v - u|| * cell_size

    which is the objective stated in every instance's
    ``metadata.cost_model.edge_weight_formula``.
    """

    def __init__(self, cost_grid: np.ndarray, cell_size_m: float,
                 excavation_rate: float):
        self.shape = cost_grid.shape
        self.cell_size = float(cell_size_m)
        self.exc = float(excavation_rate)

        nx, ny, nz = cost_grid.shape
        flat_cost = cost_grid.ravel()
        passable = flat_cost < _SENTINEL_TEST

        self.node_flat = np.flatnonzero(passable).astype(np.int64)
        self.n_nodes = self.node_flat.size

        self.ijk = np.empty((self.n_nodes, 3), dtype=np.int32)
        self.ijk[:, 0] = self.node_flat // (ny * nz)
        self.ijk[:, 1] = (self.node_flat // nz) % ny
        self.ijk[:, 2] = self.node_flat % nz

        self.flat_to_node = np.full(nx * ny * nz, -1, dtype=np.int64)
        self.flat_to_node[self.node_flat] = np.arange(self.n_nodes)

        rows, cols, data, dzs = [], [], [], []
        for dx, dy, dz, steps in _neighbour_offsets():
            nc = self.ijk.astype(np.int64) + (dx, dy, dz)
            ok = ((nc[:, 0] >= 0) & (nc[:, 0] < nx)
                  & (nc[:, 1] >= 0) & (nc[:, 1] < ny)
                  & (nc[:, 2] >= 0) & (nc[:, 2] < nz))
            nc_safe = np.clip(nc, 0, (nx - 1, ny - 1, nz - 1))
            n_flat = (nc_safe[:, 0] * ny + nc_safe[:, 1]) * nz + nc_safe[:, 2]
            dst = self.flat_to_node[n_flat]
            valid = ok & (dst >= 0)
            if not valid.any():
                continue
            src = np.flatnonzero(valid)
            w = ((flat_cost[n_flat[valid]].astype(np.float64) + self.exc)
                 * (steps * self.cell_size))
            rows.append(src)
            cols.append(dst[valid])
            data.append(w)
            dzs.append(np.full(src.size, dz, dtype=np.int8))

        self._rows = np.concatenate(rows)
        self._cols = np.concatenate(cols)
        self._data = np.concatenate(data)
        self._dz = np.concatenate(dzs)

    def csr(self, direction: str) -> csr_matrix:
        """
        A direction-filtered CSR view of the graph.

        ``descent``    dz <= 0, ramps only ever go down
        ``ascent``     dz >= 0, used for the 3-D zone fields
        ``horizontal`` dz == 0, crosscuts at constant RL
        ``any``        unfiltered
        """
        if direction == "descent":
            keep = self._dz <= 0
        elif direction == "ascent":
            keep = self._dz >= 0
        elif direction == "horizontal":
            keep = self._dz == 0
        elif direction == "any":
            keep = slice(None)
        else:
            raise ValueError(f"unknown direction {direction!r}")
        return csr_matrix(
            (self._data[keep], (self._rows[keep], self._cols[keep])),
            shape=(self.n_nodes, self.n_nodes), dtype=np.float64)

    def multi_source(self, sources, direction: str) -> np.ndarray:
        """
        Multi-source shortest-path distances as a dense grid-shaped field.

        A virtual node with zero-weight arcs to every source reduces the
        multi-source problem to a single-source Dijkstra, which is what
        ``scipy.sparse.csgraph`` provides.  Unreachable voxels are ``inf``.
        """
        g = self.csr(direction).tocoo()
        src_nodes = np.unique([self.flat_to_node[self._flat(v)] for v in sources])
        src_nodes = src_nodes[src_nodes >= 0]
        if src_nodes.size == 0:
            raise ValueError("no source voxel is passable")

        n = self.n_nodes
        rows = np.concatenate([g.row, np.full(src_nodes.size, n, dtype=np.int64)])
        cols = np.concatenate([g.col, src_nodes])
        data = np.concatenate([g.data, np.zeros(src_nodes.size)])
        big = csr_matrix((data, (rows, cols)), shape=(n + 1, n + 1))

        dist = sp_dijkstra(big, directed=True, indices=n)[:n]

        field = np.full(self.shape, np.inf, dtype=np.float64)
        field.ravel()[self.node_flat] = dist
        return field

    def _flat(self, ijk) -> int:
        nx, ny, nz = self.shape
        return int((ijk[0] * ny + ijk[1]) * nz + ijk[2])


# ---------------------------------------------------------------------------
# Junction search -- independent reimplementation
# ---------------------------------------------------------------------------
def _argmin_over(summed: np.ndarray, passable: np.ndarray,
                 iz_range: tuple[int, int] | None = None):
    """Argmin of a summed distance field over passable voxels, optionally
    restricted to an RL slice.  Returns (voxel, value) or (None, inf)."""
    mask = np.isfinite(summed) & passable
    if iz_range is not None:
        rl = np.zeros(summed.shape, dtype=bool)
        rl[:, :, iz_range[0]:iz_range[1] + 1] = True
        restricted = mask & rl
        # Mirrors find_junction_at_rl: relax the RL slice if it is empty.
        mask = restricted if restricted.any() else mask
    if not mask.any():
        return None, math.inf
    masked = np.where(mask, summed, np.inf)
    flat = int(np.argmin(masked))
    v = tuple(int(c) for c in np.unravel_index(flat, summed.shape))
    return v, float(summed[v])


def _group_levels(zones: list[dict], zone_voxels, min_rl: float,
                  cell_size: float) -> list[list[int]]:
    """
    Group zone indices by median voxel RL, reproducing
    run_steiner_poly.py:group_zones_by_level (shallowest level first).
    """
    def med_rl(zi):
        return float(np.median([min_rl + v[2] * cell_size
                                for v in zone_voxels[zi]]))

    order = sorted(range(len(zones)), key=lambda zi: -med_rl(zi))
    levels: list[list[int]] = []
    for zi in order:
        rl = med_rl(zi)
        for level in levels:
            if abs(rl - float(np.mean([med_rl(z) for z in level]))) <= LEVEL_TOL_M:
                level.append(zi)
                break
        else:
            levels.append([zi])
    return levels


def _level_iz_target(level: list[int], zone_voxels) -> int:
    """Median iz over every voxel of the zones at one level."""
    all_iz = [v[2] for zi in level for v in zone_voxels[zi]]
    return int(np.median(all_iz)) if all_iz else 0


# ---------------------------------------------------------------------------
# The verifier
# ---------------------------------------------------------------------------
class ExactnessFailure(RuntimeError):
    """Raised when an exact-labelled family does not survive verification."""


def verify_instance(instance: dict, families: list[dict],
                    verbose: bool = True) -> dict[str, dict]:
    """
    Independently verify every exact-labelled family of one instance.

    Parameters
    ----------
    instance : dict from ``loader.load_instance``
    families : the solver's per-family records from ``mineopt_adapter``
    verbose  : print a line per family

    Returns
    -------
    dict keyed by family name, each value an ``exactness_check`` record ready
    to embed in ``reference.json``.

    Raises
    ------
    ExactnessFailure
        If any exact family's independently recomputed optimum is cheaper than
        what the solver reported, or if the costs disagree beyond tolerance.
    """
    md = instance["metadata"]
    cm = md["cost_model"]
    g = md["grid"]

    cost_grid = instance["cost_grid"]
    passable = instance["passable"]
    cell = float(g["cell_size_m"])
    min_rl = float(g["min_coords_m"][2])

    graph = VoxelGraph(cost_grid, cell, cm["excavation_rate_per_m"])

    portal_df = graph.multi_source(instance["portal_voxels"], "descent")
    zone_voxels = instance["zone_voxels"]
    ascent = [graph.multi_source(zv, "ascent") for zv in zone_voxels]
    horiz = [graph.multi_source(zv, "horizontal") for zv in zone_voxels]

    levels = _group_levels(md["zones"], zone_voxels, min_rl, cell)
    by_family = {f["family"]: f for f in families}
    checks: dict[str, dict] = {}

    def _record(name, ours_cost, ours_juncs, theirs, note=""):
        their_cost = float(theirs["cost"])
        their_juncs = [tuple(j) for j in theirs["junctions_voxel"]]
        rel = (abs(ours_cost - their_cost) / their_cost
               if their_cost > 0 else math.inf)
        junc_match = ours_juncs == their_juncs
        ok = rel <= COST_RTOL

        if not ok and ours_cost < their_cost:
            raise ExactnessFailure(
                f"{instance['instance_id']}/{name}: independent recomputation "
                f"found a CHEAPER optimum ({ours_cost:,.2f}) than the solver "
                f"reported ({their_cost:,.2f}), relative delta {rel:.3e}. The "
                f"'exact' label is not justified.")
        if not ok:
            raise ExactnessFailure(
                f"{instance['instance_id']}/{name}: independent recomputation "
                f"disagrees with the solver beyond tolerance: "
                f"{ours_cost:,.2f} vs {their_cost:,.2f} "
                f"(relative delta {rel:.3e} > {COST_RTOL:.0e}).")

        rec = {
            "performed": True,
            "independent_argmin_matches": True,
            "junction_voxel_matches": junc_match,
            "max_rel_cost_delta": float(rel),
            "independent_cost": round(ours_cost, 4),
            "method": ("independent float64 CSR graph + "
                       "scipy.sparse.csgraph.dijkstra + independent argmin; "
                       "shares no code or distance fields with the reference "
                       "solver"),
        }
        if note:
            rec["note"] = note
        if not junc_match:
            rec["note"] = (
                (note + " " if note else "")
                + "Junction voxels differ from the solver's but the objective "
                  "values agree within tolerance, so this is a tie on a "
                  "discrete grid, not a suboptimality.")
        checks[name] = rec
        if verbose:
            flag = "OK " if junc_match else "TIE"
            print(f"      {flag} {name:<26} independent {ours_cost:>14,.0f}  "
                  f"solver {their_cost:>14,.0f}  rel {rel:.2e}")

    # ── single_junction: one argmin over the ascent fields of ALL zones ──────
    if "single_junction" in by_family:
        summed = portal_df.copy()
        for a in ascent:
            summed = summed + a
        junc, _ = _argmin_over(summed, passable)
        if junc is not None:
            cost = float(portal_df[junc]) + sum(float(a[junc]) for a in ascent)
            _record("single_junction", cost, [junc],
                    by_family["single_junction"])

    # ── sublevel_fan: one RL-restricted argmin per sublevel ──────────────────
    if "sublevel_fan" in by_family:
        total, juncs, ok = 0.0, [], True
        for level in levels:
            summed = portal_df.copy()
            for zi in level:
                summed = summed + horiz[zi]
            iz = _level_iz_target(level, zone_voxels)
            nz = cost_grid.shape[2]
            junc, _ = _argmin_over(
                summed, passable,
                (max(0, iz - IZ_TOL), min(nz - 1, iz + IZ_TOL)))
            if junc is None:
                ok = False
                break
            total += float(portal_df[junc]) + sum(float(horiz[zi][junc])
                                                  for zi in level)
            juncs.append(junc)
        if ok:
            _record("sublevel_fan", total, juncs, by_family["sublevel_fan"])

    # ── two_branch / three_branch: exhaustive over level partitions ──────────
    from itertools import combinations

    for k, name in ((2, "two_branch"), (3, "three_branch")):
        if name not in by_family or len(levels) < k:
            continue
        best_cost, best_juncs = math.inf, None

        for splits in combinations(range(1, len(levels)), k - 1):
            bounds = [0] + list(splits) + [len(levels)]
            total, juncs, ok = 0.0, [], True

            for bi in range(k):
                group_levels = levels[bounds[bi]:bounds[bi + 1]]
                group_zones = [z for lz in group_levels for z in lz]
                if not group_zones:
                    ok = False
                    break

                # The crosscut field of a zone is finite only at its own RL, so
                # summing across sublevels yields inf everywhere. The solver
                # falls back to the 3-D ascent fields for multi-level branches
                # (run_steiner_poly.py:602-609); mirror that exactly.
                use_xcut = len(group_levels) == 1
                fields = horiz if use_xcut else ascent

                summed = portal_df.copy()
                for zi in group_zones:
                    summed = summed + fields[zi]

                if use_xcut:
                    iz = _level_iz_target(group_zones, zone_voxels)
                    nz = cost_grid.shape[2]
                    junc, _ = _argmin_over(
                        summed, passable,
                        (max(0, iz - IZ_TOL), min(nz - 1, iz + IZ_TOL)))
                else:
                    junc, _ = _argmin_over(summed, passable)

                if junc is None:
                    ok = False
                    break
                total += float(portal_df[junc]) + sum(
                    float(fields[zi][junc]) for zi in group_zones)
                juncs.append(junc)

            if ok and total < best_cost:
                best_cost, best_juncs = total, juncs

        if best_juncs is not None:
            _record(name, best_cost, best_juncs, by_family[name],
                    note=f"Exhaustive over all "
                         f"{math.comb(len(levels) - 1, k - 1)} consecutive "
                         f"level partitions.")

    return checks
