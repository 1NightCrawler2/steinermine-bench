#!/usr/bin/env python
"""
A minimal, self-contained SteinerMineBench solver.
==================================================

SPDX-License-Identifier: MIT

This is a worked example of the whole third-party workflow: load an instance,
solve it with your own code, emit a submission JSON, score it.  It depends only
on numpy, scipy and the loader -- no MineOptimizer, no reference solver.

The strategy is deliberately simple, so the gap it reports is real rather than
flattering: place a single junction at the voxel minimising

    d_descent(portal, v) + sum_k d_ascent(zone_k, v)

This is the ``single_junction`` family.  It is optimal within that family but
usually beaten by the chained topologies, so expect positive gaps of a few tens
of percent on most instances -- which is exactly the point of the exercise.

Two details that are easy to get wrong, both worth copying
----------------------------------------------------------
**Direction convention.** The edge weight charges the voxel being entered, so a
segment's cost depends on traversal direction.  The benchmark convention is
that every segment is costed *from its own terminal towards the junction*: the
ramp over descent arcs (dz <= 0) from the portal, and each zone leg over
**ascent** arcs (dz >= 0) from the zone.  Costing a zone leg the other way round
is a defensible reading of the same physical drive, but it shifts the total by
about 1 percent -- enough to swamp the gaps being measured.  See
``metadata["cost_model"]["segment_direction_convention"]``.

**Multi-source Dijkstra.** Terminals are voxel *sets* (a zone here has 336
voxels).  Passing them all to ``scipy``'s ``indices`` runs one Dijkstra per
source.  Attach a single virtual node with zero-cost arcs to the whole set
instead: one Dijkstra, same answer, two orders of magnitude faster.

Use it as the skeleton for your own solver: replace ``solve()`` and keep
everything else.

Usage
-----
    python examples/minimal_solver.py > submission.json
    python score.py --submission submission.json

    python examples/minimal_solver.py --only scale-130k zones-04 > sub.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loader import list_instances, load_instance      # noqa: E402

SENTINEL_TEST = 0.9e9


def build_graphs(cost_grid, cell_size_m, excavation_rate):
    """
    CSR graphs over passable voxels for the descent and ascent arc sets.

    Edge weight is the published objective, charging the voxel entered:
    ``w(u,v) = (cost_grid[v] + excavation_rate) * ||v-u|| * cell_size_m``.
    """
    nx, ny, nz = cost_grid.shape
    flat = cost_grid.ravel().astype(np.float64)
    node_flat = np.flatnonzero(flat < SENTINEL_TEST).astype(np.int64)
    n = node_flat.size

    lookup = np.full(nx * ny * nz, -1, dtype=np.int64)
    lookup[node_flat] = np.arange(n)

    ijk = np.stack([node_flat // (ny * nz),
                    (node_flat // nz) % ny,
                    node_flat % nz], axis=1)

    rows, cols, data, dzs = [], [], [], []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                length = math.sqrt(dx * dx + dy * dy + dz * dz) * cell_size_m
                nc = ijk + (dx, dy, dz)
                ok = ((nc[:, 0] >= 0) & (nc[:, 0] < nx)
                      & (nc[:, 1] >= 0) & (nc[:, 1] < ny)
                      & (nc[:, 2] >= 0) & (nc[:, 2] < nz))
                safe = np.clip(nc, 0, (nx - 1, ny - 1, nz - 1))
                nflat = (safe[:, 0] * ny + safe[:, 1]) * nz + safe[:, 2]
                dst = lookup[nflat]
                valid = ok & (dst >= 0)
                if not valid.any():
                    continue
                src = np.flatnonzero(valid)
                rows.append(src)
                cols.append(dst[valid])
                data.append((flat[nflat[valid]] + excavation_rate) * length)
                dzs.append(np.full(src.size, dz, dtype=np.int8))

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.concatenate(data)
    dzs = np.concatenate(dzs)

    def sub(keep):
        return csr_matrix((data[keep], (rows[keep], cols[keep])), shape=(n, n))

    return sub(dzs <= 0), sub(dzs >= 0), lookup, node_flat, n


def multi_source(graph, source_nodes, n):
    """
    One Dijkstra from a whole terminal set, via a virtual zero-cost super-source.

    Passing the source list to scipy's ``indices`` instead would run one
    Dijkstra per voxel -- hundreds of them for a production zone.
    """
    g = graph.tocoo()
    rows = np.concatenate([g.row, np.full(source_nodes.size, n, dtype=np.int64)])
    cols = np.concatenate([g.col, source_nodes])
    vals = np.concatenate([g.data, np.zeros(source_nodes.size)])
    big = csr_matrix((vals, (rows, cols)), shape=(n + 1, n + 1))
    return dijkstra(big, directed=True, indices=n)[:n]


def solve(instance: dict) -> dict:
    """Place one junction minimising portal distance plus total zone distance."""
    md = instance["metadata"]
    cell = float(md["grid"]["cell_size_m"])
    exc = md["cost_model"]["excavation_rate_per_m"]
    cost_grid = instance["cost_grid"]
    nx, ny, nz = cost_grid.shape

    descent, ascent, lookup, node_flat, n = build_graphs(cost_grid, cell, exc)

    def nodes_of(voxels):
        f = np.array([(v[0] * ny + v[1]) * nz + v[2] for v in voxels],
                     dtype=np.int64)
        idx = lookup[f]
        return np.unique(idx[idx >= 0])

    # Ramp: portal -> v over descent arcs.
    total = multi_source(descent, nodes_of(instance["portal_voxels"]), n)
    # Zone legs: zone -> v over ascent arcs, per the direction convention.
    for zv in instance["zone_voxels"]:
        total = total + multi_source(ascent, nodes_of(zv), n)

    if not np.isfinite(total).any():
        raise RuntimeError("no voxel reaches the portal and every zone")

    best = int(np.nanargmin(np.where(np.isfinite(total), total, np.inf)))
    flat = int(node_flat[best])
    junction = (flat // (ny * nz), (flat // nz) % ny, flat % nz)

    return {
        "cost": float(total[best]),
        "topology": "single_junction",
        "junctions_voxel": [[int(c) for c in junction]],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("Usage")[0].strip(),
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", metavar="ID",
                   help="solve only these instances (default: all with a bundle)")
    p.add_argument("--skip", nargs="+", metavar="ID", default=["scale-129m"],
                   help="instances to skip (default: the 129 M-voxel rung, "
                        "which needs far more memory than this example allows)")
    args = p.parse_args(argv)

    ids = args.only or [i for i in list_instances() if i not in set(args.skip)]

    results = {}
    for iid in ids:
        print(f"solving {iid} ...", file=sys.stderr, flush=True)
        t0 = time.time()
        try:
            r = solve(load_instance(iid))
        except (MemoryError, RuntimeError) as exc:
            print(f"  skipped: {exc}", file=sys.stderr)
            continue
        r["runtime_s"] = round(time.time() - t0, 2)
        results[iid] = r
        print(f"  cost {r['cost']:,.0f}  ({r['runtime_s']} s)", file=sys.stderr)

    json.dump({
        "solver": "minimal_solver (single-junction argmin) v1.0",
        "authors": "SteinerMineBench example",
        "notes": ("Reference implementation of the submission workflow. Places "
                  "one junction minimising d(portal,v) + sum_k d(v,zone_k) on "
                  "the descent graph. Optimal within the single_junction "
                  "family; expect positive gaps against the chained "
                  "topologies."),
        "results": results,
    }, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
