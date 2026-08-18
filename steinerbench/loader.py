"""
SteinerMineBench loader.
========================

SPDX-License-Identifier: MIT

Ten-line usage
--------------
>>> from loader import load_instance
>>> inst = load_instance("xgrid-f2x-qpoor")
>>> inst["cost_grid"].shape                       # (126, 82, 100)
(126, 82, 100)
>>> inst["cost_grid"].dtype                       # float32, dollars per metre
dtype('float32')
>>> portal = inst["portal_voxels"][0]             # (i, j, k) tuple
>>> zones = inst["zone_voxels"]                   # list of lists of (i, j, k)
>>> inst["metadata"]["cost_model"]["excavation_rate_per_m"]
1000.0
>>> ref = inst["reference"]
>>> ref["reference_type"], ref["reference_cost"]  # 'exact' | 'best_known' | 'unsolved'
('...', ...)

The cost grid holds the geotechnical SUPPORT cost per metre.  The excavation
rate is deliberately NOT baked in; the objective's edge weight is

    w(u, v) = (cost_grid[v] + excavation_rate_per_m) * effective_length_m(u, v)

with ``effective_length_m`` the Euclidean distance between voxel centres over
the 26-neighbourhood.  See ``metadata["cost_model"]`` on any instance.

Impassable voxels (above the ground surface) carry ``1e9``.  Test passability
with ``inst["passable"]`` rather than an equality check on the sentinel.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from steinerbench import spec, tiers
from steinerbench.grid import read_cost_grid_npz

#: Root of the instance library.  Override with ``set_instances_dir`` if you
#: keep the bundles somewhere other than alongside this package.
_INSTANCES_DIR = Path(__file__).resolve().parent.parent / "instances"


def set_instances_dir(path) -> None:
    """Point the loader at a different instance library."""
    global _INSTANCES_DIR
    _INSTANCES_DIR = Path(path)
    load_instance.cache_clear()


def instances_dir() -> Path:
    """Directory the loader reads instance bundles from."""
    return _INSTANCES_DIR


def instance_path(instance_id: str) -> Path:
    """Directory holding one instance bundle."""
    spec.get(instance_id)
    return _INSTANCES_DIR / instance_id


def list_instances(group: str | None = None) -> list[str]:
    """
    Instance ids present on disk, optionally filtered to one group.

    ``group`` accepts either the letter (``"A"``) or the name
    (``"crossing_grid"``).
    """
    wanted = spec.select(group=group) if group else spec.INSTANCE_IDS
    return [i for i in wanted if (_INSTANCES_DIR / i / "metadata.json").exists()]


# ---------------------------------------------------------------------------
# Coordinate helpers -- so nobody has to rediscover the +0.5 convention
# ---------------------------------------------------------------------------
def world_from_voxel(ijk, metadata) -> np.ndarray:
    """
    Voxel index (i, j, k) -> world position (EAST, NORTH, RL) in metres.

    Accepts a single triple or an (N, 3) array.  RL is positive upward.
    """
    g = metadata["grid"]
    mn = np.asarray(g["min_coords_m"], dtype=np.float64)
    return mn + (np.asarray(ijk, dtype=np.float64) + 0.5) * g["cell_size_m"]


def voxel_from_world(world, metadata) -> np.ndarray:
    """
    World position (EAST, NORTH, RL) in metres -> voxel index (i, j, k).

    Accepts a single triple or an (N, 3) array.  Results are clipped into the
    grid.
    """
    g = metadata["grid"]
    mn = np.asarray(g["min_coords_m"], dtype=np.float64)
    dims = np.asarray(g["dims"], dtype=np.int64)
    idx = np.floor((np.asarray(world, dtype=np.float64) - mn)
                   / g["cell_size_m"]).astype(np.int64)
    return np.clip(idx, 0, dims - 1)


# ---------------------------------------------------------------------------
# Terminal voxel reconstruction
# ---------------------------------------------------------------------------
def _portal_voxels(metadata, passable) -> list[tuple[int, int, int]]:
    """Rebuild the portal terminal voxel set from metadata + the grid."""
    from steinerbench import geology
    s = spec.get(metadata["instance_id"])
    g = metadata["grid"]
    p = geology.portal_terminal(s, tuple(g["dims"]), g["cell_size_m"],
                                tuple(g["min_coords_m"]), passable)
    return p["_voxels"]


def _zone_voxels(metadata, passable) -> list[list[tuple[int, int, int]]]:
    """Rebuild each production zone's terminal voxel set."""
    from steinerbench import geology
    s = spec.get(metadata["instance_id"])
    g = metadata["grid"]
    zones = geology.zone_terminals(s, tuple(g["dims"]), g["cell_size_m"],
                                   tuple(g["min_coords_m"]), passable)
    return [z["_voxels"] for z in zones]


# ---------------------------------------------------------------------------
# The public entry point
# ---------------------------------------------------------------------------
@lru_cache(maxsize=4)
def load_instance(instance_id: str) -> dict:
    """
    Load one benchmark instance.

    Parameters
    ----------
    instance_id : str
        One of ``steinerbench.spec.INSTANCE_IDS``, e.g. ``"xgrid-f2x-qpoor"``.

    Returns
    -------
    dict with keys

    ``cost_grid``     float32 (nx, ny, nz), geotechnical support cost in $/m;
                      ``1e9`` marks voxels above ground surface
    ``passable``      bool   (nx, ny, nz), True where the voxel is excavatable
    ``fault_count``   uint8  (nx, ny, nz), distinct faults intersecting a voxel
    ``surface_rl``    float32 (nx, ny), ground surface RL in metres per column
    ``tier_index``    uint8  (nx, ny, nz), index into the cost tier schedule
    ``portal_voxels`` list of (i, j, k), the portal terminal set
    ``zone_voxels``   list of lists of (i, j, k), one per production zone
    ``metadata``      dict, the parsed ``metadata.json``
    ``reference``     dict, the parsed ``reference.json``; ``reference_type``
                      is ``"unsolved"`` if no reference has been computed yet

    Example
    -------
    >>> inst = load_instance("scale-130k")            # doctest: +SKIP
    >>> cost = inst["cost_grid"]                      # doctest: +SKIP
    >>> cheapest_passable = cost[inst["passable"]].min()   # doctest: +SKIP

    Results are cached, so repeated calls for the same instance are free.  The
    returned arrays are shared between callers -- copy before mutating.
    """
    path = instance_path(instance_id)
    if not path.exists():
        raise FileNotFoundError(
            f"instance bundle not found: {path}\n"
            f"Generate it with:  python generate.py --only {instance_id}")

    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    grid = read_cost_grid_npz(path / "cost_grid.npz")

    dims = tuple(metadata["grid"]["dims"])
    if grid["tier_index"].shape != dims:
        raise ValueError(
            f"{instance_id}: cost_grid.npz shape {grid['tier_index'].shape} "
            f"does not match metadata dims {dims}")

    passable = grid["tier_index"] != tiers.SENTINEL_TIER

    ref_path = path / "reference.json"
    if ref_path.exists():
        reference = json.loads(ref_path.read_text(encoding="utf-8"))
    else:
        reference = {
            "instance_id": instance_id,
            "reference_type": "unsolved",
            "reference_cost": None,
            "note": ("No reference solution present. Run "
                     f"`python solve_reference.py --only {instance_id}`."),
        }

    return {
        "instance_id": instance_id,
        "cost_grid": grid["cost_grid"],
        "passable": passable,
        "fault_count": grid["fault_count"],
        "surface_rl": grid["surface_rl"],
        "tier_index": grid["tier_index"],
        "portal_voxels": _portal_voxels(metadata, passable),
        "zone_voxels": _zone_voxels(metadata, passable),
        "metadata": metadata,
        "reference": reference,
        "path": path,
    }


def load_reference_paths(instance_id: str) -> dict:
    """
    Load the reference solution's voxel paths (``reference_paths.npz``).

    Returns a dict mapping ``"<family>/<segment index>"`` to an (M, 3) int32
    array of voxel indices.  Raises FileNotFoundError if no reference has been
    computed for this instance.
    """
    path = instance_path(instance_id) / "reference_paths.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"no reference paths for {instance_id}: {path}\n"
            f"Run:  python solve_reference.py --only {instance_id}")
    with np.load(path) as z:
        return {k: z[k] for k in z.files}
