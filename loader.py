"""
SteinerMineBench loader -- public entry point.
=============================================

SPDX-License-Identifier: MIT

This module is a thin re-export of :mod:`steinerbench.loader` so that the
documented one-liner works from the repository root with no installation step:

>>> from loader import load_instance
>>> inst = load_instance("xgrid-f2x-qpoor")
>>> inst["cost_grid"].shape
(126, 82, 100)
>>> inst["metadata"]["grid"]["cell_size_m"]
5.0
>>> len(inst["zone_voxels"])
4
>>> inst["reference"]["reference_type"] in {"exact", "best_known", "unsolved"}
True

See :func:`steinerbench.loader.load_instance` for the full return contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from steinerbench.loader import (  # noqa: E402
    instance_path,
    instances_dir,
    list_instances,
    load_instance,
    load_reference_paths,
    set_instances_dir,
    voxel_from_world,
    world_from_voxel,
)

__all__ = [
    "load_instance",
    "load_reference_paths",
    "list_instances",
    "instance_path",
    "instances_dir",
    "set_instances_dir",
    "world_from_voxel",
    "voxel_from_world",
]
