"""
SteinerMineBench -- a benchmark suite for geotechnically-weighted Steiner trees
on 3-D voxel cost grids (underground mine ramp network optimisation).

SPDX-License-Identifier: MIT

All benchmark instances are SYNTHETIC and procedurally generated.  They are not
derived from any real or operating mine.  See ``steinerbench.spec``.
"""

from __future__ import annotations

from steinerbench.spec import BENCHMARK_VERSION, INSTANCE_IDS

__version__ = BENCHMARK_VERSION
__all__ = ["BENCHMARK_VERSION", "INSTANCE_IDS", "__version__"]
