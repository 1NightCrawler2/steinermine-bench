"""
Frozen Barton Q-system cost tier schedule for SteinerMineBench.
==============================================================

SPDX-License-Identifier: MIT

This module is a *frozen* copy of the geotechnical cost model used by the
MineOptimizer reference solver (``config.Q_COST_TIERS``,
``config.FAULT_SINGLE_MIN_COST``, ``config.FAULT_MULTI_MIN_COST``).  It is
duplicated rather than imported so that the published benchmark stays
self-contained and byte-stable even if the upstream solver's configuration
changes.  ``assert_matches_mineoptimizer()`` cross-checks the two whenever
MineOptimizer happens to be importable.

Cost model
----------
Support cost per metre of drive is a step function of the Barton Q value::

    Q <= 0.04   ->  $7,585.4 /m   Exceptionally poor
    Q <= 0.10   ->  $5,679.5 /m   Extremely poor
    Q <= 0.34   ->  $4,460.9 /m   Very poor
    Q <= 1.181  ->  $3,059.0 /m   Poor
    Q <= 6.0    ->  $1,806.9 /m   Fair
    Q >  6.0    ->  $1,111.9 /m   Good

Fault intercepts impose a *floor* (not a multiplier) on the support cost,
because the structural instability of a fault is independent of the host
rock's Q::

    1 distinct fault      ->  cost = max(tier_cost, $3,059.0/m)
    2+ distinct faults    ->  cost = max(tier_cost, $4,460.9/m)

Both floors are themselves tier costs (tier index 3 and 2 respectively), so
applying the floor maps a tier index onto another tier index.  That is what
makes the uint8 ``tier_index`` storage format in ``cost_grid.npz`` lossless.

Units
-----
``cost_per_m``      : US dollars per linear metre of excavated drive ($/m)
``q_max``           : dimensionless Barton Q rock mass quality upper bound
"""

from __future__ import annotations

import numpy as np

# Ordered WORST-ROCK-FIRST.  Do not re-sort: the tier index stored in
# cost_grid.npz is an index into this list, and the support-class labels
# below are positionally aligned with it.
Q_COST_TIERS: list[tuple[float, float]] = [
    (0.04, 7585.4),
    (0.10, 5679.5),
    (0.34, 4460.9),
    (1.181, 3059.0),
    (6.0, 1806.9),
    (float("inf"), 1111.9),
]

FAULT_SINGLE_MIN_COST = 3059.0   # $/m -- one distinct fault intercept
FAULT_MULTI_MIN_COST = 4460.9    # $/m -- two or more distinct fault intercepts

# Excavation cost added to every edge by the reference solver, independent of
# rock quality.  NOT baked into the shipped cost grid -- see metadata.json
# cost_model.excavation_rate_in_grid.
EXCAVATION_RATE = 1000.0         # $/m

# Barton Q support classes, positionally aligned with Q_COST_TIERS.
SUPPORT_CLASSES: list[str] = [
    "Exceptionally poor (Q<0.04): steel sets + shotcrete + spiling",
    "Extremely poor (Q 0.04-0.10): heavy systematic bolt+mesh+shotcrete",
    "Very poor (Q 0.10-0.34): systematic bolt+mesh+shotcrete",
    "Poor (Q 0.34-1.18): systematic bolting + shotcrete",
    "Fair (Q 1.18-6.0): spot bolting",
    "Good (Q>6.0): unsupported / spot bolting",
]

N_TIERS = len(Q_COST_TIERS)
TIER_COSTS = np.array([c for _, c in Q_COST_TIERS], dtype=np.float64)
Q_BREAKS = np.array([q for q, _ in Q_COST_TIERS], dtype=np.float64)

#: uint8 value marking a voxel that is not excavatable (above ground surface).
SENTINEL_TIER = np.uint8(255)
#: Float cost written at sentinel voxels in the reconstructed cost grid ($/m).
SENTINEL_COST = 1e9
#: A voxel is passable iff ``cost < PASSABLE_FRACTION * SENTINEL_COST``.
PASSABLE_FRACTION = 0.9

# Floors expressed as tier indices, for the lossless tier-index pipeline.
FAULT_SINGLE_MIN_TIER = int(np.argmin(np.abs(TIER_COSTS - FAULT_SINGLE_MIN_COST)))
FAULT_MULTI_MIN_TIER = int(np.argmin(np.abs(TIER_COSTS - FAULT_MULTI_MIN_COST)))


def q_to_tier_index(q: np.ndarray) -> np.ndarray:
    """
    Map Barton Q values to tier indices (0 = worst rock, 5 = best rock).

    Uses the same top-to-bottom ``q <= threshold`` evaluation order as
    ``wp1_voxel/pipeline.py:map_q_to_tier_cost``.  Non-finite Q maps to the
    worst tier (index 0), matching the upstream "far from data -> worst tier"
    conservatism.

    Parameters
    ----------
    q : ndarray
        Barton Q values, any shape.

    Returns
    -------
    ndarray of uint8, same shape as ``q``.
    """
    q = np.asarray(q, dtype=np.float64)
    # np.searchsorted on the ascending Q_BREAKS gives, for each q, the index of
    # the first threshold that q does not exceed -- exactly the tier index.
    # side="left" so that q == threshold lands in the `q <= threshold` tier.
    idx = np.searchsorted(Q_BREAKS, q, side="left")
    idx = np.clip(idx, 0, N_TIERS - 1).astype(np.uint8)
    idx[~np.isfinite(q)] = 0
    return idx


def apply_fault_floor(tier_index: np.ndarray, fault_count: np.ndarray) -> np.ndarray:
    """
    Apply the fault-intercept cost floor in tier-index space.

    Because a lower tier index means worse rock and higher cost,
    ``cost = max(tier_cost, floor)`` becomes ``tier = min(tier, floor_tier)``.

    Parameters
    ----------
    tier_index  : uint8 ndarray of tier indices (255 = sentinel, left untouched)
    fault_count : integer ndarray, number of DISTINCT faults per voxel

    Returns
    -------
    uint8 ndarray, same shape.
    """
    out = np.asarray(tier_index).copy()
    live = out != SENTINEL_TIER
    single = live & (np.asarray(fault_count) == 1)
    multi = live & (np.asarray(fault_count) >= 2)
    out[single] = np.minimum(out[single], np.uint8(FAULT_SINGLE_MIN_TIER))
    out[multi] = np.minimum(out[multi], np.uint8(FAULT_MULTI_MIN_TIER))
    return out


def tier_index_to_cost(tier_index: np.ndarray) -> np.ndarray:
    """
    Expand uint8 tier indices to a float32 cost grid in $/m.

    Sentinel voxels (255) become ``SENTINEL_COST``.
    """
    ti = np.asarray(tier_index)
    cost = np.empty(ti.shape, dtype=np.float32)
    live = ti != SENTINEL_TIER
    cost[live] = TIER_COSTS[ti[live]].astype(np.float32)
    cost[~live] = np.float32(SENTINEL_COST)
    return cost


def tier_schedule_records() -> list[dict]:
    """The tier schedule as JSON-serialisable records for metadata.json."""
    records = []
    lower = 0.0
    for i, ((q_max, cost), cls) in enumerate(zip(Q_COST_TIERS, SUPPORT_CLASSES)):
        records.append({
            "tier_index": i,
            "q_min_exclusive": lower,
            "q_max_inclusive": None if q_max == float("inf") else q_max,
            "cost_per_m": cost,
            "support_class": cls,
        })
        lower = q_max
    return records


def assert_matches_mineoptimizer(config_module=None) -> None:
    """
    Cross-check this frozen schedule against a live MineOptimizer ``config``.

    Raises AssertionError on any divergence.  Intended for CI and for the
    verification step in the benchmark build, not for normal use -- the
    benchmark never imports MineOptimizer at load time.

    Example
    -------
    >>> import config                                        # doctest: +SKIP
    >>> from steinerbench.tiers import assert_matches_mineoptimizer
    >>> assert_matches_mineoptimizer(config)                 # doctest: +SKIP
    """
    if config_module is None:
        import config as config_module  # type: ignore[no-redef]

    assert list(config_module.Q_COST_TIERS) == Q_COST_TIERS, (
        f"Q_COST_TIERS diverged.\n  frozen:   {Q_COST_TIERS}\n"
        f"  upstream: {list(config_module.Q_COST_TIERS)}"
    )
    assert config_module.FAULT_SINGLE_MIN_COST == FAULT_SINGLE_MIN_COST, (
        f"FAULT_SINGLE_MIN_COST diverged: {config_module.FAULT_SINGLE_MIN_COST} "
        f"!= {FAULT_SINGLE_MIN_COST}"
    )
    assert config_module.FAULT_MULTI_MIN_COST == FAULT_MULTI_MIN_COST, (
        f"FAULT_MULTI_MIN_COST diverged: {config_module.FAULT_MULTI_MIN_COST} "
        f"!= {FAULT_MULTI_MIN_COST}"
    )
    assert config_module.EXCAVATION_RATE == EXCAVATION_RATE, (
        f"EXCAVATION_RATE diverged: {config_module.EXCAVATION_RATE} "
        f"!= {EXCAVATION_RATE}"
    )
    # Both floors must be exactly representable as tier costs, else the uint8
    # tier_index storage format would be lossy.
    assert TIER_COSTS[FAULT_SINGLE_MIN_TIER] == FAULT_SINGLE_MIN_COST
    assert TIER_COSTS[FAULT_MULTI_MIN_TIER] == FAULT_MULTI_MIN_COST
