"""
WP3 — Geometric lower bounds on a buildable ramp network
========================================================

Why this module exists
----------------------
The voxel Steiner objective is a **relaxation**: it optimises over descending
paths with no grade limit and no turning circle, so its optimum is reached by
geometry no machine can drive.  A lower bound computed on that relaxation
(``steinerbench.lower_bound.pairwise_divergence``) is valid, but it is bounding
the wrong problem — measured against a *constrained* network it reads +200 % to
+340 %, and almost all of that is the relaxation being irrelevant rather than the
search being poor.

This module bounds the constrained problem directly, from the one constraint the
relaxation throws away: **a drive cannot descend faster than its grade limit.**
That single fact forces a minimum centreline length, which forces a minimum cost,
and it does so without any search at all — the bound is trigonometry over the
portal RL, the zone RLs and the cheapest rock in the model.

The bounds proved here
----------------------
1. ``min_length_for_drop`` — the length floor.  Partition any centreline into
   segments; segment *i* has length ``len_i`` and vertical change ``dz_i``, and
   the grade limit says ``|dz_i| <= len_i * sin(theta_max)`` with
   ``theta_max = atan(max_grade)``.  Summing, and using
   ``sum|dz_i| >= |sum dz_i| = drop``::

       sum len_i  >=  drop / sin(atan(max_grade))

   This holds for *any* grade-feasible path, of any shape, at any sampling
   density, including spirals and switchbacks — nothing about the path enters
   beyond its net drop.

2. ``required_length_m`` — the same, strengthened by the straight-line distance,
   since a path between two points is never shorter than the distance between
   them.  The maximum of two valid floors is a valid floor.

3. ``geometric_grade_floor`` — the capex floor.  The network must reach every
   zone, so its total length is at least the length of the single longest
   required path, and every metre costs at least the cheapest support tariff
   plus the excavation rate.

4. ``haulage_floor`` — the operating-cost floor, bounding the *same* per-zone
   haul model ``opex.annotate_opex`` evaluates, term by term.  See that
   function's docstring for the term-by-term conservatism argument, which is
   what makes it a floor of the reported number rather than of a similar one.

Where it must NOT be used
-------------------------
**Never on the raw track.**  A raw network descends at 45 degrees and is
legitimately cheaper than any grade-feasible ramp — on the published benchmark
this floor exceeds the raw reference cost on 19 of 25 instances, and that is
correct, not a bug.  ``track_bound`` refuses to combine the two, so the mistake
cannot be made by accident.

Units
-----
``*_rl_m``, ``drop_m``, ``*_length_m``  : metres
``max_grade``                          : dimensionless rise/run (0.20 = 20 %)
``unit_cost_per_m``, ``*_rate_per_m``   : US dollars per metre of drive
``tonnes``                             : tonnes (not Mt)

This module is a *frozen* copy of ``wp3_steiner/geometric_bound.py`` in the
MineOptimizer solver.  It is duplicated rather than imported so that the
published benchmark stays self-contained and byte-stable even if the upstream
solver changes, in the same pattern as ``steinerbench/tiers.py``.  The two are
cross-checked by ``assert_matches_mineoptimizer()`` at the bottom of this file
whenever MineOptimizer happens to be importable.  Keep the core identical: the
cross-check compares both the returned values and the function source text.
"""

from __future__ import annotations

import math

__all__ = [
    "METHOD_NAME", "METHOD_NAME_TOTAL", "RAW_METHOD_NAME",
    "min_length_for_drop", "required_length_m", "select_binding_zone",
    "geometric_grade_floor", "haulage_floor", "track_bound",
]

#: Method labels written into ``lower_bound.method``.
METHOD_NAME = "geometric_grade_floor"
METHOD_NAME_TOTAL = "geometric_grade_floor+opex_floor"
RAW_METHOD_NAME = "raw_relaxation"

#: A voxel step is at most a body diagonal, ``cell * sqrt(3)``, so a path of
#: length L occupies at least ``L / sqrt(3)`` metres of *voxel-counted* length.
#: The ventilation term is charged per voxel, so this is the safe conversion.
_VOXEL_LENGTH_SAFETY = math.sqrt(3.0)


def min_length_for_drop(drop_m: float, max_grade: float) -> float:
    """
    Least centreline length that can shed ``drop_m`` at grade <= ``max_grade``.

    ``L >= drop / sin(atan(g))``.  Returns 0.0 for a non-positive drop (a level
    or rising connection has no length forced by grade).

    Raises
    ------
    ValueError
        If ``max_grade <= 0`` — no descent is possible at all, so no finite
        length suffices and a caller asking for one has a bug upstream.
    """
    if max_grade <= 0.0:
        raise ValueError(f"max_grade must be positive, got {max_grade!r}")
    if drop_m <= 0.0:
        return 0.0
    return drop_m / math.sin(math.atan(max_grade))


def required_length_m(drop_m: float, max_grade: float,
                      plan_gap_m: float = 0.0) -> dict:
    """
    Least path length from a portal to a target, honouring both floors.

    ``plan_gap_m`` is the horizontal distance to the target; the straight-line
    floor is then ``hypot(plan_gap_m, drop_m)``.  Pass 0.0 to use the grade
    floor alone (still valid, just weaker).

    Returns a record carrying both candidates and which one binds, because the
    answer to "why is the bound this big" is usually "because of that one".
    """
    grade_len = min_length_for_drop(drop_m, max_grade)
    straight = math.hypot(max(plan_gap_m, 0.0), max(drop_m, 0.0))
    value = max(grade_len, straight)
    return {
        "value": value,
        "grade_length_m": grade_len,
        "straight_line_m": straight,
        "binds": "grade" if grade_len >= straight else "straight_line",
    }


def select_binding_zone(zones: list[dict], portal_rl_m: float,
                        max_grade: float) -> tuple[int, dict]:
    """
    The zone that forces the most network length, and its length record.

    ``zones`` is a list of dicts with ``top_rl_m`` (the HIGHEST RL of the zone
    body — the part closest to the portal, so the smallest drop and therefore
    the conservative choice) and optionally ``plan_gap_m`` (horizontal distance
    from the portal body to the NEAREST point of the zone body) and ``label``.

    Valid because the network must contain a path to every zone, so its length
    is at least the largest per-zone requirement.

    Returns ``(index, record)`` where ``record`` also carries ``drop_m`` and
    ``label``.  Raises ValueError on an empty zone list.
    """
    if not zones:
        raise ValueError("no zones supplied; a network with no terminal to "
                         "reach has no geometric floor")
    best_i, best = -1, None
    for i, z in enumerate(zones):
        drop = portal_rl_m - float(z["top_rl_m"])
        rec = required_length_m(drop, max_grade, float(z.get("plan_gap_m", 0.0)))
        rec["drop_m"] = drop
        rec["label"] = z.get("label")
        if best is None or rec["value"] > best["value"]:
            best_i, best = i, rec
    return best_i, best


def geometric_grade_floor(*, portal_rl_m: float, zones: list[dict],
                          max_grade: float, unit_cost_per_m: float,
                          excavation_rate_per_m: float,
                          n_portals: int = 0,
                          portal_establishment_cost: float = 0.0,
                          opex_floor: float = 0.0) -> dict:
    """
    Lower bound on the cost of a grade-feasible network reaching every zone.

    Parameters
    ----------
    portal_rl_m
        The LOWEST RL of the portal terminal body (collar RL minus its radius).
        Using the lowest point minimises the drop and so minimises the bound —
        the conservative choice, and the necessary one: a ramp may leave from
        any voxel of the portal body.
    zones
        See :func:`select_binding_zone`.  Only ``top_rl_m`` is required.
    unit_cost_per_m
        The cheapest support tariff over *passable* ground in this model.  Every
        metre of drive costs at least this plus the excavation rate.
    n_portals, portal_establishment_cost, opex_floor
        Additive terms for the ``total`` track; all default to zero so the
        capex-track call is the plain trigonometric floor.

    Returns
    -------
    dict
        An audit record echoing every input, so the value can be re-derived and
        checked from the record alone with no grid, no metadata and no code.
    """
    if unit_cost_per_m < 0 or excavation_rate_per_m < 0:
        raise ValueError("costs per metre must be non-negative")

    idx, binding = select_binding_zone(zones, portal_rl_m, max_grade)
    length = binding["value"]
    per_m = unit_cost_per_m + excavation_rate_per_m
    capex = length * per_m
    portals = max(int(n_portals), 0) * float(portal_establishment_cost)
    value = capex + portals + float(opex_floor)

    return {
        "value": value,
        "method": METHOD_NAME_TOTAL if (portals or opex_floor) else METHOD_NAME,
        "capex_floor": capex,
        "portal_establishment": portals,
        "opex_floor": float(opex_floor),
        "min_length_m": length,
        "length_binds_on": binding["binds"],
        "binding_zone": binding.get("label"),
        "binding_zone_index": idx,
        "drop_m": binding["drop_m"],
        "grade_length_m": binding["grade_length_m"],
        "straight_line_m": binding["straight_line_m"],
        "max_grade": max_grade,
        "sin_theta_max": math.sin(math.atan(max_grade)),
        "portal_rl_m": portal_rl_m,
        "zone_top_rl_m": portal_rl_m - binding["drop_m"],
        "unit_cost_per_m": unit_cost_per_m,
        "excavation_rate_per_m": excavation_rate_per_m,
        "cost_per_m": per_m,
        "n_portals": int(n_portals),
        "portal_establishment_cost": float(portal_establishment_cost),
        "derivation": (
            f"L >= {binding['drop_m']:.1f} m / sin(atan({max_grade:g})) "
            f"= {binding['grade_length_m']:.1f} m; straight line "
            f"{binding['straight_line_m']:.1f} m; L >= {length:.1f} m. "
            f"cost >= {length:.1f} * (${unit_cost_per_m:,.1f} + "
            f"${excavation_rate_per_m:,.1f}) = ${capex:,.0f}"
            + (f" + {int(n_portals)} portal(s) ${portals:,.0f}" if portals else "")
            + (f" + opex floor ${opex_floor:,.0f}" if opex_floor else "")
            + f" = ${value:,.0f}"),
    }


def haulage_floor(zones: list[dict], *, portal_rl_m: float, max_grade: float,
                  opex_model: dict) -> dict:
    """
    Lower bound on the life-of-mine operating cost of ANY network reaching
    every zone, bounding ``wp3_steiner.opex.annotate_opex`` term by term.

    That function computes, per zone::

        haul_m  = min over the zone's voxels of the network distance from a portal
        lift_m  = max(highest portal-source RL - median zone-voxel RL, 0)
        cost_k  = tonnes_k * (haul_m/1000 * haul_cost_per_t_km
                              + lift_m * haul_vert_cost_per_t_m)

    and separately charges ventilation and pumping per metre of network per
    year.  Each term is bounded below as follows, and every choice is the
    pessimistic one:

    ``haul_m``
        A network distance is at least the straight-line distance and at least
        the grade-forced length.  Both are measured to the zone's ``top_rl_m``
        (its closest point to the portal) and from the portal body's lowest RL,
        so no shorter route can exist.
    ``lift_m``
        The model measures lift to the zone's *median* RL from the *highest*
        portal source.  We use ``centre_rl_m`` (that same median) and the
        portal's lowest RL, so the bound cannot exceed the charged lift.
    ``vent_pump``
        Charged per voxel at ``cell`` metres each; a voxel step is at most a
        body diagonal, so a path of length L spans at least ``L / sqrt(3)``
        voxel-metres.  Pumping is dropped entirely (it is non-negative).

    ``opex_model`` supplies ``haul_cost_per_t_km``, ``haul_vert_cost_per_t_m``,
    ``vent_cost_per_m_yr`` and ``mine_life_years``.  It is passed in rather than
    imported so the benchmark and the case study can each bound the rates they
    actually charge.

    Zones need ``top_rl_m``, ``centre_rl_m``, ``tonnes`` and optionally
    ``plan_gap_m`` and ``label``.
    """
    a_km = float(opex_model["haul_cost_per_t_km"])
    b_lift = float(opex_model["haul_vert_cost_per_t_m"])
    vent = float(opex_model.get("vent_cost_per_m_yr", 0.0))
    years = float(opex_model.get("mine_life_years", 0.0))

    haulage = 0.0
    per_zone = []
    longest = 0.0
    for z in zones:
        drop_top = portal_rl_m - float(z["top_rl_m"])
        rec = required_length_m(drop_top, max_grade,
                                float(z.get("plan_gap_m", 0.0)))
        lift = max(portal_rl_m - float(z["centre_rl_m"]), 0.0)
        tonnes = float(z.get("tonnes", 0.0))
        cost = tonnes * (rec["value"] / 1000.0 * a_km + lift * b_lift)
        haulage += cost
        longest = max(longest, rec["value"])
        per_zone.append({
            "label": z.get("label"), "tonnes": tonnes,
            "min_haul_m": rec["value"], "min_lift_m": lift,
            "binds": rec["binds"], "cost": cost,
        })

    vent_floor = longest / _VOXEL_LENGTH_SAFETY * vent * years

    return {
        "value": haulage + vent_floor,
        "haulage_floor": haulage,
        "vent_pump_floor": vent_floor,
        "longest_required_length_m": longest,
        "voxel_length_safety_divisor": _VOXEL_LENGTH_SAFETY,
        "opex_model": {k: v for k, v in opex_model.items() if k != "note"},
        "per_zone": per_zone,
        "pumping_included": False,
        "derivation": (
            f"haulage >= sum_k tonnes_k * (L_k/1000 * ${a_km:g}/t-km + "
            f"lift_k * ${b_lift:g}/t-m) = ${haulage:,.0f}; "
            f"vent >= {longest:.1f}/sqrt(3) m * ${vent:g}/m-yr * {years:g} yr "
            f"= ${vent_floor:,.0f}; pumping dropped (non-negative)"),
    }


def track_bound(track: str, *, pairwise: dict | None = None,
                floor: dict | None = None,
                conditional: dict | None = None) -> dict:
    """
    Combine the available bounds for one cost track.  The ONLY place the
    per-track semantics live.

    ``raw``
        The pairwise-divergence relaxation bound, alone.  Passing ``floor``
        raises — the grade floor is *not* valid against a raw network, which
        may descend at 45 degrees, and on the published suite it exceeds the
        raw reference cost on most instances.  This guard is the structural
        reason that error cannot ship.
    ``constrained`` / ``total``
        ``max`` of the relaxation bound and the geometric floor.  The maximum
        of valid bounds is valid.

    ``conditional`` carries values that are *believed* to bound the optimum but
    are not proved to — the raw best-known cost is the motivating case.  They
    are recorded in ``components`` and named in ``conditional_components``, and
    they never contribute to ``value``, so ``valid`` stays honest.
    """
    if track not in ("raw", "constrained", "total"):
        raise ValueError(f"unknown track {track!r}")
    if track == "raw" and floor is not None:
        raise ValueError(
            "the geometric grade floor is NOT a valid bound on the raw track: "
            "a raw network is not grade-feasible and is legitimately cheaper "
            "than the floor (it exceeds the raw reference cost on most "
            "benchmark instances). Pass floor=None for track='raw'.")
    if pairwise is None and floor is None:
        raise ValueError("no bound supplied")

    components: dict[str, float] = {}
    if pairwise is not None:
        components[RAW_METHOD_NAME] = float(pairwise["value"])
    if floor is not None:
        components[floor.get("method", METHOD_NAME)] = float(floor["value"])

    method = max(components, key=components.__getitem__)
    value = components[method]

    conditional_names: list[str] = []
    if conditional:
        for name, val in conditional.items():
            components[name] = float(val)
            conditional_names.append(name)

    record = {
        "value": value,
        "method": method,
        "valid": True,
        "track": track,
        "components": components,
    }
    if conditional_names:
        record["conditional_components"] = sorted(conditional_names)
    if floor is not None:
        record["audit"] = floor
    return record


# ---------------------------------------------------------------------------
# Cross-check against the canonical implementation
# ---------------------------------------------------------------------------

#: Functions whose source text must match the canonical module exactly.
_FROZEN_CORE = ("min_length_for_drop", "required_length_m",
                "select_binding_zone", "geometric_grade_floor",
                "haulage_floor", "track_bound")

#: Inputs the behavioural check evaluates in both implementations.  Spans
#: shallow/steep grades, zero and large drops, both tier extremes, and the
#: optional total-track terms.
_BATTERY = [
    dict(portal_rl_m=400.0, zones=[dict(label="A", top_rl_m=400.0)],
         max_grade=0.20, unit_cost_per_m=1111.9, excavation_rate_per_m=1000.0),
    dict(portal_rl_m=400.0, zones=[dict(label="A", top_rl_m=399.0)],
         max_grade=0.08, unit_cost_per_m=7585.4, excavation_rate_per_m=1000.0),
    dict(portal_rl_m=402.5, zones=[dict(label="A", top_rl_m=145.0)],
         max_grade=0.20, unit_cost_per_m=1806.9, excavation_rate_per_m=1000.0),
    dict(portal_rl_m=402.5, zones=[dict(label="A", top_rl_m=145.0, plan_gap_m=2000.0)],
         max_grade=0.20, unit_cost_per_m=1806.9, excavation_rate_per_m=1000.0),
    dict(portal_rl_m=1468.4, zones=[dict(label="A", top_rl_m=1237.4)],
         max_grade=0.35, unit_cost_per_m=1111.9, excavation_rate_per_m=1000.0),
    dict(portal_rl_m=400.0,
         zones=[dict(label="A", top_rl_m=325.0), dict(label="B", top_rl_m=145.0),
                dict(label="C", top_rl_m=205.0, plan_gap_m=900.0)],
         max_grade=0.20, unit_cost_per_m=3059.0, excavation_rate_per_m=1000.0),
    dict(portal_rl_m=400.0, zones=[dict(label="A", top_rl_m=145.0)],
         max_grade=0.20, unit_cost_per_m=1806.9, excavation_rate_per_m=1000.0,
         n_portals=3, portal_establishment_cost=1_500_000.0, opex_floor=6_540_000.0),
]

_HAUL_BATTERY = [
    ([dict(label="A", top_rl_m=325.0, centre_rl_m=320.0, tonnes=1_400_000.0),
      dict(label="B", top_rl_m=145.0, centre_rl_m=140.0, tonnes=1_250_000.0,
           plan_gap_m=110.5)],
     dict(portal_rl_m=392.5, max_grade=0.20,
          opex_model=dict(haul_cost_per_t_km=0.35, haul_vert_cost_per_t_m=0.012,
                          vent_cost_per_m_yr=12.0, mine_life_years=10.0))),
    ([dict(label="Z", top_rl_m=1237.4, centre_rl_m=1230.0, tonnes=900_000.0)],
     dict(portal_rl_m=1468.4, max_grade=0.20,
          opex_model=dict(haul_cost_per_t_km=1.0, haul_vert_cost_per_t_m=0.03,
                          vent_cost_per_m_yr=15.0, mine_life_years=10.0))),
]


def assert_matches_mineoptimizer(module=None) -> None:
    """
    Cross-check this frozen copy against the live MineOptimizer implementation.

    Behavioural, not constant-equality: this module is a set of functions, not a
    table, so equality of a few constants would prove nothing.  Every function
    in ``_FROZEN_CORE`` is compared by source text, and both implementations are
    evaluated over ``_BATTERY`` and ``_HAUL_BATTERY`` with every returned field
    required to match exactly.

    Raises AssertionError on any divergence.  Intended for CI and for the
    verification step in the benchmark build, not for normal use -- the
    benchmark never imports MineOptimizer at load time.

    Example
    -------
    >>> from wp3_steiner import geometric_bound as up      # doctest: +SKIP
    >>> from steinerbench.geometric_bound import assert_matches_mineoptimizer
    >>> assert_matches_mineoptimizer(up)                   # doctest: +SKIP
    """
    import inspect
    import sys as _sys

    if module is None:
        from wp3_steiner import geometric_bound as module  # type: ignore

    this = _sys.modules[__name__]

    for name in _FROZEN_CORE:
        mine = inspect.getsource(getattr(this, name))
        theirs = inspect.getsource(getattr(module, name))
        assert mine == theirs, (
            f"geometric_bound.{name} diverged from MineOptimizer.\n"
            f"--- frozen ---\n{mine}\n--- upstream ---\n{theirs}")

    for kw in _BATTERY:
        a = geometric_grade_floor(**kw)
        b = module.geometric_grade_floor(**kw)
        assert a == b, f"geometric_grade_floor diverged on {kw}:\n{a}\n{b}"

    for zones, kw in _HAUL_BATTERY:
        a = haulage_floor(zones, **kw)
        b = module.haulage_floor(zones, **kw)
        assert a == b, f"haulage_floor diverged on {kw}:\n{a}\n{b}"

    for track in ("raw", "constrained", "total"):
        pw = {"value": 2_306_029.0}
        fl = None if track == "raw" else geometric_grade_floor(**_BATTERY[2])
        a = track_bound(track, pairwise=pw, floor=fl)
        b = module.track_bound(track, pairwise=pw, floor=fl)
        assert a == b, f"track_bound({track}) diverged:\n{a}\n{b}"

    # The guard itself must be present in both.
    for mod in (this, module):
        try:
            mod.track_bound("raw", pairwise={"value": 1.0},
                            floor={"value": 2.0, "method": METHOD_NAME})
        except ValueError:
            pass
        else:                                    # pragma: no cover
            raise AssertionError(
                f"{mod.__name__}.track_bound accepted a floor on the raw track")
