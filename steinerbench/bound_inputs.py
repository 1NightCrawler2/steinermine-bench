"""
Extract geometric-bound inputs from a SteinerMineBench instance.
================================================================

SPDX-License-Identifier: MIT

``geometric_bound.py`` is deliberately ignorant of where its numbers come from —
it takes resolved RLs and rates and does trigonometry.  This module is the
benchmark's adapter into it, and it reads **metadata.json alone**.

That is a design constraint, not an accident.  Because no grid is needed, every
bound in the suite can be recomputed and audited in seconds
(``solve_reference.py --recompute-bounds-only``), and a reader can re-derive a
published bound from the bundle's smallest file without a solver, without numpy
and without trusting us.

Conventions, and why each is the conservative choice
----------------------------------------------------
``portal_rl_m``   ``min(collar_rl - radius_m)`` over portals.  The portal
                  terminal is a *body*, and a ramp may leave from any of its
                  voxels.  Where the body is clipped by topography the true
                  lowest voxel sits higher than this (on ``portal-north``,
                  397.5 against a formula value of 392.5), so the formula
                  understates the drop and therefore understates the bound.
                  Safe, and it keeps the computation metadata-only.

``top_rl_m``      ``sublevel_rl_m + rl_thickness_m/2`` — the highest RL of the
                  zone slab, i.e. its closest point to the portal, giving the
                  smallest drop.  Voxel *centres* actually top out half a cell
                  below this, so again the formula understates the drop.

                  NOTE the units trap: ``rl_thickness_m`` is a FULL thickness
                  here (the slab spans ``rl ± t/2``), while the case study's
                  ``terminals_poly.json`` uses ``rl_thickness`` as a HALF
                  thickness (``wp2_pathfinder/select_polygons.py``: "voxels
                  within ± this value").  This is exactly why
                  ``geometric_bound`` takes resolved RLs and never a thickness.

``centre_rl_m``   ``sublevel_rl_m`` — the slab centre, which is the median zone
                  voxel RL the haulage model charges lift against.

``plan_gap_m``    Horizontal distance from the portal body to the nearest point
                  of the zone polygon.  Nearest, not centroid: the haulage model
                  takes the *shortest* route to any voxel of the zone.

``unit_cost_per_m``
                  The cheapest tier that actually OCCURS in this instance, read
                  from ``cost_model.tier_histogram_passable``.  This equals
                  ``min(cost_grid[passable])`` — ``validate.py`` already asserts
                  the histogram matches the array — and is strictly tighter than
                  the global cheapest tier (1806.9 against 1111.9 on the twelve
                  instances with no tier-5 rock).
"""

from __future__ import annotations

import math

__all__ = ["floor_inputs_from_metadata", "plan_gap_to_polygon"]


def _point_segment_distance(px, py, ax, ay, bx, by) -> float:
    """Distance from (px,py) to the segment (ax,ay)-(bx,by)."""
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    vv = vx * vx + vy * vy
    if vv <= 0.0:
        return math.hypot(wx, wy)
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / vv))
    return math.hypot(px - (ax + t * vx), py - (ay + t * vy))


def plan_gap_to_polygon(point_xy, polygon, inset_m: float = 0.0) -> float:
    """
    Horizontal distance from ``point_xy`` to the nearest point of ``polygon``
    (a list of [x, y] vertices), less ``inset_m``, clamped at zero.

    Returns 0.0 when the point lies inside the polygon — a portal directly over
    a zone forces no horizontal run at all, and the grade term carries the
    bound by itself.
    """
    px, py = float(point_xy[0]), float(point_xy[1])
    n = len(polygon)
    if n == 0:
        return 0.0

    inside = False
    best = math.inf
    for i in range(n):
        ax, ay = float(polygon[i][0]), float(polygon[i][1])
        bx, by = float(polygon[(i + 1) % n][0]), float(polygon[(i + 1) % n][1])
        best = min(best, _point_segment_distance(px, py, ax, ay, bx, by))
        if (ay > py) != (by > py):
            xx = ax + (py - ay) / (by - ay) * (bx - ax)
            if px < xx:
                inside = not inside
    if inside:
        return 0.0
    return max(best - max(inset_m, 0.0), 0.0)


def floor_inputs_from_metadata(metadata: dict) -> dict:
    """
    Every input ``geometric_bound.geometric_grade_floor`` and ``haulage_floor``
    need for one instance, from ``metadata.json`` alone.

    Returns a dict with ``portal_rl_m``, ``zones`` (ready to pass straight
    through), ``max_grade``, ``unit_cost_per_m``, ``excavation_rate_per_m``,
    ``opex_model``, ``n_portals`` and ``portal_establishment_cost``.
    """
    portals = metadata["portals"]
    if not portals:
        raise ValueError("instance metadata declares no portal")

    # Lowest reachable portal RL, and the plan position of the portal that
    # attains it (the one a ramp would leave from).
    lowest = min(portals, key=lambda p: p["world_m"][2] - p.get("radius_m", 0.0))
    portal_rl_m = float(lowest["world_m"][2]) - float(lowest.get("radius_m", 0.0))
    portal_xy = (float(lowest["world_m"][0]), float(lowest["world_m"][1]))
    portal_radius = float(lowest.get("radius_m", 0.0))

    zones = []
    for z in metadata["zones"]:
        rl = float(z["sublevel_rl_m"])
        half = float(z.get("rl_thickness_m", 0.0)) / 2.0
        poly = z.get("polygon_plan_m") or []
        zones.append({
            "label": z.get("label"),
            "top_rl_m": rl + half,
            "centre_rl_m": rl,
            "plan_gap_m": plan_gap_to_polygon(portal_xy, poly, portal_radius),
            "tonnes": float(z.get("tonnage_mt", 0.0)) * 1e6,
        })

    cm = metadata["cost_model"]
    hist = cm.get("tier_histogram_passable") or []
    sched = cm["tier_schedule"]
    present = [sched[i]["cost_per_m"] for i, n in enumerate(hist) if n > 0]
    if not present:                       # no histogram shipped: fall back wide
        present = [t["cost_per_m"] for t in sched]

    om = dict(metadata["tracks"]["opex_model"])
    return {
        "portal_rl_m": portal_rl_m,
        "zones": zones,
        "max_grade": float(metadata["tracks"]["geometric_standard"]["max_grade"]),
        "unit_cost_per_m": float(min(present)),
        "excavation_rate_per_m": float(cm["excavation_rate_per_m"]),
        "opex_model": om,
        # Every built network surfaces somewhere, so one establishment charge
        # is forced.  More would be a claim about the solution, not a bound.
        "n_portals": 1,
        "portal_establishment_cost": float(om.get("portal_establishment_cost", 0.0)),
    }
