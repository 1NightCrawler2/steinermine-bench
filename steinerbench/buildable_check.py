"""
Buildability verifier for the SteinerMineBench `constrained` track.
==================================================================

SPDX-License-Identifier: MIT

Self-contained: numpy only, no scipy, no solver.  That is the point.  A
"constrained" track whose feasibility can only be judged by the reference
solver is not a benchmark, it is an assertion -- anyone must be able to take a
submitted centreline and decide, from the shipped bundle alone, whether it
could be excavated.

Why the track exists
--------------------
The v1 objective minimises support + excavation over a 26-connected descent
graph with no geometric constraint.  Measured on v1's own reference solutions,
100 % of descending steps exceed a 20 % grade limit and the median step grade
is 100 % (45 degrees).  The v1 optimum is therefore the optimum of a
RELAXATION.  Its cost is a real lower bound and a useful diagnostic, but it is
not the cost of anything anyone can build, and ranking topologies on it flips
the winner on 15 of the 23 solved instances relative to ranking on buildable
cost.

The five checks
---------------
1. ``grade``        every step at or below ``max_grade``
2. ``turn_radius``  plan-view curvature at or above ``min_turn_radius_m``
3. ``swept``        the full tunnel cross-section stays in excavatable ground,
                    not merely the centreline
4. ``continuity``   segment ends meet and the network is connected
5. ``separation``   distinct openings keep a rock pillar between them, except
                    where they legitimately share a junction

Two of these carry documented judgement calls; both are stated rather than
buried, because they change verdicts:

* ``min_pillar_m`` is a PLACEHOLDER (5 m).  A site sets it from ground
  conditions and opening span.
* the junction exemption.  Two drives leaving one junction at R_min need about
  ``sqrt(R * (pillar + span))`` metres -- roughly 16 m at the shipped standard
  -- before a full pillar between them is geometrically possible at all.
  Inside that distance they are one flared excavation, not two openings sharing
  rock.  Breaches are therefore reported WITH their distance from the shared
  junction, so a reader can separate a junction-flare finding (a design detail
  this model does not carry) from two independent drives running side by side.

Usage
-----
    from steinerbench.buildable_check import verify_network
    from loader import load_instance

    inst = load_instance("haul-t20")
    rep = verify_network(segments, inst)      # segments: [(label, (M,3) world m)]
    print(rep.summary())
    rep.ok        # -> bool
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import spec

_SENT = 1e9 * 0.9

#: Angular samples around the tunnel cross-section for the swept check.
_N_RING = 8

#: Multiplier on max(tolerance, pillar) giving the junction flare radius.
_JUNCTION_RADIUS_FACTOR = 3.0


# ---------------------------------------------------------------------------
# Geometry helpers (inlined so the module has no internal dependencies)
# ---------------------------------------------------------------------------
def _dedupe(P, tol=1e-6):
    if len(P) <= 1:
        return P
    keep = [0]
    for i in range(1, len(P)):
        if np.linalg.norm(P[i] - P[keep[-1]]) > tol:
            keep.append(i)
    return P[keep]


def _cum_arclen(P):
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def _resample_uniform(P, step):
    d = _cum_arclen(P)
    total = d[-1]
    if total < 1e-9:
        return P
    n = max(2, int(round(total / max(step, 1e-6))) + 1)
    s = np.linspace(0.0, total, n)
    out = np.empty((n, P.shape[1]))
    for j in range(P.shape[1]):
        out[:, j] = np.interp(s, d, P[:, j])
    return out


def _circumradius_min(P2):
    """Minimum circumradius over consecutive triples of a plan-view curve.

    Circumradius, not ``L / dtheta``.  The latter under-reads by
    ``sinc(dtheta/2)``: on a known 25 m circle it gives 24.9844 at four samples
    and 24.9983 at twelve, while the circumradius gives 25.0000 at every
    sampling density.  A verifier that under-reads condemns correct geometry by
    millimetres, and -- worse -- a planner and a checker using different
    estimators cannot be reconciled by tuning either one.
    """
    if len(P2) < 3:
        return float("inf")
    a, b, c = P2[:-2], P2[1:-1], P2[2:]
    ab = np.linalg.norm(b - a, axis=1)
    bc = np.linalg.norm(c - b, axis=1)
    ca = np.linalg.norm(a - c, axis=1)
    area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                        - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    valid = (area > 1e-9) & (ab > 1e-9) & (bc > 1e-9)
    if not valid.any():
        return float("inf")
    R = (ab[valid] * bc[valid] * ca[valid]) / (4.0 * area[valid])
    return float(R.min())


# ---------------------------------------------------------------------------
@dataclass
class Violation:
    check: str
    segment: str
    detail: str
    value: float = 0.0
    limit: float = 0.0
    where: tuple | None = None

    def __str__(self) -> str:
        loc = ("" if self.where is None else
               f"  @ E{self.where[0]:.0f} N{self.where[1]:.0f} "
               f"RL{self.where[2]:.0f}")
        return f"[{self.check}] {self.segment}: {self.detail}{loc}"


@dataclass
class VerifyReport:
    ok: bool = True
    violations: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def add(self, v: Violation) -> None:
        self.violations.append(v)
        self.ok = False

    def by_check(self) -> dict:
        out: dict = {}
        for v in self.violations:
            out[v.check] = out.get(v.check, 0) + 1
        return out

    def summary(self, max_examples: int = 5) -> str:
        lines = [("BUILDABLE" if self.ok else "NOT BUILDABLE")
                 + f"  ({len(self.violations)} violation(s))"]
        for k, v in sorted(self.stats.items()):
            lines.append(f"    {k:<24} {v}")
        if self.violations:
            lines.append("  by check:")
            for k, n in sorted(self.by_check().items(), key=lambda kv: -kv[1]):
                lines.append(f"    {k:<24} {n}")
            for v in self.violations[:max_examples]:
                lines.append(f"    {v}")
            if len(self.violations) > max_examples:
                lines.append(f"    ... and "
                             f"{len(self.violations) - max_examples} more")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def check_grade(name, C, max_grade, rep, tol=1e-6):
    """A step with vertical extent and no horizontal run is a shaft, not a ramp."""
    if len(C) < 2:
        return 0.0
    d = np.diff(C, axis=0)
    horiz = np.hypot(d[:, 0], d[:, 1])
    vert = np.abs(d[:, 2])
    moving = vert > 1e-9
    if not moving.any():
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        grade = np.where(horiz > 1e-9, vert / np.maximum(horiz, 1e-12), np.inf)
    grade = np.where(moving, grade, 0.0)
    for i in np.flatnonzero(grade > max_grade + tol):
        g = grade[i]
        rep.add(Violation(
            "grade", name,
            ("vertical step (infinite grade)" if not np.isfinite(g)
             else f"grade {g * 100:.1f}% exceeds {max_grade * 100:.0f}%"),
            float(g if np.isfinite(g) else 1e9), max_grade,
            tuple(float(x) for x in C[i])))
    finite = grade[np.isfinite(grade)]
    return float(finite.max()) if finite.size else float("inf")


def check_turn_radius(name, C, min_r, rep, turnout_m=0.0):
    """Plan curvature, with a bounded exemption at each end.

    A segment endpoint is a junction, and a junction is a locally widened
    turnout where a vehicle manoeuvres at low speed; the running-ramp turning
    circle is not the governing geometry there.
    """
    P = np.asarray(C, dtype=float)
    plan = _dedupe(P[:, :2])
    if len(plan) < 3:
        return float("inf")
    seg = np.linalg.norm(np.diff(plan, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    reach = min(turnout_m, 0.25 * s[-1]) if s[-1] > 0 else 0.0
    inner = (s >= reach) & (s <= s[-1] - reach)
    idx = np.flatnonzero(inner)
    if len(idx) < 3:
        return float("inf")
    R = _circumradius_min(plan[idx])
    if R < min_r - 1e-6:
        rep.add(Violation(
            "turn_radius", name,
            f"plan radius {R:.1f} m below {min_r:.0f} m "
            f"(short by {min_r - R:.1f} m)", R, min_r,
            tuple(float(x) for x in P[len(P) // 2])))
    return R


def check_swept(name, C, cost_grid, meta, span_m, rep, step_m=2.0,
                portals=(), portal_reach_m=0.0, max_report=6):
    """Sweep the tunnel cross-section, not the centreline.

    A centreline sampled one voxel per step can thread a 5 m bore through a
    fault block or through air between samples.  Two failure modes are
    distinguished because they have different fixes: breaking out above the
    topmost excavatable voxel is insufficient CROWN COVER (raise the cover),
    while entering non-excavatable ground is an OBSTRUCTION (route around).
    """
    P = np.asarray(C, dtype=float)
    if len(P) < 2:
        return 0, 0
    cs = float(meta["cell_size_m"])
    mn = np.asarray(meta["min_coords_m"], dtype=float)
    dims = np.asarray(cost_grid.shape)
    R = span_m / 2.0
    S = _resample_uniform(P, max(step_m, 1e-6))
    tang = np.gradient(S, axis=0)
    n_bad = n_daylight = 0
    PC = np.asarray(portals, dtype=float) if len(portals) else None

    for i in range(len(S)):
        t = tang[i]
        nt = np.linalg.norm(t)
        if nt < 1e-9:
            continue
        t = t / nt
        # Any two vectors orthogonal to the tangent span the bore's section.
        ref = np.array([0.0, 0.0, 1.0]) if abs(t[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = np.cross(t, ref)
        u /= max(np.linalg.norm(u), 1e-12)
        v = np.cross(t, u)
        for a in np.linspace(0.0, 2.0 * math.pi, _N_RING, endpoint=False):
            p = S[i] + R * (math.cos(a) * u + math.sin(a) * v)
            ijk = np.floor((p - mn) / cs).astype(int)
            outside = bool(np.any(ijk < 0) or np.any(ijk >= dims))
            blocked = outside or not (cost_grid[tuple(ijk)] < _SENT)
            if not blocked:
                continue
            # Above the top excavatable voxel in this column = crown break-out.
            crown = False
            if not outside:
                col = cost_grid[ijk[0], ijk[1], :]
                ok = np.flatnonzero(col < _SENT)
                crown = bool(ok.size and ijk[2] > ok.max())
            if crown and PC is not None and portal_reach_m > 0.0:
                if np.linalg.norm(PC - S[i], axis=1).min() <= portal_reach_m:
                    n_daylight += 1          # a portal IS a break-out to surface
                    continue
            n_bad += 1
            if n_bad <= max_report:
                why = ("bore leaves the model" if outside else
                       "bore breaks out above the topmost excavatable voxel "
                       "-- insufficient crown cover" if crown else
                       "bore enters non-excavatable ground")
                rep.add(Violation("swept", name, why, 0.0, 0.0,
                                  tuple(float(x) for x in S[i])))
    return n_bad, n_daylight


def check_continuity(segments, rep, tol_m, terminal_groups=None):
    """Ends must meet, and the network must hang together.

    ``terminal_groups`` is ``[(label, points)]`` -- the voxel body of one portal
    or one zone.  A terminal is an EXCAVATION, not a point: a portal collar is
    a sphere metres across and a stope is a block, so two openings that both
    reach the same body are joined through it.  Clustering on proximity alone
    misses that, and reports two declines leaving one portal from voxels 16 m
    apart as two disconnected networks -- a statement about the terminal's
    diameter rather than about the design.
    """
    ends, n_zero = [], 0
    for name, C in segments:
        C = np.asarray(C, dtype=float)
        if len(C) < 2:
            n_zero += 1
            continue
        ends.append((name, C[0], C[-1]))
    if not ends:
        return 0, n_zero

    pts, owner = [], []
    for name, a, b in ends:
        pts.extend([a, b])
        owner.extend([name, name])
    pts = np.asarray(pts)
    cluster = -np.ones(len(pts), dtype=int)
    nc = 0
    for i in range(len(pts)):
        if cluster[i] >= 0:
            continue
        cluster[np.linalg.norm(pts - pts[i], axis=1) <= tol_m] = nc
        nc += 1

    for _lbl, gpts in (terminal_groups or []):
        G = np.asarray(gpts, dtype=float)
        if not len(G):
            continue
        inside = np.flatnonzero(np.array(
            [np.linalg.norm(G - p, axis=1).min() <= tol_m for p in pts]))
        if len(inside) < 2:
            continue
        keep = cluster[inside].min()
        cluster[np.isin(cluster, np.unique(cluster[inside]))] = keep
    uniq = {c: i for i, c in enumerate(sorted(set(cluster.tolist())))}
    cluster = np.array([uniq[c] for c in cluster], dtype=int)
    nc = len(uniq)

    seg_of: dict = {}
    for i, name in enumerate(owner):
        seg_of.setdefault(name, []).append(cluster[i])
    adj: dict = {c: {c} for c in range(nc)}
    for _name, cs_ in seg_of.items():
        for a in cs_:
            for b in cs_:
                adj[a].add(b)
                adj[b].add(a)
    seen, stack = set(), [0] if nc else []
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        stack.extend(adj.get(x, ()))
    if len(seen) < nc:
        rep.add(Violation("continuity", "<network>",
                          f"network is not connected: {nc - len(seen)} of {nc} "
                          f"endpoint clusters unreachable from the rest"))
    loose = sum(1 for c in range(nc)
                if int(np.count_nonzero(cluster == c)) == 1)
    return loose, n_zero


def check_separation(segments, rep, pillar_m, tol_m, step_m=4.0, max_report=10):
    """Distinct openings must leave rock between them."""
    if pillar_m <= 0:
        return float("inf")
    prepped = []
    for name, C in segments:
        C = np.asarray(C, dtype=float)
        if len(C) >= 2:
            prepped.append((name, _resample_uniform(C, step_m), C[0], C[-1]))
    worst, n_bad = float("inf"), 0
    jr = _JUNCTION_RADIUS_FACTOR * max(tol_m, pillar_m)
    # A voxel is a cube: a ramp may end anywhere inside its goal cube while a
    # crosscut starts at the centre, so a genuinely shared junction can show
    # endpoints up to a cube diagonal apart.
    share_tol = tol_m * math.sqrt(3.0)

    for i in range(len(prepped)):
        ni, Pi, ai, bi = prepped[i]
        for j in range(i + 1, len(prepped)):
            nj, Pj, aj, bj = prepped[j]
            shared = [p for p in (ai, bi) for q in (aj, bj)
                      if np.linalg.norm(p - q) <= share_tol]
            D = np.linalg.norm(Pi[:, None, :] - Pj[None, :, :], axis=2)
            if shared:
                mi = np.zeros(len(Pi), dtype=bool)
                mj = np.zeros(len(Pj), dtype=bool)
                for s in shared:
                    mi |= np.linalg.norm(Pi - s, axis=1) <= jr
                    mj |= np.linalg.norm(Pj - s, axis=1) <= jr
                D[mi, :] = np.inf
                D[:, mj] = np.inf
            if not np.isfinite(D).any():
                continue
            m = float(D.min())
            worst = min(worst, m)
            if m < pillar_m:
                n_bad += 1
                if n_bad <= max_report:
                    a, _b = np.unravel_index(int(np.argmin(D)), D.shape)
                    near_j = (min(float(np.linalg.norm(Pi[a] - s))
                                  for s in shared) if shared else None)
                    ctx = ("" if near_j is None else
                           f"; {near_j:.0f} m from the junction they share")
                    rep.add(Violation(
                        "separation", f"{ni} vs {nj}",
                        f"openings {m:.1f} m apart, below the "
                        f"{pillar_m:.0f} m pillar{ctx}", m, float(pillar_m),
                        tuple(float(x) for x in Pi[a])))
    if n_bad > max_report:
        rep.add(Violation("separation", "<network>",
                          f"...and {n_bad - max_report} further pairs too close",
                          float(n_bad), 0.0, None))
    return worst


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _level_drive(name):
    n = str(name).strip().lower()
    return n.startswith("crosscut") or n.startswith("drive")


def verify_network(segments, instance, standard=None, resample_m=2.0):
    """Verify a network against an instance bundle.

    ``segments``  ``[(label, (M,3) array of world metres)]``
    ``instance``  the dict from ``loader.load_instance``
    ``standard``  overrides for ``spec.GEOMETRIC_STANDARD`` (optional)

    Level drives are exempt from the RAMP turning circle: they are unsmoothed
    voxel staircases and a crosscut design step does not exist in this model.
    They are still checked for grade, swept passability, continuity and
    separation -- the checks that do apply.
    """
    std = dict(spec.GEOMETRIC_STANDARD)
    std.update(standard or {})
    md = instance["metadata"]
    grid = instance["cost_grid"]
    meta = {"cell_size_m": float(md["grid"]["cell_size_m"]),
            "min_coords_m": np.asarray(md["grid"]["min_coords_m"], dtype=float)}
    cs = meta["cell_size_m"]

    rep = VerifyReport()
    worst_grade, worst_R = 0.0, float("inf")
    swept_bad = daylight = 0
    total_len = 0.0

    portals = [np.asarray(v, dtype=float) * cs + meta["min_coords_m"] + cs * 0.5
               for v in instance["portal_voxels"]]
    groups = [("portal", np.asarray(portals))] if portals else []
    for n, zv in enumerate(instance["zone_voxels"], start=1):
        groups.append((f"Z{n}", np.asarray(zv, dtype=float) * cs
                       + meta["min_coords_m"] + cs * 0.5))

    for name, C in segments:
        C = np.asarray(C, dtype=float)
        if len(C) < 2:
            continue
        total_len += float(_cum_arclen(C)[-1])
        worst_grade = max(worst_grade, check_grade(name, C, std["max_grade"], rep))
        if not _level_drive(name):
            worst_R = min(worst_R, check_turn_radius(
                name, C, std["min_turn_radius_m"], rep,
                turnout_m=2.0 * std["tunnel_span_m"]))
        nb, nd = check_swept(name, C, grid, meta, std["tunnel_span_m"], rep,
                             step_m=resample_m, portals=portals,
                             portal_reach_m=std["portal_reach_m"])
        swept_bad += nb
        daylight += nd

    loose, n_zero = check_continuity(segments, rep, cs * math.sqrt(3.0), groups)
    worst_sep = check_separation(segments, rep, std["min_pillar_m"], cs)

    rep.stats = {
        "segments": len(segments),
        "total_length_m": round(total_len, 1),
        "worst_grade": round(worst_grade, 4),
        "worst_turn_radius_m": (None if not np.isfinite(worst_R)
                                else round(worst_R, 1)),
        "bore_samples_out_of_ground": swept_bad,
        "portal_daylight_samples": daylight,
        "terminal_endpoints": loose,
        "zero_length_segments": n_zero,
        "worst_separation_m": (None if not np.isfinite(worst_sep)
                               else round(worst_sep, 1)),
        "standard": std,
    }
    return rep
