"""
Resolution-independent synthetic geology for SteinerMineBench.
==============================================================

SPDX-License-Identifier: MIT

SYNTHETIC DATA.  Everything in this module is procedurally generated.  Nothing
here is derived from, measured at, or descriptive of any real or operating mine.

Design principle: resolution independence
-----------------------------------------
The Barton Q field is drawn ONCE on a fixed world-space lattice
(``spec.Q_REFERENCE_LATTICE_M`` = 10 m), lightly smoothed there, and then
trilinearly interpolated to voxel centres.  All structural features (barrier
slab, competent corridors, fault damage halos, ground surface) are closed-form
predicates on world coordinates with metre-valued parameters.

Consequently every cell size in the scale ladder samples *the same geology* and
differs only in discretisation.  A naive per-voxel RNG, or a fault halo
specified in voxels rather than metres, would silently change the underlying
problem between rungs and destroy the controlled experiment.

Divergence from the MineOptimizer production pipeline (deliberate)
------------------------------------------------------------------
``wp1_voxel/pipeline.py:1167-1176`` widens the fault cost floor with a 3x3x3
*voxel* dilation, to compensate for sparse rasterisation of its fault block
model.  That halo is 30 m wide at a 10 m cell and 3 m wide at a 1 m cell.  This
module instead specifies the damage halo in metres
(``damage_half_width_m``) and applies no voxel dilation, so the fault footprint
is identical at every resolution.  ``metadata.json`` records this rule verbatim.

Units
-----
All lengths metres (m); Q is the dimensionless Barton rock mass quality index.
Axis order is (EAST, NORTH, RL) with RL positive upward.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from steinerbench import spec

#: Smoothing applied on the reference lattice, in lattice nodes.  At a 10 m
#: lattice this is a ~12 m correlation length, comparable to the kriging range
#: the upstream pipeline fits to real drillhole data.
_LATTICE_SMOOTH_NODES = 1.2

#: Q is clipped to this range before tiering (matches tools/make_synthetic.py).
Q_CLIP = (0.001, 50.0)

#: Voxels are processed in slabs of at most this many cells along EAST, so the
#: 129 M-voxel instance never materialises more than a few hundred MB at once.
_SLAB_TARGET_VOXELS = 4_000_000


# ---------------------------------------------------------------------------
# Ground surface
# ---------------------------------------------------------------------------
def surface_rl(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Analytic ground surface RL (m) at plan position(s) (x, y).

    A broad ENE-trending ridge with a superimposed ripple and a mild westward
    tilt.  Ranges roughly RL 385-490 over the domain, leaving every production
    sublevel (RL 140-320) well below surface.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return (415.0
            + 55.0 * np.exp(-(((y - 230.0) / 130.0) ** 2))
            + 18.0 * np.sin(2.0 * np.pi * x / 420.0)
            - 12.0 * (x / 630.0))


def surface_rl_grid(dims, cell_size_m: float, min_coords) -> np.ndarray:
    """Ground surface RL (m) per plan column, shape (nx, ny), float32."""
    nx, ny, _ = dims
    xs = min_coords[0] + (np.arange(nx) + 0.5) * cell_size_m
    ys = min_coords[1] + (np.arange(ny) + 0.5) * cell_size_m
    return surface_rl(xs[:, None], ys[None, :]).astype(np.float32)


# ---------------------------------------------------------------------------
# Random background Q field on the fixed reference lattice
# ---------------------------------------------------------------------------
def _reference_lattice(min_coords):
    """Node coordinates of the fixed reference lattice (one node of margin)."""
    step = spec.Q_REFERENCE_LATTICE_M
    axes = []
    for ax in range(3):
        n = int(np.ceil(spec.DOMAIN_SIZE_M[ax] / step)) + 3
        axes.append(min_coords[ax] - step + np.arange(n) * step)
    return axes, step


def _correlated_normal(rng, shape, sigma_nodes: float) -> np.ndarray:
    """
    A spatially correlated standard-normal field on the lattice.

    White noise is smoothed to impose a correlation length, then rescaled back
    to unit variance.  The rescale is essential: Gaussian smoothing of white
    noise in 3-D suppresses the standard deviation by roughly an order of
    magnitude at sigma = 1.2 nodes, which would collapse the Q field onto a
    near-constant value and leave most cost tiers unpopulated.
    """
    field = gaussian_filter(rng.standard_normal(shape), sigma=sigma_nodes,
                            mode="nearest")
    std = field.std()
    return field / std if std > 0 else field


def _lattice_log_q(q_regime: str, seed: int, min_coords) -> tuple[np.ndarray, list, float]:
    """
    Draw log10(Q) on the reference lattice.

    Two independent correlated fields are used: one selects the lognormal lobe
    (so a mixed regime gets spatially coherent competent and poor domains rather
    than salt-and-pepper noise), the other supplies the within-lobe deviate.
    Both carry the same correlation length, so the marginal Q distribution
    matches the regime specification while the field stays geologically smooth.

    Returns (log10_q_nodes, axes, step).
    """
    axes, step = _reference_lattice(min_coords)
    shape = tuple(len(a) for a in axes)

    rng = np.random.default_rng(seed)
    lobes = spec.Q_REGIMES[q_regime]["lobes"]
    weights = np.array([w for w, _, _ in lobes], dtype=np.float64)
    weights = weights / weights.sum()

    if len(lobes) == 1:
        pick = np.zeros(shape, dtype=np.int64)
    else:
        # Map a correlated normal through its own CDF to a uniform, then bucket
        # it by the cumulative lobe weights.  Domains come out contiguous.
        from scipy.special import ndtr
        u = ndtr(_correlated_normal(rng, shape, _LATTICE_SMOOTH_NODES))
        pick = np.searchsorted(np.cumsum(weights)[:-1], u, side="right")

    deviate = _correlated_normal(rng, shape, _LATTICE_SMOOTH_NODES)

    medians = np.array([m for _, m, _ in lobes], dtype=np.float64)
    sigmas = np.array([s for _, _, s in lobes], dtype=np.float64)

    # Lognormal in natural log, converted to log10 for interpolation.
    ln_q = np.log(medians[pick]) + sigmas[pick] * deviate
    return ln_q / np.log(10.0), axes, step


def _interp_lattice(log10_q_nodes, axes, step, xs, ys, zs) -> np.ndarray:
    """
    Trilinear interpolation of the lattice field onto a voxel-centre meshgrid.

    ``xs``/``ys``/``zs`` are 1-D coordinate vectors; the result has shape
    (len(xs), len(ys), len(zs)).  Implemented directly rather than via
    ``map_coordinates`` so the three separable weight vectors are formed once
    and broadcast, which is both faster and far lighter on memory.
    """
    def _weights(coords, origin, n_nodes):
        t = (np.asarray(coords, dtype=np.float64) - origin) / step
        i0 = np.floor(t).astype(np.int64)
        frac = t - i0
        i0 = np.clip(i0, 0, n_nodes - 2)
        frac = np.clip(t - i0, 0.0, 1.0)
        return i0, frac

    ix, fx = _weights(xs, axes[0][0], len(axes[0]))
    iy, fy = _weights(ys, axes[1][0], len(axes[1]))
    iz, fz = _weights(zs, axes[2][0], len(axes[2]))

    out = np.zeros((len(xs), len(ys), len(zs)), dtype=np.float64)
    for dx in (0, 1):
        wx = (fx if dx else 1.0 - fx)[:, None, None]
        for dy in (0, 1):
            wy = (fy if dy else 1.0 - fy)[None, :, None]
            for dz in (0, 1):
                wz = (fz if dz else 1.0 - fz)[None, None, :]
                out += wx * wy * wz * log10_q_nodes[
                    np.ix_(ix + dx, iy + dy, iz + dz)]
    return out


# ---------------------------------------------------------------------------
# Fault geometry
# ---------------------------------------------------------------------------
def _dip_unit_vector(dip_direction_deg: float) -> tuple[float, float]:
    """Horizontal unit vector (east, north) pointing down-dip."""
    az = np.deg2rad(dip_direction_deg)
    return float(np.sin(az)), float(np.cos(az))


def _dist_to_polyline(px: np.ndarray, py: np.ndarray,
                      trace: np.ndarray) -> np.ndarray:
    """
    Minimum 2-D distance from each point to a polyline, vectorised over points.

    ``trace`` is (S+1, 2); the result has the shape of ``px``.
    """
    best = np.full(px.shape, np.inf)
    for s in range(len(trace) - 1):
        ax, ay = trace[s]
        bx, by = trace[s + 1]
        vx, vy = bx - ax, by - ay
        seg_len_sq = vx * vx + vy * vy
        if seg_len_sq == 0.0:
            d = np.hypot(px - ax, py - ay)
        else:
            t = np.clip(((px - ax) * vx + (py - ay) * vy) / seg_len_sq, 0.0, 1.0)
            d = np.hypot(px - (ax + t * vx), py - (ay + t * vy))
        np.minimum(best, d, out=best)
    return best


def fault_perpendicular_distance(fault: dict, xs, ys, zs) -> np.ndarray:
    """
    Perpendicular distance (m) from every voxel centre to a dipping fault plane.

    The plan-view trace is taken as the plane's intersection with the horizontal
    plane at the TOP of the fault's RL range.  At depth the plane migrates
    down-dip by ``(z_ref - z) / tan(dip)``; back-projecting each voxel by that
    offset reduces the problem to a 2-D point-to-polyline distance, which is
    then converted from plan distance to true perpendicular distance by a factor
    of ``sin(dip)``.

    Result shape is (len(xs), len(ys), len(zs)).
    """
    trace = np.asarray(fault["trace_plan_m"], dtype=np.float64)
    dip = np.deg2rad(fault["dip_deg"])
    ux, uy = _dip_unit_vector(fault["dip_direction_deg"])
    z_ref = float(fault["rl_range_m"][1])

    # Down-dip lateral migration per voxel, one value per RL layer.
    shift = (z_ref - np.asarray(zs, dtype=np.float64)) / np.tan(dip)  # (nz,)

    px = xs[:, None, None] - ux * shift[None, None, :]
    py = ys[None, :, None] - uy * shift[None, None, :]
    px, py = np.broadcast_arrays(px, py)

    d_plan = _dist_to_polyline(px, py, trace)
    return d_plan * np.sin(dip)


# ---------------------------------------------------------------------------
# Slab-wise assembly
# ---------------------------------------------------------------------------
def _slabs(nx: int, ny: int, nz: int):
    """Yield (i0, i1) EAST-index ranges sized to bound peak memory."""
    per_column = max(1, ny * nz)
    width = max(1, min(nx, _SLAB_TARGET_VOXELS // per_column))
    for i0 in range(0, nx, width):
        yield i0, min(nx, i0 + width)


def build_fields(instance_spec: dict, dims, cell_size_m: float, min_coords,
                 want_q: bool = False):
    """
    Build the Q field, fault count and above-surface mask for one instance.

    Parameters
    ----------
    instance_spec : dict from ``spec.INSTANCES``
    dims          : (nx, ny, nz)
    cell_size_m   : voxel edge length (m)
    min_coords    : (east, north, rl) of the grid origin (m)
    want_q        : also return the float32 Q field (large; off by default)

    Returns
    -------
    dict with keys ``fault_count`` (uint8), ``above_surface`` (bool),
    ``tier_index`` (uint8, pre-fault-floor), ``surface_rl`` (float32 (nx, ny)),
    and optionally ``q`` (float32).
    """
    from steinerbench.tiers import q_to_tier_index, SENTINEL_TIER

    nx, ny, nz = dims
    faults = spec.FAULT_SYSTEMS[instance_spec["fault_system"]]

    log10_q_nodes, axes, step = _lattice_log_q(
        instance_spec["q_regime"], instance_spec["q_seed"], min_coords)

    ys = min_coords[1] + (np.arange(ny) + 0.5) * cell_size_m
    zs = min_coords[2] + (np.arange(nz) + 0.5) * cell_size_m

    tier_index = np.empty((nx, ny, nz), dtype=np.uint8)
    fault_count = np.zeros((nx, ny, nz), dtype=np.uint8)
    above_surface = np.empty((nx, ny, nz), dtype=bool)
    q_out = np.empty((nx, ny, nz), dtype=np.float32) if want_q else None

    log10 = np.log10

    for i0, i1 in _slabs(nx, ny, nz):
        xs = min_coords[0] + (np.arange(i0, i1) + 0.5) * cell_size_m

        # ── background rock mass, interpolated from the fixed lattice ────────
        lq = _interp_lattice(log10_q_nodes, axes, step, xs, ys, zs)

        # ── barrier slab, breached by the competent windows ──────────────────
        # The windows are HOLES in the lid, not a multiplier stacked on top of
        # it: applying both would leave the breach at barrier x window, which
        # merely cancels the lid instead of opening it. So the barrier is
        # applied everywhere the windows are not, and the windows then get their
        # own competence boost on top of the untouched background.
        rl_lo, rl_hi = spec.BARRIER_SLAB["rl_range_m"]
        in_barrier = (zs >= rl_lo) & (zs <= rl_hi)

        window_e = np.zeros(len(xs), dtype=bool)
        for win in spec.COMPETENT_WINDOWS:
            e_lo, e_hi = win["east_range_m"]
            window_e |= (xs >= e_lo) & (xs <= e_hi)

        lid_ix = np.flatnonzero(~window_e)
        lid_iz = np.flatnonzero(in_barrier)
        if lid_ix.size and lid_iz.size:
            lq[np.ix_(lid_ix, np.arange(len(ys)), lid_iz)] += \
                log10(spec.BARRIER_SLAB["q_multiplier"])

        # Competent breaches, confined to the barrier's RL band plus a margin so
        # the trade-off is purely "detour laterally to a breach, or drive
        # straight through the lid".
        for win in spec.COMPETENT_WINDOWS:
            e_lo, e_hi = win["east_range_m"]
            r_lo, r_hi = win["rl_range_m"]
            ix = np.flatnonzero((xs >= e_lo) & (xs <= e_hi))
            iz = np.flatnonzero((zs >= r_lo) & (zs <= r_hi))
            if ix.size and iz.size:
                lq[np.ix_(ix, np.arange(len(ys)), iz)] += log10(win["q_multiplier"])

        # ── fault damage halos (metre-specified, resolution independent) ─────
        for f in faults:
            d_perp = fault_perpendicular_distance(f, xs, ys, zs)
            f_lo, f_hi = f["rl_range_m"]
            in_rl = (zs >= f_lo) & (zs <= f_hi)
            hit = (d_perp <= f["damage_half_width_m"]) & in_rl[None, None, :]
            lq[hit] += log10(f["q_multiplier"])
            # Count DISTINCT faults per voxel, as pipeline.py:step2_voxelize does.
            fault_count[i0:i1][hit] += 1

        q = np.clip(10.0 ** lq, *Q_CLIP)
        if q_out is not None:
            q_out[i0:i1] = q.astype(np.float32)

        ti = q_to_tier_index(q)

        # ── above-surface voxels are not excavatable ─────────────────────────
        surf = surface_rl(xs[:, None], ys[None, :])
        above = zs[None, None, :] > surf[:, :, None]
        ti[above] = SENTINEL_TIER

        tier_index[i0:i1] = ti
        above_surface[i0:i1] = above

    # A voxel above ground surface has no fault support requirement.
    fault_count[above_surface] = 0

    out = {
        "tier_index": tier_index,
        "fault_count": fault_count,
        "above_surface": above_surface,
        "surface_rl": surface_rl_grid(dims, cell_size_m, min_coords),
    }
    if q_out is not None:
        out["q"] = q_out
    return out


# ---------------------------------------------------------------------------
# Terminals: portal and production zones
# ---------------------------------------------------------------------------
def _voxel_of(world, cell_size_m, min_coords, dims):
    return tuple(
        int(np.clip(int((world[ax] - min_coords[ax]) / cell_size_m), 0, dims[ax] - 1))
        for ax in range(3)
    )


def portal_terminal(instance_spec: dict, dims, cell_size_m: float, min_coords,
                    passable: np.ndarray) -> dict:
    """
    Place the portal: the topmost passable voxel in the target plan column, plus
    a sphere of ``spec.PORTAL_RADIUS_M`` about it clipped to passable voxels
    lying within ``spec.SURFACE_BAND_M`` of the ground surface.

    Raises RuntimeError if the site yields no feasible voxel, which would make
    the instance unsolvable.
    """
    site = spec.PORTAL_SITES[instance_spec["portal_site"]]
    px, py = site["plan_m"]
    nx, ny, nz = dims

    i = int(np.clip(int((px - min_coords[0]) / cell_size_m), 0, nx - 1))
    j = int(np.clip(int((py - min_coords[1]) / cell_size_m), 0, ny - 1))

    column = passable[i, j, :]
    if not column.any():
        raise RuntimeError(
            f"portal site {instance_spec['portal_site']!r} column ({i},{j}) has "
            f"no passable voxel")
    k = int(np.flatnonzero(column)[-1])

    centre_world = [min_coords[ax] + (v + 0.5) * cell_size_m
                    for ax, v in zip(range(3), (i, j, k))]

    r_vox = int(np.ceil(spec.PORTAL_RADIUS_M / cell_size_m))
    i_lo, i_hi = max(0, i - r_vox), min(nx, i + r_vox + 1)
    j_lo, j_hi = max(0, j - r_vox), min(ny, j + r_vox + 1)
    k_lo, k_hi = max(0, k - r_vox), min(nz, k + r_vox + 1)

    ii, jj, kk = np.meshgrid(np.arange(i_lo, i_hi), np.arange(j_lo, j_hi),
                             np.arange(k_lo, k_hi), indexing="ij")
    wx = min_coords[0] + (ii + 0.5) * cell_size_m
    wy = min_coords[1] + (jj + 0.5) * cell_size_m
    wz = min_coords[2] + (kk + 0.5) * cell_size_m
    dist = np.sqrt((wx - centre_world[0]) ** 2 + (wy - centre_world[1]) ** 2
                   + (wz - centre_world[2]) ** 2)

    near_surface = (surface_rl(wx, wy) - wz) <= spec.SURFACE_BAND_M
    sel = (dist <= spec.PORTAL_RADIUS_M) & near_surface & passable[i_lo:i_hi,
                                                                   j_lo:j_hi,
                                                                   k_lo:k_hi]
    voxels = [(int(a), int(b), int(c))
              for a, b, c in zip(ii[sel], jj[sel], kk[sel])]
    if not voxels:
        voxels = [(i, j, k)]

    return {
        "label": site["label"],
        "site": instance_spec["portal_site"],
        "plan_m": [float(px), float(py)],
        "world_m": [float(v) for v in centre_world],
        "voxel": [i, j, k],
        "radius_m": spec.PORTAL_RADIUS_M,
        "n_voxels": len(voxels),
        "_voxels": voxels,
    }


def zone_terminals(instance_spec: dict, dims, cell_size_m: float, min_coords,
                   passable: np.ndarray) -> list[dict]:
    """
    Build production-zone terminals: plan rectangles extruded +/- half the slab
    thickness about their sublevel RL, clipped to passable voxels.
    """
    nx, ny, nz = dims
    half_e, half_n = spec.ZONE_PLAN_HALF_M
    half_rl = spec.ZONE_RL_THICKNESS_M / 2.0

    xs = min_coords[0] + (np.arange(nx) + 0.5) * cell_size_m
    ys = min_coords[1] + (np.arange(ny) + 0.5) * cell_size_m
    zs = min_coords[2] + (np.arange(nz) + 0.5) * cell_size_m

    zones = []
    for n, (cx, cy, rl, tonnage, grade) in enumerate(
            spec.zone_entries(instance_spec), start=1):

        in_x = np.flatnonzero(np.abs(xs - cx) <= half_e)
        in_y = np.flatnonzero(np.abs(ys - cy) <= half_n)
        in_z = np.flatnonzero(np.abs(zs - rl) <= half_rl)
        if in_x.size == 0 or in_y.size == 0 or in_z.size == 0:
            raise RuntimeError(f"zone Z{n} is empty at cell size {cell_size_m} m")

        block = passable[np.ix_(in_x, in_y, in_z)]
        ii, jj, kk = np.meshgrid(in_x, in_y, in_z, indexing="ij")
        voxels = [(int(a), int(b), int(c))
                  for a, b, c in zip(ii[block], jj[block], kk[block])]
        if not voxels:
            raise RuntimeError(f"zone Z{n} has no passable voxel")

        zones.append({
            "label": f"Z{n}",
            "polygon_plan_m": [
                [cx - half_e, cy - half_n], [cx + half_e, cy - half_n],
                [cx + half_e, cy + half_n], [cx - half_e, cy + half_n],
            ],
            "centre_plan_m": [cx, cy],
            "sublevel_rl_m": rl,
            "rl_thickness_m": spec.ZONE_RL_THICKNESS_M,
            "tonnage_mt": tonnage,
            "mean_grade_g_t": grade,
            "n_voxels": len(voxels),
            "_voxels": voxels,
        })
    return zones


def n_sublevels(zones: list[dict], tol_m: float = 15.0) -> int:
    """
    Distinct sublevels among the zones, using the reference solver's 15 m
    grouping tolerance (``run_steiner_poly.py:group_zones_by_level``).  This
    determines which topology families the solver will evaluate.
    """
    rls = sorted({z["sublevel_rl_m"] for z in zones}, reverse=True)
    levels: list[float] = []
    for rl in rls:
        if not levels or abs(levels[-1] - rl) > tol_m:
            levels.append(rl)
    return len(levels)
