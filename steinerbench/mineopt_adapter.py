"""
Adapter that drives the MineOptimizer WP3 solver on a SteinerMineBench instance.
================================================================================

SPDX-License-Identifier: MIT

This module does NOT reimplement the solver.  It synthesises the three artefacts
WP3 expects, points the solver's configuration at a scratch directory, and calls
``wp3_steiner.run_steiner_poly.main()``.  WP1 (kriging) and WP2 (A*) are skipped
entirely, so a reference run never touches the upstream project's real data and
never introduces geostatistical variability.

Why it has to be done this way
------------------------------
``run_steiner_poly.py`` reads its configuration with ``from config import ...``
at module import time and freezes several paths as module constants
(``POLY_PKL``, ``STEINER_POLY_PKL``, line 110-111).  It also creates
``OUTPUT_REPORTS`` as an import side effect.  Every override therefore has to
land on the ``config`` module **before** ``run_steiner_poly`` is first imported.

Controls applied to every reference run
---------------------------------------
``MINEOPT_SKIP_LOCAL=1``
    Ignore the user's gitignored ``config_local.py``.  Without this a reference
    run silently depends on one developer's local calibration.
``MINEOPT_FLOOD_ENGINE=scipy`` + ``MINEOPT_FORCE_CPU=1`` (**default**)
    True Dijkstra on the CPU: no relaxation tolerance, no CuPy reduction-order
    variance, bit-identical on any machine.  A frozen reference has to be
    reproducible by whoever downloads it, so this is the default and it is what
    lets an instance carry ``reference_type: "exact"``.
``MINEOPT_BENCH_GPU=1`` (opt-in)
    Switches to the ``stencil`` engine on the GPU.  Measured on ``zones-04``:
    ~1.9x faster (227 s vs 459 s), but it moves the CONSTRAINED track by +5.9 %
    and flips its winning family, because that track runs through
    ``lattice_gpu.plan_leg`` -- a different planner implementation, not merely
    different float ordering -- seeded by a flood that relaxes to 1e-3.  A
    reference solved this way is **not bit-reproducible**, cannot be ``exact``,
    and records ``environment.bit_reproducible = false`` with the GPU, driver
    and CuPy version so a reader can see what produced the number.
    Used for ``make_3d_views.py``, whose solves only feed the viewer.
``MINEOPT_FORCE_CELL_SIZE=<cell>``
    Match ``config.CELL_SIZE`` to the instance.
``config.OPEX_IN_EDGE_WEIGHTS = False``
    Otherwise ``main()`` folds a depth-dependent ventilation and pumping term
    into the cost grid and the objective stops being pure geotechnical support
    cost.
``config.LATTICE_TIME_BUDGET_S = inf``
    v2's normative track is defined by the CONSTRAINED planner, so a reference
    run has to use it.  v1 disabled the planner instead (``--no-lattice``) for a
    sound reason -- a per-leg WALL-CLOCK budget makes the answer depend on the
    machine, which is fatal for a frozen reference.  Removing the clock rather
    than the planner keeps both properties: ``LATTICE_MAX_EXPAND`` caps NODE
    EXPANSIONS, so the search still terminates, and it terminates in the same
    place everywhere.

Because ``config`` is a module singleton whose ``CELL_SIZE`` is fixed at import,
each instance must run in its own process.  ``solve_reference.py`` invokes this
module as a subprocess entry point, one per instance.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

BENCH_ROOT = Path(__file__).resolve().parent.parent


def _force_utf8_streams() -> None:
    """
    Make stdout/stderr UTF-8 tolerant.

    The reference solver prints arrows and box characters in its progress
    output.  On Windows, redirecting that to a file gives stdout the cp1252
    codec, and the first arrow raises UnicodeEncodeError from deep inside a
    print() -- killing an otherwise successful solve.  Reconfiguring here fixes
    it however this module is invoked.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def find_mineoptimizer(explicit: str | None = None) -> Path:
    """
    Locate the MineOptimizer checkout that provides the reference solver.

    Search order: the ``--mineoptimizer`` argument, the ``MINEOPTIMIZER_ROOT``
    environment variable, then the benchmark repo's parent directory (the usual
    layout, with ``steinermine-bench/`` sitting inside the project).
    """
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("MINEOPTIMIZER_ROOT"):
        candidates.append(Path(os.environ["MINEOPTIMIZER_ROOT"]))
    candidates.append(BENCH_ROOT.parent)

    for c in candidates:
        c = c.expanduser().resolve()
        if (c / "config.py").exists() and (c / "wp3_steiner").is_dir():
            return c
    raise FileNotFoundError(
        "Could not locate a MineOptimizer checkout (needs config.py and "
        "wp3_steiner/).\nTried:\n  " + "\n  ".join(str(c) for c in candidates) +
        "\nSet MINEOPTIMIZER_ROOT or pass --mineoptimizer.\n\n"
        "Note: the reference solver is only needed to REGENERATE references. "
        "Loading instances and scoring your own solver need neither it nor "
        "this module.")


# ---------------------------------------------------------------------------
# Building the artefacts WP3 expects
# ---------------------------------------------------------------------------
def write_wp3_inputs(instance: dict, scratch: Path) -> dict:
    """
    Write the four files WP3 reads, into ``scratch``.

    ``cost_grid.npy``      float32 support cost per metre
    ``fault_grid.npy``     int32 distinct-fault count (viewer + portal filter)
    ``surface_rl.npy``     float32 ground surface RL per plan column
    ``grid_metadata.npy``  pickled dict, schema per wp1_voxel/pipeline.py:1205
    ``wp2_poly_results.pkl`` the terminal groups, standing in for a WP2 run

    ``path_matrix`` is deliberately absent: WP3 only reads it under
    ``--corridor`` (run_steiner_poly.py:2289), which reference runs never use.
    """
    scratch.mkdir(parents=True, exist_ok=True)
    md = instance["metadata"]
    g = md["grid"]

    np.save(scratch / "cost_grid.npy", instance["cost_grid"])
    np.save(scratch / "fault_grid.npy",
            instance["fault_count"].astype(np.int32))
    np.save(scratch / "surface_rl.npy", instance["surface_rl"])

    meta = {
        "min_coords": np.asarray(g["min_coords_m"], dtype=np.float64),
        "cell_size": float(g["cell_size_m"]),
        "dims": np.asarray(g["dims"], dtype=np.int64),
        "sentinel": float(g["sentinel_cost_per_m"]),
        "surface_source": "steinerminebench_synthetic",
        "estimation": "steinerminebench_analytic_q_field",
    }
    np.save(scratch / "grid_metadata.npy", meta, allow_pickle=True)

    groups = [{
        "label": md["portals"][0]["label"],
        "voxels": [tuple(int(c) for c in v) for v in instance["portal_voxels"]],
        "type": "portal",
    }]
    for zmeta, zvox in zip(md["zones"], instance["zone_voxels"]):
        groups.append({
            "label": zmeta["label"],
            "voxels": [tuple(int(c) for c in v) for v in zvox],
            "type": "zone",
            "tonnage_mt": zmeta.get("tonnage_mt"),
            "mean_grade": zmeta.get("mean_grade_g_t"),
            "sublevel_rl": zmeta.get("sublevel_rl_m"),
        })

    with open(scratch / "wp2_poly_results.pkl", "wb") as fh:
        pickle.dump({"groups": groups, "meta": meta, "no_portal_zones": []}, fh)

    return meta


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------
#: Maps a solver topology label prefix to its canonical benchmark family name.
#:
#: The solver decorates labels with level counts, e.g. ``"sublevel_fan (4
#: levels)"``.  Note that the hybrid family reports itself as
#: ``"hybrid_chained_fan (2L chain + 2L branch)"``
#: (run_steiner_poly.py:1330) -- WITHOUT the trailing ``_branch`` that its
#: evaluator's name carries.  Prefixes are tested longest-first so
#: ``hybrid_chained_fan`` can never be shadowed by ``chained_fan``.
_FAMILY_PREFIXES = [
    ("hybrid_chained_fan", "hybrid_chained_fan_branch"),
    ("sequential_ramp", "sequential_ramp"),
    ("single_junction", "single_junction"),
    ("sublevel_fan", "sublevel_fan"),
    ("three_branch", "three_branch"),
    ("chained_fan", "chained_fan"),
    ("two_branch", "two_branch"),
    # v2 families. The first four sink one or two declines in a chosen plan
    # column and reach each level by horizontal crosscut; they exist because
    # under a grade limit the classic branching families all inherit the same
    # impossible vertical demand, and on every instance solved so far only the
    # decline families verify. `conventional_decline` uses no cost information
    # at all -- it applies the draughting rule -- so it is the control, not an
    # optimiser. `steiner_insertion` adds free branch points that serve no zone.
    ("spiral_decline", "spiral_decline"),
    ("switchback_decline", "switchback_decline"),
    ("conventional_decline", "conventional_decline"),
    ("twin_decline", "twin_decline"),
    ("steiner_insertion", "steiner_insertion"),
]
_FAMILY_PREFIXES.sort(key=lambda kv: -len(kv[0]))


def canonical_family(topology_string: str) -> str | None:
    """Recover the canonical family name from a solver topology label."""
    s = topology_string.strip()
    for prefix, family in _FAMILY_PREFIXES:
        if s.startswith(prefix):
            return family
    return None


def support_class_lengths(paths, cost_grid, cell_size_m, tier_costs,
                          support_classes) -> tuple[dict, float]:
    """
    Metres of drive in each Barton Q support class, plus the total length.

    Length is accumulated per step as the Euclidean distance between voxel
    centres, and attributed to the support class of the voxel being entered --
    the same convention the edge weight uses.
    """
    lengths = {cls: 0.0 for cls in support_classes}
    total = 0.0
    for path in paths:
        for a, b in zip(path[:-1], path[1:]):
            d = np.sqrt(sum((b[ax] - a[ax]) ** 2 for ax in range(3))) * cell_size_m
            c = float(cost_grid[b[0], b[1], b[2]])
            idx = int(np.argmin(np.abs(np.asarray(tier_costs) - c)))
            lengths[support_classes[idx]] += d
            total += d
    return lengths, total


def extract_results(all_topologies, meta, cost_grid, tier_costs,
                    support_classes) -> list[dict]:
    """Reduce the solver's topology dicts to compact, JSON-friendly records."""
    mn = np.asarray(meta["min_coords"], dtype=np.float64)
    cs = float(meta["cell_size"])
    out = []

    for t in all_topologies:
        if t is None:
            continue
        family = canonical_family(t.get("topology", ""))
        if family is None:
            # Never drop a family silently: a reference that quietly omits a
            # topology would understate the benchmark's best-known cost.
            raise RuntimeError(
                f"cannot map solver topology label {t.get('topology')!r} to a "
                f"benchmark family. The upstream solver has added or renamed a "
                f"topology; update _FAMILY_PREFIXES in mineopt_adapter.py.")

        sps = [tuple(int(c) for c in sp) for sp in t.get("steiner_points", [])]
        paths = [[tuple(int(c) for c in v) for v in p]
                 for p in t.get("paths", []) if p]
        cls_len, total_len = support_class_lengths(
            paths, cost_grid, cs, tier_costs, support_classes)

        bd = t.get("buildable") or {}
        out.append({
            "family": family,
            "topology_label": t.get("topology"),
            "cost": float(t["cost"]),
            "cost_buildable": (float(t["cost_buildable"])
                               if t.get("cost_buildable") is not None else None),
            "junctions_voxel": [list(sp) for sp in sps],
            "junctions_world_m": [
                [float(mn[ax] + (sp[ax] + 0.5) * cs) for ax in range(3)]
                for sp in sps],
            "support_class_length_m": {k: round(v, 3)
                                       for k, v in cls_len.items() if v > 0},
            "total_length_m": round(total_len, 3),
            "buildable_summary": {
                "ramp_length_m": bd.get("ramp_length"),
                "worst_grade": bd.get("worst_grade"),
                "worst_min_radius_m": bd.get("worst_min_radius"),
                "n_spiral_ramps": bd.get("n_spiral_ramps"),
                "method": bd.get("method"),
            } if bd else None,
            # v2: a cost is only a CONSTRAINED cost if the geometry that
            # produced it was planned under the standard AND independently
            # verified.  `cost_buildable` on its own is just an estimate the
            # geometric post-processor produced, which is what v1 shipped and
            # what the constrained track exists to replace.
            "constrained_feasible": bool(bd.get("feasible")) if bd else False,
            "verified": bool(bd.get("verified")) if bd else False,
            "verify": bd.get("verify"),
            "cost_constrained": (float(t["cost_buildable"])
                                 if bd and bd.get("feasible")
                                 and bd.get("verified")
                                 and t.get("cost_buildable") is not None
                                 else None),
            # `basis` and `network_length_m` ship so a reader can see WHICH
            # geometry the operating cost was measured on without re-running
            # anything -- the v2.0.0 references were silently raw-basis.
            "opex": {k: (float(v) if isinstance(v, (int, float)) else v)
                     for k, v in (t.get("opex") or {}).items()
                     if k in ("haulage", "vent_pump", "total",
                              "mine_life_years", "basis", "network_length_m",
                              "n_portal_exits", "portal_establishment")} or None,
            "cost_total": (float(t["cost_total"])
                           if bd and bd.get("feasible") and bd.get("verified")
                           and t.get("cost_total") is not None else None),
            "segment_labels": list(t.get("segment_labels", [])),
            "_paths": paths,
            "_buildable_lines": [
                (str(lbl), np.asarray(P, dtype=float).tolist())
                for lbl, P in (t.get("buildable_lines") or [])],
        })
    return out


# ---------------------------------------------------------------------------
# The worker: one instance, one process
# ---------------------------------------------------------------------------
def run_instance(instance_id: str, scratch: Path, mineopt_root: Path,
                 production_buffer_m: float = 0.0) -> dict:
    """
    Run the reference solver on one instance.  Must be called in a fresh
    process: it mutates ``os.environ`` and the ``config`` module singleton.

    ``production_buffer_m`` is the ramp standoff from every production zone
    (``config.PRODUCTION_BUFFER_M``).  It is a parameter of the INSTANCE FAMILY,
    not of the solver: at 5 m the winning family and cost both move, so a
    reference solved under one standoff is not comparable with one solved under
    another.  It is passed explicitly and recorded in ``environment`` so a
    reference always states which it was.
    """
    sys.path.insert(0, str(BENCH_ROOT))
    from steinerbench import spec, tiers
    from steinerbench.loader import load_instance

    instance = load_instance(instance_id)
    md = instance["metadata"]
    cell = float(md["grid"]["cell_size_m"])

    # ── 1. Environment, before `config` is imported ──────────────────────────
    os.environ["MINEOPT_SKIP_LOCAL"] = "1"
    os.environ["MINEOPT_FORCE_CELL_SIZE"] = repr(cell)
    # Deterministic CPU by default; GPU only when explicitly asked for.
    #
    # A frozen reference has to be reproducible by whoever downloads it, so the
    # default is scipy's true Dijkstra on the CPU: no relaxation tolerance, no
    # CuPy reduction-order variance, bit-identical on any machine.
    #
    # The GPU path was measured on `zones-04` and is NOT a last-digits effect.
    # It is ~1.9x faster (227 s vs 459 s) but moves the CONSTRAINED track by
    # +5.9% and flips its winning family (sequential_ramp -> chained_fan),
    # because that track runs through lattice_gpu.plan_leg -- a different
    # planner implementation, not merely different float ordering -- seeded by a
    # stencil flood that relaxes to 1e-3.  The constrained track is the
    # normative one, so its stability is worth more than the 2x.
    #
    # MINEOPT_BENCH_GPU=1 opts in.  It is used for `make_3d_views.py`, whose
    # solves exist only to draw pictures and whose costs nobody cites.
    if os.environ.get("MINEOPT_BENCH_GPU") == "1":
        os.environ["MINEOPT_FLOOD_ENGINE"] = "stencil"
        os.environ.pop("MINEOPT_FORCE_CPU", None)
    else:
        os.environ["MINEOPT_FLOOD_ENGINE"] = "scipy"
        os.environ["MINEOPT_FORCE_CPU"] = "1"
    # A reference solve must never inherit a developer's workspace: the paths
    # below are overridden to the scratch dir anyway, but leaving the variable
    # set would make config resolve `config_local.py` from that workspace and
    # silently apply another dataset's calibration to a frozen reference.
    os.environ.pop("MINEOPT_WORKSPACE", None)
    # Same reasoning, one level down.  `run_steiner_poly.py` takes its standoff
    # default from $WP3_PRODUCTION_BUFFER_M before falling back to config, so a
    # developer who exported it for a case-study run would silently re-solve the
    # frozen references under a standoff that nothing in the bundle records.
    # The standoff arrives as an argument below; it may not arrive by ambience.
    os.environ.pop("WP3_PRODUCTION_BUFFER_M", None)

    sys.path.insert(0, str(mineopt_root))
    import config  # noqa: E402

    tiers.assert_matches_mineoptimizer(config)

    # ── 2. Redirect every path, before `run_steiner_poly` is imported ────────
    scratch = scratch.resolve()
    reports = scratch / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    config.DATA_PROCESSED = scratch
    config.COST_GRID_NPY = scratch / "cost_grid.npy"
    config.GRID_META_NPY = scratch / "grid_metadata.npy"
    config.FAULT_GRID_NPY = scratch / "fault_grid.npy"
    config.SURFACE_RL_NPY = scratch / "surface_rl.npy"
    config.OUTPUT_REPORTS = reports
    config.OPEX_IN_EDGE_WEIGHTS = False

    # The benchmark PUBLISHES its operating-cost rates in every
    # metadata.json (`tracks.opex_model`, written from spec.OPEX_MODEL), so a
    # submitter computes the `total` track from them.  Until 2026-08-14 they
    # were written but never applied: the solver charged MineOptimizer's own
    # config rates ($1.00/t-km, $0.03/t-m) while the bundle declared $0.35 and
    # $0.012, making every reference total ~2.7x what its own published model
    # gives -- so any conforming submission would have scored as a NEW BEST.
    # Apply them here, before wp3 (and opex.py with it) is imported.
    for _key, _attr in (("haul_cost_per_t_km", "HAUL_COST_PER_T_KM"),
                        ("haul_vert_cost_per_t_m", "HAUL_VERT_COST_PER_T_M"),
                        ("vent_cost_per_m_yr", "VENT_COST_PER_M_YR"),
                        ("pump_cost_per_m_yr_m_depth", "PUMP_COST_PER_M_YR_M_DEPTH"),
                        ("mine_life_years", "MINE_LIFE_YEARS")):
        setattr(config, _attr, float(spec.OPEX_MODEL[_key]))
    config.PORTAL_ESTABLISHMENT_COST = float(
        spec.OPEX_MODEL["portal_establishment_cost"])

    # v2 runs the CONSTRAINED planner, because the normative track is defined
    # by it.  v1 passed --no-lattice for a good reason -- a per-leg WALL-CLOCK
    # budget makes the result depend on the machine, which is fatal for a
    # frozen reference.  The fix is to remove the clock, not the planner:
    # LATTICE_MAX_EXPAND is a cap on NODE EXPANSIONS, so the search still
    # terminates, and it terminates at the same place on every machine.
    config.LATTICE_TIME_BUDGET_S = float("inf")
    config.LATTICE_STRICT = True

    if abs(config.CELL_SIZE - cell) > 1e-9:
        raise RuntimeError(
            f"config.CELL_SIZE is {config.CELL_SIZE} but instance {instance_id} "
            f"needs {cell}; MINEOPT_FORCE_CELL_SIZE did not take effect")

    write_wp3_inputs(instance, scratch)

    # ── 3. Import and drive the solver ───────────────────────────────────────
    from wp3_steiner import run_steiner_poly as wp3  # noqa: E402

    if wp3.POLY_PKL.parent != scratch:
        raise RuntimeError(
            f"WP3 resolved its input path to {wp3.POLY_PKL}, not {scratch}; "
            f"config was mutated too late")

    argv = sys.argv
    t0 = time.time()
    try:
        sys.argv = ["run_steiner_poly.py", "--auto",
                    "--production-buffer-m", repr(float(production_buffer_m))]
        wp3.main()
    finally:
        sys.argv = argv
    runtime = time.time() - t0

    # ── 4. Harvest ───────────────────────────────────────────────────────────
    results_pkl = scratch / "wp3_poly_steiner_results.pkl"
    if not results_pkl.exists():
        # `wp3.main()` returned NORMALLY and still wrote nothing: every family
        # was rejected, so there is no network to cost.  Under a standoff that
        # is a result about the instance -- the constraint is binding to the
        # point of infeasibility -- not a solver malfunction, and reporting it
        # as a crash would lose exactly the finding a standoff sweep is for.
        # A genuine crash raises out of main() and never reaches here.
        return {
            "instance_id": instance_id,
            "families": [],
            "infeasible": True,
            "infeasible_reason": (
                f"no topology family could be built with a "
                f"{production_buffer_m:g} m production standoff"
                if production_buffer_m else
                "no topology family could be built"),
            "total_runtime_s": round(runtime, 2),
            "environment": {"production_buffer_m": float(production_buffer_m)},
        }
    with open(results_pkl, "rb") as fh:
        res = pickle.load(fh)

    families = extract_results(
        res["all_topologies"], res["meta"], instance["cost_grid"],
        tiers.TIER_COSTS, tiers.SUPPORT_CLASSES)

    import platform
    import scipy

    return {
        "instance_id": instance_id,
        "families": families,
        "total_runtime_s": round(runtime, 2),
        "environment": {
            "solver": "MineOptimizer WP3 (wp3_steiner/run_steiner_poly.py)",
            "solver_commit": _git_commit(mineopt_root),
            "flood_engine": config.MINEOPT_FLOOD_ENGINE,
            "opex_in_edge_weights": config.OPEX_IN_EDGE_WEIGHTS,
            "lattice_refinement": True,
            "lattice_time_budget_s": None,
            "lattice_max_expand": config.LATTICE_MAX_EXPAND,
            "production_buffer_m": float(production_buffer_m),
            "buildable_method": (
                "constrained heading-augmented A* on all candidates, then "
                "independent verification; the per-leg cap is node expansions, "
                "not wall clock, so it does not depend on machine speed"),
            "cell_size_m": cell,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            # Reproducibility disclosure.  With the GPU/stencil path these
            # numbers are NOT bit-reproducible on other hardware, so the reader
            # needs to know exactly what produced them.
            **_compute_environment(),
        },
    }


def _compute_environment() -> dict:
    """
    What actually executed the solve, and whether the result is reproducible.

    The CPU/scipy path is exact and machine-independent; the GPU/stencil path is
    neither, so the GPU model, driver and CuPy version become part of the
    provenance a reader needs in order to interpret -- or reproduce -- the cost.
    """
    import os as _os
    cpu_only = _os.environ.get("MINEOPT_BENCH_GPU") != "1"
    env = {
        "compute": "cpu" if cpu_only else "gpu",
        "bit_reproducible": bool(cpu_only),
        "reproducibility_note": (
            "scipy true Dijkstra on CPU: bit-reproducible on any machine"
            if cpu_only else
            "stencil (Bellman-Ford, 1e-3 tolerance, sweep cap) on GPU, with "
            "the CuPy lattice planner: measured to move the constrained track "
            "by ~6% and to change its winning family, so this reference is "
            "best_known and is not reproducible on other hardware"),
    }
    if cpu_only:
        return env
    try:
        import cupy                                            # noqa: F401
        env["cupy"] = cupy.__version__
        d = cupy.cuda.runtime.getDeviceProperties(0)
        env["gpu"] = d["name"].decode() if isinstance(d["name"], bytes) else str(d["name"])
        env["cuda_runtime"] = cupy.cuda.runtime.runtimeGetVersion()
    except Exception as exc:                # noqa: BLE001 -- provenance, not control flow
        env["gpu"] = f"unavailable ({exc.__class__.__name__})"
    return env


def _git_commit(root: Path) -> str:
    import subprocess
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=15)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Worker: run the MineOptimizer reference solver on one "
                    "SteinerMineBench instance. Normally invoked by "
                    "solve_reference.py, not directly.")
    p.add_argument("--instance", required=True)
    p.add_argument("--scratch", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path,
                   help="pickle path for the harvested result")
    p.add_argument("--mineoptimizer", default=None,
                   help="path to the MineOptimizer checkout")
    p.add_argument("--production-buffer-m", type=float, default=0.0, metavar="M",
                   help="ramp standoff from every production zone, in metres "
                        "(default 0, the standoff the shipped references were "
                        "solved under). Recorded in environment."
                        "production_buffer_m.")
    args = p.parse_args(argv)

    _force_utf8_streams()
    root = find_mineoptimizer(args.mineoptimizer)
    result = run_instance(args.instance, args.scratch, root,
                          production_buffer_m=args.production_buffer_m)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump(result, fh)
    print(f"\n[adapter] harvested {len(result['families'])} families -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
