#!/usr/bin/env python
"""
solve_reference.py -- compute reference solutions for SteinerMineBench.
======================================================================

SPDX-License-Identifier: MIT

Runs the MineOptimizer WP3 reference solver on each instance and writes
``reference.json`` and ``reference_paths.npz`` into the instance bundle.

For every family the solver labels EXACT, an independent recomputation
(``steinerbench.verify_exact``) rebuilds the graph, the Dijkstra fields and the
argmin from scratch, sharing no code with the solver, and asserts agreement.
A failure aborts the run rather than silently downgrading the label -- that
assertion is what earns an instance its ``exact`` badge.

You only need this script to REGENERATE references.  Loading instances and
scoring your own solver need neither it nor a MineOptimizer checkout.

Usage
-----
    python solve_reference.py --all                  # every solvable instance
    python solve_reference.py --only zones-04
    python solve_reference.py --group A --resume     # skip ones already done

    # rewrite every bound block from metadata + the stored costs, no solving:
    python solve_reference.py --all --recompute-bounds-only
    python solve_reference.py --all --recompute-bounds-only --check   # CI

Each instance runs in its own subprocess: ``config.CELL_SIZE`` is fixed when the
solver's configuration module is first imported, so a single process cannot
serve instances at different resolutions.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import json
import os
import pickle
import shutil
import subprocess
import sys
import threading
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from steinerbench import spec                                    # noqa: E402
from steinerbench.loader import instance_path, load_instance     # noqa: E402
from steinerbench.lower_bound import (                           # noqa: E402
    FORBIDDEN_BOUND_METHODS, compute_lower_bound, compute_track_bounds,
)
from steinerbench.spec import EXACT_FAMILIES                     # noqa: E402
from steinerbench.verify_exact import (                          # noqa: E402
    ExactnessFailure, verify_instance,
)

OBJECTIVE = (
    "Total raw voxel cost of a portal-rooted, monotonically descending network "
    "reaching every production zone, with edge weight "
    "w(u,v) = (cost_grid[v] + excavation_rate_per_m) * effective_length_m(u,v) "
    "over the 26-neighbourhood. Excludes buildability post-processing, "
    "operating cost and portal establishment."
)


def run_solver_subprocess(instance_id: str, mineoptimizer: str | None,
                          keep_log: Path | None,
                          production_buffer_m: float = 0.0) -> dict:
    """Run the adapter worker for one instance and return its harvested dict."""
    scratch = Path(tempfile.mkdtemp(prefix=f"sbref-{instance_id}-"))
    out = scratch / "result.pkl"
    cmd = [sys.executable, "-m", "steinerbench.mineopt_adapter",
           "--instance", instance_id, "--scratch", str(scratch),
           "--out", str(out),
           "--production-buffer-m", repr(float(production_buffer_m))]
    if mineoptimizer:
        cmd += ["--mineoptimizer", mineoptimizer]

    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True,
                              text=True, encoding="utf-8", errors="replace")
        if keep_log:
            keep_log.parent.mkdir(parents=True, exist_ok=True)
            keep_log.write_text(proc.stdout + "\n" + proc.stderr,
                                encoding="utf-8")
        if proc.returncode != 0 or not out.exists():
            tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-25:])
            raise RuntimeError(
                f"reference solver failed for {instance_id} "
                f"(exit {proc.returncode}). Last output:\n{tail}")
        with open(out, "rb") as fh:
            return pickle.load(fh)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def build_reference(instance_id: str, harvested: dict, checks: dict,
                    lb: dict) -> tuple[dict, dict]:
    """Assemble ``reference.json`` and the path archive for one instance."""
    inst = load_instance(instance_id)
    applicable = set(inst["metadata"]["topology_families"]["applicable"])
    search_kind = {name: kind for name, _, kind in spec.FAMILIES}

    families = [f for f in harvested["families"] if f["family"] in applicable]
    if not families:
        raise RuntimeError(f"{instance_id}: solver returned no applicable family")

    # A standoff can make a family STRUCTURALLY unbuildable rather than merely
    # unsolved -- `sequential_ramp` chains ramps that terminate at the orebody by
    # construction, and a standoff forbids ramps in ore -- so its absence is a
    # result about the constrained problem, not a gap in the search.
    #
    # At 0 m the guard stands: there, a missing applicable family means the
    # solver failed to return one it should have, and a reference built without
    # it would quietly claim a best-known cost that is not the best known.
    _standoff = float((harvested.get("environment") or {})
                      .get("production_buffer_m") or 0.0)
    missing = applicable - {f["family"] for f in families}
    if missing and _standoff <= 0:
        raise RuntimeError(
            f"{instance_id}: solver did not return applicable families "
            f"{sorted(missing)}; the reference would understate the best-known "
            f"cost")

    best = min(families, key=lambda f: f["cost"])
    buildable = [f for f in families if f.get("cost_buildable") is not None]
    best_build = min(buildable, key=lambda f: f["cost_buildable"]) if buildable else None

    # v2 tracks. A candidate qualifies only if it was planned under the
    # geometric standard AND passed the independent verifier -- an unplanned
    # candidate's `cost_buildable` is the geometric estimate, which is exactly
    # the quantity the constrained track replaces.
    admissible = spec.constrained_admissible(inst["metadata"]["grid"]["cell_size_m"])
    con = [f for f in families if f.get("cost_constrained") is not None]
    tot = [f for f in families if f.get("cost_total") is not None]
    best_con = min(con, key=lambda f: f["cost_constrained"]) if con and admissible else None
    best_tot = min(tot, key=lambda f: f["cost_total"]) if tot and admissible else None

    # An instance is 'exact' only when its cheapest family is one of the
    # exact-search families, that family survived independent verification, AND
    # the solve itself was reproducible.
    #
    # That last clause matters only under MINEOPT_BENCH_GPU=1.  An `exact` label
    # is a claim that anyone re-running the solver lands on the same number; the
    # stencil engine relaxes to a 1e-3 tolerance under a sweep cap and CuPy's
    # reduction order varies across GPUs, so on that path the claim cannot be
    # made and every reference is `best_known`.  The default CPU path can be
    # exact, and is.
    check = checks.get(best["family"])
    reproducible = os.environ.get("MINEOPT_BENCH_GPU") != "1"
    # A production standoff restricts where a ramp may go, so the network it
    # yields is the optimum of a DIFFERENT problem from the one this instance
    # publishes. `exact` is a claim about the published problem, so a standoff
    # run can only ever be `best_known` -- regardless of how well it verified.
    # (`_standoff` is read once, above, where the applicable-family guard needs it.)
    is_exact = (reproducible
                and _standoff == 0.0
                and best["family"] in EXACT_FAMILIES
                and check is not None
                and check.get("independent_argmin_matches") is True)

    # Per-track bounds. The relaxation bound `lb` was computed once, on the
    # grid; the geometric floors are trigonometry over metadata and cost
    # nothing, which is what makes --recompute-bounds-only possible.
    #
    # Round to the SAME precision the reference stores, because
    # `--recompute-bounds-only --check` later re-derives these bounds from the
    # stored `reference_cost` -- i.e. from the rounded value.  Feeding the
    # unrounded cost here made the two paths disagree in the last digits of
    # `raw_best_known_conditional`, so --check failed on every freshly solved
    # instance and could never do the job it exists for.  The affected component
    # is conditional and never contributes to any bound's `value`, so this
    # changes no reported bound or gap.
    bounds = compute_track_bounds(inst["metadata"], lb,
                                  reference_cost_raw=round(best["cost"], 4))

    # Per-track consistency. Each bound is checked against the cost of ITS OWN
    # track: the raw bound against the raw reference, the constrained bound
    # against the constrained reference. Checking every bound against the raw
    # cost -- what v2.0.0 did, when there was only one bound -- would fire on
    # 19 of 25 instances here, because the grade floor is legitimately above a
    # raw network's cost. That is the whole reason the floor is not applied to
    # the raw track.
    _track_cost = {
        "raw": best["cost"],
        "constrained": best_con["cost_constrained"] if best_con else None,
        "total": best_tot["cost_total"] if best_tot else None,
    }
    for _track, _cost in _track_cost.items():
        if _cost is None:
            continue
        _b = bounds.get(_track)
        if _b and _b["value"] > _cost * (1 + 1e-9):
            raise RuntimeError(
                f"{instance_id}: the {_track} lower bound {_b['value']:,.2f} "
                f"({_b['method']}) exceeds the {_track} reference cost "
                f"{_cost:,.2f}. One of them is wrong; refusing to ship an "
                f"inconsistent bundle.")

    per_family = []
    paths_archive: dict[str, np.ndarray] = {}
    for f in sorted(families, key=lambda x: x["cost"]):
        rec = {
            "family": f["family"],
            "search": search_kind[f["family"]],
            "cost": round(f["cost"], 4),
            "cost_buildable": (round(f["cost_buildable"], 4)
                               if f["cost_buildable"] is not None else None),
            "total_length_m": f["total_length_m"],
            "junctions_voxel": f["junctions_voxel"],
            "junctions_world_m": [[round(c, 3) for c in w]
                                  for w in f["junctions_world_m"]],
            "support_class_length_m": f["support_class_length_m"],
            "buildable_summary": f["buildable_summary"],
            "topology_label": f["topology_label"],
            "constrained_feasible": f.get("constrained_feasible", False),
            "verified": f.get("verified", False),
            "verify": f.get("verify"),
            "cost_constrained": (round(f["cost_constrained"], 4)
                                 if f.get("cost_constrained") is not None
                                 else None),
            "cost_total": (round(f["cost_total"], 4)
                           if f.get("cost_total") is not None else None),
            "opex": f.get("opex"),
        }
        if f["family"] in checks:
            rec["exactness_check"] = checks[f["family"]]
        elif f["family"] in EXACT_FAMILIES:
            rec["exactness_check"] = {
                "performed": False,
                "note": "Exact-search family, but no independent check ran.",
            }
        per_family.append(rec)

        for si, path in enumerate(f["_paths"]):
            if path:
                paths_archive[f"{f['family']}/{si}"] = np.asarray(
                    path, dtype=np.int32)

    reference = {
        "instance_id": instance_id,
        "schema_version": spec.SCHEMA_VERSION,
        "benchmark_version": spec.BENCHMARK_VERSION,
        # v1 shipped `exact` where the cheapest family used a closed-form
        # argmin.  That label describes the RAW track and only the raw track: a
        # constrained search with junction iteration is heuristic, so no
        # constrained reference is ever provably optimal.  The label is
        # therefore reported per track rather than once for the bundle.
        "reference_type": "best_known",
        "reference_type_raw": "exact" if is_exact else "best_known",
        "reference_type_constrained": ("best_known" if best_con else
                                       "none" if admissible else
                                       "not_admissible"),
        "normative_track": spec.NORMATIVE_TRACK,
        "objective": OBJECTIVE,
        "reference_cost": round(best["cost"], 4),
        "reference_cost_buildable": (round(best_build["cost_buildable"], 4)
                                     if best_build else None),
        "reference_cost_constrained": (round(best_con["cost_constrained"], 4)
                                       if best_con else None),
        "reference_cost_total": (round(best_tot["cost_total"], 4)
                                 if best_tot else None),
        "constrained_admissible": admissible,
        "n_families_verified": sum(1 for f in families if f.get("verified")),
        "n_families_planned": sum(1 for f in families
                                  if f.get("constrained_feasible")),
        "constrained_note": (
            "reference_cost_constrained is the cheapest network that was "
            "planned under the geometric standard AND passed the shipped "
            "verifier. Families that only carry cost_buildable were not "
            "planned constrained; their figure is a geometric estimate for a "
            "network nobody has shown can be built, retained as evidence and "
            "never used as a reference."
            if admissible and con else
            "No family reached a goal under the constrained planner on this "
            "instance, so nothing was verified and there is no constrained "
            "reference. Note this is NOT the discretisation floor -- the cell "
            "is fine relative to the turning radius. Every family reports "
            "unplannable legs, which points at the planner's node-expansion "
            "cap (LATTICE_MAX_EXPAND) binding on a state space this large "
            "before the search reaches a goal. Treat the absence as a limit of "
            "the reference solver at this size, not as a property of the "
            "instance."
            if admissible and not con else
            "This rung is too coarse to carry a constrained reference "
            "(min_turn_radius_m / cell_size_m < "
            f"{spec.MIN_RADIUS_CELLS_FOR_CONSTRAINED:g}); it is scored on the "
            "raw track only."),
        "best_topology": best["family"],
        "best_topology_buildable": best_build["family"] if best_build else None,
        "best_topology_constrained": best_con["family"] if best_con else None,
        "best_topology_total": best_tot["family"] if best_tot else None,
        # `lower_bound` keeps its v2.0.0 name and its RAW semantics, because
        # every downstream reader and all 28 instance READMEs name it. The
        # per-track bounds sit beside it.
        "lower_bound": bounds["raw"],
        "lower_bounds": bounds,
        "gap_to_lower_bound": (round((best["cost"] - lb["value"]) / lb["value"], 6)
                               if lb["value"] > 0 else None),
        "gaps_to_lower_bound": {
            t: (round((c - bounds[t]["value"]) / bounds[t]["value"], 6)
                if c is not None and bounds[t]["value"] > 0 else None)
            for t, c in _track_cost.items()
        },
        "exactness_note": (
            f"The cheapest family ({best['family']}) uses a closed-form argmin "
            f"over all passable voxels and was independently recomputed from "
            f"scratch; reference_cost is optimal for it."
            if is_exact else
            (f"The cheapest family ({best['family']}) uses a closed-form "
             f"argmin, but this reference was solved under MINEOPT_BENCH_GPU, "
             f"which relaxes to a 1e-3 tolerance under a sweep cap and whose "
             f"floating-point reduction order is not reproducible across "
             f"hardware. No exactness claim is made; see "
             f"environment.reproducibility_note. Re-solve on the default CPU "
             f"path for an exact, bit-reproducible reference."
             if not reproducible and best["family"] in EXACT_FAMILIES else
             f"The cheapest family ({best['family']}) uses a "
             f"{search_kind[best['family']]} search, so reference_cost is a "
             f"best-known bound and may be improvable. Exact-search families "
             f"on this instance were still independently verified; see "
             f"per_family[].exactness_check.")),
        "per_family": per_family,
        # Applicable families the solver could not build. Empty at 0 m (the guard
        # above refuses the reference instead); under a standoff this is a
        # RESULT -- the constraint eliminated the topology -- and it must be
        # stated, or a reader comparing arms sees a field of 11 against 12 with
        # no explanation and reads the difference as a search failure.
        "families_unbuildable": sorted(missing),
        "production_buffer_m": _standoff,
        "environment": harvested["environment"],
        "solver_runtime_s": harvested["total_runtime_s"],
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return reference, paths_archive


def unsolved_reference(instance_id: str) -> dict:
    """
    The stub reference shipped for instances with no computed solution.

    It still carries a **real, valid lower bound** on the constrained and total
    tracks: the geometric floor is trigonometry over the portal and zone RLs, so
    it needs neither a solution nor a grid.  An open frontier that ships with a
    proven floor is a better open problem than one that ships with nothing --
    a submitter knows immediately what they have to beat.  There is no raw bound
    here, because that one does need the grid.
    """
    inst = load_instance(instance_id)
    md = inst["metadata"]
    g = md["grid"]
    bounds = compute_track_bounds(
        md, {"value": 0.0, "method": "none", "valid": True, "track": "raw",
             "components": {}, "note": "not computed for an unsolved instance"})
    return {
        "instance_id": instance_id,
        "schema_version": spec.SCHEMA_VERSION,
        "benchmark_version": spec.BENCHMARK_VERSION,
        "reference_type": "unsolved",
        "objective": OBJECTIVE,
        "reference_cost": None,
        "reference_cost_buildable": None,
        "best_topology": None,
        "lower_bounds": {t: bounds[t] for t in ("constrained", "total")},
        "gaps_to_lower_bound": {"raw": None, "constrained": None, "total": None},
        "note": (
            f"No reference solution. "
            + (f"At {g['n_voxels']:,} voxels ({g['n_passable']:,} passable) a "
               f"multi-source Dijkstra over the 26-neighbour graph needs "
               f"roughly 22 GB for the CSR structure alone, beyond the machine "
               f"used to build this suite -- a MEMORY limit. "
               if g["n_voxels"] > 5e7 else
               f"At {g['n_voxels']:,} voxels ({g['n_passable']:,} passable) the "
               f"grid is small enough to solve, but the constrained planner "
               f"reaches no goal on any of the twelve families, while the same "
               f"geology at a 5 m cell plans every leg. The radius-to-cell "
               f"ratio here is the finest in the suite, so this is not the "
               f"discretisation floor -- it is a SEARCH-BUDGET limit "
               f"(LATTICE_MAX_EXPAND binding on a heading-augmented state space "
               f"of this size). Nothing here says no network exists, only that "
               f"this solver did not find one. ")
            + f"The grid, terminals and cost model are fully specified, so this "
              f"instance stands as an open frontier: the first valid solution "
              f"submitted becomes its best-known reference. score.py reports "
              f"submissions for it without an optimality gap. The constrained "
              f"and total lower bounds ARE computed and valid -- they follow "
              f"from the grade limit and the cheapest rock in the model, not "
              f"from a search -- so a submission can still be judged against a "
              f"floor."),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def out_dir(instance_id: str, args) -> Path:
    """
    Where this run's reference is written.

    Default is the instance bundle -- the shipped references live with the
    instances they describe.  ``--out-root`` diverts a VARIANT solve (a
    different production standoff, say) into a parallel tree instead.  A variant
    is not a better answer to the same question, it is the answer to a different
    one: writing it into ``instances/`` would silently redefine what the
    published bundle's reference means, and nothing downstream -- score.py,
    make_tables.py, a submitter's gap calculation -- would notice.
    """
    if getattr(args, "out_root", None):
        d = Path(args.out_root) / instance_id
        d.mkdir(parents=True, exist_ok=True)
        return d
    return instance_path(instance_id)


def solve_one(instance_id: str, args) -> str:
    """Solve one instance and write its reference. Returns a status word."""
    dest = out_dir(instance_id, args)
    ref_path = dest / "reference.json"

    if args.resume and ref_path.exists():
        existing = json.loads(ref_path.read_text(encoding="utf-8"))
        if existing.get("reference_type") != "unsolved" or \
                instance_id in spec.UNSOLVED_INSTANCES:
            return "skipped"

    if instance_id in spec.UNSOLVED_INSTANCES:
        ref_path.write_text(
            json.dumps(unsolved_reference(instance_id), indent=2) + "\n",
            encoding="utf-8")
        print("    reference_type = unsolved (open scaling frontier)")
        return "unsolved"

    t0 = time.time()
    print("    running reference solver ...", flush=True)
    log = (Path(args.logs) / f"{instance_id}.log") if args.logs else None
    harvested = run_solver_subprocess(
        instance_id, args.mineoptimizer, log,
        production_buffer_m=getattr(args, "production_buffer_m", 0.0))
    print(f"    solver: {len(harvested['families'])} families in "
          f"{harvested['total_runtime_s']:.1f} s")

    if harvested.get("infeasible"):
        # A binding constraint, recorded as a result. The lower bounds still
        # hold -- they come from the grade limit and the cheapest rock, not from
        # a search -- so the record stays comparable with the 0 m reference.
        buf = getattr(args, "production_buffer_m", 0.0)
        ref = unsolved_reference(instance_id)
        ref["reference_type"] = "infeasible"
        ref["production_buffer_m"] = float(buf)
        ref["note"] = (
            f"INFEASIBLE under a {buf:g} m production standoff: "
            f"{harvested['infeasible_reason']}. The same instance solves at "
            f"0 m; see the published reference. The lower bounds below are "
            f"unaffected by the standoff (they follow from the grade limit and "
            f"the cheapest rock in the model), so they remain valid floors.")
        ref_path.write_text(json.dumps(ref, indent=2) + "\n", encoding="utf-8")
        print(f"    -> INFEASIBLE at {buf:g} m standoff "
              f"[{time.time() - t0:.1f} s]")
        return "unsolved"

    inst = load_instance(instance_id)

    # The independent recomputation still runs on the GPU path -- it is a real
    # check that the reported junction is the argmin of an independently
    # rebuilt field, and a gross error would still show up.  What changes is the
    # verdict on a MISMATCH: on the deterministic CPU path a mismatch is a bug
    # and must stop the run, whereas on the GPU path a last-digits disagreement
    # is an expected consequence of a 1e-3 relaxation tolerance and must not.
    # It is recorded either way, and no instance claims exactness on this path.
    print("    independent exactness verification ...")
    _buf = float(getattr(args, "production_buffer_m", 0.0) or 0.0)
    try:
        checks = verify_instance(inst, harvested["families"], verbose=True)
    except ExactnessFailure as exc:
        # A standoff run is not solving the published problem. `verify_instance`
        # recomputes the junction argmin on the instance's own (unbuffered) grid,
        # so as soon as the standoff pushes the optimum off that argmin the
        # verifier finds something cheaper and refuses the exact label -- which is
        # correct, and is the whole point of an independent check. It is not a
        # solver fault and must not abort the run: a constrained sub-problem
        # simply cannot claim exactness against the unconstrained one, and
        # build_reference below downgrades it to best_known for that reason.
        if _buf > 0:
            print(f"    exactness not claimable under a {_buf:g} m standoff "
                  f"(the verifier recomputes on the unbuffered grid): {exc}")
            checks = {}
        elif os.environ.get("MINEOPT_BENCH_GPU") != "1":
            print(f"    EXACTNESS FAILURE: {exc}")
            raise
        else:
            print(f"    exactness mismatch (expected under MINEOPT_BENCH_GPU, "
                  f"not fatal): {exc}")
            checks = {}

    print("    lower bound ...")
    lb = compute_lower_bound(inst, verbose=True)

    reference, paths = build_reference(instance_id, harvested, checks, lb)

    ref_path.write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(dest / "reference_paths.npz", **paths)

    print(f"    -> {reference['reference_type']}  "
          f"cost ${reference['reference_cost']:,.0f} "
          f"({reference['best_topology']})  [{time.time() - t0:.1f} s]")
    for _t in ("raw", "constrained", "total"):
        _b = reference["lower_bounds"].get(_t)
        _g = reference["gaps_to_lower_bound"].get(_t)
        if not _b:
            continue
        print(f"       {_t:11} bound ${_b['value']:>12,.0f}  ({_b['method']})"
              + (f"  gap {_g * 100:6.1f}%" if _g is not None else "  gap -"))
    return "solved"


def recompute_bounds(instance_id: str, args) -> str:
    """
    Rewrite one instance's bound blocks WITHOUT solving anything.

    The relaxation bound is reused from the existing reference (it costs a pair
    of Dijkstra floods over the whole grid and does not depend on anything this
    change touches); the geometric floors are recomputed from ``metadata.json``.
    So the whole suite is seconds, not hours.

    Also strips any ``wong_dual_ascent`` component left over from v2.0.0 and
    re-derives ``method``/``value`` as the argmax over what survives.  On every
    instance but ``scale-130k`` that is a no-op, pairwise divergence having been
    the argmax; on ``scale-130k`` -- the 10 m rung, where a sparse-graph method
    finally had a sparse graph -- the reported raw bound drops back to pairwise.

    With ``--check`` nothing is written and any difference is an error, which is
    what makes bound drift a CI failure rather than a surprise at release time.
    """
    dest = instance_path(instance_id)
    ref_path = dest / "reference.json"
    if not ref_path.exists():
        print("    no reference.json -- nothing to recompute")
        return "skipped"

    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    before = json.dumps(ref.get("lower_bound"), sort_keys=True)
    md = json.loads((dest / "metadata.json").read_text(encoding="utf-8"))

    raw = dict(ref.get("lower_bound") or {})
    comps = {k: v for k, v in (raw.get("components") or {}).items()
             if k not in FORBIDDEN_BOUND_METHODS}
    if comps:
        raw["components"] = comps
        raw["method"] = max(comps, key=comps.__getitem__)
        raw["value"] = comps[raw["method"]]
    elif "value" not in raw:
        # An unsolved instance has no relaxation bound; the floors still apply.
        raw = {"value": 0.0, "method": "none", "components": {},
               "note": "not computed for an unsolved instance"}
    raw["track"] = "raw"
    raw["valid"] = True

    cost_raw = ref.get("reference_cost")
    bounds = compute_track_bounds(md, raw, reference_cost_raw=cost_raw)
    costs = {"raw": cost_raw,
             "constrained": ref.get("reference_cost_constrained"),
             "total": ref.get("reference_cost_total")}
    gaps = {t: (round((c - bounds[t]["value"]) / bounds[t]["value"], 6)
                if c is not None and bounds[t]["value"] > 0 else None)
            for t, c in costs.items()}

    for t, c in costs.items():
        if c is not None and bounds[t]["value"] > c * (1 + 1e-9):
            raise RuntimeError(
                f"{instance_id}: recomputed {t} bound {bounds[t]['value']:,.2f} "
                f"exceeds the {t} reference cost {c:,.2f}")

    # Relative tolerance: `components` is stored rounded to 4 dp while `value`
    # keeps full precision, so an exact comparison reports every instance as
    # changed and the one that really did change stops standing out.
    was = (json.loads(before) or {}).get("value") if before != "null" else None
    moved = (was is not None
             and abs(was - raw["value"]) > 1e-6 * max(1.0, abs(was)))
    print(f"    raw bound ${raw['value']:>12,.0f} ({raw['method']})"
          + (f"   <- CHANGED from ${was:,.0f}, a dropped component was the argmax"
             if moved else "   unchanged"))
    for t in ("constrained", "total"):
        g = gaps[t]
        print(f"    {t:11} ${bounds[t]['value']:>12,.0f} ({bounds[t]['method']})"
              + (f"   gap {g * 100:6.1f}%" if g is not None else "   gap -"))

    if args.check:
        stale = (json.dumps(ref.get("lower_bounds"), sort_keys=True)
                 != json.dumps(bounds, sort_keys=True)
                 or json.dumps(ref.get("gaps_to_lower_bound"), sort_keys=True)
                 != json.dumps(gaps, sort_keys=True))
        if stale:
            raise RuntimeError(
                f"{instance_id}: the stored bounds differ from a fresh "
                f"recomputation. Run without --check to refresh them.")
        return "skipped"

    ref["lower_bound"] = bounds["raw"]
    ref["lower_bounds"] = bounds
    ref["gap_to_lower_bound"] = gaps["raw"]
    ref["gaps_to_lower_bound"] = gaps
    # The bound blocks ARE the schema change, so a recomputed reference is a
    # v2.1.0 reference by definition.
    ref["schema_version"] = spec.SCHEMA_VERSION
    ref["benchmark_version"] = spec.BENCHMARK_VERSION
    ref["bounds_recomputed_utc"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    ref_path.write_text(json.dumps(ref, indent=2) + "\n", encoding="utf-8")
    return "solved"


class _FanOutStdout:
    """
    Thread-local stdout so parallel instance solves stay readable.

    Every worker prints progress as it goes.  Run several at once and those
    lines interleave into something no one can read, and worse, an error ends up
    attributed to whichever instance printed last.  Installing this once as
    ``sys.stdout`` lets each worker thread capture its own output into a buffer,
    which is then flushed as one block when that instance finishes; threads with
    no buffer registered (the main thread) write straight through.

    ``contextlib.redirect_stdout`` cannot do this -- it swaps the global, so
    with threads the last one in wins and output is lost.
    """

    def __init__(self, real):
        self._real = real
        self._local = threading.local()

    def capture(self):
        self._local.buf = io.StringIO()

    def release(self) -> str:
        buf = getattr(self._local, "buf", None)
        self._local.buf = None
        return buf.getvalue() if buf else ""

    def _target(self):
        return getattr(self._local, "buf", None) or self._real

    def write(self, s):
        return self._target().write(s)

    def flush(self):
        return self._target().flush()

    def isatty(self):
        return False


def _run_parallel(ids, args, worker, tally, jobs: int) -> bool:
    """Solve `ids` `jobs`-at-a-time.  Returns False if the run should stop."""
    fan = _FanOutStdout(sys.stdout)
    sys.stdout = fan
    lock = threading.Lock()
    done = 0
    stop = False

    def one(iid):
        fan.capture()
        try:
            return iid, worker(iid, args), None, fan.release()
        except ExactnessFailure as exc:
            return iid, None, exc, fan.release()
        except Exception as exc:                              # noqa: BLE001
            return iid, None, exc, fan.release()

    try:
        with cf.ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(one, iid) for iid in ids]
            for fut in cf.as_completed(futures):
                iid, status, exc, out = fut.result()
                with lock:
                    done += 1
                    print(f"\n[{done}/{len(ids)}] {iid}", file=fan._real)
                    if out:
                        print(out.rstrip(), file=fan._real)
                    if exc is None:
                        tally[status].append(iid)
                    else:
                        tally["failed"].append(iid)
                        print(f"    FAILED: {exc}", file=fan._real)
                        if isinstance(exc, ExactnessFailure) or \
                                not args.continue_on_error:
                            stop = True
                            for f in futures:
                                f.cancel()
    finally:
        sys.stdout = fan._real
    return not stop


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("Usage")[0].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--all", action="store_true")
    sel.add_argument("--only", nargs="+", metavar="ID")
    sel.add_argument("--group", metavar="A|B|C|D|E")

    p.add_argument("--resume", action="store_true",
                   help="skip instances that already have a reference")
    p.add_argument("--recompute-bounds-only", action="store_true",
                   help="rewrite the lower-bound blocks from metadata and the "
                        "stored costs, without solving anything (seconds)")
    p.add_argument("--check", action="store_true",
                   help="with --recompute-bounds-only: verify the stored "
                        "bounds match a fresh recomputation and write nothing")
    p.add_argument("--mineoptimizer", default=None,
                   help="path to the MineOptimizer checkout providing WP3")
    p.add_argument("--logs", default=None, metavar="DIR",
                   help="keep the full solver log per instance in DIR")
    p.add_argument("--continue-on-error", action="store_true",
                   help="report and move on instead of stopping at the first "
                        "failure (exactness failures always stop the run)")
    p.add_argument("--production-buffer-m", type=float, default=0.0, metavar="M",
                   help="ramp standoff from every production zone, in metres. "
                        "0 (the default) is what every shipped reference was "
                        "solved under. Any other value changes the problem, not "
                        "just the answer, so it REQUIRES --out-root.")
    p.add_argument("--out-root", default=None, metavar="DIR",
                   help="write reference.json / reference_paths.npz under "
                        "DIR/<instance_id>/ instead of into the instance "
                        "bundle. Use this for every variant solve so the "
                        "published references stay the 0 m ones.")
    p.add_argument("--jobs", "-j", type=int, default=1, metavar="N",
                   help="solve N instances concurrently. Each solve is a "
                        "separate subprocess on one core, so this is close to "
                        "linear until RAM or cores run out; 26 of the 28 "
                        "instances are <=1M voxels. Determinism is unaffected: "
                        "the per-instance result does not depend on what else "
                        "is running.")
    args = p.parse_args(argv)
    if args.jobs < 1:
        p.error("--jobs must be >= 1")
    # A standoff changes WHERE the network may go, so it changes the reference
    # cost and can change the winning family. Letting it overwrite instances/
    # would leave the bundle claiming a 0 m reference while holding a 5 m one,
    # and the only trace would be one field in `environment`. Refuse instead.
    if args.production_buffer_m and not args.out_root:
        p.error(f"--production-buffer-m {args.production_buffer_m:g} needs "
                f"--out-root: a non-zero standoff is a different problem from "
                f"the published references and must not overwrite them "
                f"(e.g. --out-root ../runs/benchmark/variants/buffer5)")
    if args.production_buffer_m < 0:
        p.error("--production-buffer-m must be >= 0")
    # The search budget decides what the solver is ALLOWED TO PROVE, so a run at
    # a reduced cap can report a dearer reference (or lose a family's constrained
    # cost to the geometric fallback) on the same instance. That is a different
    # number for the same question, which makes it a variant too -- and unlike a
    # standoff it arrives through the environment, where it is easy to leave set
    # from a previous experiment and never notice. A 0 m re-solve at a reduced cap
    # would otherwise overwrite the published references in place.
    _cap = os.environ.get("MINEOPT_LATTICE_MAX_EXPAND")
    if _cap and not args.out_root:
        p.error(f"MINEOPT_LATTICE_MAX_EXPAND={_cap} is set in the environment "
                f"but the published default is 4000000. A reduced cap is a "
                f"variant solve and needs --out-root; unset the variable to "
                f"write the canonical references.")

    if args.only:
        ids = spec.select(only=args.only)
    elif args.group:
        ids = spec.select(group=args.group)
    elif args.all:
        ids = spec.select()
    else:
        p.print_help()
        return 2

    tally: dict[str, list[str]] = {"solved": [], "skipped": [], "unsolved": [],
                                   "failed": []}
    t_start = time.time()

    worker = recompute_bounds if args.recompute_bounds_only else solve_one

    # Bound recomputation is seconds of pure metadata arithmetic -- parallelism
    # would only cost readability.
    jobs = 1 if args.recompute_bounds_only else min(args.jobs, len(ids))
    if jobs > 1:
        print(f"  running {jobs} instances concurrently "
              f"({len(ids)} total, one subprocess each)")
        if not _run_parallel(ids, args, worker, tally, jobs):
            print(f"\n  stopped early after a failure")
    else:
        for n, iid in enumerate(ids, 1):
            print(f"\n[{n}/{len(ids)}] {iid}")
            try:
                tally[worker(iid, args)].append(iid)
            except ExactnessFailure:
                raise
            except Exception as exc:  # noqa: BLE001 - reported, then re-raised or skipped
                tally["failed"].append(iid)
                print(f"    FAILED: {exc}")
                if not args.continue_on_error:
                    return 1

    print(f"\n{'=' * 62}")
    print(f"  solved   {len(tally['solved']):>3}")
    print(f"  unsolved {len(tally['unsolved']):>3}"
          + (f"  ({', '.join(tally['unsolved'])})" if tally["unsolved"] else ""))
    print(f"  skipped  {len(tally['skipped']):>3}")
    print(f"  failed   {len(tally['failed']):>3}"
          + (f"  ({', '.join(tally['failed'])})" if tally["failed"] else ""))
    print(f"  elapsed  {time.time() - t_start:.1f} s")
    print("=" * 62)
    return 1 if tally["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
