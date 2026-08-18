#!/usr/bin/env python
"""
score.py - score a solver against the frozen SteinerMineBench references.
=========================================================================

SPDX-License-Identifier: MIT

The graded quantity is the **optimality gap**

    optimality_gap = (solver_cost - reference_cost) / reference_cost

so 0 means you matched the reference and a negative value means you beat it.

Three tracks (v2)
-----------------
``--track constrained`` (default, NORMATIVE)
    Buildable capex: the voxel cost integrated along a centreline that
    satisfies ``metadata.tracks.geometric_standard`` -- grade <= 20 %, turn
    radius >= 25 m, a passable 5 m swept volume, a connected network and a rock
    pillar between distinct openings. Submissions are re-checked with the
    shipped verifier (``steinerbench/buildable_check.py``, numpy only), so this
    does not depend on any solver's post-processor.

``--track raw`` (diagnostic, unchanged from v1)
    Pure voxel Steiner cost under monotone descent and nothing else.
    Solver-agnostic and exactly reproducible from the instance bundle: see
    ``metadata.cost_model.edge_weight_formula``. It is no longer normative
    because it is the optimum of a RELAXATION: measured on the v1 references,
    100 % of descending steps exceed a 20 % grade limit and the median step
    grade is 100 % (45 deg). It is retained because it is exactly reproducible
    and because the gap between it and a buildable answer is itself a result.

``--track total``
    Constrained capex plus life-of-mine operating cost under
    ``metadata.tracks.opex_model``. This is the only track on which Group E
    varies: those instances differ solely in tonnage, so their geometry and
    their capex are identical and only haulage moves.

Negative gaps
-------------
Not all improvements mean the same thing, and they are never quietly averaged in:

* on a ``best_known`` instance, a negative gap is a **new best known bound** --
  exactly what this benchmark is for. Submit it (see the README) and it becomes
  the new reference.
* on an ``exact`` instance, a negative gap is an **error**: the reference is
  provably optimal for its family, so either the submitted solution is
  infeasible or the reference is wrong. Both possibilities are flagged loudly.

Usage
-----
    python score.py                                  # sanity-check the references
    python score.py --submission mysolver.json
    python score.py --submission mysolver.json --track raw
    python score.py --submission mysolver.json --csv results.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from steinerbench import spec                                 # noqa: E402
from steinerbench.loader import instance_path, list_instances  # noqa: E402

#: Per track, so a file can never mix them. The single `results.csv` v2.0.0
#: shipped had been written on one track and read as another -- its
#: `reference_cost` column held constrained costs beside raw gaps.
def default_csv(track: str) -> Path:
    return ROOT / f"results_{track}.csv"

#: Relative tolerance below which a submission counts as having *matched* the
#: reference rather than beaten or lost to it. The reference solver accumulates
#: edge weights in float32 and reference.json stores costs rounded to 4 decimal
#: places, so an independent float64 implementation of the same topology lands
#: a few parts in 1e9 away. Anything inside 1e-6 is numerically the same answer,
#: and in particular is never reported as a new best-known bound.
MATCH_RTOL = 1e-6


def _bound(ref: dict, track: str) -> dict:
    """
    The lower-bound record for one track, tolerating a v2.0.0 bundle.

    Pre-2.1.0 references carry only `lower_bound`, which is the RAW bound. It is
    returned for the raw track and withheld for the others, because quoting a
    relaxation bound against a constrained cost is the misreading this version
    removes -- better an empty column than a number that means something else.
    """
    per_track = ref.get("lower_bounds") or {}
    if per_track:
        return per_track.get(track) or {}
    return (ref.get("lower_bound") or {}) if track == "raw" else {}


def _load_reference(instance_id: str) -> dict | None:
    p = instance_path(instance_id) / "reference.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _validate_submission(sub: dict, path: Path) -> None:
    """Validate against schemas/submission.schema.json when jsonschema is present."""
    try:
        import jsonschema
    except ImportError:
        if not isinstance(sub.get("results"), dict) or not sub.get("solver"):
            raise SystemExit(
                f"{path}: a submission needs a 'solver' string and a 'results' "
                f"object keyed by instance_id. Install jsonschema for full "
                f"validation: pip install jsonschema")
        return
    schema = json.loads(
        (ROOT / "schemas" / "submission.schema.json").read_text(encoding="utf-8"))
    try:
        jsonschema.validate(sub, schema)
    except jsonschema.ValidationError as exc:
        loc = "/".join(str(p) for p in exc.absolute_path) or "(root)"
        raise SystemExit(f"{path}: invalid submission at {loc}: {exc.message}")


def collect_rows(submission: dict | None, track: str) -> list[dict]:
    """Build one row per instance that has a reference on disk."""
    cost_key = {"raw": "reference_cost",
                "constrained": "reference_cost_constrained",
                "total": "reference_cost_total"}[track]
    sub_key = {"raw": "cost",
               "constrained": "cost_constrained",
               "total": "cost_total"}[track]
    results = (submission or {}).get("results", {})

    rows = []
    for iid in list_instances():
        ref = _load_reference(iid)
        if ref is None:
            continue
        s = spec.get(iid)
        ref_cost = ref.get(cost_key)
        entry = results.get(iid)
        solver_cost = entry.get(sub_key) if entry else None

        gap = status = None
        if solver_cost is not None and ref_cost:
            gap = (solver_cost - ref_cost) / ref_cost
            if abs(gap) <= MATCH_RTOL:
                status = "matched"
            elif gap < 0:
                status = ("INVALID" if ref.get("reference_type") == "exact"
                          else "NEW BEST")
            else:
                status = "ok"
        elif entry is not None and ref_cost is None:
            status = "no reference"
        elif submission is not None:
            status = "unattempted"

        rows.append({
            "instance_id": iid,
            "group": s["group"],
            "voxels": ref and None,
            "reference_type": ref.get("reference_type"),
            "reference_cost": ref_cost,
            # Bound and gap are BOTH per-track, and are read from the track's
            # own record rather than recomputed here. v2.0.0 divided the
            # constrained cost by the RAW bound, which measured the relaxation's
            # irrelevance (+200-340%) far more than it measured any search;
            # v2.1.0 gives the constrained and total tracks a bound that keeps
            # the grade limit. Showing the RAW winner beside a constrained
            # reference would invite the reader to attribute one track's answer
            # to the other -- the confusion this benchmark version exists to
            # remove -- so the winner is per-track too.
            "lower_bound": _bound(ref, track).get("value"),
            "lower_bound_method": _bound(ref, track).get("method"),
            "gap_to_lower_bound": (ref.get("gaps_to_lower_bound") or {}).get(track),
            "best_topology": (
                ref.get("best_topology") if track == "raw"
                else ref.get(f"best_topology_{track}")),
            "solver_cost": solver_cost,
            "solver_topology": (entry or {}).get("topology"),
            "optimality_gap": gap,
            "runtime_s": (entry or {}).get("runtime_s"),
            "status": status,
        })
    return rows


def print_leaderboard(rows: list[dict], submission: dict | None,
                      track: str) -> None:
    scored = [r for r in rows if r["optimality_gap"] is not None]

    if submission:
        hdr = (f"{'instance':<18} {'type':<11} {'reference':>14} "
               f"{'submitted':>14} {'gap %':>9} {'LB gap %':>9} "
               f"{'time s':>8}  status")
    else:
        hdr = (f"{'instance':<18} {'type':<11} {'reference':>14} "
               f"{'lower bound':>14} {'LB gap %':>9} {'topology':<26}")

    title = (f"SteinerMineBench - {submission['solver']}"
             if submission else "SteinerMineBench - frozen references")
    print(f"\n{title}   [track: {track}]")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    for r in rows:
        ref = r["reference_cost"]
        ref_s = f"{ref:>14,.0f}" if ref else f"{'-':>14}"
        lbg = r["gap_to_lower_bound"]
        lbg_s = f"{lbg * 100:>9.1f}" if lbg is not None else f"{'-':>9}"

        if submission:
            sub_s = (f"{r['solver_cost']:>14,.0f}"
                     if r["solver_cost"] is not None else f"{'-':>14}")
            gap_s = (f"{r['optimality_gap'] * 100:>9.2f}"
                     if r["optimality_gap"] is not None else f"{'-':>9}")
            rt = f"{r['runtime_s']:>8.1f}" if r["runtime_s"] is not None else f"{'-':>8}"
            print(f"{r['instance_id']:<18} {r['reference_type']:<11} {ref_s} "
                  f"{sub_s} {gap_s} {lbg_s} {rt}  {r['status'] or ''}")
        else:
            lb = r["lower_bound"]
            lb_s = f"{lb:>14,.0f}" if lb else f"{'-':>14}"
            print(f"{r['instance_id']:<18} {r['reference_type']:<11} {ref_s} "
                  f"{lb_s} {lbg_s} {str(r['best_topology'] or '-'):<26}")

    print("-" * len(hdr))

    if submission and scored:
        gaps = [r["optimality_gap"] for r in scored]
        mean = sum(gaps) / len(gaps)
        print(f"\n  instances scored     {len(scored)} of {len(rows)}")
        print(f"  mean optimality gap  {mean * 100:+.3f} %")
        print(f"  worst gap            {max(gaps) * 100:+.3f} %  "
              f"({max(scored, key=lambda r: r['optimality_gap'])['instance_id']})")
        print(f"  best gap             {min(gaps) * 100:+.3f} %  "
              f"({min(scored, key=lambda r: r['optimality_gap'])['instance_id']})")
        matched = sum(1 for r in scored if r["status"] == "matched")
        print(f"  matched reference    {matched}")

        new_best = [r for r in scored if r["status"] == "NEW BEST"]
        invalid = [r for r in scored if r["status"] == "INVALID"]

        if new_best:
            print(f"\n  *** {len(new_best)} NEW BEST KNOWN BOUND(S) ***")
            for r in new_best:
                print(f"      {r['instance_id']}: {r['solver_cost']:,.0f} beats "
                      f"{r['reference_cost']:,.0f} by "
                      f"{-r['optimality_gap'] * 100:.2f} %")
            print("      These instances are labelled 'best_known', so an "
                  "improvement is expected and welcome.")
            print("      Open a pull request including paths_voxel for each, so "
                  "the network can be re-costed and verified. See the README.")

        if invalid:
            print(f"\n  !!! {len(invalid)} SUBMISSION(S) BEAT AN 'exact' "
                  f"REFERENCE !!!")
            for r in invalid:
                print(f"      {r['instance_id']}: {r['solver_cost']:,.0f} < "
                      f"{r['reference_cost']:,.0f}")
            print("      An 'exact' reference was independently recomputed and "
                  "is optimal for its topology family, so either the submitted")
            print("      solution is infeasible (check monotone descent, "
                  "26-connectivity, and that every zone is reached), or the")
            print("      reference is wrong. Please report it either way - a "
                  "genuine counterexample is a benchmark bug worth fixing.")

        unattempted = [r for r in rows if r["status"] == "unattempted"]
        if unattempted:
            print(f"\n  not attempted: {len(unattempted)} "
                  f"({', '.join(r['instance_id'] for r in unattempted)})")

    elif not submission:
        by_type: dict[str, int] = {}
        for r in rows:
            by_type[r["reference_type"]] = by_type.get(r["reference_type"], 0) + 1
        print(f"\n  {len(rows)} instance(s): "
              + ", ".join(f"{n} {t}" for t, n in sorted(by_type.items())))
        lbg = [r["gap_to_lower_bound"] for r in rows
               if r["gap_to_lower_bound"] is not None]
        if lbg:
            print(f"  gap to lower bound: mean {sum(lbg) / len(lbg) * 100:.1f} %, "
                  f"min {min(lbg) * 100:.1f} %, max {max(lbg) * 100:.1f} %")
        print("\n  Pass --submission your_results.json to score your own solver.")


def write_csv(rows: list[dict], path: Path, track: str) -> None:
    import csv
    # `track` is written into every row below, so it has to be declared here
    # too -- extrasaction="ignore" was silently dropping it, which is how a
    # track-mixed results.csv became indistinguishable from a clean one.
    fields = ["instance_id", "track", "group", "reference_type",
              "reference_cost", "lower_bound", "lower_bound_method",
              "gap_to_lower_bound", "best_topology", "solver_cost",
              "solver_topology", "optimality_gap", "runtime_s", "status"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "track": track})
    print(f"\n  wrote {path}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("Usage")[0].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--submission", type=Path, metavar="JSON",
                   help="your solver's results; see schemas/submission.schema.json")
    p.add_argument("--track", choices=("constrained", "raw", "total"),
                   default="constrained",
                   help="which cost to grade (default: constrained, the normative one)")
    p.add_argument("--csv", type=Path, default=None,
                   help="results CSV path (default: results_<track>.csv)")
    p.add_argument("--no-csv", action="store_true")
    p.add_argument("--fail-on-invalid", action="store_true",
                   help="exit non-zero if any submission beats an exact reference")
    args = p.parse_args(argv)

    submission = None
    if args.submission:
        submission = json.loads(args.submission.read_text(encoding="utf-8"))
        _validate_submission(submission, args.submission)

    rows = collect_rows(submission, args.track)
    if not rows:
        print("No references found. Run:  python solve_reference.py --all")
        return 1

    print_leaderboard(rows, submission, args.track)
    if not args.no_csv:
        write_csv(rows, args.csv or default_csv(args.track), args.track)

    if args.fail_on_invalid and any(r["status"] == "INVALID" for r in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
