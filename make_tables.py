#!/usr/bin/env python
"""
make_tables.py -- emit paper-ready tables from the frozen instance set.
======================================================================

SPDX-License-Identifier: MIT

Writes into ``tables/``:

``instance_manifest.csv``   every instance with its axis values, size,
                            reference type, reference cost, bound and gap
``instance_manifest.tex``   the same as a booktabs LaTeX table
``results_table.tex``       the reference solver's per-family costs, formatted
                            for direct \\input into a manuscript

Both .tex files are self-contained tabulars: they assume ``booktabs`` and
``siunitx`` are loaded in the preamble and define no macros of their own.

Usage
-----
    python make_tables.py
    python make_tables.py --out-dir /path/to/paper/tables
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from steinerbench import spec                                  # noqa: E402
from steinerbench.loader import instance_path, list_instances  # noqa: E402

GROUP_LABEL = {
    "crossing_grid": "Crossing grid",
    "portal_sweep": "Portal sweep",
    "zone_count": "Zone count",
    "scale_ladder": "Scale ladder",
    "haulage_ratio": "Haulage ratio",
}

FAMILY_SHORT = {
    "spiral_decline": "Spiral decline",
    "switchback_decline": "Switchback decline",
    "twin_decline": "Twin decline",
    "conventional_decline": "Conventional decline",
    "steiner_insertion": "Steiner insertion",
    "sublevel_fan": "Sublevel fan",
    "two_branch": "Two-branch",
    "three_branch": "Three-branch",
    "single_junction": "Single junction",
    "sequential_ramp": "Sequential ramp",
    "chained_fan": "Chained fan",
    "hybrid_chained_fan_branch": "Hybrid chain+branch",
}

AXIS_LABEL = {
    "fault_system": {"none": "none", "single": "1 fault",
                     "two_crossing": "2 crossing", "conjugate_pair": "conjugate"},
    "q_regime": {"competent": "competent", "mixed": "mixed",
                 "poor_dominated": "poor"},
}


def tex_escape(s: str) -> str:
    """Escape the LaTeX specials that can appear in our labels."""
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("&", r"\&"),
                 ("%", r"\%"), ("#", r"\#"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def _tb(ref: dict, track: str) -> dict:
    """One track's lower-bound record, empty on a pre-v2.1.0 bundle."""
    return ((ref.get("lower_bounds") or {}).get(track)) or {}


def axis_summary(s: dict) -> str:
    """Human-readable value of the axis this instance varies."""
    parts = []
    for k, v in s["varied_axis"]["value"].items():
        parts.append(str(AXIS_LABEL.get(k, {}).get(v, v)))
    return ", ".join(parts)


def gather() -> list[dict]:
    rows = []
    for iid in list_instances():
        s = spec.get(iid)
        md = json.loads((instance_path(iid) / "metadata.json")
                        .read_text(encoding="utf-8"))
        rp = instance_path(iid) / "reference.json"
        ref = json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else {}

        g, tf = md["grid"], md["topology_families"]
        lb = (ref.get("lower_bound") or {}).get("value")
        rows.append({
            "instance_id": iid,
            "group": s["group"],
            "group_label": GROUP_LABEL[s["group"]],
            "axis": ", ".join(s["varied_axis"]["axis"]),
            "axis_value": axis_summary(s),
            "cell_size_m": g["cell_size_m"],
            "nx": g["dims"][0], "ny": g["dims"][1], "nz": g["dims"][2],
            "n_voxels": g["n_voxels"],
            "n_passable": g["n_passable"],
            "n_zones": tf["n_zones"],
            "n_sublevels": tf["n_sublevels"],
            "n_families": len(tf["applicable"]),
            "reference_type": ref.get("reference_type", "unsolved"),
            "reference_cost": ref.get("reference_cost"),
            "reference_cost_buildable": ref.get("reference_cost_buildable"),
            "best_topology": ref.get("best_topology"),
            # The unsuffixed pair stays RAW so anything reading this CSV keeps
            # working; the per-track columns are the ones to quote.
            "lower_bound": lb,
            "lower_bound_method": (ref.get("lower_bound") or {}).get("method"),
            "gap_to_lower_bound": ref.get("gap_to_lower_bound"),
            "lower_bound_raw": lb,
            "gap_to_lower_bound_raw": ref.get("gap_to_lower_bound"),
            "reference_cost_constrained": ref.get("reference_cost_constrained"),
            "lower_bound_constrained": _tb(ref, "constrained").get("value"),
            "lower_bound_method_constrained": _tb(ref, "constrained").get("method"),
            "gap_to_lower_bound_constrained": (
                ref.get("gaps_to_lower_bound") or {}).get("constrained"),
            "reference_cost_total": ref.get("reference_cost_total"),
            "lower_bound_total": _tb(ref, "total").get("value"),
            "gap_to_lower_bound_total": (
                ref.get("gaps_to_lower_bound") or {}).get("total"),
            "grid_sha256": md["checksums"]["cost_grid.npz"],
            "seed": md["generation"]["seed"],
            "_ref": ref,
        })
    return rows


def write_csv(rows, out: Path) -> None:
    fields = [k for k in rows[0] if not k.startswith("_")]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {out}")


def _fmt_money(v) -> str:
    return r"\num{" + f"{v:.0f}" + "}" if v else "--"


def write_manifest_tex(rows, out: Path) -> None:
    lines = [
        r"% SteinerMineBench instance manifest - generated by make_tables.py",
        r"% Requires: \usepackage{booktabs} \usepackage{siunitx}",
        r"\begin{tabular}{llrrrrlrrrr}",
        r"\toprule",
        r"Instance & Axis value & Cell & Voxels & Zones & Lev. & Ref.\ type "
        r"& Ref.\ cost & Gap\textsubscript{raw} & Constr.\ cost "
        r"& Gap\textsubscript{con} \\",
        r" & & (m) & & & & & (\$) & (\%) & (\$) & (\%) \\",
        r"\midrule",
    ]
    current = None
    for r in rows:
        if r["group"] != current:
            current = r["group"]
            lines.append(r"\addlinespace")
            lines.append(r"\multicolumn{11}{l}{\itshape "
                         + tex_escape(r["group_label"])
                         + r" -- varying " + tex_escape(r["axis"]) + r"} \\")
            lines.append(r"\addlinespace[2pt]")
        gap = (f"{r['gap_to_lower_bound_raw'] * 100:.1f}"
               if r["gap_to_lower_bound_raw"] is not None else "--")
        gap_c = (f"{r['gap_to_lower_bound_constrained'] * 100:.1f}"
                 if r["gap_to_lower_bound_constrained"] is not None else "--")
        lines.append(
            f"{tex_escape(r['instance_id'])} & {tex_escape(r['axis_value'])} & "
            f"{r['cell_size_m']:g} & \\num{{{r['n_voxels']}}} & {r['n_zones']} & "
            f"{r['n_sublevels']} & {tex_escape(r['reference_type'])} & "
            f"{_fmt_money(r['reference_cost'])} & {gap} & "
            f"{_fmt_money(r['reference_cost_constrained'])} & {gap_c} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {out}")


def write_results_tex(rows, out: Path) -> None:
    """Per-family reference costs, one column per topology family."""
    fams = [f for f, _, _ in spec.FAMILIES]
    header = " & ".join(FAMILY_SHORT[f].replace(" ", r"\,") for f in fams)

    lines = [
        r"% SteinerMineBench reference-solver results - generated by make_tables.py",
        r"% Requires: \usepackage{booktabs} \usepackage{siunitx}",
        r"% Costs are raw voxel Steiner cost in US dollars; boldface marks the",
        r"% cheapest family on each instance. '--' means the family is not",
        r"% applicable at that instance's sublevel count.",
        r"% The Bound column is the RAW-track bound, because the per-family",
        r"% costs beside it are raw costs. The constrained-track bound is a",
        r"% different quantity and lives in the manifest table; putting the two",
        r"% side by side is the cross-track confusion v2 exists to remove.",
        r"\begin{tabular}{l" + "r" * len(fams) + r"r}",
        r"\toprule",
        r"Instance & " + header + r" & Bound\textsubscript{raw} \\",
        r"\midrule",
    ]

    for r in rows:
        ref = r["_ref"]
        if not ref.get("per_family"):
            lines.append(tex_escape(r["instance_id"]) + " & "
                         + " & ".join(["--"] * len(fams)) + r" & -- \\")
            continue
        costs = {f["family"]: f["cost"] for f in ref["per_family"]}
        best = ref["best_topology"]
        cells = []
        for f in fams:
            if f not in costs:
                cells.append("--")
            elif f == best:
                cells.append(r"\bfseries " + _fmt_money(costs[f]))
            else:
                cells.append(_fmt_money(costs[f]))
        lines.append(tex_escape(r["instance_id"]) + " & " + " & ".join(cells)
                     + " & " + _fmt_money(r["lower_bound"]) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {out}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("Usage")[0].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=ROOT / "tables")
    args = p.parse_args(argv)

    rows = gather()
    if not rows:
        print("No instance bundles found. Run:  python generate.py --all")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out_dir / "instance_manifest.csv")
    write_manifest_tex(rows, args.out_dir / "instance_manifest.tex")
    write_results_tex(rows, args.out_dir / "results_table.tex")

    n_ref = sum(1 for r in rows if r["reference_cost"] is not None)
    print(f"\n  {len(rows)} instances, {n_ref} with a reference solution")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
