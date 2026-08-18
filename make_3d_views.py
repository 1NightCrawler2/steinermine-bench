#!/usr/bin/env python
"""
make_3d_views.py -- one interactive 3-D page per benchmark instance.
====================================================================

SPDX-License-Identifier: MIT

Writes a self-contained HTML file per instance showing **every** topology
family the solver evaluated -- selectable from a dropdown -- drawn against that
instance's cost field, with the verifier's verdict and every violation located
on the alignment.

    python make_3d_views.py                  # all admissible instances
    python make_3d_views.py --only zones-08
    python make_3d_views.py --force          # re-render ones already written

Why it re-solves rather than reading the bundle
-----------------------------------------------
``reference.json`` stores costs and junctions, and ``reference_paths.npz``
stores the RAW voxel polylines -- but the viewer needs the *buildable
centrelines*, which are the smoothed, spiral-inserted geometry the constrained
planner emitted. ``solve_reference.py`` runs each instance in a subprocess whose
scratch directory is deleted on the way out, so that geometry does not survive
the reference run. Rather than change the reference pipeline's contract, this
script does its own solve and keeps the scratch.

Output goes to ``<MineOptimizer>/output/views/bench/`` and is NOT part of the
shipped bundle: the pages are ~500 kB each because they embed the cost cloud,
and the whole point of the repository staying ~10 MB is that it needs no
out-of-band download.
"""
from __future__ import annotations

import argparse
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from steinerbench import spec                                    # noqa: E402
from steinerbench.mineopt_adapter import (                       # noqa: E402
    _force_utf8_streams, find_mineoptimizer,
)

#: scale-8m is 8.3 M voxels and scale-129m has no reference; neither is worth a
#: browser page. scale-130k is kept even though it carries no CONSTRAINED
#: reference, because seeing why a too-coarse grid fails is the point of it.
SKIP = {"scale-8m", "scale-129m"}


def _render(iid: str, pkl: Path, out: Path, mineopt: Path) -> str:
    """Draw one page from an already-solved result. Cheap; no solver needed."""
    import numpy as np
    sys.path.insert(0, str(mineopt))
    from steinerbench.loader import load_instance
    from wp3_steiner.view_3d_html import (collect, write_html, encode_volume,
                                          encode_cloud, encode_terminals)
    with open(pkl, "rb") as fh:
        res = pickle.load(fh)
    inst = load_instance(iid)
    grid = inst["cost_grid"]
    fault = inst.get("fault_count")

    cands = collect(res, grid)
    n_ok = sum(1 for c in cands if not c["infeasible"])
    cell = float(res["meta"]["cell_size"])
    note = ("" if spec.constrained_admissible(cell) else
            "  ·  NOT constrained-admissible at this cell size "
            "(R_min/cell < 5): geometry shown is the unconstrained estimate")
    write_html(cands, out, f"SteinerMineBench — {iid}",
               f"{n_ok} of {len(cands)} topologies buildable  ·  "
               f"cell {cell:g} m  ·  grade ≤ 20 %, R ≥ 25 m{note}",
               volume=encode_volume(grid, fault, res["meta"]),
               cloud=encode_cloud(grid, res["meta"]),
               terminals=encode_terminals(res, res["meta"]))
    return f"{n_ok}/{len(cands)} buildable  ({out.stat().st_size / 1024:.0f} kB)"


def render_one(iid: str, out_dir: Path, mineopt: Path, force: bool,
               resolve: bool = False, use_gpu: bool = False) -> str:
    out = out_dir / f"{iid}.html"

    # The solve is the expensive part and the render is nearly free, so the
    # solved result is cached beside the page. A change to the VIEWER -- and
    # there has already been one that altered the buildable count -- then costs
    # a re-render rather than a re-solve of the whole suite.
    cache = out_dir / "_solved" / f"{iid}.pkl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and not resolve:
        if out.exists() and not force:
            return "skipped (exists)"
        return _render(iid, cache, out, mineopt) + "  [cached solve]"

    scratch = Path(tempfile.mkdtemp(prefix=f"view3d-{iid}-"))
    try:
        # Solve in a subprocess: config.CELL_SIZE is frozen at import, so one
        # process cannot serve instances at different resolutions.
        code = (
            "import sys, pickle;"
            f"sys.path.insert(0, r'{ROOT}');"
            f"sys.path.insert(0, r'{mineopt}');"
            "from steinerbench.mineopt_adapter import run_instance, _force_utf8_streams;"
            "_force_utf8_streams();"
            f"h = run_instance(r'{iid}', __import__('pathlib').Path(r'{scratch}'),"
            f" __import__('pathlib').Path(r'{mineopt}'));"
            "print('SOLVED')"
        )
        # CPU by default, matching solve_reference.py, because these pages
        # DISPLAY buildability verdicts and family costs that a reader will
        # compare against results_*.csv.  The GPU path is ~1.9x faster but its
        # lattice planner disagrees: generated on GPU, all four haul-* pages
        # reported "0/12 buildable" while their CPU references carry 6/6
        # verified families and a constrained cost.  A page that contradicts the
        # published table is worse than a slow one.  --gpu opts in anyway for a
        # quick look, and then the page is NOT consistent with the references.
        env = dict(os.environ)
        if use_gpu:
            env["MINEOPT_BENCH_GPU"] = "1"
        p = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                           capture_output=True, text=True, env=env,
                           encoding="utf-8", errors="replace")
        pkl = scratch / "wp3_poly_steiner_results.pkl"
        if not pkl.exists():
            tail = "\n".join((p.stdout + p.stderr).splitlines()[-6:])
            return f"FAILED\n        {tail}"
        shutil.copy2(pkl, cache)
        return _render(iid, cache, out, mineopt)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def write_index(out_dir: Path, rows):
    """A plain contents page, so the set is navigable without a file browser."""
    items = "\n".join(
        f'  <li><a href="{iid}.html">{iid}</a>'
        f'<span class="g">{grp}</span><span class="s">{status}</span></li>'
        for iid, grp, status in rows)
    (out_dir / "index.html").write_text(f"""<!doctype html>
<meta charset="utf-8"><title>SteinerMineBench — 3-D views</title>
<style>
:root{{color-scheme:light dark}}
body{{font:14px/1.6 ui-sans-serif,system-ui,sans-serif;max-width:52rem;
  margin:3rem auto;padding:0 1.5rem}}
h1{{font-size:1.15rem;letter-spacing:.01em}}
p{{color:#6b7280;font-size:.9rem}}
ul{{list-style:none;padding:0;display:grid;gap:.15rem}}
li{{display:grid;grid-template-columns:1fr auto auto;gap:1rem;
  padding:.45rem .6rem;border-radius:6px}}
li:nth-child(odd){{background:color-mix(in srgb,currentColor 5%,transparent)}}
a{{text-decoration:none;font-weight:600}}
.g{{color:#6b7280;font-size:.82rem}}
.s{{font-variant-numeric:tabular-nums;font-size:.82rem;color:#6b7280;
  min-width:11rem;text-align:right}}
</style>
<h1>SteinerMineBench — ramp networks in 3-D</h1>
<p>One page per instance. Every topology family the solver evaluated is
selectable from the dropdown, drawn against that instance's cost field, with
the verifier's verdict and each violation marked on the alignment. Orbit with
drag, zoom with the wheel, pan with shift-drag.</p>
<ul>
{items}
</ul>
""", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="+", metavar="ID")
    ap.add_argument("--force", action="store_true",
                    help="re-render pages that already exist (uses the cached "
                         "solve, so this is fast)")
    ap.add_argument("--resolve", action="store_true",
                    help="discard the cached solve and run the solver again")
    ap.add_argument("--gpu", action="store_true",
                    help="solve on the GPU (~1.9x faster). NOT consistent with "
                         "the references: the GPU lattice planner disagrees, "
                         "and every haul-* page then reports 0/12 buildable "
                         "against a reference with 6/6 verified. Quick looks "
                         "only, never for a published bundle.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    _force_utf8_streams()
    mineopt = find_mineoptimizer()
    out_dir = Path(a.out) if a.out else mineopt / "output" / "views" / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = a.only or [i for i in spec.INSTANCE_IDS if i not in SKIP]
    rows = []
    for n, iid in enumerate(ids, 1):
        print(f"[{n}/{len(ids)}] {iid} ... ", end="", flush=True)
        status = render_one(iid, out_dir, mineopt, a.force, a.resolve, a.gpu)
        print(status)
        rows.append((iid, spec.get(iid)["group"], status))

    write_index(out_dir, rows)
    print(f"\nwrote {len(rows)} view(s) to {out_dir}")
    print(f"  open {out_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
