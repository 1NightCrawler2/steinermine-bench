#!/usr/bin/env python
"""
generate.py -- reproduce the entire SteinerMineBench instance set from scratch.
==============================================================================

SPDX-License-Identifier: MIT

The shipped instance set is frozen, but this generator exists so the set is
reproducible and reviewable: every array in every bundle is a deterministic
function of the seed and parameters recorded in that bundle's ``metadata.json``.

All generated data is SYNTHETIC.  It is procedurally constructed to be
geotechnically representative and is NOT derived from any operating mine.

Usage
-----
    python generate.py --all                    # rebuild every bundle
    python generate.py --only xgrid-f2x-qpoor   # rebuild one
    python generate.py --group A                # rebuild the crossing grid
    python generate.py --check                  # verify determinism, write nothing
    python generate.py --emit-q xgrid-f2x-qpoor # dump the raw Q field for inspection
    python generate.py --list                   # show the manifest and exit

``--check`` regenerates each instance into a temporary directory and compares
the SHA-256 of ``cost_grid.npz`` against the checksum recorded in the shipped
``metadata.json``.  It touches nothing under ``instances/``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from steinerbench import spec                      # noqa: E402
from steinerbench.grid import (                    # noqa: E402
    build_instance, sha256_file, write_bundle, write_cost_grid_npz,
)

ROOT = Path(__file__).resolve().parent
INSTANCES_DIR = ROOT / "instances"


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def cmd_list() -> int:
    hdr = (f"{'instance_id':<18} {'group':<14} {'cell':>5} {'voxels':>13} "
           f"{'levels':>6} {'zones':>5}  {'git':<4} varied axis")
    print(hdr)
    print("-" * len(hdr))
    for iid, s in spec.INSTANCES.items():
        d = spec.dims_for(s["cell_size_m"])
        entries = spec.zone_entries(s)
        n_zones = len(entries)
        n_lev = len({e[2] for e in entries})
        varied = ", ".join(f"{k}={v}" for k, v in s["varied_axis"]["value"].items())
        print(f"{iid:<18} {s['group']:<14} {s['cell_size_m']:>5.1f} "
              f"{d[0]*d[1]*d[2]:>13,} {n_lev:>6} {n_zones:>5}  "
              f"{'yes' if s['shipped_in_git'] else 'no':<4} {varied}")
    print(f"\n{len(spec.INSTANCES)} instances, benchmark version "
          f"{spec.BENCHMARK_VERSION}")
    return 0


def cmd_emit_q(instance_id: str, out: Path) -> int:
    print(f"Building Q field for {instance_id} ...")
    built = build_instance(instance_id, want_q=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, built["q"])
    q = built["q"][built["passable"]]
    print(f"  wrote {out}  shape={built['q'].shape}  "
          f"({_human_bytes(out.stat().st_size)})")
    print(f"  passable Q: min={q.min():.4f} p10={np.percentile(q, 10):.4f} "
          f"median={np.median(q):.4f} p90={np.percentile(q, 90):.4f} "
          f"max={q.max():.4f}")
    return 0


def cmd_generate(ids: list[str], out_dir: Path, force: bool) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    t_start = time.time()

    for n, iid in enumerate(ids, 1):
        s = spec.get(iid)
        d = spec.dims_for(s["cell_size_m"])
        n_vox = d[0] * d[1] * d[2]
        dest = out_dir / iid

        if dest.exists() and not force:
            print(f"[{n}/{len(ids)}] {iid}: exists, skipping (use --force to "
                  f"overwrite)")
            continue

        print(f"[{n}/{len(ids)}] {iid}  {d[0]}x{d[1]}x{d[2]} = {n_vox:,} voxels "
              f"@ {s['cell_size_m']:g} m ... ", end="", flush=True)
        t0 = time.time()
        built = build_instance(iid)
        meta = write_bundle(built, dest)
        size = meta["checksums"]["size_bytes"]
        total_bytes += size
        print(f"{_human_bytes(size)}  ({time.time() - t0:.1f} s)")

        if not s["shipped_in_git"]:
            print(f"        NOTE: too large for git; distribute via Zenodo. "
                  f"sha256 = {meta['checksums']['cost_grid.npz'][7:23]}...")

    print(f"\nWrote {len(ids)} instance(s), {_human_bytes(total_bytes)} total, "
          f"in {time.time() - t_start:.1f} s")
    return 0


def cmd_check(ids: list[str], out_dir: Path) -> int:
    """Regenerate into a scratch dir and diff checksums against the shipped set."""
    failures: list[str] = []
    missing: list[str] = []
    tmp = Path(tempfile.mkdtemp(prefix="steinerbench-check-"))
    print(f"Determinism check against {out_dir}\n(scratch: {tmp})\n")
    try:
        for n, iid in enumerate(ids, 1):
            meta_path = out_dir / iid / "metadata.json"
            if not meta_path.exists():
                missing.append(iid)
                print(f"[{n}/{len(ids)}] {iid}: MISSING (not generated yet)")
                continue

            shipped = json.loads(meta_path.read_text(encoding="utf-8"))
            expected = shipped["checksums"]["cost_grid.npz"]

            built = build_instance(iid)
            npz = tmp / f"{iid}.npz"
            write_cost_grid_npz(npz, built["tier_index"], built["fault_count"],
                                built["surface_rl"])
            actual = f"sha256:{sha256_file(npz)}"
            npz.unlink()

            if actual == expected:
                print(f"[{n}/{len(ids)}] {iid}: OK  {actual[7:23]}...")
            else:
                failures.append(iid)
                print(f"[{n}/{len(ids)}] {iid}: MISMATCH\n"
                      f"        expected {expected}\n        actual   {actual}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if failures:
        print(f"FAIL: {len(failures)} instance(s) did not reproduce: "
              f"{', '.join(failures)}")
        return 1
    if missing:
        print(f"INCOMPLETE: {len(missing)} instance(s) not generated: "
              f"{', '.join(missing)}")
        return 1
    print(f"OK: all {len(ids)} instance(s) reproduce bit-for-bit.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("Usage")[0].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--all", action="store_true",
                     help="generate every instance in the suite")
    sel.add_argument("--only", nargs="+", metavar="ID",
                     help="generate only these instance ids")
    sel.add_argument("--group", metavar="A|B|C|D|E",
                     help="generate one group (A crossing grid, B portal "
                          "sweep, C zone count, D scale ladder, E haulage "
                          "ratio)")
    sel.add_argument("--list", action="store_true",
                     help="print the instance manifest and exit")
    sel.add_argument("--emit-q", metavar="ID",
                     help="write the raw Barton Q field for one instance")

    p.add_argument("--check", action="store_true",
                   help="verify the shipped set reproduces; write nothing")
    p.add_argument("--force", action="store_true",
                   help="overwrite existing bundles")
    p.add_argument("--out", type=Path, default=INSTANCES_DIR,
                   help="output directory (default: ./instances)")
    p.add_argument("--skip-large", action="store_true",
                   help="skip instances not shipped in git (the 129 M rung)")
    args = p.parse_args(argv)

    if args.list:
        return cmd_list()
    if args.emit_q:
        spec.get(args.emit_q)
        return cmd_emit_q(args.emit_q, args.out / args.emit_q / "q_field.npy")

    if args.only:
        ids = spec.select(only=args.only)
    elif args.group:
        ids = spec.select(group=args.group)
    elif args.all or args.check:
        ids = spec.select()
    else:
        p.print_help()
        print("\nNothing selected. Use --all, --only, --group, --check or --list.")
        return 2

    if args.skip_large:
        ids = [i for i in ids if spec.get(i)["shipped_in_git"]]

    if args.check:
        return cmd_check(ids, args.out)
    return cmd_generate(ids, args.out, args.force)


if __name__ == "__main__":
    raise SystemExit(main())
