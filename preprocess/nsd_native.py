#!/usr/bin/env python3
"""NSD native-from-raw core: the shared from-raw kernel driven over NSD's raw BIDS, STREAMING each
run -- an NSD subject has hundreds of ~235 MB raw runs, so nothing is pre-staged.

Reuses ``openneuro`` verbatim for the anat and the per-run geometry, adding two NSD specifics:
(1) stream the raw run to a temp file before the kernel's run stage (mcflirt needs a filename);
(2) stamp ``ses``/``run_num`` into the run's sidecar entry, so the cococlip planner can look up its
COCO trial onsets. The acquisition TR is the RAW 1.6 s, which the finalizer resamples to 1.0 s.

Standalone-but-composing: imports nothing from the package, and sibling-imports ``openneuro`` (also
package-free) -- legal because the driver invokes these scripts BY PATH.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openneuro  # noqa: E402  -- the shared from-raw kernel


def preflight(cfg=None) -> None:
    openneuro.preflight(cfg)


def _stream_to(uri: str, dest: Path, storage_options: dict | None) -> None:
    """Copy an s3:// object to a local file (mcflirt needs a filename; NSD raw is not pre-staged)."""
    import s3fs

    fs = s3fs.S3FileSystem(**(storage_options or {}))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with fs.open(uri, "rb", block_size=16 * 2**20) as src, open(dest, "wb") as out:
        shutil.copyfileobj(src, out, 16 * 2**20)


def _anat(item, raw_dir, out_bids, cfg) -> None:
    work = Path(raw_dir) / "work"
    t1_raw = Path(raw_dir) / item["sources"]["t1"]["name"]
    openneuro._anat(t1_raw, out_bids, work, item["key"])


def _one_run(item, run_idx, raw_dir, out_bids, cfg) -> None:
    work = Path(raw_dir) / "work"
    r = next(x for x in item["sources"]["eval_runs"] if int(x["idx"]) == int(run_idx))
    tmp = work / f"raw_run_{r['idx']:05d}.nii.gz"
    _stream_to(r["uri"], tmp, cfg.get("input_storage_options"))
    try:
        spec = {"path": tmp, "idx": r["idx"], "run": r["run"], "tr_acq": r["tr_acq"]}
        openneuro._one_run(spec, out_bids, work, item["key"])
    finally:
        tmp.unlink(missing_ok=True)  # free disk immediately -- hundreds of runs per subject
    # Stamp ses/run_num into the run entry the kernel wrote: the cococlip eval_clips planner keys on them.
    p = work / f"run_{r['idx']:05d}.json"
    entry = json.loads(p.read_text())
    entry["ses"], entry["run_num"] = int(r["ses"]), int(r["run_num"])
    p.write_text(json.dumps(entry))


def _finish(item, raw_dir, out_bids, cfg) -> None:
    work = Path(raw_dir) / "work"
    t1_raw = Path(raw_dir) / item["sources"]["t1"]["name"]
    openneuro._finish(t1_raw, out_bids, work, item["key"], item["sources"].get("meta", {}))


def process(item: dict, raw_dir: str, out_bids: str, cfg: dict, stage="all", run_idx=None) -> None:
    if stage == "anat":
        _anat(item, raw_dir, out_bids, cfg)
    elif stage == "run":
        _one_run(item, run_idx, raw_dir, out_bids, cfg)
    elif stage == "finish":
        _finish(item, raw_dir, out_bids, cfg)
    else:
        _anat(item, raw_dir, out_bids, cfg)
        for r in item["sources"]["eval_runs"]:
            try:
                _one_run(item, r["idx"], raw_dir, out_bids, cfg)
            except Exception as exc:
                print(f"WARN nsd_native {item['key']} run {r['idx']} failed ({exc}) -- skipping run")
        _finish(item, raw_dir, out_bids, cfg)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="NSD native-from-raw core (streaming)")
    ap.add_argument("--item", default=None)
    ap.add_argument("raw", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--cfg", default=None)
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--stage", choices=("all", "anat", "run", "finish"), default="all")
    ap.add_argument("--run-idx", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = json.loads(Path(args.cfg).read_text()) if args.cfg else {}
    if args.preflight:
        preflight(cfg)
        return
    if args.stage == "run" and args.run_idx is None:
        ap.error("--stage run needs --run-idx")
    process(json.loads(Path(args.item).read_text()), args.raw, args.out, cfg,
            stage=args.stage, run_idx=args.run_idx)


if __name__ == "__main__":
    main()
