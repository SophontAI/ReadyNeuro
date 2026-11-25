#!/usr/bin/env python3
"""Real end-to-end smoke of ONE dataset core, in isolation from the driver's build loop.

    python scripts/native_smoke.py --config dataset_configs/<ds>-native.yaml --subject <key> \
        [--set key=value ...] [--runs a,b] [--expect-runs N] [--expect-tr 1.6] [--scratch DIR]

discover -> download_item -> ``--preflight`` -> the core's ``--item/--cfg`` subprocess CLI
(stage=all) -> assert the intermediate BIDS is well formed. Unlike a driver run it does NOT evict
the per-subject scratch, so a failure can be attributed between the core and the finalizer.

Covers the from-raw kernel family only: it asserts ``spaces == ["native"]``, so it does not
validate the vendor/mni cores. ``--set`` mirrors the driver's override loop, which is what makes
this usable on the expensive datasets -- NSD ships uncapped, so a real NSD smoke needs
``--set eval_max_runs=2``.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

# Anchor on THIS file, not the CWD, so the smoke runs from anywhere.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from readyneuro.builder import driver, wrappers  # noqa: E402
from readyneuro.io import INDEX_SUFFIX  # noqa: E402

ap = argparse.ArgumentParser(description="real end-to-end smoke of one native-space dataset core")
ap.add_argument("--config", required=True)
ap.add_argument("--subject", required=True)
ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                help="override a config key (VALUE parsed as YAML); repeatable, as in driver.main")
ap.add_argument("--runs", default=None, help="comma-separated run restriction")
ap.add_argument("--expect-runs", type=int, default=None, help="require exactly N runs in the sidecar")
ap.add_argument("--expect-tr", type=float, default=None,
                help="require every run's sidecar tr_acq to equal this ACQUISITION TR. For NSD "
                     "native that is the raw 1.6 s -- the vendor product is already 1.0 s, and "
                     "confusing the two is a silent wrong-TR corruption")
ap.add_argument("--expect-ses", action="store_true",
                help="require every run to carry a non-null `ses` (NSD: the COCO clip planner keys "
                     "trial onsets off it)")
ap.add_argument("--scratch", default="/tmp/native_smoke")
a = ap.parse_args()


def log(*x):
    print(f"[{time.strftime('%H:%M:%S')}]", *x, flush=True)


cfg = yaml.safe_load(Path(a.config).read_text())
for kv in a.set:
    k, v = kv.split("=", 1)
    cfg[k] = yaml.safe_load(v)
cfg["subjects"] = [a.subject]
if a.runs:
    cfg["runs"] = a.runs.split(",")
driver.resolve_storage_options(cfg)

core = wrappers.load_core(cfg["dataset"])
CORE = ROOT / "preprocess" / f"{core.preproc}.py"
log(f"dataset={cfg['dataset']} preproc={core.preproc} staged={core.staged}")
log(f"core script: {CORE} {'(FOUND)' if CORE.exists() else '(MISSING!)'}")

item = core.discover(cfg)[0]
key = item["key"]
src = item["sources"]
# `runs` is what the driver PRE-STAGES; `eval_runs` is what a core STREAMS itself (NSD -- hundreds
# of runs per subject, pre-staging them would be hundreds of GB). Report whichever is populated.
n_runs = len(src["runs"]) or len(src.get("eval_runs") or ())
log("item key:", key, "| deliverables:", [(d["kind"], d["space"]) for d in item["deliverables"]],
    "| runs:", n_runs)

SCRATCH = Path(a.scratch) / cfg["dataset"]
raw_root, out_bids = SCRATCH / "scratch", SCRATCH / "out" / f"{key}dir"
raw_root.mkdir(parents=True, exist_ok=True)
out_bids.mkdir(parents=True, exist_ok=True)
log("download_item (raw T1 + BOLD run(s)) ...")
t0 = time.time()
driver.download_item(item, str(raw_root), cfg)
raw_dir = raw_root / key / "raw"
log("staged in %.0fs:" % (time.time() - t0), sorted(p.name for p in raw_dir.iterdir()))

ip, cp = SCRATCH / "item.json", SCRATCH / "cfg.json"
ip.write_text(json.dumps(item, default=str))
cp.write_text(json.dumps(cfg, default=str))
pf = subprocess.run([sys.executable, str(CORE), "--preflight", "--cfg", str(cp)],
                    capture_output=True, text=True)
assert pf.returncode == 0, f"preflight failed: {pf.stderr[-600:]}"

log("running core stage=all ...")
t0 = time.time()
proc = subprocess.run([sys.executable, str(CORE), str(raw_dir), str(out_bids),
                       "--item", str(ip), "--cfg", str(cp)], capture_output=True, text=True)
log("core rc %d in %.0fs" % (proc.returncode, time.time() - t0))
if proc.stdout.strip():
    # A core can exit 0 having DROPPED a run (it prints WARN and continues), so this is not noise.
    log("STDOUT tail:\n" + proc.stdout[-1500:])
if proc.returncode != 0:
    log("STDERR tail:\n" + proc.stderr[-4000:])
    sys.exit(1)

# The kernel's own label rule (preprocess/openneuro.py `sanitize_label`), inlined rather than
# imported so this stays a plain consumer of the core's CLI contract.
label = re.sub(r"[^A-Za-z0-9]", "", key)
sub = out_bids / f"sub-{label}"
anat = sub / "anat"
b = np.asanyarray(nib.load(anat / f"sub-{label}_space-native_desc-brain_mask.nii.gz").dataobj) > 0
g = np.asanyarray(nib.load(anat / f"sub-{label}_space-native_label-GM_mask.nii.gz").dataobj) > 0
c = np.asanyarray(nib.load(anat / f"sub-{label}_space-native_label-corticalGM_mask.nii.gz").dataobj) > 0
t1 = nib.load(anat / f"sub-{label}_space-native_desc-preproc_T1w.nii.gz")
nested = (g & ~b).sum() == 0 and (c & ~g).sum() == 0
log(f"native T1 {t1.shape} | brain {int(b.sum())} gm {int(g.sum())} cort {int(c.sum())} nested={nested}")

idx = json.loads((sub / f"sub-{label}_{INDEX_SUFFIX}").read_text())
log("sidecar spaces:", idx["spaces"],
    "| runs:", [(r["idx"], r.get("run"), r["tr_acq"]) for r in idx["runs"]])

runs_ok = len(idx["runs"]) > 0
for r in idx["runs"]:
    stem = f"sub-{label}_run-{r['idx']:02d}_space-native"
    bold = nib.load(sub / "func" / f"{stem}_desc-preproc_bold.nii")
    fg = np.asanyarray(nib.load(sub / "func" / f"{stem}_label-GM_mask.nii.gz").dataobj) > 0
    log(f"  run {r['idx']} ({r.get('run')}): ses={r.get('ses')} tr_acq={r['tr_acq']} "
        f"BOLD {bold.shape} func_gm {int(fg.sum())}")
    runs_ok = runs_ok and bold.ndim == 4 and int(fg.sum()) > 500
    if a.expect_tr is not None and r["tr_acq"] != a.expect_tr:
        log(f"  FAIL: run {r['idx']} tr_acq={r['tr_acq']}, expected {a.expect_tr}")
        runs_ok = False
    if a.expect_ses and r.get("ses") is None:
        log(f"  FAIL: run {r['idx']} has no `ses`")
        runs_ok = False

if a.expect_runs is not None and len(idx["runs"]) != a.expect_runs:
    log(f"FAIL: sidecar has {len(idx['runs'])} runs, expected {a.expect_runs}")
    runs_ok = False

ok = int(b.sum()) > 100_000 and nested and idx["spaces"] == ["native"] and runs_ok
log("SMOKE RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
