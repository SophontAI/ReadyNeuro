#!/usr/bin/env bash
# Real end-to-end smoke of the OpenNeuro preprocessing on ONE downloaded subject.
#
#   source scripts/setup_preproc.sh                 # env + FreeSurfer license (once)
#   scripts/smoke_openneuro.sh [ACCESSION] [SUB] [OUT_DIR]
#
# Anonymously downloads a subject's T1 + first BOLD (plus its json sidecar, any dataset-level
# *_bold.json for the TR, and participants.tsv) from s3://openneuro.org into a temp raw-BIDS tree,
# runs preprocess/openneuro.py on it (real N4 -> SynthSeg -> rigid-register -> resample), and lists
# the output. Needs the setup env (FS license!) and FMRI_PYTHON pointing at the project venv.
set -euo pipefail

ACC="${1:-ds000102}"
SUB="${2:-sub-01}"
OUT="${3:-$(mktemp -d)/smoke_out}"
PY="${FMRI_PYTHON:-python3}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$(mktemp -d)/$ACC"

echo "== downloading $ACC/$SUB from s3://openneuro.org (anon) =="
"$PY" - "$ACC" "$SUB" "$RAW" <<'EOF'
import s3fs, sys
from pathlib import Path
acc, sub, raw = sys.argv[1], sys.argv[2], Path(sys.argv[3])
fs = s3fs.S3FileSystem(anon=True)
base = f"openneuro.org/{acc}/{sub}"
def pick(pat):
    hits = sorted(k for k in fs.glob(f"{base}/**/{pat}")
                  if "/derivatives/" not in k and "/sourcedata/" not in k)
    return hits[0] if hits else None
t1, bold = pick("*_T1w.nii*"), pick("*_bold.nii*")
assert t1 and bold, f"no T1w/bold found under s3://{base}"
def get(key, rel):
    dst = raw / rel; dst.parent.mkdir(parents=True, exist_ok=True); fs.get(key, str(dst))
get(t1, f"{sub}/anat/{t1.split('/')[-1]}")
get(bold, f"{sub}/func/{bold.split('/')[-1]}")
sidecar = bold.replace(".nii.gz", ".json").replace(".nii", ".json")
if fs.exists(sidecar):
    get(sidecar, f"{sub}/func/{sidecar.split('/')[-1]}")
for j in fs.glob(f"openneuro.org/{acc}/*_bold.json"):   # dataset-level TR (BIDS inheritance)
    fs.get(j, str(raw / j.split('/')[-1]))
if fs.exists(f"openneuro.org/{acc}/participants.tsv"):
    fs.get(f"openneuro.org/{acc}/participants.tsv", str(raw / "participants.tsv"))
print("  downloaded T1:", t1.split('/')[-1], "| bold:", bold.split('/')[-1])
EOF

echo "== running preprocess/openneuro.py -> $OUT =="
"$PY" "$ROOT/preprocess/openneuro.py" "$RAW/$SUB" "$OUT" --accession "$ACC"

echo "== output BIDS tree =="
find "$OUT" -type f | sort
echo "== smoke OK =="
