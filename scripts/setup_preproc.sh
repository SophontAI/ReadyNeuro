#!/usr/bin/env bash
# Environment setup + verification for the standalone preprocessing scripts (preprocess/*.py).
# They shell out to FreeSurfer (mri_synthseg / mri_synthmorph / mri_convert / mri_mask / lta_convert),
# FSL (mcflirt) and ANTs (N4BiasFieldCorrection), and fetch the MNI template via templateflow.
#
#   source scripts/setup_preproc.sh   # export the env vars into your shell + verify (recommended)
#   bash   scripts/setup_preproc.sh   # verify only (a subshell cannot export into your shell)
#
# It derives FREESURFER_HOME / FSLDIR from the tools already on PATH, finds a FreeSurfer license,
# sets FSLOUTPUTTYPE + TEMPLATEFLOW_HOME, prefetches the rigid-registration target template, and
# prints a ready / not-ready summary. Point FMRI_PYTHON at a python with the deps.
set -u

_root_of() { command -v "$1" >/dev/null 2>&1 && dirname "$(dirname "$(command -v "$1")")"; }

command -v mri_synthseg >/dev/null 2>&1 && export FREESURFER_HOME="${FREESURFER_HOME:-$(_root_of mri_synthseg)}"
command -v mcflirt      >/dev/null 2>&1 && export FSLDIR="${FSLDIR:-$(_root_of mcflirt)}"
export FSLOUTPUTTYPE="${FSLOUTPUTTYPE:-NIFTI_GZ}"
export TEMPLATEFLOW_HOME="${TEMPLATEFLOW_HOME:-$HOME/.cache/templateflow}"
if [ -z "${FS_LICENSE:-}" ]; then
  for c in "${FREESURFER_HOME:-/nonexistent}/license.txt" "${FREESURFER_HOME:-/nonexistent}/.license" \
           "$HOME/license.txt" "$HOME/.license"; do
    [ -f "$c" ] && export FS_LICENSE="$c" && break
  done
fi

ok=1
echo "== preprocessing environment =="
for t in N4BiasFieldCorrection mri_synthseg mri_synthmorph mri_convert mri_mask lta_convert mcflirt; do
  if command -v "$t" >/dev/null 2>&1; then printf "  ok    %-22s %s\n" "$t" "$(command -v "$t")"
  else printf "  MISS  %-22s not on PATH\n" "$t"; ok=0; fi
done
printf "  FREESURFER_HOME   = %s\n" "${FREESURFER_HOME:-<unset>}"
printf "  FSLDIR            = %s\n" "${FSLDIR:-<unset>}"
printf "  FSLOUTPUTTYPE     = %s\n" "$FSLOUTPUTTYPE"
printf "  TEMPLATEFLOW_HOME = %s\n" "$TEMPLATEFLOW_HOME"
if [ -n "${FS_LICENSE:-}" ] && [ -f "${FS_LICENSE:-}" ]; then
  printf "  ok    FS_LICENSE           %s\n" "$FS_LICENSE"
else
  printf "  MISS  FreeSurfer license -- mri_synthseg REQUIRES one.\n"
  printf "        Register (free): https://surfer.nmr.mgh.harvard.edu/registration.html\n"
  printf "        Save it to \$FREESURFER_HOME/license.txt (or set FS_LICENSE), then re-source.\n"
  ok=0
fi

PY="${FMRI_PYTHON:-python3}"
if "$PY" - <<'EOF' 2>/dev/null
import importlib.util as u, sys
sys.exit(1 if [m for m in ("nibabel","numpy","nitransforms","scipy","templateflow") if u.find_spec(m) is None] else 0)
EOF
then
  printf "  ok    python deps (%s)\n" "$PY"
  printf "        prefetching MNI152NLin2009cAsym (res-1 T1w + brain) into TEMPLATEFLOW_HOME ...\n"
  TEMPLATEFLOW_HOME="$TEMPLATEFLOW_HOME" "$PY" - <<'EOF' && printf "  ok    template cached\n" || { printf "  MISS  template prefetch failed (network?)\n"; ok=0; }
import templateflow.api as tf
for desc in (None, "brain"):
    tf.get("MNI152NLin2009cAsym", resolution=1, suffix="T1w", desc=desc)
EOF
else
  printf "  MISS  python deps -- install the project (pip install '.[build]') or set FMRI_PYTHON to it\n"; ok=0
fi

[ "$ok" = 1 ] && echo "== ready ==" || { echo "== NOT ready: fix the MISS lines above =="; return 1 2>/dev/null || exit 1; }
