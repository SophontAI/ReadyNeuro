"""PPMI (Parkinson's) diagnosis-eval discovery over the fMRIPrep derivatives.

The curated cohort, its splits and its first-run BOLD paths come from the benchmark suite's
metadata; the derivatives themselves are on a private mirror (PPMI is access-controlled).
"""

from __future__ import annotations

import csv
import io
import json
import sys
from dataclasses import dataclass
from functools import lru_cache, partial

from readyneuro.io import BENCHMARK_METADATA_ROOT, fetch, is_s3, path_exists, s3_filesystem


# --- locations --------------------------------------------------------------
DERIV_ROOT = "s3://<EVAL_ROOT>/PPMI/fmriprep"
_META = f"{BENCHMARK_METADATA_ROOT}/PPMI/metadata"

# The temporal contract (target_tr / common_tr / window_frames) lives in dataset_configs/ppmi*.yaml
# and is read there by `finalize_eval` -- never re-declared here, because a second copy drifts.

# fMRIPrep derivative suffixes (the curated path always ends in ``_bold.nii.gz``).
_BOLD_MNI = "_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
_SIDECAR = "_space-MNI152NLin6Asym_res-2_desc-preproc_bold.json"  # RepetitionTime lives here


# --- metadata (pinned Brainmarks commit on GitHub; $BRAINMARKS_METADATA_URL overrides) ---
#: The benchmark-metadata fetcher (local stage + 429 backoff), bound to PPMI's metadata dir.
_fetch = partial(fetch, _META)


@lru_cache
def curated_split() -> dict[str, str]:
    """``sub -> split`` from ``PPMI_curated.csv`` (columns ``Subject``, ``split``)."""
    rows = csv.DictReader(io.StringIO(_fetch("PPMI_curated.csv").decode()))
    return {r["Subject"].strip(): r["split"].strip() for r in rows}


@lru_cache
def dx_map() -> dict[str, str]:
    """``sub -> "PD"|"Prodromal"`` from ``targets/ppmi_target_map_dx.json`` -- the eval label. The
    curated cohort has no healthy controls, so this is PD vs Prodromal, not PD vs HC."""
    return {str(k): str(v) for k, v in json.loads(_fetch("targets/ppmi_target_map_dx.json").decode()).items()}


@lru_cache
def first_bold_path() -> dict[str, str]:
    """``sub -> first bold rel-path`` from ``PPMI_curated_paths.txt``.

    The file lists every usable run (multiple sessions/subject); the benchmark keeps the FIRST run
    per subject, so we take the first occurrence in file order. Rel-path form:
    ``sub-XXXX/ses-YYYYMMDD/func/sub-XXXX_..._dir-<PE>_bold.nii.gz``."""
    out: dict[str, str] = {}
    for line in _fetch("PPMI_curated_paths.txt").decode().splitlines():
        line = line.strip()
        if not line:
            continue
        sub = line.split("/", 1)[0].removeprefix("sub-")  # sub-999001/... -> 999001
        out.setdefault(sub, line)
    return out


# --- work items -------------------------------------------------------------
@dataclass(frozen=True)
class PPMIWork:
    """One subject's eval unit: bare ``sub`` id, its ``split``, and the first-run BOLD
    rel-path (rooted at ``DERIV_ROOT``)."""

    sub: str
    split: str
    bold_rel: str


def discover_work(
    *, only_subjects: list[str] | None = None, n_subjects: int | None = None
) -> list[PPMIWork]:
    """All curated subjects paired to (split, first-run bold path). ``--subjects`` filter +
    ``--n-subjects`` cap. Curated subjects with no listed path are dropped (logged nowhere;
    they don't occur in practice -- the curated CSV and paths file share the cohort)."""
    split_of = curated_split()
    paths = first_bold_path()
    subs = sorted(split_of)
    if only_subjects:
        keep = {str(x) for x in only_subjects}  # str: a YAML-parsed numeric id (subjects=[999001]) still matches
        subs = [s for s in subs if s in keep]
    work = [
        PPMIWork(sub=s, split=split_of[s], bold_rel=paths[s]) for s in subs if s in paths
    ]
    return work[:n_subjects] if n_subjects else work


# --- derivative URIs (bold by suffix-swap; anat at subject level) -----------
def _swap(bold_rel: str, suffix: str) -> str:
    return bold_rel.replace("_bold.nii.gz", suffix)


def bold_uri(work: PPMIWork) -> str:
    """The shipped MNI preproc BOLD -- the eval's frame source."""
    return f"{DERIV_ROOT}/{_swap(work.bold_rel, _BOLD_MNI)}"


def sidecar_uri(work: PPMIWork) -> str:
    """The fMRIPrep bold JSON sidecar carrying ``RepetitionTime`` (the acquisition TR)."""
    return f"{DERIV_ROOT}/{_swap(work.bold_rel, _SIDECAR)}"


_ANAT_SUFFIX = {
    "aparcaseg": "_desc-aparcaseg_dseg.nii.gz",
    "t1w": "_desc-preproc_T1w.nii.gz",
    "fwd_h5": "_from-T1w_to-MNI152NLin6Asym_mode-image_xfm.h5",
}


def _func_session(sub: str) -> str:
    """The subject's first-run func session token, e.g. ``ses-20000101`` (from the bold
    rel-path ``sub-XXXX/ses-YYYYMMDD/func/...``)."""
    return first_bold_path()[sub].split("/")[1]


def anat_stem(sub: str, storage_options: dict | None = None) -> str:
    """Resolve the subject's anat **stem prefix**, handling fMRIPrep's two anat layouts:
    subject-level ``sub-XXXX/anat/sub-XXXX`` (multi-session -> one unbiased-template anat)
    vs session-level ``sub-XXXX/ses-YYYYMMDD/anat/sub-XXXX_ses-YYYYMMDD`` (single-session).

    Probes the subject-level aparcaseg once; on miss, falls back to the first-run session
    (which, for a single-session subject, is the only session and carries the anat). Append
    a suffix with `anat_uri`."""
    subj = f"{DERIV_ROOT}/sub-{sub}/anat/sub-{sub}"
    if path_exists(f"{subj}{_ANAT_SUFFIX['aparcaseg']}", storage_options):
        return subj
    ses = _func_session(sub)
    return f"{DERIV_ROOT}/sub-{sub}/{ses}/anat/sub-{sub}_{ses}"


def anat_uri(stem: str, kind: str) -> str:
    """Append an anat/transform suffix to a resolved `anat_stem`. ``kind`` in {``aparcaseg``,
    ``t1w``, ``fwd_h5``}."""
    return f"{stem}{_ANAT_SUFFIX[kind]}"


def read_repetition_time(sidecar_uri_: str, storage_options: dict | None = None) -> float:
    """Read ``RepetitionTime`` (seconds) from a fMRIPrep bold JSON sidecar on R2/local.

    fMRIPrep's derivative NIfTI header TR can be unreliable, so the sidecar is read instead; the
    finalizer resamples FROM this TR."""
    opener = s3_filesystem(storage_options).open if is_s3(sidecar_uri_) else open
    with opener(sidecar_uri_, "rb") as f:
        meta = json.loads(f.read().decode())
    tr = meta.get("RepetitionTime")
    if tr is None:
        raise ValueError(f"{sidecar_uri_}: no RepetitionTime in sidecar")
    return float(tr)


# --- the driver contract ----------------------------------------------------
# PPMI keys its anat by SUBJECT while ABIDE/ADHD200 key theirs by (site, subject), so it cannot use
# `fmriprep_dx.discover_mni_only`; only the staged-input names are shared.
from readyneuro.builder.discover import fmriprep_dx as _dx



def subject_meta(sub: str) -> dict:
    """Best-effort PD/Prodromal diagnosis; ``{}`` if absent."""
    return _dx.dx_meta(sys.modules[__name__], sub)


def discover(cfg: dict) -> list[dict]:
    """One eval item per curated subject. The acquisition TR is read from the fMRIPrep sidecar HERE,
    into the manifest, so the core needs no remote read; ``sources.runs`` stays empty because the one
    REST run's frames come from the staged ``aux`` BOLD."""
    if list(cfg["spaces"]) != ["mni"]:
        raise ValueError(f"{__name__} builds only 'mni'; for native space use the *_native dataset (got {cfg['spaces']!r})")
    so = cfg.get("input_storage_options")
    works = discover_work(only_subjects=cfg.get("subjects"), n_subjects=cfg.get("n_subjects"))

    items: list[dict] = []
    for w in works:
        stem = anat_stem(w.sub, so)
        items.append(
            {
                "key": w.sub,
                "deliverables": [{"kind": "eval_arrow", "space": "mni", "split": w.split}],
                "sources": {
                    "t1": {"name": _dx.T1W, "uri": anat_uri(stem, "t1w")},
                    "aux": [{"name": _dx.FWD_H5, "uri": anat_uri(stem, "fwd_h5")},
                            {"name": _dx.BOLD_MNI, "uri": bold_uri(w)}],
                    "runs": [],
                    "tr_acq": read_repetition_time(sidecar_uri(w), so),
                    "meta": subject_meta(w.sub),
                },
            }
        )
    return items
