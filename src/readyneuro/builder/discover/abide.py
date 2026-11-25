"""ABIDE (Autism Brain Imaging Data Exchange) diagnosis-eval discovery, MNI derivatives.

The mirrored fMRIPrep derivatives are SITE-PARTITIONED (``<Site>/sub-XXXX/{func,anat}/``, anat at
subject level -- no session/run entities), and fMRIPrep saved MNI BOLD only. The eval keeps ONE REST
window per subject (the benchmark suite's first run); site TRs span 1.5-3.0 s, which the finalizer
harmonizes to a common 3.0 s band (``common_tr`` in the config) before resampling to 1.0 s.

The Autism/Control label is joined consumer-side, so discovery needs only
``sub -> (split, site, bold path)``.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from functools import lru_cache, partial

from readyneuro.io import BENCHMARK_METADATA_ROOT, fetch, is_s3, s3_filesystem


# --- locations --------------------------------------------------------------
DERIV_ROOT = "s3://<EVAL_ROOT>/ABIDE/fmriprep"
_META = f"{BENCHMARK_METADATA_ROOT}/ABIDE/metadata"

# The temporal contract (target_tr / common_tr / window_frames) lives in dataset_configs/abide*.yaml
# and is read there by `finalize_eval` -- never re-declared here, because a second copy drifts.

# fMRIPrep derivative suffixes (the curated path always ends in ``_bold.nii.gz``).
_BOLD_MNI = "_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
_SIDECAR = "_space-MNI152NLin6Asym_res-2_desc-preproc_bold.json"  # RepetitionTime lives here

#: The shipped ``_desc-aparcaseg_dseg`` is deliberately NOT listed: it is FreeSurfer DK, where every
#: dataset here carries SynthSeg DKT. One segmenter across the corpus.
_ANAT_SUFFIX = {
    "t1w": "_desc-preproc_T1w.nii.gz",
    "fwd_h5": "_from-T1w_to-MNI152NLin6Asym_mode-image_xfm.h5",
}


# --- metadata (pinned Brainmarks commit on GitHub; $BRAINMARKS_METADATA_URL overrides) ---
#: The benchmark-metadata fetcher (local stage + 429 backoff), bound to ABIDE's metadata dir.
_fetch = partial(fetch, _META)


@lru_cache
def curated_split() -> dict[str, str]:
    """``sub -> split`` from ``ABIDE_curated.csv`` (columns include ``sub``, ``split``).
    ``sub`` is a zero-padded string id (``0050004``); read as str, never int-cast."""
    rows = csv.DictReader(io.StringIO(_fetch("ABIDE_curated.csv").decode()))
    return {r["sub"].strip(): r["split"].strip() for r in rows}


@lru_cache
def dx_map() -> dict[str, str]:
    """``sub -> "Autism"|"Control"`` from ``targets/abide_target_map_dx.json`` -- the eval label,
    joined consumer-side; carried here only for `subject_meta`."""
    raw = json.loads(_fetch("targets/abide_target_map_dx.json").decode())
    return {str(k): str(v) for k, v in raw.items()}


@lru_cache
def first_bold_paths() -> list[str]:
    """The curated first-run BOLD rel-paths (one per subject), site-prefixed:
    ``<Site>/sub-XXXX/func/sub-XXXX_task-rest_run-1_bold.nii.gz``."""
    return [
        line.strip()
        for line in _fetch("ABIDE_curated_paths.txt").decode().splitlines()
        if line.strip()
    ]


# --- work items -------------------------------------------------------------
@dataclass(frozen=True)
class AbideWork:
    """One subject's eval unit: bare ``sub`` id, its ``split``, the ``site`` it was scanned at
    (the top path component -- fMRIPrep is laid out ``<Site>/sub-XXXX/...``), and the
    first-run BOLD rel-path (rooted at ``DERIV_ROOT``, ending ``_bold.nii.gz``)."""

    sub: str
    split: str
    site: str
    bold_rel: str


def discover_work(
    *, only_subjects: list[str] | None = None, n_subjects: int | None = None
) -> list[AbideWork]:
    """All curated subjects paired to (split, site, first-run bold path). ``sub`` and
    ``site`` are parsed from each path (``<Site>/sub-XXXX/...``: site is component 0, sub is
    component 1 after ``sub-``). A path whose subject is not in the curated split is dropped
    (they share the cohort in practice)."""
    split_of = curated_split()
    keep = {str(x) for x in only_subjects} if only_subjects else None  # str: match a YAML-parsed numeric id
    work: list[AbideWork] = []
    for rel in first_bold_paths():
        parts = rel.split("/")
        site, sub = parts[0], parts[1].removeprefix("sub-")
        if sub not in split_of:
            continue
        if keep is not None and sub not in keep:
            continue
        work.append(AbideWork(sub=sub, split=split_of[sub], site=site, bold_rel=rel))
    work.sort(key=lambda w: w.sub)
    return work[:n_subjects] if n_subjects else work


# --- derivative URIs --------------------------------------------------------
def _swap(bold_rel: str, suffix: str) -> str:
    return bold_rel.replace("_bold.nii.gz", suffix)


def bold_uri(work: AbideWork) -> str:
    """The staged MNI preproc BOLD -- the eval's frame source (fMRIPrep saved no ``space-T1w``
    run for this cohort)."""
    return f"{DERIV_ROOT}/{_swap(work.bold_rel, _BOLD_MNI)}"


def sidecar_uri(work: AbideWork) -> str:
    """The MNI bold JSON sidecar carrying ``RepetitionTime`` (the acquisition TR)."""
    return f"{DERIV_ROOT}/{_swap(work.bold_rel, _SIDECAR)}"


def anat_stem(work: AbideWork, storage_options: dict | None = None) -> str:
    """ABIDE anat is SUBJECT-level: ``<Site>/sub-XXXX/anat/sub-XXXX`` (no ses-/run- entities).
    ``storage_options`` is accepted for interface symmetry with ADHD200, whose stem must be listed."""
    return f"{DERIV_ROOT}/{work.site}/sub-{work.sub}/anat/sub-{work.sub}"


def anat_uri(stem: str, kind: str) -> str:
    """Append an anat/transform suffix to a resolved `anat_stem`. ``kind`` in {``t1w``, ``fwd_h5``}."""
    return f"{stem}{_ANAT_SUFFIX[kind]}"


def read_repetition_time(sidecar_uri_: str, storage_options: dict | None = None) -> float:
    """Read ``RepetitionTime`` (seconds) from a fMRIPrep bold JSON sidecar (the derivative NIfTI
    header TR is unreliable, so the sidecar is the source)."""
    opener = s3_filesystem(storage_options).open if is_s3(sidecar_uri_) else open
    with opener(sidecar_uri_, "rb") as f:
        meta = json.loads(f.read().decode())
    tr = meta.get("RepetitionTime")
    if tr is None:
        raise ValueError(f"{sidecar_uri_}: no RepetitionTime in sidecar")
    return float(tr)


# --- the driver contract, via the shared fMRIPrep wiring --------------------
import sys

from readyneuro.builder.discover import fmriprep_dx as _dxd



def subject_meta(sub: str) -> dict:
    """Best-effort Autism/Control diagnosis; {} if absent."""
    return _dxd.dx_meta(sys.modules[__name__], sub)


def discover(cfg: dict) -> list[dict]:
    return _dxd.discover_mni_only(cfg, sys.modules[__name__])
