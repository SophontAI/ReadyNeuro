"""ADHD-200 diagnosis-eval discovery, MNI derivatives.

The mirrored fMRIPrep derivatives are site-partitioned AND sessioned
(``<Site>/sub-XXXX/ses-1/{func,anat}/``); like ABIDE, fMRIPrep saved MNI BOLD only. Two quirks:

  * **All metadata is in one CSV.** ``ADHD200_curated.csv`` carries ``sub``/``split``/``site`` and
    the site-prefixed ``path`` column directly -- no separate paths file and no target-map JSON (the
    ``dx`` column IS the label). Curation keeps the first run only.
  * **anat filenames carry ``ses-``/``run-`` entities** that are not derivable from the func stem,
    so `anat_stem` LISTS the anat dir and strips a known suffix instead of string-swapping.

Site TRs span 1.96-2.5 s and the config harmonizes to a common band before the 1.0 s resample. No
motion/FD filter is applied: the curated set is used as-is (ABIDE arrives FD-filtered upstream).
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from functools import lru_cache, partial

from readyneuro.io import BENCHMARK_METADATA_ROOT, fetch, is_s3, s3_filesystem


# --- locations --------------------------------------------------------------
DERIV_ROOT = "s3://<EVAL_ROOT>/ADHD200/fmriprep"
_META = f"{BENCHMARK_METADATA_ROOT}/ADHD200/metadata"

# The temporal contract lives in dataset_configs/adhd200*.yaml and is read there by `finalize_eval`
# -- never re-declared here, because a second copy drifts.

_BOLD_MNI = "_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
_SIDECAR = "_space-MNI152NLin6Asym_res-2_desc-preproc_bold.json"

#: ``aparcaseg`` is not a segmentation source (one segmenter across the corpus: SynthSeg DKT); its
#: suffix survives only as the `anat_stem` listing probe, being the one anat file guaranteed present
#: to read the ses-/run- entities off.
_ANAT_SUFFIX = {
    "aparcaseg": "_desc-aparcaseg_dseg.nii.gz",  # stem probe only
    "t1w": "_desc-preproc_T1w.nii.gz",
    "fwd_h5": "_from-T1w_to-MNI152NLin6Asym_mode-image_xfm.h5",
}


# --- metadata (pinned Brainmarks commit on GitHub; $BRAINMARKS_METADATA_URL overrides) ---
#: The benchmark-metadata fetcher (local stage + 429 backoff), bound to ADHD200's metadata dir.
_fetch = partial(fetch, _META)


@lru_cache
def curated_rows() -> tuple[dict, ...]:
    """Every row of ``ADHD200_curated.csv`` (``sub``/``split``/``site``/``path``/``dx``).
    Tuple, not list, so the `lru_cache` value cannot be mutated by a caller."""
    rows = csv.DictReader(io.StringIO(_fetch("ADHD200_curated.csv").decode()))
    return tuple(rows)


@lru_cache
def dx_map() -> dict[str, str]:
    """``sub -> dx`` straight from the curated CSV's ``dx`` column -- ADHD200 ships no target-map
    JSON (unlike ABIDE/PPMI). ``sub`` is a zero-padded str, never int-cast."""
    return {r["sub"].strip(): r["dx"].strip() for r in curated_rows() if r.get("dx")}


# --- work items -------------------------------------------------------------
@dataclass(frozen=True)
class Adhd200Work:
    """One subject's eval unit: bare ``sub`` id, its ``split``, the ``site`` it was scanned at,
    and the first-run BOLD rel-path (rooted at ``DERIV_ROOT``, ending ``_bold.nii.gz``)."""

    sub: str
    split: str
    site: str
    bold_rel: str


def discover_work(
    *, only_subjects: list[str] | None = None, n_subjects: int | None = None
) -> list[Adhd200Work]:
    """All curated subjects from ``ADHD200_curated.csv`` (``sub`` dtype=str -- zero-padded
    ids, never int-cast). ``site`` and ``bold_rel`` come from the ``site``/``path`` columns
    (``<Site>/sub-XXXX/ses-1/func/...``)."""
    keep = {str(x) for x in only_subjects} if only_subjects else None  # str: match a YAML-parsed numeric id
    work: list[Adhd200Work] = []
    for r in curated_rows():
        sub = r["sub"].strip()
        if keep is not None and sub not in keep:
            continue
        work.append(
            Adhd200Work(
                sub=sub,
                split=r["split"].strip(),
                site=r["site"].strip(),
                bold_rel=r["path"].strip(),
            )
        )
    work.sort(key=lambda w: w.sub)
    return work[:n_subjects] if n_subjects else work


# --- derivative URIs --------------------------------------------------------
def _swap(bold_rel: str, suffix: str) -> str:
    return bold_rel.replace("_bold.nii.gz", suffix)


def bold_uri(work: Adhd200Work) -> str:
    """The staged MNI preproc BOLD -- the eval's frame source (fMRIPrep saved no ``space-T1w``
    run for this cohort)."""
    return f"{DERIV_ROOT}/{_swap(work.bold_rel, _BOLD_MNI)}"


def sidecar_uri(work: Adhd200Work) -> str:
    """The MNI bold JSON sidecar carrying ``RepetitionTime`` (the acquisition TR)."""
    return f"{DERIV_ROOT}/{_swap(work.bold_rel, _SIDECAR)}"


def _ses_token(work: Adhd200Work) -> str:
    """The session token from the bold rel-path (``<Site>/sub-XXXX/ses-1/func/...``)."""
    return work.bold_rel.split("/")[2]


def anat_stem(work: Adhd200Work, storage_options: dict | None = None) -> str:
    """Resolve the subject's anat **stem prefix** by **listing** the anat dir for the
    ``aparcaseg`` -- its ``sub-XXXX_ses-1_run-N`` prefix carries ``ses-``/``run-`` entities
    that are not derivable from the func stem (fMRIPrep picks a specific anat run). Append a
    suffix with `anat_uri`."""
    ses = _ses_token(work)
    # Site + subject dir come from bold_rel (which bold_uri/sidecar_uri already use), NOT work.site:
    # the curated cohort MERGES the Peking scan-waves into "Peking" while the fMRIPrep derivatives keep
    # them SPLIT (Peking_1/2/3), so work.site names a nonexistent anat dir for the Peking subjects.
    site, sub_dir = work.bold_rel.split("/")[:2]
    anat_dir = f"{DERIV_ROOT}/{site}/{sub_dir}/{ses}/anat"
    fs = s3_filesystem(storage_options)
    suf = _ANAT_SUFFIX["aparcaseg"]
    hits = sorted(
        p for p in fs.ls(anat_dir.removeprefix("s3://"), detail=False) if p.endswith(suf)
    )
    if not hits:
        raise FileNotFoundError(f"no {suf} under {anat_dir}")
    stem_name = hits[0].rsplit("/", 1)[-1][: -len(suf)]  # sub-XXXX_ses-1_run-N
    return f"{anat_dir}/{stem_name}"


def anat_uri(stem: str, kind: str) -> str:
    """Append an anat/transform suffix to a resolved `anat_stem`. ``kind`` in {``t1w``, ``fwd_h5``}."""
    return f"{stem}{_ANAT_SUFFIX[kind]}"


def read_repetition_time(sidecar_uri_: str, storage_options: dict | None = None) -> float:
    """Read ``RepetitionTime`` (seconds) from a fMRIPrep bold JSON sidecar on R2/local."""
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
    """Best-effort ADHD/Control diagnosis; {} if absent."""
    return _dxd.dx_meta(sys.modules[__name__], sub)


def discover(cfg: dict) -> list[dict]:
    return _dxd.discover_mni_only(cfg, sys.modules[__name__])
