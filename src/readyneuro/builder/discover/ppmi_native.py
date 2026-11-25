"""PPMI native-from-raw discovery.

Reuses the vendor PPMI sampling layer (the curated cohort, splits, first-run paths, demographics, and
the acquisition TR off the derivative sidecar) but re-points the staged inputs at the RAW unprocessed
BIDS on the same private mirror. Emits one ``native``-only eval item per subject in the shape the
from-raw kernel reads: ``sources.t1`` = the raw T1, ``sources.runs`` = the raw REST run.
"""

from __future__ import annotations

from readyneuro.builder.discover import ppmi as _ppmi
from readyneuro.io import path_exists, s3_filesystem

# Raw unprocessed BIDS (dcm2bids-converted). The curated ``bold_rel`` IS the raw rel-path -- the
# fMRIPrep derivative paths are suffix-swaps of it -- so raw BOLD = ``RAW_ROOT/{bold_rel}``.
RAW_ROOT = "s3://<EVAL_ROOT>/PPMI/bids"


# Staged input names (the from-raw kernel opens ``{raw}/{name}`` verbatim; PPMI raw is gzipped).
_T1 = "t1w.nii.gz"
_BOLD = "run-00_bold.nii.gz"

subject_meta = _ppmi.subject_meta  # the same PD/Prodromal label


def raw_bold_uri(bold_rel: str) -> str:
    return f"{RAW_ROOT}/{bold_rel}"


def raw_t1_uri(sub: str, storage_options: dict | None = None) -> str:
    """The raw unprocessed T1w for the first-run session. PPMI raw BIDS keeps anat under the session
    (``sub-XXXX/ses-YYYYMMDD/anat/sub-XXXX_ses-YYYYMMDD_T1w.nii.gz``); probe that deterministic path,
    and on a miss LIST the anat dir and take the first ``_T1w``."""
    ses = _ppmi._func_session(sub)
    direct = f"{RAW_ROOT}/sub-{sub}/{ses}/anat/sub-{sub}_{ses}_T1w.nii.gz"
    if path_exists(direct, storage_options):
        return direct
    anat_dir = f"{RAW_ROOT}/sub-{sub}/{ses}/anat"
    cands = sorted(p for p in s3_filesystem(storage_options).ls(anat_dir.removeprefix("s3://"),
                                                                detail=False)
                   if p.endswith("_T1w.nii.gz"))
    if not cands:
        raise FileNotFoundError(f"no raw T1w under {anat_dir}")
    return f"s3://{cands[0]}"


def discover(cfg: dict) -> list[dict]:
    """One ``native``-only eval item per curated subject, in the from-raw kernel's source shape. The
    acquisition TR is read from the derivative sidecar (the same acquisition as the raw run) into the
    manifest, so the core needs no remote read."""
    so = cfg.get("input_storage_options")
    works = _ppmi.discover_work(only_subjects=cfg.get("subjects"), n_subjects=cfg.get("n_subjects"))
    items: list[dict] = []
    for w in works:
        tr = _ppmi.read_repetition_time(_ppmi.sidecar_uri(w), so)
        items.append(
            {
                "key": w.sub,
                "deliverables": [{"kind": "eval_arrow", "space": "native", "split": w.split}],
                "sources": {
                    "t1": {"name": _T1, "uri": raw_t1_uri(w.sub, so)},
                    "aux": [],
                    "runs": [{"idx": 0, "run": "REST", "name": _BOLD,
                              "uri": raw_bold_uri(w.bold_rel), "tr_acq": tr}],
                    "meta": subject_meta(w.sub),
                },
            }
        )
    return items
