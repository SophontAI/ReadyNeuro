"""ABIDE native-from-raw discovery.

Reuses the vendor ABIDE sampling layer (the curated cohort, splits, sites, first-run bold rel-paths,
diagnosis) but re-points the staged inputs at the public raw BIDS. The curated ``bold_rel`` ENDS in
``_bold.nii.gz`` -- the raw BIDS name -- so the raw BOLD is ``RAW_ROOT/{bold_rel}`` directly; the raw
T1w is LISTED for, because its filename varies by site (plain / run- / defaced).

The acquisition TR is read from the derivative sidecar, i.e. the SAME value the mni build uses, so
the two arms of the native/mni comparison stay TR-consistent.
"""

from __future__ import annotations

import logging

from readyneuro.builder.discover import abide as _vendor
from readyneuro.io import profile_storage_options, s3_filesystem

_log = logging.getLogger(__name__)

RAW_ROOT = "s3://fcp-indi/data/Projects/ABIDE/RawDataBIDS"  # public, anonymous
DERIV_PROFILE = "r2"  # the credentialed mirror the TR sidecar is read from
_T1 = "t1w.nii.gz"
_BOLD = "run-00_bold.nii.gz"

subject_meta = _vendor.subject_meta  # re-export the Autism/Control diagnosis


def raw_bold_uri(work) -> str:
    return f"{RAW_ROOT}/{work.bold_rel}"


def raw_t1_uri(work, storage_options: dict | None = None) -> str:
    """The raw T1w from the subject-level anat dir, LISTED for because names vary by site
    (``sub-XXXX_T1w`` / ``sub-XXXX_run-1_T1w`` / defaced).

    Site + subject dir come from ``bold_rel`` -- the authoritative raw path -- so anat and BOLD always
    resolve to the SAME raw site dir, robust to a curated/raw site-label mismatch."""
    site, sub_dir = work.bold_rel.split("/")[:2]
    anat_dir = f"{RAW_ROOT}/{site}/{sub_dir}/anat"
    cands = sorted(p for p in s3_filesystem(storage_options).ls(anat_dir.removeprefix("s3://"),
                                                                detail=False)
                   if p.endswith("_T1w.nii.gz"))
    if not cands:
        raise FileNotFoundError(f"no raw T1w under {anat_dir}")
    return f"s3://{cands[0]}"


def discover(cfg: dict) -> list[dict]:
    """One ``native``-only eval item per curated subject, in the from-raw kernel's source shape."""
    anon_so = cfg.get("input_storage_options")            # the public raw bucket: anonymous reads
    deriv_so = profile_storage_options(DERIV_PROFILE)     # the mirror holding the TR sidecar
    works = _vendor.discover_work(only_subjects=cfg.get("subjects"), n_subjects=cfg.get("n_subjects"))
    items: list[dict] = []
    dropped = 0
    for w in works:
        try:
            t1_uri = raw_t1_uri(w, anon_so)          # resolve the raw T1w FIRST -- drop the subject if
        except FileNotFoundError:                    # it has none (matches the openneuro drop-subject
            dropped += 1                             # policy; keeps one bad subject from aborting the scan)
            continue
        tr = _vendor.read_repetition_time(_vendor.sidecar_uri(w), deriv_so)
        items.append(
            {
                "key": w.sub,
                "deliverables": [{"kind": "eval_arrow", "space": "native", "split": w.split}],
                "sources": {
                    "t1": {"name": _T1, "uri": t1_uri},
                    "aux": [],
                    "runs": [{"idx": 0, "run": "REST", "name": _BOLD,
                              "uri": raw_bold_uri(w), "tr_acq": tr}],
                    "meta": subject_meta(w.sub),
                },
            }
        )
    if dropped:
        _log.warning("abide_native: dropped %d/%d subjects with no raw T1w", dropped, len(works))
    return items
