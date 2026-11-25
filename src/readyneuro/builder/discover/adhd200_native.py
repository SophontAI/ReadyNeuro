"""ADHD-200 native-from-raw discovery.

Reuses the vendor ADHD200 sampling layer (the curated cohort, splits, sites, first-run bold
rel-paths, diagnosis) but re-points the staged inputs at the public raw BIDS. The curated
``bold_rel`` ENDS in ``_bold.nii.gz`` -- the raw BIDS name the fMRIPrep paths are suffix-swaps of --
so the raw BOLD is ``RAW_ROOT/{bold_rel}`` directly; the raw T1w is LISTED for in the session's anat
dir. The acquisition TR is read from the derivative sidecar, the SAME value the mni build uses, so
the two arms stay TR-consistent.
"""

from __future__ import annotations

import logging

from readyneuro.builder.discover import adhd200 as _vendor
from readyneuro.io import profile_storage_options, s3_filesystem

_log = logging.getLogger(__name__)

RAW_ROOT = "s3://fcp-indi/data/Projects/ADHD200/RawDataBIDS"  # public, anonymous
DERIV_PROFILE = "r2"  # the credentialed mirror the TR sidecar is read from
_T1 = "t1w.nii.gz"
_BOLD = "run-00_bold.nii.gz"

subject_meta = _vendor.subject_meta  # re-export the ADHD/Control diagnosis


def raw_bold_uri(work) -> str:
    return f"{RAW_ROOT}/{work.bold_rel}"


def raw_t1_uri(work, storage_options: dict | None = None) -> str:
    """The raw T1w for the subject's session -- listed (the run/desc entities are not derivable from the
    func stem): ``RawDataBIDS/<Site>/sub-XXXX/ses-1/anat/sub-XXXX_ses-1_run-N_T1w.nii.gz``.

    The site dir + session are taken from ``bold_rel`` (the authoritative raw path), NOT ``work.site``:
    the curated cohort MERGES the Peking scan-waves into ``Peking`` while the raw BIDS keeps them SPLIT
    (``Peking_1/2/3``), so ``work.site`` names a nonexistent dir for the 85 Peking subjects. ``bold_rel``
    already carries the real split-wave dir (e.g. ``Peking_1/sub-XXXX/ses-1/func/...``)."""
    site, sub_dir, ses = work.bold_rel.split("/")[:3]
    anat_dir = f"{RAW_ROOT}/{site}/{sub_dir}/{ses}/anat"
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
        _log.warning("adhd200_native: dropped %d/%d subjects with no raw T1w", dropped, len(works))
    return items
