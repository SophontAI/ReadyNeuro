"""AABC native-from-raw discovery (EVAL-ONLY, REST).

Reuses the vendor AABC sampling layer (the eval split, the sampled visit, the phenotype
demographics) but re-points the FUNCTIONAL input at the RAW unprocessed scanner-space REST zips
(104x104x72 @ 2 mm, TR 0.8 s, oblique LAS, aligned to nothing), which the shared from-raw kernel
processes. The vendor ``mni`` product is a SEPARATE build into the same eval_root.

Two AABC specifics vs the other ``*_native`` cores:

* **There is no raw structural upload.** The anatomical input stays the STRUCT zip, from which the
  kernel reads ``T1w/T1w_acpc_dc.nii.gz`` -- AC-PC and gradient-distortion corrected, but NOT
  bias-corrected, so the kernel's own N4 step still does real work. This build is "raw BOLD +
  vendor T1", not raw-everything.
* **The eval window spans TWO raw runs.** The benchmark suite windows the REST concat at ``[0:500]``,
  and that concat runs in acquisition order REST1_AP, REST1_PA, REST2_AP, REST2_PA at 478 frames
  each -- so frames 0-477 are REST1_AP and 478-499 are REST1_PA. For a complete session only those
  two runs are read; the core falls through to the next available REST run when one is missing (see
  ``REST_RUNS``), and resamples what it reads into ONE 500-frame series.

The raw BOLD is NOT pre-staged: the members are streamed out of the ~2.34 GB session zip by the core
(~1.06 GB read instead of 2.34 GB), so they ride in ``sources.eval_runs`` rather than
``sources.runs``, which the driver would download whole.
"""

from __future__ import annotations

import s3fs

from readyneuro.builder.discover import aabc as _vendor


# --- the raw (unprocessed, scanner-space) fMRI mirror ------------------------
RAW_FUNC_PREFIX = "s3://<RAW_ROOT>/aabc-unproc-fmri"
_RFMRI_SUFFIX = "_UnprocRfmri.zip"   # the 4 REST runs; `_UnprocTfmri.zip` (task) is never read

#: EVERY REST run, in concat order. The ``[0:500]`` window normally stops inside the second (478 +
#: 22) and the core stops as soon as it is full, so runs 3/4 cost nothing for a complete session.
#:
#: They are LISTED because a session is not guaranteed to have all four: 2 of the 584 eval sessions
#: have REST1_AP + REST2_AP and no REST1_PA, and hardcoding the first two made the member lookup
#: raise ``KeyError`` and drop the subject. The vendor concat also takes whatever runs exist in this
#: order, so falling through to the next available run reproduces its window rather than degrading it.
REST_RUNS = ("rfMRI_REST1_AP", "rfMRI_REST1_PA", "rfMRI_REST2_AP", "rfMRI_REST2_PA")

NATIVE_TR = _vendor.NATIVE_TR        # 0.8 s acquisition TR (the finalizer resamples FROM this)

STRUCT_ZIP = _vendor.STRUCT_ZIP      # "struct.zip" -- staged whole; the kernel reads ONE member out of it
#: The anatomical member: AC-PC + gradient-distortion corrected, NOT bias-corrected (`_restore` is),
#: so the from-raw kernel's N4 is not a no-op. The vendor core uses `_restore` because it does no N4.
STRUCT_T1_MEMBER = "T1w/T1w_acpc_dc.nii.gz"

subject_meta = _vendor.subject_meta  # the same age/sex target maps (bare-subject keyed)


def rfmri_zip_uri(session: str) -> str:
    return f"{RAW_FUNC_PREFIX}/{session}{_RFMRI_SUFFIX}"


def raw_bold_member(session: str, run: str) -> str:
    """The member path of one raw REST run inside its session zip (entries are rooted at ``SESSION/``)."""
    return f"{session}/unprocessed/{run}/{session}_{run}.nii.gz"


def struct_t1_member(session: str) -> str:
    return _vendor.member(session, STRUCT_T1_MEMBER)


def list_raw_sessions(storage_options: dict | None = None) -> frozenset[str]:
    """Sessions with a raw REST zip present -- a session without one is SKIPPED rather than
    discovered into a manifest that then fails per-subject.

    Uncached on purpose: ``discover`` calls it once and the driver caches the whole manifest, while
    ``storage_options`` carries a nested dict and is not hashable as an ``lru_cache`` key."""
    fs = s3fs.S3FileSystem(**(storage_options or {}))
    out = set()
    for p in fs.ls(RAW_FUNC_PREFIX.removeprefix("s3://"), detail=False):
        name = p.rsplit("/", 1)[-1]
        if name.endswith(_RFMRI_SUFFIX):
            out.add(name[: -len(_RFMRI_SUFFIX)])
    return frozenset(out)


def discover(cfg: dict) -> list[dict]:
    """One ``native``-only eval item per eval subject whose sampled visit has a raw REST zip.

    Keyed by the BARE subject -- the vendor core's key -- so the two stores line up and a
    leave-one-subject-out probe groups a subject's visits. ``sources.runs`` stays empty; the two REST
    members are streamed by the core out of ``sources.eval_runs``."""
    so = cfg.get("input_storage_options")
    works = _vendor.discover_work(so, split_path=cfg.get("aabc_eval_split"),
                                  only_subjects=cfg.get("subjects"), n_sessions=cfg.get("n_sessions"))
    available = list_raw_sessions(so)
    works = [w for w in works if w.session in available]

    items: list[dict] = []
    for w in works:
        items.append(
            {
                "key": w.sub,
                "deliverables": [{"kind": "eval_arrow", "space": "native", "split": w.split}],
                "sources": {
                    "t1": {"name": STRUCT_ZIP, "uri": _vendor.struct_zip_uri(w.session),
                           "member": struct_t1_member(w.session)},
                    "aux": [],
                    "runs": [],
                    "eval_runs": [
                        {"idx": i, "run": r, "tr_acq": NATIVE_TR,
                         "uri": rfmri_zip_uri(w.session), "member": raw_bold_member(w.session, r)}
                        for i, r in enumerate(REST_RUNS)
                    ],
                    "session": w.session,
                    "meta": subject_meta(w.sub, cfg),
                },
            }
        )
    return items
