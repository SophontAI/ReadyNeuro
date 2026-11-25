"""HCP-YA native-from-raw discovery, reusing the vendor split / task21 / demographics layer.

Same subjects, splits and eval plan as ``discover.hcp_ya``, but the staged inputs are the RAW
``unprocessed/3T`` tier -- the raw MPRAGE T1w and the raw per-run BOLD -- which ``preprocess/openneuro``
processes from scratch. The run NAMES are preserved, so the reused ``eval_clips`` task21 gate fires
on exactly the same runs as the vendor build.
"""

from __future__ import annotations

import s3fs

from readyneuro.builder.discover import hcp_ya as _vendor
from readyneuro import hcp_ya

#: The run-subset knob changes the per-item run list, so it must key the manifest cache.
DISCOVERY_KEYS = ("runs",)

NATIVE_TR_3T = _vendor.NATIVE_TR_3T  # 0.72 s HCP 3T acquisition TR (the finalizer resamples FROM this)
_RAW = "unprocessed/3T"

subject_meta = _vendor.subject_meta   # the same phenotype table
eval_clips = _vendor.eval_clips       # the same task21 planner (it gates on the run name)


def raw_list_runs(sub: str, *, storage_options: dict | None = None) -> list[str]:
    """The subject's 3T BOLD run dirs under ``unprocessed/3T`` -- the RAW twin of
    `hcp_ya.list_runs`, which lists the MNINonLinear derivatives. ``RUN_3T`` excludes the
    T1w_MPR / T2w_SPC / Diffusion / fieldmap dirs."""
    fs = s3fs.S3FileSystem(**(storage_options or {}))
    base = f"{hcp_ya.HCP_S3.removeprefix('s3://')}/{sub}/{_RAW}"
    runs = []
    for p in fs.ls(base, detail=False):
        run = p.rstrip("/").rsplit("/", 1)[-1]
        if hcp_ya.RUN_3T.match(run) and fs.exists(f"{base}/{run}/{sub}_3T_{run}.nii.gz"):
            runs.append(run)
    return sorted(runs)


def raw_bold_uri(sub: str, run: str) -> str:
    return f"{hcp_ya.HCP_S3}/{sub}/{_RAW}/{run}/{sub}_3T_{run}.nii.gz"


def raw_t1_uri(sub: str) -> str:
    return f"{hcp_ya.HCP_S3}/{sub}/{_RAW}/T1w_MPR1/{sub}_3T_T1w_MPR1.nii.gz"


def discover(cfg: dict) -> list[dict]:
    """One item per subject: a pretrain deliverable when it is in a pretrain batch, an eval
    deliverable when it is in an eval batch AND has all six task21 tasks -- ``native`` only, since the
    vendor FNIRT ``mni`` product is a separate build. Empty deliverables -> no item."""
    so = cfg.get("input_storage_options")
    subs = cfg.get("subjects") or hcp_ya.discover_subjects(cfg.get("n_subjects"), None)
    only = set(cfg["runs"]) if cfg.get("runs") else None

    items: list[dict] = []
    for sub in subs:
        try:
            runs = raw_list_runs(sub, storage_options=so)
        except (PermissionError, FileNotFoundError):
            continue
        if only:
            runs = [r for r in runs if r in only]
        if not runs:
            continue
        pre_split, eval_split = hcp_ya.assign_splits(sub)
        do_eval = eval_split is not None and hcp_ya.has_all_task21(runs)
        deliverables: list[dict] = []
        if pre_split is not None:
            deliverables.append({"kind": "pretrain_tar", "space": "native", "split": pre_split})
        if do_eval:
            deliverables.append({"kind": "eval_arrow", "space": "native", "split": eval_split})
        if not deliverables:
            continue
        items.append(
            {
                "key": sub,
                "deliverables": deliverables,
                "sources": {
                    "t1": {"name": "t1w.nii.gz", "uri": raw_t1_uri(sub)},
                    "aux": [],
                    "runs": [
                        {"idx": i, "run": r, "name": f"run-{i:02d}_bold.nii.gz",
                         "uri": raw_bold_uri(sub, r), "tr_acq": NATIVE_TR_3T}
                        for i, r in enumerate(runs)
                    ],
                    "meta": subject_meta(sub),
                },
            }
        )
    return items
