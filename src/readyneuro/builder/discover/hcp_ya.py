"""HCP-YA discovery (dual-product), demographics, and the task21 eval-clip planner.

``discover`` emits ONE item per subject carrying BOTH products: a pretrain deliverable per space, and
an eval deliverable per space when the subject is in an eval batch AND has all six task21 tasks.
``eval_clips`` is the per-dataset seam `finalize_eval` reads."""

from __future__ import annotations

from functools import lru_cache

from readyneuro import hcp_ya

#: The run-subset knob changes the per-item run list (a subject with no surviving run emits no
#: deliverable), so it must key the manifest cache.
DISCOVERY_KEYS = ("runs",)

NATIVE_TR_3T = 0.72  # HCP 3T acquisition TR (the finalizer resamples FROM this to target_tr)

# Staged input names under a subject's raw dir (the driver's `hcp_ya.t1` + `hcp_ya.aux`).
T1_ACPC = "T1w_acpc.nii.gz"            # native acpc T1 (T1w_acpc_dc_restore) -- the SynthSeg input
T1_MNI = "T1w_mni.nii.gz"             # shipped MNI T1_restore (mni anat)
WARP_ACPC2STD = "acpc_dc2standard.nii.gz"  # acpc seg -> MNI (mni mask warp)


# --- demographics -----------------------------------------------------------
@lru_cache
def _pheno_map() -> dict:
    """sub -> best-effort demographics + behavioral targets from the benchmark suite's phenotype
    table (sex / age band + fluid-IQ / flanker / neuroticism; NaN dropped). Fetched once."""
    import math

    def _v(x):
        return None if x is None or (isinstance(x, float) and math.isnan(x)) else x

    df = hcp_ya.pheno_targets()
    return {
        str(r["Subject"]): {
            k: v for k, v in {
                "sex": _v(r.get("Gender")), "age": _v(r.get("Age")),
                "neofac_n": _v(r.get("NEOFAC_N")), "flanker": _v(r.get("Flanker_Unadj")),
                "pmat24": _v(r.get("PMAT24_A_CR")),
            }.items() if v is not None
        }
        for _, r in df.iterrows()
    }


def subject_meta(sub: str) -> dict:
    return dict(_pheno_map().get(str(sub), {}))


# --- discovery --------------------------------------------------------------
def _aux_sources(sub: str) -> list[dict]:
    """The anatomical inputs beyond the acpc T1, staged by the driver as ``sources.aux``: the
    shipped MNI T1 + the acpc->MNI mask warp."""
    return [{"name": T1_MNI, "uri": hcp_ya.t1_uri(sub, "mni")},
            {"name": WARP_ACPC2STD, "uri": hcp_ya.warp_uri(sub, "acpc_dc2standard")}]


def discover(cfg: dict) -> list[dict]:
    """Enumerate work items -- ONE per subject, carrying BOTH products (dual-product one-pass): a
    pretrain deliverable per space when the subject is in a pretrain batch, an eval deliverable per
    space when it is in an eval batch AND has all six task21 tasks. Empty deliverables -> no item."""
    if list(cfg["spaces"]) != ["mni"]:
        raise ValueError(f"{__name__} builds only 'mni'; for native space use the *_native dataset (got {cfg['spaces']!r})")
    so = cfg.get("input_storage_options")
    subs = cfg.get("subjects") or hcp_ya.discover_subjects(cfg.get("n_subjects"), None)
    spaces = list(cfg["spaces"])
    only = set(cfg["runs"]) if cfg.get("runs") else None

    items: list[dict] = []
    for sub in subs:
        try:
            runs = hcp_ya.list_runs(sub, storage_options=so)
        except (PermissionError, FileNotFoundError):
            continue
        if only:
            runs = [r for r in runs if r in only]
        if not runs:
            continue
        pre_split, eval_split = hcp_ya.assign_splits(sub)
        do_eval = eval_split is not None and hcp_ya.has_all_task21(runs)
        deliverables: list[dict] = []
        for sp in spaces:
            if pre_split is not None:
                deliverables.append({"kind": "pretrain_tar", "space": sp, "split": pre_split})
            if do_eval:
                deliverables.append({"kind": "eval_arrow", "space": sp, "split": eval_split})
        if not deliverables:
            continue
        items.append(
            {
                "key": sub,
                "deliverables": deliverables,
                "sources": {
                    "t1": {"name": T1_ACPC, "uri": hcp_ya.t1_uri(sub, "native")},
                    "aux": _aux_sources(sub),
                    "runs": [
                        {"idx": i, "run": r, "name": f"run-{i:02d}_bold.nii.gz",
                         "uri": hcp_ya.bold_uri(sub, r), "tr_acq": NATIVE_TR_3T}
                        for i, r in enumerate(runs)
                    ],
                    "meta": subject_meta(sub),
                },
            }
        )
    return items


# --- eval clip planner (the finalize_eval seam) -----------------------------
def eval_clips(sidecar: dict, run: dict, cfg: dict) -> dict:
    """The task21 clip plan for one sidecar run -- `finalize_eval`'s per-dataset seam.

    Gates on the task21 LR runs, dropping RL / rest with ``clips: []`` BEFORE the finalizer gathers
    their BOLD (the core wrote every run, for the pretrain product), then joins 16-frame windows at
    the benchmark suite's event onsets, keyed by task."""
    rk = run["run"]  # e.g. "tfMRI_EMOTION_LR"
    if not (rk.startswith("tfMRI_") and rk.endswith("_LR")):
        return {"clips": []}
    task = rk.split("_")[1]
    if task not in hcp_ya.TASK21_TASKS:
        return {"clips": []}
    clip_list = hcp_ya.subject_run_clips(sidecar["key"], task, meta_df=hcp_ya.metadata())
    return {"clips": clip_list, "n_frames": hcp_ya.NUM_FRAMES, "extra": {"task": task}}
