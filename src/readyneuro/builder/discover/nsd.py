"""NSD discovery (vendor func1mm) + the cococlip eval-clip planner.

One eval item per subject with EMPTY ``sources.runs`` -- a subject has hundreds of runs, so nothing
is pre-staged; the per-run URIs ride in ``sources.eval_runs`` and the core streams them."""

from __future__ import annotations

from readyneuro import nsd

#: `eval_max_runs` changes which runs a manifest contains, so it MUST key the cache: a capped smoke
#: and an uncapped production run sharing one results_dir would otherwise replay whichever cached
#: first, silently -- the signature guard only fires on keys it knows about.
DISCOVERY_KEYS = ("eval_max_runs",)

T1 = "t1.nii.gz"                       # T1_to_func1mm (the SynthSeg input; already the 1 mm func grid, RAS)
XFM = "func1pt0-to-MNI.nii.gz"        # NSD's func1pt0->MNI coord volume (staged for the mni warp)


def discover(cfg: dict) -> list[dict]:
    """One eval item per subject; the split is its held-out-subject assignment. ``sources.runs`` is
    EMPTY (the driver pre-stages nothing); the per-run BOLD URIs ride in ``sources.eval_runs``, which
    the core copies into the sidecar for the streaming finalizer. The func1pt0->MNI coord volume is
    staged as ``aux``."""
    if list(cfg["spaces"]) != ["mni"]:
        raise ValueError(f"{__name__} builds only 'mni'; for native space use the *_native dataset (got {cfg['spaces']!r})")
    by_sub = nsd.runs_by_subject()
    subs = sorted(by_sub)
    if cfg.get("subjects"):
        subs = [s for s in subs if s in set(cfg["subjects"])]
    if cfg.get("n_subjects"):
        subs = subs[: int(cfg["n_subjects"])]
    spaces = list(cfg["spaces"])

    items: list[dict] = []
    for sub in subs:
        runs = by_sub[sub]
        split = runs[0][2]  # subject-disjoint: all of a subject's runs share one split
        aux = [{"name": XFM, "uri": nsd.mni_transform_uri(sub)}]
        # `eval_max_runs` bounds a subject to its first N runs. NSD ships HUNDREDS of runs per
        # subject at ~950 MB each, so an uncapped subject is ~500 GB. The cap is applied HERE, where
        # the run list is built, rather than anywhere downstream that could be bypassed.
        if cfg.get("eval_max_runs"):
            runs = runs[: int(cfg["eval_max_runs"])]
        eval_runs = [
            {"idx": i, "run": nsd.run_key(ses, run), "ses": int(ses), "run_num": int(run),
             "tr_acq": nsd.TARGET_TR, "uri": nsd.bold_uri(sub, ses, run)}
            for i, (ses, run, _) in enumerate(runs)
        ]
        items.append(
            {
                "key": sub,
                "deliverables": [{"kind": "eval_arrow", "space": sp, "split": split} for sp in spaces],
                "sources": {"t1": {"name": T1, "uri": nsd.t1_uri(sub)}, "aux": aux,
                            "runs": [], "eval_runs": eval_runs, "meta": {}},
            }
        )
    return items


def eval_clips(sidecar: dict, run: dict, cfg: dict) -> dict:
    """The cococlip plan for one run -- 16-frame windows at each COCO trial's onset, labelled by the
    24-way ``category_id`` (== ``cond_id``). ``t_run=None`` returns every trial; the finalizer bounds
    each window to the resampled run length. ``extra={}``: the finalizer's own per-run identity
    (``ses``/``run_num`` -> the run key) is what lands in ``task``."""
    clip_list = nsd.subject_run_clips(
        nsd.cococlip_metadata(), sidecar["key"], int(run["ses"]), int(run["run_num"]), t_run=None)
    return {"clips": clip_list, "n_frames": nsd.NUM_FRAMES, "extra": {}}
