"""The eval finalizer: one subject's preprocessed BIDS -> a per-clip f16 Arrow eval store.

Imports the shared front half FROM `finalize_pretrain` (one-directional, acyclic). The anat row is
the SAME 1 mm T1 layer the pretraining bundle packs, so a frozen encoder sees an identical struct
layer at eval and at train. This module adds the eval tail: clip planning, the fixed diagnosis
window, and the anat row.

ONE per-dataset seam, read off the dataset's `wrappers.Core`:
``eval_clips(sidecar, run, cfg) -> {clips, n_frames, extra}`` -- pure metadata, so ``clips == []``
drops a non-contributing run BEFORE its BOLD is gathered. A dataset with no planner gets the
default: one unlabelled whole-window REST clip.

Every clip row also carries the RUN IDENTITY it came from. That is not decoration: it is the only
thing that lets an audit join a stored row back to the metadata row that produced it. It was missing
once, and a build that paired 30% of its native runs with the WRONG acquisition passed every QC path
there was.
"""

from __future__ import annotations

from dataclasses import replace
from math import ceil

import numpy as np

from readyneuro.builder.finalize_pretrain import (
    _coverage_ok, _load_subject, _passes_gm_gate, _run_vol, log)
from readyneuro.tools import codec, eval_arrow
from readyneuro.io import ValidVolume


def _tile_to(vol: ValidVolume, target_frames: int) -> ValidVolume:
    """Tile a short run's time axis up to ``target_frames`` -- the benchmark suite's pad-if-short.

    This is the 1.0 s-resample fencepost: a genuine 300 s REST run at a 1.5 s TR resamples to 299
    frames (199 x 1.5 = 298.5 s spanned), and dropping it on ``end < window_frames`` excluded whole
    single-TR sites that the benchmark keeps. Padding preserves the store's FIXED window length, so
    the reader and the collate are unchanged -- only more subjects ship."""
    T = vol.n_frames
    tiled = np.tile(vol.values, (1, ceil(target_frames / T)))[:, :target_frames]
    return replace(vol, values=tiled)


def _run_identity(run: dict) -> dict:
    """The identity columns (``task``/``ses``/``run``) stamped on EVERY clip row a sidecar run yields.

    Read off the sidecar run entry the core wrote -- the entry already carries whatever identity its
    dataset has. Two shapes:

    - Runs with NUMERIC session/run numbers (both NSD builds stamp ``ses``/``run_num``, precisely so
      the cococlip planner can join on them) -> those ints plus the matching `eval_arrow.run_key` in
      ``task``. ``run`` is the METADATA run number, NOT the raw-BIDS acquisition index: the metadata
      is the namespace every downstream join uses.
    - Anything else (HCP names its runs ``tfMRI_EMOTION_LR``; the REST cores have one ``REST`` run)
      -> the schema's -1 sentinel for the two int columns, with the run's own key as ``task``.

    PRECEDENCE: the caller merges this UNDER a planner's ``extra``, so a planner that puts a LABEL in
    ``task`` wins -- HCP's ``task`` is its task21 class name, contractually identical to the
    benchmark's. Identity only fills what a planner left blank."""
    ses, num = run.get("ses"), run.get("run_num")
    if ses is None or num is None:
        return {"task": str(run.get("run", "")), "ses": -1, "run": -1}
    return {"task": eval_arrow.run_key(ses, num), "ses": int(ses), "run": int(num)}


def finalize_eval(bids_root: str, cfg: dict, *, split: str, space: str, core) -> list[str]:
    """Slice one subject's preprocessed BIDS into a per-clip eval Arrow store for ``(split, space)``.
    Returns the produced arrow uris (empty if the subject has no eval clips).

    Two config knobs: ``common_tr`` (cross-TR Nyquist harmonization, set by the multi-site diagnosis
    cohorts and unset elsewhere) and ``window_frames`` (the fixed 5-min diagnosis window -- clip a
    longer run, tile-pad a shorter one, and drop only a run below ``min_frames`` of real data). Both
    keep the stored per-clip ``nyquist_tr`` honest."""
    sub = _load_subject(bids_root, space)
    if sub is None:
        return []
    if not _passes_gm_gate(sub, cfg, space):  # reject a degenerate subject BEFORE any Arrow is written
        return []

    anat_rec = codec.pack(sub.t1_vv, sub.gm1, sub.cortical1, sub.seg1, quant="f16")  # == the pretrain bundle's T1 layer
    target_tr = float(cfg.get("target_tr", 1.0))
    common_tr = cfg.get("common_tr")                                    # None = single-stage resample
    window_frames = cfg.get("window_frames")                           # None = whole resampled run
    planner = core.eval_clips

    root = cfg["eval_root"]

    def _run_rec_batches():
        """Yield one run's clip recs at a time, so the writer emits a record batch per run and the
        whole subject's clips (hundreds of GB for an uncapped NSD subject) never coexist in
        memory."""
        for r in sub.runs:
            if not _coverage_ok(bids_root, sub, r, cfg, space):        # reject low-fMRI-coverage run before any gather
                continue
            # The run identity every clip of this run inherits, merged UNDER a planner's `extra` so
            # a planner's `task` LABEL still wins.
            ident = _run_identity(r)
            if planner is None:                                        # default: whole-window REST clip
                vol = _run_vol(bids_root, sub, r, space, target_tr=target_tr, common_tr=common_tr)
                end = int(vol.n_frames)
                if window_frames is not None:
                    # Fix the run to EXACTLY window_frames, matching the benchmark: clip if long,
                    # tile-pad if short, never drop for length. Drop only a DEGENERATE run (below
                    # `min_frames` of real data -- padding that would be mostly tiling).
                    wf = int(window_frames)
                    min_frames = int(cfg.get("min_frames", wf // 2))
                    if end < min_frames:
                        log.warning("%s/%s: run resampled to %d frames < min_frames %d -- dropping "
                                    "(too short for a %d-frame window without >2x tiling)",
                                    sub.key, space, end, min_frames, wf)
                        continue
                    if end < wf:
                        log.info("%s/%s: run %d frames -> tile-padded to window_frames %d "
                                 "(a 300 s run resamples to 299 at 1.0 s)", sub.key, space, end, wf)
                        vol = _tile_to(vol, wf)
                    end = wf
                clips = [{"start": 0, "end": end, "cond": "", "cond_id": -1}]
                n_frames, extra = end, {**ident, "task": "REST"}
            else:
                plan = planner(sub.sidecar, r, cfg)                    # metadata only (cheap)
                if not plan.get("clips"):                             # run contributes nothing -- no gather
                    continue
                vol = _run_vol(bids_root, sub, r, space, target_tr=target_tr, common_tr=common_tr)
                clips, n_frames = plan["clips"], int(plan["n_frames"])
                extra = {**ident, **plan.get("extra", {})}
            yield codec.slice_clips(vol, sub.fgm, sub.fcortical, clips, seg_img=sub.fseg,
                                    n_frames=n_frames, extra=extra)

    produced = eval_arrow.write_subject_streamed(
        root, space, split, sub.key, _run_rec_batches(), anat_rec,
        storage_options=cfg.get("out_storage_options"))
    if not produced:  # every run dropped / too short -> no deliverable (matches the legacy eval skip contract)
        log.warning("%s/%s: no eval clips (all runs dropped/too short) -- shipping nothing", sub.key, space)
    return produced
