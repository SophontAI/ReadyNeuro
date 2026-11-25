"""The finalizer's three gates. Each one, if it stops working, corrupts a store silently.

* **BIDS completeness.** A space the core CLAIMED (it is in the sidecar) but did not finish must
  fail loud. A dropped mask save would otherwise ship a half-formed shard that reads fine.
* **GM viability.** A subject whose func GM is empty (failed registration, empty-FOV BOLD, bad
  segmentation) must be rejected BEFORE any writer opens, so it leaves no tar, no Arrow and no
  ``.done`` sentinel for a resume to trust.
* **fMRI coverage.** A run whose BOLD covers less than ``min_func_coverage`` of the anatomical
  brain is usually a bad EPI->T1 registration. It has to be dropped, not averaged in.

Plus the two things that decide what a stored row CONTAINS -- the func-mask union over runs and the
sidecar acquisition TR -- and one readback of the pretraining tar, the only artifact this release
publishes.
"""

from __future__ import annotations

import io
import json
import tarfile
import types
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from brainmarks.datasets import eval_store
from readyneuro.builder import finalize_eval, finalize_pretrain
from readyneuro import nsd
from readyneuro.io import BidsLayout
from readyneuro.tools import eval_arrow


def _cfg(tmp_path, **kw):
    # min_func_gm_voxels=1 keeps the GM gate ACTIVE without rejecting this toy grid; production
    # runs the 500 default.
    return {"dataset": "fake", "eval_root": str(tmp_path / "eval"),
            "pretrain_root": str(tmp_path / "pre"), "results_dir": str(tmp_path / "results"),
            "target_tr": 1.0, "min_func_gm_voxels": 1, **kw}


def _load(cfg, kind: str):
    """Read back through the CONSUMER (`brainmarks.datasets.eval_store`), not the writer."""
    where = eval_arrow.func_dir if kind == "func" else eval_arrow.anat_dir
    return eval_store.load_arrow_dir(where(cfg["eval_root"], "native", "test"))


def _finalize_eval(out_bids, cfg):
    return finalize_eval.finalize_eval(out_bids, cfg, split="test", space="native",
                                       core=types.SimpleNamespace(eval_clips=None))


# --- BIDS completeness ------------------------------------------------------
def test_a_claimed_space_missing_a_mask_fails_loud(bids, tmp_path):
    out_bids = bids("subD")
    BidsLayout(out_bids).gm_mask_path("subD", "native").unlink()
    with pytest.raises(RuntimeError, match="BIDS incomplete"):
        _finalize_eval(out_bids, _cfg(tmp_path))


def test_an_unclaimed_space_is_a_soft_skip(bids, tmp_path):
    """A subject that simply lacks a space ships nothing and does not raise -- that is a genuinely
    modality-less subject, not a broken one."""
    out_bids = bids("subU")
    assert finalize_eval.finalize_eval(out_bids, _cfg(tmp_path), split="test", space="mni",
                                       core=types.SimpleNamespace(eval_clips=None)) == []


def test_non_nested_anat_masks_fail_loud(bids, tmp_path):
    """cortical must stay inside gm inside brain. A GM mask reaching past the brain is a core bug,
    and packing it would silently store rows the model's tissue streams disagree about."""
    out_bids = bids("subN")
    layout = BidsLayout(out_bids)
    gm_path = layout.gm_mask_path("subN", "native")
    img = nib.load(gm_path)
    nib.save(nib.Nifti1Image(np.ones(img.shape, np.uint8), img.affine), gm_path)
    with pytest.raises(RuntimeError, match="not nested"):
        _finalize_eval(out_bids, _cfg(tmp_path))


# --- GM viability -----------------------------------------------------------
def test_an_empty_gm_subject_ships_nothing_and_writes_nothing(bids, tmp_path):
    """The load-bearing part is that the output root does not even get CREATED: the writers are the
    only things that create it, so this fails the instant the gate moves past one."""
    out_bids = bids("subG")
    cfg = _cfg(tmp_path, min_func_gm_voxels=10**6)
    assert _finalize_eval(out_bids, cfg) == []
    assert not Path(cfg["eval_root"]).exists()
    assert finalize_pretrain.finalize_pretrain(out_bids, cfg, split="train", space="native") is None
    assert not Path(cfg["pretrain_root"]).exists()


# --- fMRI coverage ----------------------------------------------------------
def test_a_run_below_sixty_percent_coverage_is_dropped(bids, tmp_path):
    out_bids = bids("subC", runs=[
        {"idx": 0, "run": "good", "cov_slabs": 3},   # 75% -> kept
        {"idx": 1, "run": "bad", "cov_slabs": 2},    # 50% -> dropped
    ], n_t=8)
    cfg = _cfg(tmp_path, min_func_coverage=0.60)
    assert len(_finalize_eval(out_bids, cfg)) == 2
    func = _load(cfg, "func")
    assert func.num_rows == 1


def test_every_run_below_threshold_ships_nothing(bids, tmp_path):
    out_bids = bids("subZ", runs=[
        {"idx": 0, "run": "bad0", "cov_slabs": 2},
        {"idx": 1, "run": "bad1", "cov_slabs": 2},
    ], n_t=8)
    assert _finalize_eval(out_bids, _cfg(tmp_path, min_func_coverage=0.60)) == []


def test_without_an_anatomical_reference_the_coverage_check_is_skipped(bids, tmp_path):
    """A vendor core that ships no segmentation has nothing to measure coverage against, so the
    check must not fire -- it would reject every one of those subjects."""
    out_bids = bids("subS", runs=[{"idx": 0, "run": "bad", "cov_slabs": 2}], with_dseg=False)
    cfg = _cfg(tmp_path, min_func_coverage=0.60)
    assert len(_finalize_eval(out_bids, cfg)) == 2
    func = _load(cfg, "func")
    assert func.num_rows == 1


# --- the fixed diagnosis window ---------------------------------------------
def test_a_short_run_is_tile_padded_to_the_fixed_window(bids, tmp_path):
    """PPMI/ABIDE/ADHD-200 fix every scan at ``window_frames``; a 300 s run resamples to 299, and
    dropping it would exclude whole single-TR sites. Broken padding leaves rows of the wrong length
    that still decode."""
    out_bids = bids("subT", runs=[{"idx": 0, "run": "rest"}], n_t=5)
    cfg = _cfg(tmp_path, window_frames=8, min_frames=4)
    assert len(_finalize_eval(out_bids, cfg)) == 2
    row = _load(cfg, "func").with_format("numpy")[0]
    values = eval_store.decode_func_row(row)["values"]
    assert int(row["t"]) == 8 and values.shape[1] == 8
    assert np.array_equal(values[:, 5:], values[:, :3])  # the tile is periodic in the real frames


def test_a_long_run_is_clipped_to_the_fixed_window(bids, tmp_path):
    out_bids = bids("subL", runs=[{"idx": 0, "run": "rest"}], n_t=12)
    cfg = _cfg(tmp_path, window_frames=8, min_frames=4)
    assert len(_finalize_eval(out_bids, cfg)) == 2
    row = _load(cfg, "func").with_format("numpy")[0]
    assert int(row["t"]) == 8 and int(row["end"]) == 8


# --- run identity ------------------------------------------------------------
def test_every_clip_row_carries_the_identity_of_the_run_it_came_from():
    """The only thing that joins a stored row back to the metadata row that produced it. It was
    missing once, and a build that paired 30% of its native runs with the WRONG acquisition passed
    every QC path there was. The run key is asserted against a LITERAL, not against `run_key`
    itself: the zero-pad is the join, and `readyneuro.nsd.run_key` is a second copy of the format."""
    assert finalize_eval._run_identity({"ses": 21, "run_num": 3, "run": "nsdcore"}) == {
        "task": "21_03", "ses": 21, "run": 3}
    assert eval_arrow.run_key(21, 3) == nsd.run_key(21, 3) == "21_03"
    assert finalize_eval._run_identity({"run": "tfMRI_EMOTION_LR"}) == {
        "task": "tfMRI_EMOTION_LR", "ses": -1, "run": -1}


def test_a_planners_label_wins_over_the_run_key_but_ses_and_run_survive(bids, tmp_path):
    """`finalize_eval` merges the identity UNDER the planner's ``extra``. Flip that merge and a
    planner's task label is overwritten by the run key -- HCP's class name replaced by "00_-1"."""
    out_bids = bids("subP2", runs=[{"idx": 0, "run": "rest"}], n_t=8)
    cfg = _cfg(tmp_path)
    planner = lambda sidecar, run, cfg: {  # noqa: E731
        "clips": [{"start": 0, "end": 4, "cond": "", "cond_id": 2}],
        "n_frames": 4, "extra": {"task": "EMOTION"}}
    assert len(finalize_eval.finalize_eval(
        out_bids, cfg, split="test", space="native",
        core=types.SimpleNamespace(eval_clips=planner))) == 2
    row = _load(cfg, "func").with_format("numpy")[0]
    assert str(row["task"]) == "EMOTION"
    assert int(row["ses"]) == -1 and int(row["run"]) == -1


# --- what a stored row contains ----------------------------------------------
def test_the_stored_voxels_are_the_union_over_runs_not_the_intersection(bids, tmp_path):
    """The func mask is the per-mask UNION over runs -- a voxel ANY run reached is stored, and a run
    that missed it stores a flat row. An intersection still writes, still decodes and still reports a
    plausible ``num_rows``; only the voxel count moves."""
    out_bids = bids("subM", runs=[{"idx": 0, "run": "a", "cov_slabs": 4},
                                  {"idx": 1, "run": "b", "cov_slabs": 3}], n_t=8)
    cfg = _cfg(tmp_path)
    _finalize_eval(out_bids, cfg)
    row = _load(cfg, "func").with_format("numpy")[0]
    assert int(row["m"]) == 64  # |[1:5,1:5,1:5]|, not the 48 of the narrower run


def test_the_sidecar_acquisition_tr_drives_the_resample(bids, tmp_path):
    """``tr_acq`` from the sidecar, not the BOLD header -- an unstamped nibabel header reads 1.0 s,
    which is a silent wrong TR and so a silently wrong frame count for every run in the corpus."""
    out_bids = bids("subR", runs=[{"idx": 0, "run": "rest", "tr_acq": 2.0}], n_t=8)
    cfg = _cfg(tmp_path)
    _finalize_eval(out_bids, cfg)
    row = _load(cfg, "func").with_format("numpy")[0]
    assert int(row["t"]) == 15  # 8 frames @ 2 s span 14 s -> 15 samples @ 1 s (fencepost trimmed)


# --- the pretraining tar ------------------------------------------------------
def test_a_pretraining_shard_is_a_readable_webdataset_tar(bids, tmp_path):
    """The one artifact this release publishes. Every assertion here is a silent-corruption mode:
    a non-lowercase member name is a key WebDataset folds and then drops (emptying the stream), a
    wrong ``schema_version`` is unreadable by `fmrifm.pretrain_data`, a single-underscore run key
    collapses every run of a subject into one sample, and an f16 T1 under an int8 scale decodes to
    garbage."""
    out_bids = bids("subP", runs=[{"idx": 0, "run": "rfMRI_REST1_LR"},
                                  {"idx": 1, "run": "rfMRI_REST2_LR"}], n_t=8)
    cfg = _cfg(tmp_path)
    uri = finalize_pretrain.finalize_pretrain(out_bids, cfg, split="train", space="native")
    assert Path(uri).exists() and Path(uri + ".done").exists()
    with tarfile.open(uri) as tf:
        names = tf.getnames()
        sidecar = json.loads(tf.extractfile("subP.json").read())
        values = np.load(io.BytesIO(tf.extractfile("subP__rfMRI_REST1_LR.values.npy").read()))
        t1 = np.load(io.BytesIO(tf.extractfile("subP.t1_values.npy").read()))
    assert names[0] == "subP.bold_idx.npy"        # bundle first: a streaming loader needs it ahead
    # WebDataset case-folds member SUFFIXES, so the schema's `T1_` must go out lowercase.
    assert "subP.t1_values.npy" in names
    assert sum(n.endswith(".values.npy") for n in names) == 2   # one BOLD block per run
    assert sidecar["schema_version"] == 3         # a literal, not codec.SCHEMA_VERSION
    assert values.dtype == np.int8 and t1.dtype == np.int8


def test_a_tar_with_no_usable_run_leaves_no_done_sentinel(bids, tmp_path):
    """Bundle-only is not a deliverable. Mark it done and resume trusts a tar with no BOLD in it."""
    out_bids = bids("subQ", runs=[{"idx": 0, "run": "bad", "cov_slabs": 2}], n_t=8)
    cfg = _cfg(tmp_path, min_func_coverage=0.60)
    assert finalize_pretrain.finalize_pretrain(out_bids, cfg, split="train", space="native") is None
    assert not list(Path(cfg["pretrain_root"]).rglob("*.done"))
