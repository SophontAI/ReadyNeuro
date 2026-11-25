"""The producer/consumer seam of the evaluation store.

``readyneuro.tools.eval_arrow`` writes it during the build; ``brainmarks.datasets.eval_store`` reads
it during evaluation, in a different process and a different environment. Every disagreement here
is silent: a clip whose values reshape against the wrong axis still has the right number of
elements, voxel indices in the wrong order still index, and an unknown sampling period that comes
back as NaN instead of ``None`` poisons the bandwidth conditioning rather than raising.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np

from brainmarks.datasets import eval_store
from brainmarks.datasets.eval_store import _optional_float
from readyneuro.io import ValidVolume
from readyneuro.tools import codec, eval_arrow

RAD2 = np.array([[-2, 0, 0, 40], [0, 2, 0, -30], [0, 0, 2, -20], [0, 0, 0, 1]], float)
RAD1 = np.array([[-1, 0, 0, 40], [0, 1, 0, -30], [0, 0, 1, -20], [0, 0, 0, 1]], float)


def _func_rec(seed, frames):
    rng = np.random.default_rng(seed)
    gm = np.zeros((8, 8, 8), bool)
    gm[2:6, 2:6, 2:6] = True
    values = rng.standard_normal((int(gm.sum()), frames)).astype(np.float32)
    gm_img = nib.Nifti1Image(gm.astype(np.uint8), RAD2)
    rec = codec.pack(ValidVolume(values, gm, RAD2), gm_img, gm_img, quant="f16")
    return rec, gm, values


def _anat_rec(seed):
    rng = np.random.default_rng(seed)
    gm = np.zeros((16, 16, 16), bool)
    gm[4:12, 4:12, 4:12] = True
    values = rng.standard_normal(int(gm.sum())).astype(np.float32)
    gm_img = nib.Nifti1Image(gm.astype(np.uint8), RAD1)
    cort_img = nib.Nifti1Image(np.zeros((16, 16, 16), np.uint8), RAD1)
    return codec.pack(ValidVolume(values, gm, RAD1), gm_img, cort_img, quant="f16")


def _write(root, frames=6, nyquist_tr=2.5):
    rec, gm, values = _func_rec(7, frames)
    clip = {"rec": rec, "task": "REST", "cond": "c", "cond_id": -1, "start": 0, "end": frames,
            "tr": 1.0, "nyquist_tr": nyquist_tr, "n_frames": frames}
    eval_arrow.write_subject_streamed(root, "native", "test", "S1", [[clip]], _anat_rec(200))
    return gm, values


def _func_row(root):
    split = eval_store.load_arrow_dir(eval_store.func_dir(root, "native", "test"))
    return split.with_format("numpy")[0]


def test_a_written_clip_reads_back_as_the_same_window(tmp_path):
    root = str(tmp_path)
    gm, values = _write(root, frames=6)
    row = _func_row(root)
    decoded = eval_store.decode_func_row(row)

    m = int(gm.sum())
    assert decoded["values"].shape == (m, 6)
    assert np.allclose(decoded["values"].astype(np.float32), values, atol=1e-2)
    # The consumer feeds the model the transpose: (frames, voxels).
    assert decoded["values"].T.shape == (6, m)
    assert int(row["start"]) == 0 and int(row["end"]) == 6 and int(row["t"]) == 6
    assert str(row["sub"]) == "S1" and str(row["task"]) == "REST" and int(row["cond_id"]) == -1


def test_voxel_indices_survive_the_round_trip_in_c_order(tmp_path):
    root = str(tmp_path)
    gm, _ = _write(root)
    decoded = eval_store.decode_func_row(_func_row(root))
    assert decoded["idx"].dtype == np.int32
    assert np.array_equal(decoded["idx"], np.flatnonzero(gm))
    assert decoded["shape"] == (8, 8, 8)
    assert np.allclose(decoded["affine"], RAD2)


def test_the_anat_row_is_the_same_schema_without_frames(tmp_path):
    root = str(tmp_path)
    _write(root)
    split = eval_store.load_arrow_dir(eval_store.anat_dir(root, "native", "test"))
    decoded = eval_store.decode_anat_row(split.with_format("numpy")[0])
    assert decoded["values"].ndim == 1
    assert decoded["values"].shape == decoded["idx"].shape
    assert decoded["shape"] == (16, 16, 16)


def test_an_unknown_sampling_period_reads_as_none_not_nan(tmp_path):
    """Arrow float32 has no null, so the store writes NaN for UNKNOWN. The reader must turn that
    back into ``None``: a NaN riding into the sampler's ``max`` over periods NaNs the conditioning
    for a whole batch, with no error anywhere."""
    known_root, unknown_root = str(tmp_path / "known"), str(tmp_path / "unknown")
    _write(known_root, nyquist_tr=2.5)
    _write(unknown_root, nyquist_tr=None)

    known, unknown = _func_row(known_root), _func_row(unknown_root)
    assert _optional_float(known, "nyquist_tr") == 2.5
    assert np.isnan(float(unknown["nyquist_tr"]))
    assert _optional_float(unknown, "nyquist_tr") is None
