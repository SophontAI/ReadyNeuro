"""The packed store codec: voxel order, int8 round trip, and the tissue nesting invariant.

Every one of these fails silently. A voxel order other than C-order ``flatnonzero(mask)`` still
decodes to an array of the right shape, with the values attached to the wrong coordinates; an int8
scale that is off still dequantizes; a cortical mask reaching past grey matter still packs.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np

from readyneuro.io import ValidVolume
from readyneuro.tools import codec

RAD2 = np.array([[-2, 0, 0, 40], [0, 2, 0, -30], [0, 0, 2, -20], [0, 0, 0, 1]], float)


def _func(seed=0, frames=5):
    """A 4D func volume whose whole labeled region is cortical grey matter."""
    rng = np.random.default_rng(seed)
    gm = np.zeros((8, 8, 8), bool)
    gm[2:6, 2:6, 2:6] = True
    z = rng.standard_normal((int(gm.sum()), frames)).astype(np.float32)
    mask_img = nib.Nifti1Image(gm.astype(np.uint8), RAD2)
    return ValidVolume(z, gm, RAD2), mask_img, gm, z


def test_stored_voxels_are_c_order_flatnonzero_of_the_mask():
    """``idx`` is the only thing that says where a stored value lives. Any other order still has
    the right length and still indexes; the values simply land on the wrong voxels."""
    vol, mask_img, gm, _ = _func()
    rec = codec.pack(vol, mask_img, mask_img, quant="int8")

    m = int(gm.sum())
    assert rec["values"].shape == (m, 5) and rec["values"].dtype == np.int8
    assert rec["idx"].dtype == np.int32 and rec["shape"] == (8, 8, 8)
    assert np.array_equal(rec["idx"], np.flatnonzero(gm))
    assert np.allclose(rec["affine"], RAD2)


def test_int8_roundtrip_stays_within_one_quantization_step():
    """``int8 * scale`` recovers z. Full scale is the 99.9th percentile of |z|, so the unsaturated
    99% must land within one step; the tail beyond that may clip."""
    vol, mask_img, _, z = _func()
    rec = codec.pack(vol, mask_img, mask_img, quant="int8")
    assert rec["scale"] > 0
    recovered = rec["values"].astype(np.float32) * rec["scale"]
    assert np.percentile(np.abs(recovered - z), 99) <= rec["scale"]


def test_f16_pack_is_near_lossless():
    """The evaluation store packs f16 with no quantization, so ``scale`` is exactly 1.0 -- a reader
    that multiplied by a wrong scale would silently rescale every clip."""
    vol, mask_img, _, z = _func()
    rec = codec.pack(vol, mask_img, mask_img, quant="f16")
    assert rec["values"].dtype == np.float16 and rec["scale"] == 1.0
    assert np.allclose(rec["values"].astype(np.float32), z, atol=1e-2)


def test_pack_enforces_cortical_subset_of_grey_matter():
    """A cortical mask reaching past grey matter is corrected at the single pack site, so no stored
    row can ever carry ``cortical_gm=1, gm=0`` -- which the model's two tissue streams would read
    as disagreeing about the same voxel."""
    vol, _mask_img, gm, _ = _func()
    rogue_cortical = nib.Nifti1Image(np.ones((8, 8, 8), np.uint8), RAD2)
    subcortical_gm = nib.Nifti1Image((gm & ~np.roll(gm, 1, 0)).astype(np.uint8), RAD2)
    rec = codec.pack(vol, subcortical_gm, rogue_cortical, quant="f16")
    assert not np.any((rec["cortical_gm"] == 1) & (rec["gm"] == 0))


def test_a_stray_non_finite_value_cannot_poison_the_int8_scale():
    """One inf in the block would take the percentile to inf, the scale to inf, and every value in
    that block to garbage -- with no error anywhere. The clamp runs before the percentile."""
    vol, mask_img, gm, z = _func()
    values = z.copy()
    values[0, 0] = np.inf  # the first stored voxel, first frame
    poisoned = ValidVolume(values, gm, RAD2)
    rec = codec.pack(poisoned, mask_img, mask_img, quant="int8")
    assert np.isfinite(rec["scale"]) and rec["scale"] > 0
    assert np.isfinite(rec["values"].astype(np.float32) * rec["scale"]).all()


def test_a_clip_is_cut_at_its_labelled_onset_not_at_the_start_of_the_run():
    """``slice_clips`` is where every HCP-YA Task21 condition and every NSD COCO category is bound
    to its 16 frames. Ignore ``start`` and every row still has the right ``m``/``t``/``start``/
    ``end`` and the right label -- attached to the wrong window."""
    gm = np.zeros((8, 8, 8), bool)
    gm[2:6, 2:6, 2:6] = True
    # 0.5 per frame, so every frame is identifiable and none is clipped at |z| = 5.
    ramp = np.broadcast_to(0.5 * np.arange(10, dtype=np.float32), (int(gm.sum()), 10))
    vol = ValidVolume(np.ascontiguousarray(ramp), gm, RAD2, tr=1.0)
    clip = {"start": 3, "end": 7, "cond": "", "cond_id": -1}
    (rec,) = codec.slice_clips(vol, nib.Nifti1Image(gm.astype(np.uint8), RAD2),
                               nib.Nifti1Image(gm.astype(np.uint8), RAD2), [clip],
                               n_frames=4, quant="f16")
    assert np.array_equal(rec["rec"]["values"][0], np.array([1.5, 2.0, 2.5, 3.0], np.float16))
