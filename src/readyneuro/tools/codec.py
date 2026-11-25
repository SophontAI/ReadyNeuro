"""The store's wire format: packed voxels -> quantized records -> WebDataset tar members.

Every voxel layer -- on disk, in a decoded sample, everywhere -- is spelled ``f"{modality}_{field}"``
(`layer_key`), so a reader that knows a modality name knows every key it has.
"""

from __future__ import annotations

import io
import json

import nibabel as nib
import numpy as np

from readyneuro.io import ValidVolume

#: The two modality tokens the store (and the model config) spell its layers with.
BOLD, T1 = "bold", "T1"

_QUANT_PCT = 99.9  # int8 full-scale percentile: |z| at this pct maps to +-127
_CLIP_Z = 5.0  # outlier clamp on packed z-values; ONE bound, shared by every writer

# --- the layer schema -------------------------------------------------------
#: * ``values`` -- ``(M, T)`` BOLD / ``(M,)`` T1. Axis 0 IS the voxel population, both modalities.
#: * ``idx``    -- ``(M,)`` flat C-order indices into ``shape``. THE mask, stored sparse.
#: * ``shape``  -- ``(3,)`` grid shape, the only thing that makes ``idx`` dense-able.
#: * ``affine`` -- ``(4, 4)`` voxel -> world mm. Also the ONLY record of resolution (its column norms).
#: * ``gm`` / ``cortical_gm`` -- ``(M,)`` uint8 rows at those voxels, in ``values``-row order.
#:   ``brain`` is IMPLICIT: every stored row is a brain voxel (``idx`` IS the brain mask), so it is
#:   never stored, and the three masks are nested by construction (``cortical_gm ⊆ gm ⊆ brain``,
#:   enforced in `_tissue_rows`) -- the invariant could not be expressed any other way.
#: * ``seg`` -- ``(M,)`` int16 SynthSeg aseg LABEL row at those voxels (0 = unlabeled): the full
#:   segmentation the gm/cortical masks are DERIVED from, kept separate from them. A core shipping
#:   no ``dseg`` (the vendor MNI cores) packs an all-zero seg, so the field is uniform everywhere.
#:   No pretraining loader reads it, but it is stored bytes -- removing it changes the store.
LAYER_FIELDS: tuple[str, ...] = ("values", "idx", "shape", "affine", "gm", "cortical_gm", "seg")

def layer_key(modality: str, field: str) -> str:
    """The sample/bundle key holding ``field`` of ``modality``'s layer, e.g. ``"bold_values"``."""
    return f"{modality}_{field}"


#: The store's wire-format version, written into every bundle sidecar. The pretrain shard reader
#: hard-fails on a mismatch: a missing member would otherwise surface as a ``KeyError`` that
#: WebDataset's warn-and-continue handler swallows, silently emptying the stream. The eval-Arrow
#: store carries no version.
SCHEMA_VERSION = 3


def _tissue_rows(gm_img: nib.Nifti1Image, cortical_img: nib.Nifti1Image,
                 idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The ``(gm, cortical_gm)`` uint8 rows at the stored voxels ``idx`` (the brain mask, C-order),
    sampled from the gray-matter / cortical-gray-matter mask images (same grid).

    Enforces ``cortical_gm ⊆ gm`` HERE, at the single pack site, so the three masks cannot disagree
    on disk however the core wrote them. ``gm ⊆ brain`` is automatic: the rows are sampled AT the
    brain ``idx``, so a non-brain GM voxel is never stored."""
    gm = np.asanyarray(gm_img.dataobj).reshape(-1)[idx] != 0
    cortical = (np.asanyarray(cortical_img.dataobj).reshape(-1)[idx] != 0) & gm
    return gm.astype(np.uint8), cortical.astype(np.uint8)


def _seg_row(seg_img: nib.Nifti1Image | None, idx: np.ndarray) -> np.ndarray:
    """The int16 SynthSeg-label row at the stored voxels ``idx`` (the brain mask, C-order), sampled
    from the full aseg label volume (same grid). ``None`` -> an all-zero row, so every store carries
    the field even though only the from-raw native cores fill it."""
    if seg_img is None:
        return np.zeros(len(idx), dtype=np.int16)
    return np.asanyarray(seg_img.dataobj).reshape(-1)[idx].astype(np.int16)


# --- quantization -----------------------------------------------------------
def quantize_int8(values: np.ndarray) -> tuple[np.ndarray, float]:
    """Packed finite z-values -> (int8, scale) with ``scale = 1/c`` so that ``int8 * scale``
    recovers z. Full scale is the 99.9th percentile of ``|z|``, per block, after the caller's clamp.

    Finiteness guard: a stray NaN would propagate into the percentile -> a NaN ``scale`` and an
    all-garbage int8 block with no error, so non-finite inputs are zeroed first."""
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    hi = float(np.percentile(np.abs(values), _QUANT_PCT)) if values.size else 1.0
    c = 127.0 / max(hi, 1e-6)
    q = np.clip(np.rint(values * c), -127, 127).astype(np.int8)
    return q, float(1.0 / c)


def _clip_quantize(values: np.ndarray, *, quant: str) -> tuple[np.ndarray, float]:
    """The recipe every writer shares: clamp the z-values to +-`_CLIP_Z`, then quantize ->
    ``(values, scale)``.

    Both products go through here, so pretrain (int8) and eval (f16) cannot drift in value
    fidelity. ``quant="f16"`` stores float16 unquantized (``scale=1.0``)."""
    # finite guard + outlier clamp (new array; never mutate vol.values)
    values = np.clip(
        np.nan_to_num(values, nan=0.0, posinf=_CLIP_Z, neginf=-_CLIP_Z), -_CLIP_Z, _CLIP_Z
    )
    if quant == "int8":
        return quantize_int8(values)
    return values.astype(np.float16), 1.0


# --- pack: dense valid volume -> record -------------------------------------
def pack(vol: ValidVolume, gm_img: nib.Nifti1Image, cortical_img: nib.Nifti1Image,
         seg_img: nib.Nifti1Image | None = None, *, quant: str) -> dict:
    """Pack a `ValidVolume` + its gm / cortical-gm masks (+ optional seg) into a record.

    ``vol``'s mask IS the brain (``idx``); ``gm_img``/``cortical_img`` share its grid and are
    sampled at ``idx`` into the two uint8 rows (`_tissue_rows`, which enforces ``cortical ⊆ gm``).
    ``seg_img`` is the full SynthSeg aseg label volume on the same grid, sampled at ``idx`` into an
    int16 row (`_seg_row`); ``None`` packs an all-zero seg (uniform schema for a core that ships no
    ``dseg``). The inputs are z-scored, so |z|>`_CLIP_Z` is a tail/artifact and the clamp keeps a
    stray outlier (or NaN) from blowing up the int8 scale (see :func:`_clip_quantize`)."""
    mask = vol.mask
    values = np.asarray(vol.values, dtype=np.float32)  # (M,) or (M, T), C-order
    affine = np.asarray(vol.affine, dtype=np.float32)
    idx = np.flatnonzero(mask).astype(np.int32)  # C-order (the brain mask)
    gm, cortical_gm = _tissue_rows(gm_img, cortical_img, idx)
    seg = _seg_row(seg_img, idx)

    values, scale = _clip_quantize(values, quant=quant)

    return {
        "values": values,
        "idx": idx,
        "gm": gm,
        "cortical_gm": cortical_gm,
        "seg": seg,
        "affine": affine,
        "shape": tuple(int(s) for s in mask.shape),
        "scale": float(scale),
    }


# --- clip slicing: a sampled run -> per-clip records ------------------------
def slice_clips(
    vol: ValidVolume,
    gm_img: nib.Nifti1Image,
    cortical_img: nib.Nifti1Image,
    clip_list: list[dict],
    *,
    seg_img: nib.Nifti1Image | None = None,
    n_frames: int,
    quant: str = "f16",
    extra: dict | None = None,
) -> list[dict]:
    """Slice a sampled run's `ValidVolume` into per-clip records -- one per clip window
    ``[start, end)``. The clip geometry (mask/gm/cortical/seg/affine) is the run's grid, identical
    across clips, so each ``pack`` re-derives the same ``idx``/``gm``/``cortical_gm``/``seg`` over the
    sliced ``(M, end-start)`` values. Each rec is ``{"rec": <packed>, "tr", "n_frames", **clip,
    **extra}``; ``eval_arrow`` cherry-picks the label keys it needs (HCP passes
    ``extra={"task": ...}``, NSD none). Asserts each window is in-bounds -- the clip
    index already bounds ``end`` to the resampled run length, so this only fires on a
    real bug (a silent truncation otherwise)."""
    extra = extra or {}
    recs: list[dict] = []
    for c in clip_list:
        assert c["end"] <= vol.n_frames, (
            f"clip [{c['start']},{c['end']}) past run length {vol.n_frames}"
        )
        clip_vol = ValidVolume(
            vol.values[:, c["start"] : c["end"]],
            vol.mask,
            vol.affine,
            tr=vol.tr,
            nyquist_tr=vol.nyquist_tr,  # a time slice changes no rate: carry the run's limiting TR
        )
        recs.append(
            {
                "rec": pack(clip_vol, gm_img, cortical_img, seg_img, quant=quant),
                "tr": float(vol.tr),
                # The run's limiting (nyquist) TR period (`float | None`): a time slice changes no
                # rate, so the clip carries it verbatim. `None` -> `eval_arrow.func_row` writes NaN
                # (Arrow f32 has no null) and a reader treats NaN as UNKNOWN, falling back to `tr`.
                "nyquist_tr": vol.nyquist_tr,
                "n_frames": n_frames,
                **c,
                **extra,
            }
        )
    return recs


# --- npy (de)serialization helpers ------------------------------------------
def _npy_bytes(arr: np.ndarray) -> bytes:
    bio = io.BytesIO()
    np.save(bio, arr, allow_pickle=False)
    return bio.getvalue()


# --- self-contained bundle + BOLD-block codec ------------------------------------
# The store is two record kinds. A per-subject BUNDLE, written once: BOLD geometry + gm/cortical/seg
# rows, the packed 1 mm T1 + its own geometry/rows, and meta. Plus per-run BOLD BLOCKS: values +
# scale + temporal meta, carrying NO geometry. A loader re-attaches the bundle to each block, which
# is what makes a deliverable self-contained -- a block is undecodable without its subject's bundle.
#
# `shape` and T1's `scale` are scalars and ride the json sidecar instead of an npy, which is the
# only reason they are not in this tuple.
_BUNDLE_ARRAYS = (
    layer_key(BOLD, "idx"), layer_key(BOLD, "gm"), layer_key(BOLD, "cortical_gm"),
    layer_key(BOLD, "seg"), layer_key(BOLD, "affine"),
    layer_key(T1, "values"), layer_key(T1, "idx"), layer_key(T1, "gm"), layer_key(T1, "cortical_gm"),
    layer_key(T1, "seg"), layer_key(T1, "affine"),
)

#: The bundle's scalar sidecar entries: the two grid shapes + T1's dequantization scale. BOLD's
#: scale is per-BLOCK (each run quantizes its own z-values), so it is not here.
_BUNDLE_SHAPES = (layer_key(BOLD, "shape"), layer_key(T1, "shape"))
_T1_SCALE = f"{T1}_scale"


def _wire(name: str) -> str:
    """A schema name -> the tar MEMBER suffix it is written under. Lowercase, always.

    WEBDATASET CASE-FOLDS MEMBER SUFFIXES: ``wds.group_by_keys`` splits ``{key}.{suffix}`` and
    lowercases the suffix, so a member written ``sub0.T1_values.npy`` comes back keyed
    ``t1_values.npy`` (measured, not inferred). A reader asking for the schema name then raises
    ``KeyError``, which WebDataset's ``warn_and_continue`` SWALLOWS -- the bundle is dropped, its
    blocks have no geometry to attach, and the whole stream goes silently empty. So the wire format
    is lowercase and the schema is not, and the split stays confined to the tar writers.

    The json sidecar's CONTENTS are untouched (only member NAMES fold), so ``T1_scale``/``T1_shape``
    inside it keep their case."""
    return name.lower()


def _geom(brain_img: nib.Nifti1Image, gm_img: nib.Nifti1Image,
          cortical_img: nib.Nifti1Image, seg_img: nib.Nifti1Image | None = None) -> dict:
    """The codec geometry of a subject's masks: flat C-order ``idx`` (the brain voxels), the two
    ``gm``/``cortical_gm`` uint8 rows + the int16 ``seg`` row at those voxels, ``affine``, ``shape``
    -- no values (a bundle carries geometry once, blocks carry values).

    THE MASK IS ``brain != 0``, and the gm/cortical/seg rows are sampled AT that ``idx``, so the
    three masks are nested by construction and cannot disagree at the storage boundary."""
    brain = np.asanyarray(brain_img.dataobj)
    idx = np.flatnonzero(brain.reshape(-1) != 0).astype(np.int32)
    gm, cortical_gm = _tissue_rows(gm_img, cortical_img, idx)
    return {
        "idx": idx,
        "gm": gm,
        "cortical_gm": cortical_gm,
        "seg": _seg_row(seg_img, idx),
        "affine": np.asarray(brain_img.affine, np.float32),
        "shape": tuple(int(s) for s in brain.shape),
    }


def pack_bundle(
    brain_f_img: nib.Nifti1Image,
    gm_f_img: nib.Nifti1Image,
    cortical_f_img: nib.Nifti1Image,
    t1_vol,
    gm1_img: nib.Nifti1Image,
    cortical1_img: nib.Nifti1Image,
    meta: dict,
    *,
    quant: str = "int8",
    seg_f_img: nib.Nifti1Image | None = None,
    seg1_img: nib.Nifti1Image | None = None,
) -> dict:
    """One subject·space bundle: BOLD geometry (``bold_*``) + packed 1 mm T1 with its
    geometry/masks (``T1_*``) + ``meta``. ``t1_vol`` is the z-scored T1 ``ValidVolume`` (1 mm);
    ``quant`` is ``int8`` (pretrain) or ``f16`` (eval).

    There is no ``bold_values`` here -- BOLD's values are per-run and live in the blocks, which is
    the one asymmetry the schema has and the reason a bundle alone decodes nothing.

    ``brain_f_img``/``gm_f_img``/``cortical_f_img`` are the func-grid brain / gm / cortical-gm masks;
    the BOLD mask is ``brain_f != 0`` and gm/cortical ride as uint8 rows sampled at it. The T1's
    ``idx`` comes off ``t1_vol`` itself (z-scored over the 1 mm brain mask) and its masks are sampled
    at THAT ``idx`` -- the same rule. ``seg_f_img``/``seg1_img`` are the func-grid / 1 mm aseg label
    volumes, sampled into the int16 ``*_seg`` rows; ``None`` -> an all-zero seg.

    Neither layer's RESOLUTION is stored: it is the affine's column norms."""
    bold = _geom(brain_f_img, gm_f_img, cortical_f_img, seg_f_img)
    t1 = pack(t1_vol, gm1_img, cortical1_img, seg1_img, quant=quant)  # values/idx/gm/cortical_gm/seg/affine/shape/scale
    return {
        **{layer_key(BOLD, f): bold[f] for f in ("idx", "gm", "cortical_gm", "seg", "affine", "shape")},
        **{layer_key(T1, f): t1[f] for f in LAYER_FIELDS},
        _T1_SCALE: float(t1["scale"]),
        "meta": dict(meta),
    }


def pack_bold(vol, *, quant: str = "int8") -> dict:
    """One BOLD block: packed quantized ``(M, T)`` z-values + ``scale`` + temporal meta. Carries
    NO geometry -- the mask/gm/cortical/affine live in the subject bundle, re-attached at load
    (voxel size included: it is the bundle affine's column norms)."""
    values = np.asarray(vol.values, dtype=np.float32)  # (M, T) or (M,)
    values, scale = _clip_quantize(values, quant=quant)
    out = {"values": values, "scale": float(scale), "tr": float(vol.tr), "n_frames": int(vol.n_frames)}
    # OPTIONAL: `tr` is the STORED rate, which after a resample says nothing about the signal band.
    # `nyquist_tr` separates a native 1.0 s run from a 2.5 s run upsampled to 1.0 s -- identical in
    # `tr`, different in real bandwidth. Absent -> UNKNOWN, and the reader falls back to `tr`.
    if getattr(vol, "nyquist_tr", None) is not None:
        out["nyquist_tr"] = float(vol.nyquist_tr)
    # OPTIONAL (~1% of a shard): the pre-z mean/std the z-score divided out; together they invert it.
    # `mean` is the ONLY anatomy left from the run -- it IS the mean EPI on this grid -- so a shard can
    # evidence its own BOLD->T1 registration standalone. `std` is pre-resample, hence the honest tSNR
    # denominator (the stored series is TR-interpolated, which smooths variance). Absent when the caller
    # has no z-score to report (e.g. clip slicing) and from shards predating the field -- treat as optional.
    if getattr(vol, "mean", None) is not None and getattr(vol, "std", None) is not None:
        out["mean"], out["mean_scale"] = _pack_f16_scaled(vol.mean)
        out["std"], out["std_scale"] = _pack_f16_scaled(vol.std)
    return out


def _pack_f16_scaled(a: np.ndarray) -> tuple[np.ndarray, float]:
    """Raw-intensity ``(M,)`` -> ``(f16, scale)``; ``f16 * scale`` recovers it.

    A bare f16 cast is NOT safe here the way it is for z-values: these are raw scanner intensities,
    which can exceed f16's 65504 ceiling and would silently become ``inf``. Non-finite inputs are
    zeroed before the peak is taken so a stray NaN cannot poison the scale (the same failure mode
    `quantize_int8` guards)."""
    a = np.nan_to_num(np.asarray(a, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.abs(a).max()) if a.size else 0.0
    scale = peak / 30000.0 if peak > 30000.0 else 1.0  # headroom under the f16 max
    return (a / scale).astype(np.float16), float(scale)


def bundle_to_tar_members(key: str, bundle: dict) -> dict[str, bytes]:
    """Serialize a bundle to WDS members ``{key}.{name}.npy`` + a ``{key}.json`` sidecar
    (``kind=bundle``; scales/shapes/meta). ``key`` is the subject work-key (no ``.``)."""
    members = {
        f"{key}.{_wire(name)}.npy": _npy_bytes(np.asarray(bundle[name]))
        for name in _BUNDLE_ARRAYS
    }
    sidecar = {
        "kind": "bundle",
        "schema_version": SCHEMA_VERSION,
        _T1_SCALE: float(bundle[_T1_SCALE]),
        **{k: list(bundle[k]) for k in _BUNDLE_SHAPES},
        **bundle["meta"],
    }
    members[f"{key}.json"] = json.dumps(sidecar).encode()
    return members


def bold_to_tar_members(key: str, bold: dict, extra: dict | None = None) -> dict[str, bytes]:
    """Serialize a BOLD block to ``{key}.values.npy`` + ``{key}.json`` (``kind=bold``; scale,
    tr, n_frames, the optional ``nyquist_tr``, + ``extra`` labels like task/phase)."""
    sidecar = {
        "kind": "bold",
        "scale": float(bold["scale"]),
        "tr": float(bold["tr"]),
        "n_frames": int(bold["n_frames"]),
        # Mirrors `pack_bold`: the precomputed limiting TR period, only when the block knows it.
        # A reader treats a missing key as unknown and falls back to `tr`.
        **({"nyquist_tr": float(bold["nyquist_tr"])} if "nyquist_tr" in bold else {}),
        **(extra or {}),
    }
    members = {
        f"{key}.values.npy": _npy_bytes(np.asarray(bold["values"])),
    }
    # Mirrors `pack_bold`: emitted only when the caller had a z-score to report.
    if "mean" in bold and "std" in bold:
        members[f"{key}.mean.npy"] = _npy_bytes(np.asarray(bold["mean"]))
        members[f"{key}.std.npy"] = _npy_bytes(np.asarray(bold["std"]))
        sidecar["mean_scale"] = float(bold["mean_scale"])
        sidecar["std_scale"] = float(bold["std_scale"])
    members[f"{key}.json"] = json.dumps(sidecar).encode()
    return members


def requantize_bundle_t1(bundle: dict, quant: str) -> dict:
    """Re-quantize a (canonical f16) bundle's T1 to ``quant`` for a product's serialization
    (``int8`` for the pretrain tar; ``f16`` = no-op for the eval anat row). The T1 is already
    z-scored + clipped, so int8 re-quant just derives a fresh scale (NOT via `_clip_quantize`:
    re-clipping an already-clipped T1 would compound the clamp)."""
    if quant == "f16":
        return bundle
    q, scale = quantize_int8(np.asarray(bundle[layer_key(T1, "values")], np.float32))
    return {**bundle, layer_key(T1, "values"): q, _T1_SCALE: float(scale)}
