#!/usr/bin/env python3
"""NSD vendor core: the func1mm timeseries warped to MNI -> preprocessed BIDS on the shared 2 mm grid.

The VENDOR half of NSD's native-vs-vendor pair. NSD ships a bespoke ``func1pt0 -> MNI`` transform as
a COORDINATE VOLUME (per MNI voxel, the func1pt0 voxel it came from) rather than a displacement
field, so `warp_to_canonical_mni` / `warp_dense_mni` sample through it directly. Everything else is
the usual: SynthSeg on the T1, the same masks, the same 2 mm grid. The from-raw twin is
``nsd_native.py``.

Imports NOTHING from the package (a pinned invariant), so it lifts out as a standalone script.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shlex
import shutil
import subprocess
from io import BytesIO
from pathlib import Path

import nibabel as nib
import numpy as np
import nitransforms as nt
from scipy.ndimage import map_coordinates

GM_LABELS = (3, 42, 8, 47, 10, 49, 11, 50, 12, 51, 13, 52, 17, 53, 18, 54, 26, 58)
CORTICAL_GM_LABELS = (3, 42)
NEAREST, TRILINEAR, CUBIC = 0, 1, 3
CLI_ENV = {
    "FSLOUTPUTTYPE": "NIFTI_GZ",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": "1",
}
MIN_PLAUSIBLE_BRAIN_VOXELS = 100_000
# A faithful cubic reslice preserves the source T1's intensity range; `mri_convert` WITHOUT
# `-odt float` blows it up by orders of magnitude. The check is therefore scale-INVARIANT (resliced
# peak vs source peak), not an absolute cap: a derivative T1 can carry a large but faithful INU
# scale, and the z-score removes it downstream anyway.
RESLICE_OVERSHOOT_TOLERANCE = 4.0
#: BOLD lands on the 2 mm grid, which nests exactly 2x inside the 1 mm one. ONE production
#: value, so a constant rather than a knob.
BOLD_MM = 2.0
FSL_MNI_1MM_SHAPE = (182, 218, 182)
FSL_MNI_1MM_AFFINE = np.array(
    [[-1, 0, 0, 90], [0, 1, 0, -126], [0, 0, 1, -72], [0, 0, 0, 1]], dtype=float
)
_EPS = 1e-4
#: No SynthMorph / lta_convert here: this core runs no registration of its own.
REQUIRED_TOOLS = ("mri_synthseg", "mri_convert", "mri_mask")
GZIP_MAGIC = b"\x1f\x8b"
# The staged-input names, spelled the same way in `discover/nsd.py`. Change one, change both.
T1 = "t1.nii.gz"
XFM = "func1pt0-to-MNI.nii.gz"
_BADVAL_SENTINEL = 9999.0  # the transform volume's "no source voxel here" marker
_A1_CACHE: dict[str, np.ndarray] = {}


def is_s3(uri) -> bool:
    return str(uri).startswith("s3://")


def s3_filesystem(storage_options=None):
    import s3fs

    return s3fs.S3FileSystem(**(storage_options or {}))


def s3fs_nifti_read(fname, storage_options=None):
    with s3_filesystem(storage_options).open(fname) as f:
        bb = BytesIO()
        is_gz = f.read(2) == GZIP_MAGIC
        f.seek(0)
        src = gzip.open(f) if is_gz else f
        try:
            shutil.copyfileobj(src, bb)
        finally:
            if is_gz:
                src.close()
        bb.seek(0)
        fh = nib.FileHolder(fileobj=bb)
        return nib.Nifti1Image.from_file_map({"header": fh, "image": fh})


def read_nifti(fname, *, storage_options=None):
    if is_s3(fname):
        return s3fs_nifti_read(fname, storage_options)
    return nib.load(fname)


def sanitize_label(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", key)


def run_cli(command_block: str) -> None:
    for line in filter(None, (ln.strip() for ln in command_block.strip().splitlines())):
        argv = shlex.split(line)
        proc = subprocess.run(argv, capture_output=True, text=True, env={**os.environ, **CLI_ENV})
        if proc.returncode != 0:
            raise RuntimeError(f"{argv[0]} failed ({proc.returncode}):\n{proc.stderr[-2000:]}")


def preflight(cfg=None) -> None:
    missing = [t for t in REQUIRED_TOOLS if shutil.which(t) is None]
    if missing:
        raise RuntimeError(
            f"builder preflight: CLIs not on PATH: {missing}. Need FreeSurfer + (per core) FSL/ANTs; "
            f"set FREESURFER_HOME/FSLDIR and prepend their bin/ to PATH."
        )
    fs_home = os.environ.get("FREESURFER_HOME", "")
    have_license = os.environ.get("FS_LICENSE") or any(
        os.path.exists(os.path.join(fs_home, n)) for n in ("license.txt", ".license") if fs_home
    )
    if not have_license:
        raise RuntimeError(
            "builder preflight: no FreeSurfer license (mri_synthseg requires one) -- set FS_LICENSE "
            "to a license file, or place license.txt under FREESURFER_HOME."
        )


def segment(t1, out_seg) -> None:
    run_cli(f"mri_synthseg --i {t1} --o {out_seg} --robust --cpu --threads 1")


def reslice_and_skullstrip(t1, seg, out_1mm, out_brain) -> None:
    run_cli(f"""
        mri_convert -rl {seg} -rt cubic -odt float {t1} {out_1mm}
        mri_mask {out_1mm} {seg} {out_brain}
    """)


def reject_failed_segmentation(key, segmentation) -> None:
    brain_voxels = int((np.asanyarray(segmentation.dataobj) != 0).sum())
    if brain_voxels < MIN_PLAUSIBLE_BRAIN_VOXELS:
        raise RuntimeError(
            f"{key}: SynthSeg produced only {brain_voxels} brain voxels "
            f"(< {MIN_PLAUSIBLE_BRAIN_VOXELS}) -- it likely ran out of memory; "
            f"refusing to pack empty masks"
        )


def reject_corrupt_reslice(key, resliced_t1, source_t1) -> None:
    resliced_peak = float(np.abs(np.asanyarray(resliced_t1.dataobj)).max())
    source_peak = float(np.abs(np.asanyarray(source_t1.dataobj)).max())
    if resliced_peak > RESLICE_OVERSHOOT_TOLERANCE * max(source_peak, 1.0):
        raise RuntimeError(
            f"{key}: resliced T1 peak {resliced_peak:.3g} exceeds "
            f"{RESLICE_OVERSHOOT_TOLERANCE:g}x the source T1 peak {source_peak:.3g} -- the reslice "
            f"blew up the intensity range (mri_convert -rl needs -odt float); refusing to store garbage"
        )


def round_to_labels(img) -> np.ndarray:
    return np.rint(np.asanyarray(img.dataobj)).astype(np.int16)


def mask_image(mask, reference):
    return nib.Nifti1Image(np.asarray(mask, np.uint8), reference.affine)


def resample_onto(transform, img, reference, *, order):
    xfm = nt.Affine(np.asarray(transform)) if isinstance(transform, np.ndarray) else transform
    return nt.apply(xfm, img, reference=reference, order=order)


def brain_extract(t1, seg_arr):
    data = np.where(seg_arr != 0, np.asarray(t1.dataobj, np.float32), 0.0)
    return nib.Nifti1Image(data.astype(np.float32), t1.affine)


def _zooms(affine):
    return np.sqrt((np.asarray(affine, float)[:3, :3] ** 2).sum(axis=0))


def iso_ref(ref, mm):
    """An isotropic ``mm`` grid covering the same field of view as `ref`."""
    z = _zooms(ref.affine)
    mm = np.broadcast_to(np.asarray(mm, dtype=float), (3,))
    aff = np.asarray(ref.affine, dtype=float).copy()
    aff[:3, :3] = (aff[:3, :3] / z) * mm
    shape = tuple(int(np.ceil(s * zz / m - _EPS)) for s, zz, m in zip(ref.shape[:3], z, mm))
    return nib.Nifti1Image(np.zeros(shape, np.float32), aff)


def fsl_mni_1mm_grid():
    return nib.Nifti1Image(np.zeros(FSL_MNI_1MM_SHAPE, np.float32), FSL_MNI_1MM_AFFINE)


def assert_nested_2x(struct, func) -> None:
    """The func grid must nest exactly 2x inside the 1 mm grid -- `warp_dense_mni` indexes the
    transform volume at ``2*ijk``, which is only the right voxel if it does."""
    sa, fa = struct.affine, func.affine
    if not np.allclose(fa[:3, :3], 2.0 * sa[:3, :3]):
        raise ValueError(f"func 3x3 {fa[:3, :3].tolist()} != 2 x struct {sa[:3, :3].tolist()}")
    if not np.allclose(fa[:3, 3], sa[:3, 3]):
        raise ValueError(f"func origin {fa[:3, 3].tolist()} != struct origin {sa[:3, 3].tolist()}")
    expect = tuple(int(np.ceil(s / 2)) for s in struct.shape[:3])
    if tuple(func.shape[:3]) != expect:
        raise ValueError(f"func shape {tuple(func.shape[:3])} != ceil(struct/2) {expect}")


def bold_coverage(bold_on_grid):
    data = np.asarray(bold_on_grid.dataobj, np.float32)
    return (np.abs(data).max(axis=-1) > 0) if data.ndim == 4 else (np.abs(data) > 0)


def func1pt0_coords(a1):
    """The transform volume as ``(3, N)`` func1pt0 voxel coordinates + a "no source" mask. NSD stores
    them 1-based, hence the -1."""
    coords = np.stack([a1[..., c].ravel(order="F") for c in range(3)]).astype(np.float64)
    bad = np.any(coords == _BADVAL_SENTINEL, axis=0)
    return coords - 1.0, bad


def _prep_coords(coords, bad, src_shape):
    fill = (
        bad
        | (coords[0] < 0) | (coords[0] > src_shape[0] - 1)
        | (coords[1] < 0) | (coords[1] > src_shape[1] - 1)
        | (coords[2] < 0) | (coords[2] > src_shape[2] - 1)
    )
    safe = coords.copy()
    safe[:, fill] = 0.0
    return safe, fill


def _sample(src3d, safe, fill, order):
    out = map_coordinates(src3d, safe, order=order, mode="nearest")
    out[fill] = 0.0
    return out


def warp_to_canonical_mni(src_raw_img, a1, *, order, coords=None, bad=None):
    if tuple(a1.shape[:3]) != FSL_MNI_1MM_SHAPE:
        raise ValueError(f"transform grid {a1.shape[:3]} != MNI {FSL_MNI_1MM_SHAPE}")
    if coords is None:
        coords, bad = func1pt0_coords(a1)
    src = np.nan_to_num(np.asarray(src_raw_img.dataobj, dtype=np.float64))
    safe, fill = _prep_coords(coords, bad, src.shape)
    out = _sample(src, safe, fill, order).reshape(FSL_MNI_1MM_SHAPE, order="F")
    return nib.Nifti1Image(out, FSL_MNI_1MM_AFFINE)


def warp_dense_mni(bold_raw_img, func_grid, a1, *, order=1, chunk_tr=128):
    if bold_raw_img.ndim != 4:
        raise ValueError(f"expected 4D BOLD, got shape {bold_raw_img.shape}")
    if tuple(a1.shape[:3]) != FSL_MNI_1MM_SHAPE:
        raise ValueError(f"transform grid {a1.shape[:3]} != MNI {FSL_MNI_1MM_SHAPE}")
    assert_nested_2x(fsl_mni_1mm_grid(), func_grid)
    shape = tuple(func_grid.shape[:3])
    ijk = np.indices(shape).reshape(3, -1).T
    i1mm = 2 * ijk
    fc = a1[i1mm[:, 0], i1mm[:, 1], i1mm[:, 2], :].astype(np.float64)
    bad = np.any(fc == _BADVAL_SENTINEL, axis=1)
    safe, fill = _prep_coords((fc - 1.0).T, bad, bold_raw_img.shape[:3])

    n_tr = bold_raw_img.shape[3]
    vals = np.empty((ijk.shape[0], n_tr), dtype=np.float32)
    dataobj = bold_raw_img.dataobj
    for t0 in range(0, n_tr, chunk_tr):
        t1 = min(t0 + chunk_tr, n_tr)
        slab = np.nan_to_num(np.asarray(dataobj[..., t0:t1], dtype=np.float64))
        for t in range(t1 - t0):
            vals[:, t0 + t] = _sample(slab[..., t], safe, fill, order)

    dense = vals.reshape(shape + (n_tr,), order="C")
    return nib.Nifti1Image(dense.astype(np.float32), func_grid.affine)


def load_a1(raw_dir) -> np.ndarray:
    path = str(Path(raw_dir) / XFM)
    if path not in _A1_CACHE:
        _A1_CACHE[path] = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    return _A1_CACHE[path]


def _seg_func_path(work) -> Path:
    return Path(work) / "seg_func_mni.nii.gz"


def _write_anat(label, raw_dir, out_bids, work, cfg, segmentation, brain):
    """The T1 + its segmentation through NSD's func1pt0->MNI coordinate volume, then the func-grid
    segmentation the run stage reads. One `func1pt0_coords` for all three, since they share it."""
    anat = Path(out_bids) / f"sub-{label}" / "anat"
    a1 = load_a1(raw_dir)
    coords, bad = func1pt0_coords(a1)
    seg_anat_img = warp_to_canonical_mni(segmentation, a1, order=NEAREST, coords=coords, bad=bad)
    anat_ref = seg_anat_img
    anat_t1 = brain_extract(
        warp_to_canonical_mni(brain, a1, order=CUBIC, coords=coords, bad=bad),
        round_to_labels(seg_anat_img))
    seg_anat_arr = round_to_labels(seg_anat_img)
    func_grid = iso_ref(fsl_mni_1mm_grid(), BOLD_MM)
    seg_func = round_to_labels(resample_onto(np.eye(4), seg_anat_img, func_grid, order=NEAREST))

    nib.save(anat_t1, anat / f"sub-{label}_space-mni_desc-preproc_T1w.nii.gz")
    nib.save(mask_image(seg_anat_arr != 0, anat_ref),
             anat / f"sub-{label}_space-mni_desc-brain_mask.nii.gz")
    nib.save(mask_image(np.isin(seg_anat_arr, GM_LABELS), anat_ref),
             anat / f"sub-{label}_space-mni_label-GM_mask.nii.gz")
    nib.save(mask_image(np.isin(seg_anat_arr, CORTICAL_GM_LABELS), anat_ref),
             anat / f"sub-{label}_space-mni_label-corticalGM_mask.nii.gz")
    nib.save(nib.Nifti1Image(seg_func, func_grid.affine), _seg_func_path(work))


def _anat(item, raw_dir, out_bids, work, cfg) -> None:
    label = sanitize_label(item["key"])
    work.mkdir(parents=True, exist_ok=True)
    sub = Path(out_bids) / f"sub-{label}"
    (sub / "anat").mkdir(parents=True, exist_ok=True)
    (sub / "func").mkdir(parents=True, exist_ok=True)

    t1 = Path(raw_dir) / T1
    seg_path = work / "seg.nii.gz"
    t1_1mm, t1_brain = work / "t1_1mm.nii.gz", work / "t1_brain.nii.gz"
    segment(t1, seg_path)
    reslice_and_skullstrip(t1, seg_path, t1_1mm, t1_brain)
    segmentation = nib.load(seg_path)
    reject_failed_segmentation(item["key"], segmentation)
    reject_corrupt_reslice(item["key"], nib.load(t1_1mm), nib.load(t1))
    _write_anat(label, raw_dir, out_bids, work, cfg, segmentation, nib.load(t1_brain))


def _one_run(item, r, raw_dir, out_bids, work, cfg) -> dict:
    label = sanitize_label(item["key"])
    func = Path(out_bids) / f"sub-{label}" / "func"
    func.mkdir(parents=True, exist_ok=True)
    # Streamed, not pre-staged: a subject has hundreds of runs (see `discover/nsd.py`).
    raw_bold = read_nifti(r["uri"], storage_options=cfg.get("input_storage_options"))

    func_grid = iso_ref(fsl_mni_1mm_grid(), BOLD_MM)
    seg_func = round_to_labels(nib.load(_seg_func_path(work)))
    on_grid = warp_dense_mni(raw_bold, func_grid, load_a1(raw_dir))
    covered = bold_coverage(on_grid)
    stem = f"sub-{label}_run-{r['idx']:02d}_space-mni"
    nib.save(on_grid, func / f"{stem}_desc-preproc_bold.nii")
    nib.save(mask_image((seg_func != 0) & covered, func_grid), func / f"{stem}_desc-brain_mask.nii.gz")
    nib.save(mask_image(np.isin(seg_func, GM_LABELS) & covered, func_grid),
             func / f"{stem}_label-GM_mask.nii.gz")
    nib.save(mask_image(np.isin(seg_func, CORTICAL_GM_LABELS) & covered, func_grid),
             func / f"{stem}_label-corticalGM_mask.nii.gz")

    entry = {"idx": r["idx"], "run": r["run"], "ses": r["ses"], "run_num": r["run_num"],
             "tr_acq": r["tr_acq"], "native_mm": float(max(raw_bold.header.get_zooms()[:3]))}
    (Path(work) / f"run_{r['idx']:05d}.json").write_text(json.dumps(entry))
    return entry


def _finish(item, out_bids, work) -> None:
    label = sanitize_label(item["key"])
    sub = Path(out_bids) / f"sub-{label}"
    sub.mkdir(parents=True, exist_ok=True)
    written = [json.loads(p.read_text()) for p in
               sorted(Path(work).glob("run_*.json"), key=lambda q: int(q.stem.split("_")[1]))]
    # `spaces` is the CLAIMED set, not the achieved one: the finalizer keys its completeness gate on
    # it to tell a modality-less subject from a dropped save.
    (sub / f"sub-{label}_desc-fmridata.json").write_text(json.dumps(
        {"key": item["key"], "label": label, "spaces": ["mni"], "runs": written,
         "meta": {}}, indent=2))


def _preproc(item, raw_dir, out_bids, work, cfg) -> None:
    _anat(item, raw_dir, out_bids, work, cfg)
    for r in item["sources"]["eval_runs"]:
        _one_run(item, r, raw_dir, out_bids, work, cfg)
    _finish(item, out_bids, work)


def process(item: dict, raw_dir: str, out_bids: str, cfg: dict, stage="all", run_idx=None) -> None:
    work = Path(raw_dir) / "work"
    if stage == "anat":
        _anat(item, raw_dir, out_bids, work, cfg)
    elif stage == "run":
        r = next(x for x in item["sources"]["eval_runs"] if int(x["idx"]) == int(run_idx))
        _one_run(item, r, raw_dir, out_bids, work, cfg)
    elif stage == "finish":
        _finish(item, out_bids, work)
    else:
        _preproc(item, raw_dir, out_bids, work, cfg)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--item", default=None)
    ap.add_argument("raw", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--cfg", default=None)
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--stage", choices=("all", "anat", "run", "finish"), default="all")
    ap.add_argument("--run-idx", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = json.loads(Path(args.cfg).read_text()) if args.cfg else {}
    if args.preflight:
        preflight(cfg)
        return
    if args.stage == "run" and args.run_idx is None:
        ap.error("--stage run needs --run-idx")
    process(json.loads(Path(args.item).read_text()), args.raw, args.out, cfg,
            stage=args.stage, run_idx=args.run_idx)


if __name__ == "__main__":
    main()
