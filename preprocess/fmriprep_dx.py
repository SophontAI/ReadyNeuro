#!/usr/bin/env python3
"""The shared vendor core for the fMRIPrep diagnosis cohorts (PPMI / ABIDE / ADHD-200).

The VENDOR half of those datasets' native-vs-vendor pairs. fMRIPrep already put the BOLD in MNI, so
this core runs no registration of its own: SynthSeg on the subject-space T1 (one segmenter across the
whole corpus), the shipped ``T1w -> MNI`` composite ``.h5`` to carry the segmentation over, and one
resample of the first REST window onto the shared 2 mm grid. The from-raw twins run ``openneuro.py``.

Imports NOTHING from the package (a pinned invariant), so it lifts out as a standalone script.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
from math import ceil
from pathlib import Path

import nibabel as nib
import numpy as np
import nitransforms as nt

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
FSL_MNI_1MM_SHAPE = (182, 218, 182)
FSL_MNI_1MM_AFFINE = np.array(
    [[-1, 0, 0, 90], [0, 1, 0, -126], [0, 0, 1, -72], [0, 0, 0, 1]], dtype=float
)
_EPS = 1e-4
#: No SynthMorph / lta_convert here: this core runs no registration of its own.
REQUIRED_TOOLS = ("mri_synthseg", "mri_convert", "mri_mask")
#: The two geometry settings have exactly ONE production value each and are recorded in the
#: sidecar meta: the T1 (and the aseg) stay on the 1 mm grid...
T1_MM = 1.0
BOLD_MM = 2.0  # ...and BOLD lands on the 2 mm grid, which nests exactly 2x inside it
WINDOW_SECONDS = 300   # the eval window; SLICE_PAD_S of slack so the resample cannot come up short
SLICE_PAD_S = 10
# The staged-input names, spelled the same way in `discover/fmriprep_dx.py`. Change one, change both.
T1W = "t1w.nii.gz"
FWD_H5 = "from-T1w_to-MNI.h5"
BOLD_MNI = "bold_mni.nii.gz"


def run_cli(command_block):
    for line in filter(None, (ln.strip() for ln in command_block.strip().splitlines())):
        argv = shlex.split(line)
        proc = subprocess.run(
            argv, capture_output=True, text=True, env={**os.environ, **CLI_ENV}
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"{argv[0]} failed ({proc.returncode}):\n{proc.stderr[-2000:]}"
            )


def preflight(cfg):
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


def segment(t1, out_seg):
    run_cli(f"mri_synthseg --i {t1} --o {out_seg} --robust --cpu --threads 1")


def reslice_and_skullstrip(t1, seg, out_1mm, out_brain):
    run_cli(f"""
        mri_convert -rl {seg} -rt cubic -odt float {t1} {out_1mm}
        mri_mask {out_1mm} {seg} {out_brain}
    """)


def reject_failed_segmentation(key, segmentation):
    brain_voxels = int((np.asanyarray(segmentation.dataobj) != 0).sum())
    if brain_voxels < MIN_PLAUSIBLE_BRAIN_VOXELS:
        raise RuntimeError(
            f"{key}: SynthSeg produced only {brain_voxels} brain voxels "
            f"(< {MIN_PLAUSIBLE_BRAIN_VOXELS}) -- it likely ran out of memory; "
            f"refusing to pack empty masks"
        )


def reject_corrupt_reslice(key, resliced_t1, source_t1):
    resliced_peak = float(np.abs(np.asanyarray(resliced_t1.dataobj)).max())
    source_peak = float(np.abs(np.asanyarray(source_t1.dataobj)).max())
    if resliced_peak > RESLICE_OVERSHOOT_TOLERANCE * max(source_peak, 1.0):
        raise RuntimeError(
            f"{key}: resliced T1 peak {resliced_peak:.3g} exceeds "
            f"{RESLICE_OVERSHOOT_TOLERANCE:g}x the source T1 peak {source_peak:.3g} -- the reslice "
            f"blew up the intensity range (mri_convert -rl needs -odt float); refusing to store garbage"
        )


def round_to_labels(img):
    return np.rint(np.asanyarray(img.dataobj)).astype(np.int16)


def mask_image(mask, reference):
    return nib.Nifti1Image(np.asarray(mask, np.uint8), reference.affine)


def resample_onto(transform, img, reference, *, order):
    xfm = nt.Affine(np.asarray(transform)) if isinstance(transform, np.ndarray) else transform
    return nt.apply(xfm, img, reference=reference, order=order)


def itk_h5_chain(h5_path):
    """The shipped fMRIPrep ``T1w -> MNI`` composite ``.h5`` as a nitransforms chain (one affine +
    one displacement field, in that order)."""
    from nitransforms.io.itk import ITKCompositeH5
    from nitransforms.linear import Affine
    from nitransforms.manip import TransformChain
    from nitransforms.nonlinear import DenseFieldTransform

    affines, fields = [], []
    for c in ITKCompositeH5.from_filename(h5_path):
        if hasattr(c, "to_ras"):
            affines.append(Affine(c.to_ras()))
        else:
            fields.append(DenseFieldTransform(c))
    if len(affines) != 1 or len(fields) != 1:
        raise ValueError(
            f"{h5_path}: expected 1 affine + 1 displacement field, got "
            f"{len(affines)} affine(s) + {len(fields)} field(s)"
        )
    return TransformChain([fields[0], affines[0]])


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


def _func_to_moving_coords(reference, moving_affine, sel_idx=None):
    """Voxel coordinates in the moving image for each (selected) voxel of `reference`."""
    from nibabel.affines import apply_affine

    if sel_idx is None:
        ijk = np.indices(reference.shape[:3]).reshape(3, -1).T.astype(np.float64)
    else:
        ijk = np.stack(np.unravel_index(sel_idx, reference.shape[:3]), axis=1).astype(np.float64)
    world = apply_affine(np.asarray(reference.affine, float), ijk)
    return apply_affine(np.linalg.inv(np.asarray(moving_affine, float)), world).T


def _same_grid(a, b):
    return tuple(a.shape[:3]) == tuple(b.shape[:3]) and np.allclose(
        np.asarray(a.affine), np.asarray(b.affine), atol=1e-3)


def resample_bold(bold, reference, *, order=TRILINEAR, sel):
    """The shipped MNI BOLD onto the func grid. No transform: it is already in MNI space, so this is
    a grid change only -- and a no-op copy when the two grids already agree."""
    from scipy.ndimage import map_coordinates

    ref_shape = tuple(reference.shape[:3])
    if bold.ndim == 4 and _same_grid(bold, reference):
        return nib.Nifti1Image(np.asarray(bold.dataobj, np.float32), reference.affine)

    sel_idx = np.flatnonzero(np.asarray(sel, bool).reshape(-1))
    coords = _func_to_moving_coords(reference, bold.affine, sel_idx)
    dataobj = bold.dataobj
    n_t = bold.shape[3]
    out = np.zeros((int(np.prod(ref_shape)), n_t), np.float32)
    rows = sel_idx
    for t in range(n_t):
        out[rows, t] = map_coordinates(np.asarray(dataobj[..., t], np.float32), coords,
                                       order=order, mode="constant", cval=0.0)
    return nib.Nifti1Image(out.reshape(ref_shape + (n_t,)), reference.affine)


def bold_coverage(bold_on_grid):
    data = np.asarray(bold_on_grid.dataobj, np.float32)
    return (np.abs(data).max(axis=-1) > 0) if data.ndim == 4 else (np.abs(data) > 0)


def fsl_mni_1mm_grid():
    return nib.Nifti1Image(np.zeros(FSL_MNI_1MM_SHAPE, np.float32), FSL_MNI_1MM_AFFINE)


def brain_extract(t1, seg_arr):
    data = np.where(seg_arr != 0, np.asarray(t1.dataobj, np.float32), 0.0)
    return nib.Nifti1Image(data.astype(np.float32), t1.affine)


def sanitize_label(key):
    return re.sub(r"[^A-Za-z0-9]", "", key)


def n_native(cfg, tr_acq):
    """How many acquired frames cover the eval window, with slack so the resample cannot come up
    short of ``window_frames``."""
    return ceil((float(cfg.get("window_seconds", WINDOW_SECONDS)) + SLICE_PAD_S) / float(tr_acq))


def _write_anat(label, raw, work, segmentation, out_bids):
    """The shipped T1 + the segmentation, both carried into MNI through the shipped composite
    warp, plus the func-grid segmentation the run stage reads."""
    anat = Path(out_bids) / f"sub-{label}" / "anat"
    anat_ref = fsl_mni_1mm_grid()
    seg_xfm = itk_h5_chain(str(raw / FWD_H5))
    seg_anat = round_to_labels(resample_onto(seg_xfm, segmentation, anat_ref, order=NEAREST))
    anat_t1 = brain_extract(
        resample_onto(seg_xfm, nib.load(raw / T1W), anat_ref, order=CUBIC), seg_anat)
    func_grid = iso_ref(anat_ref, BOLD_MM)
    seg_func = round_to_labels(resample_onto(seg_xfm, segmentation, func_grid, order=NEAREST))
    nib.save(nib.Nifti1Image(seg_func, func_grid.affine), work / "seg_func_mni.nii.gz")

    nib.save(anat_t1, anat / f"sub-{label}_space-mni_desc-preproc_T1w.nii.gz")
    nib.save(mask_image(seg_anat != 0, anat_ref),
             anat / f"sub-{label}_space-mni_desc-brain_mask.nii.gz")
    nib.save(mask_image(np.isin(seg_anat, GM_LABELS), anat_ref),
             anat / f"sub-{label}_space-mni_label-GM_mask.nii.gz")
    nib.save(mask_image(np.isin(seg_anat, CORTICAL_GM_LABELS), anat_ref),
             anat / f"sub-{label}_space-mni_label-corticalGM_mask.nii.gz")


def _write_run(label, raw, work, out_bids, n_nat, run_idx):
    func = Path(out_bids) / f"sub-{label}" / "func"
    func_grid = iso_ref(fsl_mni_1mm_grid(), BOLD_MM)
    seg_func = np.asanyarray(nib.load(work / "seg_func_mni.nii.gz").dataobj).astype(np.int16)

    raw_bold = nib.load(raw / BOLD_MNI)
    window = raw_bold.slicer[..., : min(n_nat, raw_bold.shape[3])]
    on_grid = resample_bold(window, func_grid, order=TRILINEAR, sel=seg_func != 0)
    covered = bold_coverage(on_grid)
    stem = f"sub-{label}_run-{run_idx:02d}_space-mni"
    nib.save(on_grid, func / f"{stem}_desc-preproc_bold.nii")
    nib.save(mask_image((seg_func != 0) & covered, func_grid), func / f"{stem}_desc-brain_mask.nii.gz")
    nib.save(mask_image(np.isin(seg_func, GM_LABELS) & covered, func_grid),
             func / f"{stem}_label-GM_mask.nii.gz")
    nib.save(mask_image(np.isin(seg_func, CORTICAL_GM_LABELS) & covered, func_grid),
             func / f"{stem}_label-corticalGM_mask.nii.gz")


def _anat(item, raw_dir, out_bids, cfg):
    label = sanitize_label(item["key"])
    raw, work = Path(raw_dir), Path(raw_dir) / "work"
    work.mkdir(parents=True, exist_ok=True)
    sub = Path(out_bids) / f"sub-{label}"
    (sub / "anat").mkdir(parents=True, exist_ok=True)
    (sub / "func").mkdir(parents=True, exist_ok=True)

    seg_path = work / "seg.nii.gz"
    t1_1mm, t1_brain = work / "t1_1mm.nii.gz", work / "t1_brain.nii.gz"
    segment(raw / T1W, seg_path)
    reslice_and_skullstrip(raw / T1W, seg_path, t1_1mm, t1_brain)
    segmentation = nib.load(seg_path)
    reject_failed_segmentation(item["key"], segmentation)
    reject_corrupt_reslice(item["key"], nib.load(t1_1mm), nib.load(raw / T1W))
    _write_anat(label, raw, work, segmentation, out_bids)


def _one_run(item, raw_dir, out_bids, cfg, run_idx):
    label = sanitize_label(item["key"])
    raw, work = Path(raw_dir), Path(raw_dir) / "work"
    (Path(out_bids) / f"sub-{label}" / "func").mkdir(parents=True, exist_ok=True)
    tr_acq = float(item["sources"]["tr_acq"])
    n_nat, idx = n_native(cfg, tr_acq), int(run_idx)

    _write_run(label, raw, work, out_bids, n_nat, idx)
    entry = {"idx": idx, "run": "REST", "tr_acq": tr_acq}
    (work / f"run_{idx:02d}.json").write_text(json.dumps(entry))
    return entry


def _finish(item, out_bids, work):
    label = sanitize_label(item["key"])
    sub = Path(out_bids) / f"sub-{label}"
    sub.mkdir(parents=True, exist_ok=True)
    written = [json.loads(p.read_text()) for p in sorted(Path(work).glob("run_*.json"))]
    if not written:
        raise RuntimeError(
            f"{item['key']}: no run_*.json under {work} -- the BOLD stage never ran. A RUN-STAGED "
            f"build reaches this, because these cohorts carry their BOLD in `aux` and leave "
            f"`sources['runs']` empty, so the driver would go anat -> finish and write nothing."
        )
    meta = {**(item["sources"].get("meta") or {}), "bold_mm": BOLD_MM, "t1_mm": T1_MM}
    # `spaces` is the CLAIMED set, not the achieved one: the finalizer keys its completeness gate on
    # it to tell a modality-less subject from a dropped save.
    (sub / f"sub-{label}_desc-fmridata.json").write_text(json.dumps(
        {"key": item["key"], "label": label, "spaces": ["mni"], "runs": written,
         "meta": meta}, indent=2))


def _preproc(item, raw_dir, out_bids, cfg):
    _anat(item, raw_dir, out_bids, cfg)
    _one_run(item, raw_dir, out_bids, cfg, 0)
    _finish(item, out_bids, Path(raw_dir) / "work")


def process(item, raw_dir, out_bids, cfg, stage="all", run_idx=None):
    if stage == "anat":
        _anat(item, raw_dir, out_bids, cfg)
    elif stage == "run":
        _one_run(item, raw_dir, out_bids, cfg, run_idx)
    elif stage == "finish":
        _finish(item, out_bids, Path(raw_dir) / "work")
    else:
        _preproc(item, raw_dir, out_bids, cfg)


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
