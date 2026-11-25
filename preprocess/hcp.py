#!/usr/bin/env python3
"""HCP-YA vendor core: the shipped MNI derivatives -> preprocessed BIDS on the shared 2 mm grid.

This is the VENDOR half of the paper's native-vs-vendor comparison. It does no registration of its
own: the T1 and the BOLD already sit in the vendor's nonlinearly-normalised MNI space, so the work is
SynthSeg (one segmenter across the whole corpus), the vendor acpc->MNI warp for the segmentation, and
a resample onto the same 2 mm grid every other build lands on. The from-raw twin of this dataset runs
``openneuro.py`` instead, which is the whole point of the comparison.

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
from pathlib import Path

import nibabel as nib
import numpy as np
import nitransforms as nt
from nibabel.affines import apply_affine
from nibabel.processing import resample_from_to
from nitransforms.io import fsl as _ntfsl
from nitransforms.nonlinear import DenseFieldTransform
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
#: The two geometry settings have exactly ONE production value each and are recorded in the
#: sidecar meta: the T1 (and the aseg) stay on the 1 mm grid...
T1_MM = 1.0
BOLD_MM = 2.0  # ...and BOLD lands on the 2 mm grid, which nests exactly 2x inside it
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

# The staged-input names: the driver writes the discovered URIs under these, and the discover module
# spells them the same way. Change one side, change both. (The subject-space T1 arrives too, but the
# work item names it, so it is read from there.)
T1_MNI = "T1w_mni.nii.gz"                  # the vendor's MNI T1
WARP_ACPC2STD = "acpc_dc2standard.nii.gz"  # the vendor's acpc -> MNI warp, for the segmentation

WORK_SEG = "seg.nii.gz"
WORK_T1_1MM = "t1_acpc_1mm.nii.gz"
WORK_T1_BRAIN = "t1_brain.nii.gz"


def run_cli(command_block: str) -> None:
    for line in filter(None, (ln.strip() for ln in command_block.strip().splitlines())):
        argv = shlex.split(line)
        proc = subprocess.run(argv, capture_output=True, text=True, env={**os.environ, **CLI_ENV})
        if proc.returncode != 0:
            raise RuntimeError(f"{argv[0]} failed ({proc.returncode}):\n{proc.stderr[-2000:]}")


def preflight(cfg: dict) -> None:
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


def segment(t1: Path, out_seg: Path) -> None:
    run_cli(f"mri_synthseg --i {t1} --o {out_seg} --robust --cpu --threads 1")


def reslice_and_skullstrip(t1: Path, seg: Path, out_1mm: Path, out_brain: Path) -> None:
    run_cli(f"""
        mri_convert -rl {seg} -rt cubic -odt float {t1} {out_1mm}
        mri_mask {out_1mm} {seg} {out_brain}
    """)


def reject_failed_segmentation(key: str, segmentation) -> None:
    brain_voxels = int((np.asanyarray(segmentation.dataobj) != 0).sum())
    if brain_voxels < MIN_PLAUSIBLE_BRAIN_VOXELS:
        raise RuntimeError(
            f"{key}: SynthSeg produced only {brain_voxels} brain voxels "
            f"(< {MIN_PLAUSIBLE_BRAIN_VOXELS}) -- it likely ran out of memory; "
            f"refusing to pack empty masks"
        )


def reject_corrupt_reslice(key: str, resliced_t1, source_t1) -> None:
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


def mask_image(mask, reference) -> "nib.Nifti1Image":
    return nib.Nifti1Image(np.asarray(mask, np.uint8), reference.affine)


def resample_onto(transform, img, reference, *, order):
    xfm = nt.Affine(np.asarray(transform)) if isinstance(transform, np.ndarray) else transform
    return nt.apply(xfm, img, reference=reference, order=order)


def fsl_mni_1mm_grid() -> "nib.Nifti1Image":
    return nib.Nifti1Image(np.zeros(FSL_MNI_1MM_SHAPE, np.float32), FSL_MNI_1MM_AFFINE)


def reslice_to_fsl_mni_1mm(img: "nib.Nifti1Image", *, order: int = CUBIC) -> "nib.Nifti1Image":
    return resample_from_to(img, fsl_mni_1mm_grid(), order=order)


def _zooms(affine: np.ndarray) -> np.ndarray:
    return np.sqrt((np.asarray(affine, float)[:3, :3] ** 2).sum(axis=0))


def iso_ref(ref: "nib.Nifti1Image", mm) -> "nib.Nifti1Image":
    """An isotropic ``mm`` grid covering the same field of view as `ref`."""
    z = _zooms(ref.affine)
    mm = np.broadcast_to(np.asarray(mm, dtype=float), (3,))
    aff = np.asarray(ref.affine, dtype=float).copy()
    aff[:3, :3] = (aff[:3, :3] / z) * mm
    shape = tuple(int(np.ceil(s * zz / m - _EPS)) for s, zz, m in zip(ref.shape[:3], z, mm))
    return nib.Nifti1Image(np.zeros(shape, np.float32), aff)


def fsl_warp(warp_img: "nib.Nifti1Image"):
    aff = np.asarray(warp_img.affine, dtype=float)
    if np.linalg.det(aff[:3, :3]) >= 0:
        raise ValueError(
            "fsl_warp: field must be radiological (det<0) for the unconditional FSL->RAS x-flip"
        )
    return DenseFieldTransform(_ntfsl.FSLDisplacementsField.from_image(warp_img), is_deltas=True)


def _func_to_moving_coords(reference: "nib.Nifti1Image", moving_affine,
                           sel_idx: np.ndarray | None = None) -> np.ndarray:
    """Voxel coordinates in the moving image for each (selected) voxel of `reference`."""
    if sel_idx is None:
        ijk = np.indices(reference.shape[:3]).reshape(3, -1).T.astype(np.float64)
    else:
        ijk = np.stack(np.unravel_index(sel_idx, reference.shape[:3]), axis=1).astype(np.float64)
    world = apply_affine(np.asarray(reference.affine, float), ijk)
    return apply_affine(np.linalg.inv(np.asarray(moving_affine, float)), world).T


def _same_grid(a: "nib.Nifti1Image", b: "nib.Nifti1Image") -> bool:
    return tuple(a.shape[:3]) == tuple(b.shape[:3]) and np.allclose(
        np.asarray(a.affine), np.asarray(b.affine), atol=1e-3)


def resample_bold(bold: "nib.Nifti1Image", reference: "nib.Nifti1Image", *,
                  order: int = TRILINEAR, sel: np.ndarray) -> "nib.Nifti1Image":
    """The vendor MNI BOLD onto the func grid. No transform: it is already in MNI space, so this is
    a grid change only -- and a no-op copy when the two grids already agree."""
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


def bold_coverage(bold_on_grid: "nib.Nifti1Image") -> np.ndarray:
    data = np.asarray(bold_on_grid.dataobj, np.float32)
    return (np.abs(data).max(axis=-1) > 0) if data.ndim == 4 else (np.abs(data) > 0)


def sanitize_label(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", key)


def _seg_on_grid(seg, reference, warp) -> np.ndarray:
    return round_to_labels(resample_onto(warp, seg, reference, order=NEAREST))


def _brain_extract(t1: "nib.Nifti1Image", seg_arr: np.ndarray) -> "nib.Nifti1Image":
    data = np.where(seg_arr != 0, np.asarray(t1.dataobj, np.float32), 0.0)
    return nib.Nifti1Image(data.astype(np.float32), t1.affine)


def _write_anat_masks(seg_arr: np.ndarray, reference, anat: Path, label: str, space: str) -> None:
    nib.save(mask_image(seg_arr != 0, reference),
             anat / f"sub-{label}_space-{space}_desc-brain_mask.nii.gz")
    nib.save(mask_image(np.isin(seg_arr, GM_LABELS), reference),
             anat / f"sub-{label}_space-{space}_label-GM_mask.nii.gz")
    nib.save(mask_image(np.isin(seg_arr, CORTICAL_GM_LABELS), reference),
             anat / f"sub-{label}_space-{space}_label-corticalGM_mask.nii.gz")


def _anat_warp(raw: Path):
    """The vendor's acpc -> MNI warp, which is how the subject-space segmentation reaches MNI."""
    return fsl_warp(nib.load(raw / WARP_ACPC2STD))


def _func_refs(work: Path):
    func_grid = iso_ref(fsl_mni_1mm_grid(), BOLD_MM)
    return func_grid, round_to_labels(nib.load(work / "seg_func_mni.nii.gz"))


def _anat(item, raw: Path, out_bids, work: Path) -> None:
    """SynthSeg the subject-space T1, then carry the segmentation into MNI through the vendor warp;
    the anat T1 itself is the vendor's MNI T1, brain-extracted with that segmentation."""
    label = sanitize_label(item["key"])
    work.mkdir(parents=True, exist_ok=True)
    sub = Path(out_bids) / f"sub-{label}"
    anat, func = sub / "anat", sub / "func"
    anat.mkdir(parents=True, exist_ok=True)
    func.mkdir(parents=True, exist_ok=True)

    t1_acpc = raw / item["sources"]["t1"]["name"]
    seg_path, t1_1mm, t1_brain = work / WORK_SEG, work / WORK_T1_1MM, work / WORK_T1_BRAIN
    segment(t1_acpc, seg_path)
    reslice_and_skullstrip(t1_acpc, seg_path, t1_1mm, t1_brain)
    segmentation = nib.load(seg_path)
    reject_failed_segmentation(item["key"], segmentation)
    reject_corrupt_reslice(item["key"], nib.load(t1_1mm), nib.load(t1_acpc))

    anat_ref = fsl_mni_1mm_grid()
    warp = _anat_warp(raw)
    seg_anat = _seg_on_grid(segmentation, anat_ref, warp)
    anat_t1 = _brain_extract(reslice_to_fsl_mni_1mm(nib.load(raw / T1_MNI)), seg_anat)
    nib.save(anat_t1, anat / f"sub-{label}_space-mni_desc-preproc_T1w.nii.gz")
    _write_anat_masks(seg_anat, anat_ref, anat, label, "mni")

    func_grid = iso_ref(anat_ref, BOLD_MM)
    nib.save(nib.Nifti1Image(_seg_on_grid(segmentation, func_grid, warp), func_grid.affine),
             work / "seg_func_mni.nii.gz")


def _one_run(item, spec, raw: Path, out_bids, work: Path) -> dict:
    label = sanitize_label(item["key"])
    func = Path(out_bids) / f"sub-{label}" / "func"
    func.mkdir(parents=True, exist_ok=True)
    raw_bold = nib.load(raw / spec["name"])

    func_grid, seg_func = _func_refs(work)
    brain_f = seg_func != 0
    on_grid = resample_bold(raw_bold, func_grid, order=TRILINEAR, sel=brain_f)
    covered = bold_coverage(on_grid)
    stem = f"sub-{label}_run-{spec['idx']:02d}_space-mni"
    nib.save(on_grid, func / f"{stem}_desc-preproc_bold.nii")
    nib.save(mask_image(brain_f & covered, func_grid), func / f"{stem}_desc-brain_mask.nii.gz")
    nib.save(mask_image(np.isin(seg_func, GM_LABELS) & covered, func_grid),
             func / f"{stem}_label-GM_mask.nii.gz")
    nib.save(mask_image(np.isin(seg_func, CORTICAL_GM_LABELS) & covered, func_grid),
             func / f"{stem}_label-corticalGM_mask.nii.gz")

    entry = {"idx": spec["idx"], "run": spec["run"], "tr_acq": spec.get("tr_acq"),
             "native_mm": float(max(raw_bold.header.get_zooms()[:3]))}
    (work / f"run_{spec['idx']:05d}.json").write_text(json.dumps(entry))
    return entry


def _finish(item, out_bids, work: Path) -> None:
    label = sanitize_label(item["key"])
    sub = Path(out_bids) / f"sub-{label}"
    sub.mkdir(parents=True, exist_ok=True)
    written = [json.loads(p.read_text()) for p in
               sorted(work.glob("run_*.json"), key=lambda q: int(q.stem.split("_")[1]))]
    meta = {**(item["sources"].get("meta") or {}), "bold_mm": BOLD_MM, "t1_mm": T1_MM}
    # `spaces` is the CLAIMED set, not the achieved one: the finalizer keys its completeness gate on
    # it to tell a modality-less subject from a dropped save.
    (sub / f"sub-{label}_desc-fmridata.json").write_text(json.dumps(
        {"key": item["key"], "label": label, "spaces": ["mni"], "runs": written, "meta": meta},
        indent=2))


def _preproc(item, raw: Path, out_bids, work: Path) -> None:
    _anat(item, raw, out_bids, work)
    for spec in item["sources"]["runs"]:
        if not (raw / spec["name"]).exists():
            continue
        try:
            _one_run(item, spec, raw, out_bids, work)
        except Exception as exc:
            # Keep going on a bad RUN, but print it: the sidecar records only the runs that
            # SUCCEEDED, and the driver escalates any stdout from a zero-exit core to WARNING.
            print(f"WARN {item['key']} run {spec['idx']} ({spec['name']}) failed ({exc}) "
                  f"-- skipping run")
    _finish(item, out_bids, work)


def process(item: dict, raw_dir: str, out_bids: str, cfg: dict, stage="all", run_idx=None) -> None:
    raw, work = Path(raw_dir), Path(raw_dir) / "work"
    if stage == "anat":
        _anat(item, raw, out_bids, work)
    elif stage == "run":
        spec = next(s for s in item["sources"]["runs"] if s["idx"] == int(run_idx))
        _one_run(item, spec, raw, out_bids, work)
    elif stage == "finish":
        _finish(item, out_bids, work)
    else:
        _preproc(item, raw, out_bids, work)


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
