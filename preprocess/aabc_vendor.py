#!/usr/bin/env python3
"""AABC vendor core: the shipped MNI session zips -> preprocessed BIDS on the shared 2 mm grid.

The VENDOR half of AABC's native-vs-vendor pair. It reads two members out of the staged per-session
zips (the vendor's MNI T1 and its 500-frame REST concat), SynthSegs the subject-space T1, and carries
that segmentation into MNI through the vendor's own acpc->MNI warp. No registration of its own; the
from-raw twin is ``aabc_native.py``.

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
import zipfile
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path

import nibabel as nib
import numpy as np
import nitransforms as nt
import s3fs
from nibabel.processing import resample_from_to

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

#: The two geometry settings have exactly ONE production value each and are recorded in the
#: sidecar meta: the T1 (and the aseg) stay on the 1 mm grid...
T1_MM = 1.0
BOLD_MM = 2.0  # ...and BOLD lands on the 2 mm grid, which nests exactly 2x inside it
NATIVE_TR = 0.8              # the acquisition TR; the finalizer resamples FROM it
REST_WINDOW = 500            # the benchmark's REST window: fMRI_CONCAT_ALL[0:500]
STRUCT_ZIP = "struct.zip"    # the staged zips (the driver writes them under these names)
FUNC_ZIP = "func.zip"
# Members read out of those zips.
ACPC_T1 = "T1w/T1w_acpc_dc_restore.nii.gz"              # subject space -- the SynthSeg input
MNI_T1 = "MNINonLinear/T1w_restore.nii.gz"              # the vendor's MNI T1
ACPC2STD = "MNINonLinear/xfms/acpc_dc2standard.nii.gz"  # the vendor's acpc -> MNI warp
_CONCAT = "MNINonLinear/Results/fMRI_CONCAT_ALL"
CONCAT_BOLD = f"{_CONCAT}/fMRI_CONCAT_ALL_hp0_clean_rclean_tclean.nii.gz"

#: No SynthMorph / lta_convert here: this core runs no registration of its own.
REQUIRED_TOOLS = ("mri_synthseg", "mri_convert", "mri_mask")


def is_s3(uri) -> bool:
    return str(uri).startswith("s3://")


def s3_filesystem(storage_options=None) -> s3fs.S3FileSystem:
    return s3fs.S3FileSystem(**(storage_options or {}))


@contextmanager
def open_zip(zip_uri, storage_options=None):
    opener = s3_filesystem(storage_options).open if is_s3(zip_uri) else open
    with opener(zip_uri, "rb") as f, zipfile.ZipFile(f) as z:
        yield z


def zip_nifti(z: zipfile.ZipFile, member: str):
    bb = BytesIO()
    with z.open(member) as m:
        if member.endswith(".gz"):
            with gzip.open(m) as gz:
                shutil.copyfileobj(gz, bb)
        else:
            shutil.copyfileobj(m, bb)
    bb.seek(0)
    fh = nib.FileHolder(fileobj=bb)
    return nib.Nifti1Image.from_file_map({"header": fh, "image": fh})


def sanitize_label(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", key)


def member(session: str, rel: str) -> str:
    return f"{session}/{rel}"


def run_cli(command_block: str) -> None:
    for line in filter(None, (ln.strip() for ln in command_block.strip().splitlines())):
        argv = shlex.split(line)
        proc = subprocess.run(
            argv, capture_output=True, text=True, env={**os.environ, **CLI_ENV}
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"{argv[0]} failed ({proc.returncode}):\n{proc.stderr[-2000:]}"
            )


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
    from nitransforms.io import fsl as _ntfsl
    from nitransforms.nonlinear import DenseFieldTransform

    aff = np.asarray(warp_img.affine, dtype=float)
    if np.linalg.det(aff[:3, :3]) >= 0:
        raise ValueError(
            "fsl_warp: field must be radiological (det<0) for the unconditional FSL->RAS x-flip"
        )
    return DenseFieldTransform(_ntfsl.FSLDisplacementsField.from_image(warp_img), is_deltas=True)


def fsl_mni_1mm_grid() -> "nib.Nifti1Image":
    return nib.Nifti1Image(np.zeros(FSL_MNI_1MM_SHAPE, np.float32), FSL_MNI_1MM_AFFINE)


def reslice_to_fsl_mni_1mm(img: "nib.Nifti1Image", *, order: int = CUBIC) -> "nib.Nifti1Image":
    return resample_from_to(img, fsl_mni_1mm_grid(), order=order)


def brain_extract(t1: "nib.Nifti1Image", seg_arr: np.ndarray) -> "nib.Nifti1Image":
    data = np.where(seg_arr != 0, np.asarray(t1.dataobj, np.float32), 0.0)
    return nib.Nifti1Image(data.astype(np.float32), t1.affine)


def _func_to_moving_coords(reference: "nib.Nifti1Image", moving_affine,
                           sel_idx: np.ndarray | None = None) -> np.ndarray:
    """Voxel coordinates in the moving image for each (selected) voxel of `reference`."""
    from nibabel.affines import apply_affine

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
    """The vendor MNI concat onto the func grid. No transform: it is already in MNI space, so this is
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


def bold_coverage(bold_on_grid: "nib.Nifti1Image") -> np.ndarray:
    data = np.asarray(bold_on_grid.dataobj, np.float32)
    return (np.abs(data).max(axis=-1) > 0) if data.ndim == 4 else (np.abs(data) > 0)


def _seg_on_grid(seg, reference, warp) -> np.ndarray:
    return round_to_labels(resample_onto(warp, seg, reference, order=NEAREST))


def seg_func_path(work) -> Path:
    return Path(work) / "seg_func_mni.nii.gz"


def _write_anat(label, work, segmentation, out_bids):
    """The vendor MNI T1 + the segmentation carried into MNI through the vendor warp, plus the
    func-grid segmentation the run stage reads."""
    anat = Path(out_bids) / f"sub-{label}" / "anat"
    anat.mkdir(parents=True, exist_ok=True)
    anat_ref = fsl_mni_1mm_grid()
    warp = fsl_warp(nib.load(work / "acpc2std.nii.gz"))
    seg_anat = _seg_on_grid(segmentation, anat_ref, warp)
    anat_t1 = brain_extract(reslice_to_fsl_mni_1mm(nib.load(work / "t1_mni.nii.gz")), seg_anat)
    func_grid = iso_ref(anat_ref, BOLD_MM)
    nib.save(nib.Nifti1Image(_seg_on_grid(segmentation, func_grid, warp), func_grid.affine),
             seg_func_path(work))

    nib.save(anat_t1, anat / f"sub-{label}_space-mni_desc-preproc_T1w.nii.gz")
    nib.save(mask_image(seg_anat != 0, anat_ref),
             anat / f"sub-{label}_space-mni_desc-brain_mask.nii.gz")
    nib.save(mask_image(np.isin(seg_anat, GM_LABELS), anat_ref),
             anat / f"sub-{label}_space-mni_label-GM_mask.nii.gz")
    nib.save(mask_image(np.isin(seg_anat, CORTICAL_GM_LABELS), anat_ref),
             anat / f"sub-{label}_space-mni_label-corticalGM_mask.nii.gz")


def _write_bold(label, work, rest, out_bids, idx):
    func = Path(out_bids) / f"sub-{label}" / "func"
    func.mkdir(parents=True, exist_ok=True)
    func_grid = iso_ref(fsl_mni_1mm_grid(), BOLD_MM)
    seg_func = np.asanyarray(nib.load(seg_func_path(work)).dataobj).astype(np.int16)

    on_grid = resample_bold(rest, func_grid, order=TRILINEAR, sel=seg_func != 0)
    covered = bold_coverage(on_grid)
    stem = f"sub-{label}_run-{idx:02d}_space-mni"
    nib.save(on_grid, func / f"{stem}_desc-preproc_bold.nii")
    nib.save(mask_image((seg_func != 0) & covered, func_grid), func / f"{stem}_desc-brain_mask.nii.gz")
    nib.save(mask_image(np.isin(seg_func, GM_LABELS) & covered, func_grid),
             func / f"{stem}_label-GM_mask.nii.gz")
    nib.save(mask_image(np.isin(seg_func, CORTICAL_GM_LABELS) & covered, func_grid),
             func / f"{stem}_label-corticalGM_mask.nii.gz")


def _anat(item, raw_dir, out_bids, work):
    label, key = sanitize_label(item["key"]), item["key"]
    session = item["sources"]["session"]
    work.mkdir(parents=True, exist_ok=True)
    sub = Path(out_bids) / f"sub-{label}"
    (sub / "anat").mkdir(parents=True, exist_ok=True)
    (sub / "func").mkdir(parents=True, exist_ok=True)

    t1_acpc = work / "t1_acpc.nii.gz"
    with open_zip(str(Path(raw_dir) / STRUCT_ZIP)) as z:  # one open, three member reads
        nib.save(zip_nifti(z, member(session, ACPC_T1)), t1_acpc)
        nib.save(zip_nifti(z, member(session, MNI_T1)), work / "t1_mni.nii.gz")
        nib.save(zip_nifti(z, member(session, ACPC2STD)), work / "acpc2std.nii.gz")

    seg_path = work / "seg.nii.gz"
    t1_1mm, t1_brain = work / "t1_1mm.nii.gz", work / "t1_brain.nii.gz"
    segment(t1_acpc, seg_path)
    reslice_and_skullstrip(t1_acpc, seg_path, t1_1mm, t1_brain)
    segmentation = nib.load(seg_path)
    reject_failed_segmentation(key, segmentation)
    reject_corrupt_reslice(key, nib.load(t1_1mm), nib.load(t1_acpc))
    _write_anat(label, work, segmentation, out_bids)


def _one_run(item, raw_dir, out_bids, work, idx):
    label = sanitize_label(item["key"])
    session = item["sources"]["session"]
    with open_zip(str(Path(raw_dir) / FUNC_ZIP)) as z:
        concat = zip_nifti(z, member(session, CONCAT_BOLD))
    rest = concat.slicer[..., : min(REST_WINDOW, concat.shape[3])]  # the benchmark's REST window
    _write_bold(label, work, rest, out_bids, idx)
    entry = {"idx": int(idx), "run": "REST", "tr_acq": NATIVE_TR}
    (work / f"run_{int(idx):02d}.json").write_text(json.dumps(entry))
    return entry


def _finish(item, out_bids, work):
    label = sanitize_label(item["key"])
    sub = Path(out_bids) / f"sub-{label}"
    sub.mkdir(parents=True, exist_ok=True)
    written = [json.loads(p.read_text()) for p in sorted(Path(work).glob("run_*.json"))]
    meta = {**(item["sources"].get("meta") or {}), "bold_mm": BOLD_MM, "t1_mm": T1_MM}
    # `spaces` is the CLAIMED set, not the achieved one: the finalizer keys its completeness gate on
    # it to tell a modality-less subject from a dropped save.
    (sub / f"sub-{label}_desc-fmridata.json").write_text(json.dumps(
        {"key": item["key"], "label": label, "spaces": ["mni"],
         "runs": written, "meta": meta}, indent=2))


def _preproc(item, raw_dir, out_bids, work):
    _anat(item, raw_dir, out_bids, work)
    _one_run(item, raw_dir, out_bids, work, 0)
    _finish(item, out_bids, work)


def process(item: dict, raw_dir: str, out_bids: str, cfg: dict, stage="all", run_idx=None) -> None:
    work = Path(raw_dir) / "work"
    if stage == "anat":
        _anat(item, raw_dir, out_bids, work)
    elif stage == "run":
        _one_run(item, raw_dir, out_bids, work, int(run_idx))
    elif stage == "finish":
        _finish(item, out_bids, work)
    else:
        _preproc(item, raw_dir, out_bids, work)


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
