#!/usr/bin/env python3
"""The from-raw preprocessing kernel: one subject's raw BIDS -> preprocessed BIDS derivatives on a
shared 2 mm grid, using standard neuroimaging tools only.

THIS IS THE METHOD the paper calls native space. The T1 reaches the template by a RIGID transform --
rotation and translation only, size-preserving -- so the anatomy stays the subject's own; the fixed
output grid gives every subject a shared, consistently oriented coordinate frame, and is NOT an
anatomical normalisation.

The sequence, per subject:

  anat (once)
    1. N4 bias-field correction of the raw T1.
    2. SynthSeg ``--robust --cpu`` -> the aseg label volume (and, from it, the brain / GM /
       cortical-GM masks).
    3. ``mri_convert -rl`` the debiased T1 onto the seg's 1 mm grid, ``mri_mask`` it to the brain.
    4. ``mri_synthmorph register -m rigid`` T1-brain -> the template's brain, then ``lta_convert``
       to an FSL matrix.
  run (per BOLD run)
    5. ``mcflirt -meanvol -mats`` -> a mean EPI + one motion matrix per volume.
    6. ``mri_synthmorph register -m rigid`` mean EPI -> T1-brain.
    7. COMPOSE, never chain resamples: ``mni_to_mean = epi2t1 @ mni_to_t1``, then the per-volume
       motion matrix on top, so a func-grid point maps back into raw-BOLD space in one hop.
    8. ONE streamed resample of the raw BOLD onto the 2 mm grid (`resample_bold_to_file`).

Deliberately NO susceptibility-distortion correction and no gradient-nonlinearity unwarp: dropping
them is what makes this comparable across studies that ship no fieldmaps, and the gap versus a vendor
pipeline is exactly what the native-vs-vendor comparison measures.

This script imports NOTHING from the package -- a deliberate invariant (there is a test pinning it),
so it lifts out as a standalone file. The driver invokes it as a subprocess CLI.
"""
from __future__ import annotations

import argparse
import io
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
from nibabel.arraywriters import get_slope_inter, make_array_writer
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform
from scipy.ndimage import map_coordinates

SPACE = "native"
#: The four geometry settings have exactly ONE production value each, so they are constants here
#: rather than knobs: the output grid is templateflow's MNI152NLin2009cAsym...
TEMPLATE = "MNI152NLin2009cAsym"
T1_MM = 1.0      # ...the T1 (and the aseg) stay on its 1 mm grid...
BOLD_MM = 2.0    # ...BOLD lands on the 2 mm grid, which nests exactly 2x inside it...
N4_SHRINK = 4    # ...and N4 runs at the usual shrink factor.
MIN_PLAUSIBLE_BRAIN_VOXELS = 100_000
GM_LABELS = (3, 42, 8, 47, 10, 49, 11, 50, 12, 51, 13, 52, 17, 53, 18, 54, 26, 58)
CORTICAL_GM_LABELS = (3, 42)
REQUIRED_TOOLS = (
    "N4BiasFieldCorrection",
    "mri_synthseg",
    "mri_convert",
    "mri_mask",
    "mri_synthmorph",
    "lta_convert",
    "mcflirt",
)


def run(command_block: str) -> None:
    env = {**os.environ, "FSLOUTPUTTYPE": "NIFTI_GZ", "OMP_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": "1"}
    for line in filter(None, (ln.strip() for ln in command_block.strip().splitlines())):
        argv = shlex.split(line)
        proc = subprocess.run(argv, capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            raise RuntimeError(f"{argv[0]} failed ({proc.returncode}):\n{proc.stderr[-2000:]}")


def preflight(cfg=None) -> None:
    missing = [t for t in REQUIRED_TOOLS if shutil.which(t) is None]
    if missing:
        raise RuntimeError(f"CLIs not on PATH: {missing} -- need FreeSurfer + FSL (mcflirt) + ANTs (N4)")
    fs_home = os.environ.get("FREESURFER_HOME", "")
    licensed = os.environ.get("FS_LICENSE") or any(
        os.path.exists(os.path.join(fs_home, n)) for n in ("license.txt", ".license") if fs_home
    )
    if not licensed:
        raise RuntimeError("no FreeSurfer license -- set FS_LICENSE or place license.txt in FREESURFER_HOME")


def sanitize_label(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", key)


def to_las(img):
    return img.as_reoriented(ornt_transform(io_orientation(img.affine), axcodes2ornt(("L", "A", "S"))))


def iso_grid(ref, mm):
    z = np.sqrt((np.asarray(ref.affine, float)[:3, :3] ** 2).sum(axis=0))
    mm = np.broadcast_to(float(mm), (3,))
    aff = np.asarray(ref.affine, float).copy()
    aff[:3, :3] = (aff[:3, :3] / z) * mm
    shape = tuple(int(np.ceil(s * zz / m - 1e-4)) for s, zz, m in zip(ref.shape[:3], z, mm))
    return nib.Nifti1Image(np.zeros(shape, np.float32), aff)


def template_grid(desc: str | None = None) -> str:
    """The templateflow 1 mm template (``desc="brain"`` for its brain-extracted twin) -- the fixed
    frame the rigid T1 registration targets."""
    import templateflow.api as tf

    return str(tf.get(TEMPLATE, resolution=1, suffix="T1w", desc=desc))


def fsl_to_ras(mat, moving, reference) -> np.ndarray:
    return nt.linear.load(str(mat), fmt="fsl", moving=moving, reference=reference).matrix


def resample(transform, img, reference, order):
    xfm = nt.Affine(np.asarray(transform)) if isinstance(transform, np.ndarray) else transform
    return nt.apply(xfm, img, reference=reference, order=order)


def rigid_register(moving: Path, fixed, out_lta: Path, out_mat: Path) -> None:
    run(f"""
        mri_synthmorph register -m rigid -j 1 -t {out_lta} {moving} {fixed}
        lta_convert --inlta {out_lta} --outfsl {out_mat} --src {moving} --trg {fixed}
    """)


def label_grid(transform, seg, reference) -> np.ndarray:
    return np.rint(np.asanyarray(resample(transform, seg, reference, 0).dataobj)).astype(np.int16)


def _preproc_bold_header(shape, n_frames: int, affine, tr) -> bytes:
    """The exact 352-byte NIfTI-1 header ``nib.save`` would write for a float32 ``(X, Y, Z, T)``
    image on `affine` after ``set_tr(tr)`` -- WITHOUT materializing the data.

    Mirrors nibabel's own header prep on a zero-STRIDED broadcast view: update the header from the
    affine/dtype, then finalize ``scl_slope``/``scl_inter`` via the array writer. For float32 output
    there is no scaling and no data scan, so the header is a pure function of
    ``(shape, T, affine, tr)`` -- byte-identical for broadcast zeros and for the real volume."""
    zero = np.broadcast_to(np.float32(0.0), tuple(shape) + (int(n_frames),))
    img = set_tr(nib.Nifti1Image(zero, np.asarray(affine, float)), tr)
    img.update_header()
    hdr = img.header
    slope = hdr["scl_slope"].item() if hdr.has_data_slope else np.nan
    inter = hdr["scl_inter"].item() if hdr.has_data_intercept else np.nan
    if np.all(np.isnan((slope, inter))):  # default (unset) -> derive exactly as to_file_map does
        writer = make_array_writer(np.asanyarray(img.dataobj), hdr.get_data_dtype(),
                                   hdr.has_data_slope, hdr.has_data_intercept)
        hdr.set_slope_inter(*get_slope_inter(writer))
    buf = io.BytesIO()
    hdr.write_to(buf)
    return buf.getvalue()


def resample_bold_to_file(bold, reference, per_volume, sel, out_path, tr, order=1) -> np.ndarray:
    """Stream the per-volume BOLD resample straight to an uncompressed ``.nii`` at `out_path`,
    byte-for-byte identical to the dense equivalent (resample every frame onto `reference` at the
    in-brain voxels `sel`, then ``nib.save(set_tr(...))``) -- but NEVER holding the full
    ``(prod(shape), T)`` array, which is ~1.3 GB for a 300-frame subject (4.33 MB/frame float32 on
    the 2 mm grid).

    NIfTI-1 stores voxels Fortran-order, so frame ``t`` is a CONTIGUOUS ``prod(shape)`` block on
    disk: write the header once, then one dense frame at a time (a single reused buffer, scattered at
    `sel` and emitted ``order="F"``). Peak RAM is ONE frame plus the in-brain working set,
    independent of ``T``.

    Returns the func-grid ``in_fov`` mask (``|value|.max over time > 0``), accumulated in the SAME
    pass, so the caller needs no second dense read of the volume."""
    shape = tuple(reference.shape[:3])
    n_frames = int(bold.shape[3])
    n_vox = int(np.prod(shape))
    sel_idx = np.flatnonzero(np.asarray(sel, bool).reshape(-1))
    ijk = np.stack(np.unravel_index(sel_idx, shape), axis=1).astype(np.float64)
    world = apply_affine(np.asarray(reference.affine, float), ijk)
    inv_moving = np.linalg.inv(np.asarray(bold.affine, float))

    absmax = np.zeros(len(sel_idx), np.float32)  # running max|value| over time, at the sel voxels
    frame = np.zeros(n_vox, np.float32)          # ONE reused dense frame buffer (prod(shape) floats)
    with open(out_path, "wb") as f:
        f.write(_preproc_bold_header(shape, n_frames, reference.affine, tr))
        for t in range(n_frames):
            coords = apply_affine(inv_moving @ np.asarray(per_volume[t], float), world).T
            vals = map_coordinates(np.asarray(bold.dataobj[..., t], np.float32), coords,
                                   order=order, mode="constant", cval=0.0).astype(np.float32)
            frame[sel_idx] = vals
            f.write(frame.reshape(shape).tobytes(order="F"))  # F-order == the on-disk voxel order
            frame[sel_idx] = 0.0                              # clear only the touched rows for next frame
            np.maximum(absmax, np.abs(vals), out=absmax)
    in_fov = np.zeros(n_vox, bool)
    in_fov[sel_idx] = absmax > 0
    return in_fov.reshape(shape)


def reject_failed_segmentation(key: str, segmentation) -> None:
    brain_voxels = int((np.asanyarray(segmentation.dataobj) != 0).sum())
    if brain_voxels < MIN_PLAUSIBLE_BRAIN_VOXELS:
        raise RuntimeError(
            f"{key}: SynthSeg produced only {brain_voxels} brain voxels "
            f"(< {MIN_PLAUSIBLE_BRAIN_VOXELS}) -- it likely ran out of memory; "
            f"refusing to pack empty masks"
        )


def save_mask(mask, affine, path: Path) -> None:
    nib.save(nib.Nifti1Image(np.asarray(mask, np.uint8), affine), path)


def set_tr(img, tr):
    """Stamp the acquisition TR into a 4D BOLD header's pixdim[4] so the stored file is SELF-DESCRIBING.
    The finalizer resamples FROM this TR; an image built as ``Nifti1Image(array, affine)`` carries no
    temporal zoom, so its header defaults to 1.0 s -- a silent wrong TR if the manifest ``tr_acq`` is
    ever missing. Stamping makes the header fallback correct instead of a hidden 1.0."""
    if tr and float(tr) > 0 and img.ndim == 4:
        img.header.set_zooms(tuple(img.header.get_zooms()[:3]) + (float(tr),))
    return img


def _anat(t1_raw, out_bids, work, key):
    """Steps 1-4: debias, segment, reslice + skull-strip, and rigidly register the T1 to the
    template. Writes the anat derivatives and caches the func-grid aseg for the run stage."""
    label = sanitize_label(key)
    work.mkdir(parents=True, exist_ok=True)
    sub = Path(out_bids) / f"sub-{label}"
    anat, func = sub / "anat", sub / "func"
    anat.mkdir(parents=True, exist_ok=True)
    func.mkdir(parents=True, exist_ok=True)

    mni_t1 = nib.load(template_grid())
    mni_brain = template_grid(desc="brain")

    t1_debiased, seg = work / "t1_n4.nii.gz", work / "seg.nii.gz"
    t1_1mm, t1_brain = work / "t1_1mm.nii.gz", work / "t1_brain.nii.gz"
    t1_lta, t1_mat = work / "t1_to_mni.lta", work / "t1_to_mni.mat"

    run(f"N4BiasFieldCorrection -d 3 -s {N4_SHRINK} -i {t1_raw} -o {t1_debiased}")
    run(f"mri_synthseg --i {t1_debiased} --o {seg} --robust --cpu --threads 1")
    run(f"""
        mri_convert -rl {seg} -rt cubic -odt float {t1_debiased} {t1_1mm}
        mri_mask {t1_1mm} {seg} {t1_brain}
    """)
    rigid_register(t1_brain, mni_brain, t1_lta, t1_mat)

    struct, segmentation, brain = nib.load(t1_1mm), nib.load(seg), nib.load(t1_brain)
    reject_failed_segmentation(key, segmentation)

    mni_to_t1 = fsl_to_ras(t1_mat, struct, mni_t1)
    mni_t1 = to_las(mni_t1)
    mni_bold = iso_grid(mni_t1, BOLD_MM)
    seg_t1 = label_grid(mni_to_t1, segmentation, mni_t1)
    # The BOLD grid nests exactly 2x inside the 1 mm grid (BOLD voxel [i] IS T1 voxel [2i]), so a
    # plain [::2] subsample of the 1 mm labels is bit-identical to nearest-resampling the
    # segmentation onto the BOLD grid -- one fewer transform, and the 2x nesting is guaranteed rather
    # than hoped for. The assert is what guards that.
    step = int(round(BOLD_MM / T1_MM))
    seg_bold = seg_t1[::step, ::step, ::step]
    assert tuple(seg_bold.shape) == tuple(mni_bold.shape[:3]), (
        f"the {BOLD_MM} mm grid does not nest {step}x inside the {T1_MM} mm grid: "
        f"{seg_bold.shape} != {tuple(mni_bold.shape[:3])}")
    nib.save(nib.Nifti1Image(seg_bold, mni_bold.affine), work / "seg_bold.nii.gz")

    nib.save(resample(mni_to_t1, brain, mni_t1, 1),
             anat / f"sub-{label}_space-{SPACE}_desc-preproc_T1w.nii.gz")
    save_mask(seg_t1 != 0, mni_t1.affine, anat / f"sub-{label}_space-{SPACE}_desc-brain_mask.nii.gz")
    save_mask(np.isin(seg_t1, GM_LABELS), mni_t1.affine, anat / f"sub-{label}_space-{SPACE}_label-GM_mask.nii.gz")
    save_mask(np.isin(seg_t1, CORTICAL_GM_LABELS), mni_t1.affine,
              anat / f"sub-{label}_space-{SPACE}_label-corticalGM_mask.nii.gz")
    # The FULL aseg label volumes (int16) the three masks above are derived from, kept as a SEPARATE
    # store field (finalizer -> `codec` seg row): the anat grid for the T1 layer, the func grid for the
    # BOLD layer (subject-level -- labels are FOV-independent, one image serves every run).
    nib.save(nib.Nifti1Image(seg_t1.astype(np.int16), mni_t1.affine),
             anat / f"sub-{label}_space-{SPACE}_desc-aseg_dseg.nii.gz")
    nib.save(nib.Nifti1Image(seg_bold.astype(np.int16), mni_bold.affine),
             func / f"sub-{label}_space-{SPACE}_desc-aseg_dseg.nii.gz")


def anat_refs(work):
    """Re-read the anat stage's outputs: the struct, the func grid, the template->T1 rigid, and the
    func-grid aseg. This is the seam a staged ``run`` node crosses (a fresh subprocess)."""
    mni_t1 = nib.load(template_grid())
    struct = nib.load(work / "t1_1mm.nii.gz")
    mni_to_t1 = fsl_to_ras(work / "t1_to_mni.mat", struct, mni_t1)
    mni_bold = iso_grid(to_las(mni_t1), BOLD_MM)
    seg_bold = np.asanyarray(nib.load(work / "seg_bold.nii.gz").dataobj).astype(np.int16)
    return struct, mni_bold, mni_to_t1, seg_bold


def _one_run(spec, out_bids, work, key):
    """Steps 5-8 for one BOLD run: motion-correct, register the mean EPI to the T1, compose, and
    stream the single resample onto the func grid."""
    label = sanitize_label(key)
    func = Path(out_bids) / f"sub-{label}" / "func"
    func.mkdir(parents=True, exist_ok=True)

    struct, mni_bold, mni_to_t1, seg_bold = anat_refs(work)
    brain_bold = seg_bold != 0
    gm_bold = np.isin(seg_bold, GM_LABELS)
    cortical_bold = np.isin(seg_bold, CORTICAL_GM_LABELS)

    i, bold_raw = spec["idx"], spec["path"]
    moco = work / f"moco{i:02d}"
    run(f"mcflirt -in {bold_raw} -out {moco} -meanvol -mats")

    epi_lta, epi_mat = work / f"epi2t1_{i:02d}.lta", work / f"epi2t1_{i:02d}.mat"
    mean_epi = Path(f"{moco}_mean_reg.nii.gz")
    rigid_register(mean_epi, work / "t1_brain.nii.gz", epi_lta, epi_mat)

    mean_img, raw_bold = nib.load(mean_epi), nib.load(bold_raw)
    mni_to_mean = fsl_to_ras(epi_mat, mean_img, struct) @ mni_to_t1
    motion_mats = sorted(Path(f"{moco}.mat").glob("MAT_*"), key=lambda p: int(p.name.split("_")[1]))

    # ONE composed transform per frame: func grid -> template -> T1 -> mean EPI -> this volume. The
    # raw BOLD is interpolated exactly once, however many steps the composition has.
    per_volume = [fsl_to_ras(m, mean_img, mean_img) @ mni_to_mean for m in motion_mats]
    bold_path = func / f"sub-{label}_run-{i:02d}_space-{SPACE}_desc-preproc_bold.nii"
    in_fov = resample_bold_to_file(raw_bold, mni_bold, per_volume, brain_bold, bold_path, spec["tr_acq"])

    save_mask(brain_bold & in_fov, mni_bold.affine,
              func / f"sub-{label}_run-{i:02d}_space-{SPACE}_desc-brain_mask.nii.gz")
    save_mask(gm_bold & in_fov, mni_bold.affine,
              func / f"sub-{label}_run-{i:02d}_space-{SPACE}_label-GM_mask.nii.gz")
    save_mask(cortical_bold & in_fov, mni_bold.affine,
              func / f"sub-{label}_run-{i:02d}_space-{SPACE}_label-corticalGM_mask.nii.gz")

    entry = {"idx": i, "run": spec["run"], "tr_acq": spec["tr_acq"],
             "native_mm": float(max(raw_bold.header.get_zooms()[:3]))}
    (work / f"run_{i:05d}.json").write_text(json.dumps(entry))
    return entry


def _finish(t1_raw, out_bids, work, key, meta):
    """Write the sidecar the finalizer reads: the runs that actually landed, plus the acquired voxel
    size (which the resample erased from the geometry)."""
    label = sanitize_label(key)
    sub = Path(out_bids) / f"sub-{label}"
    sub.mkdir(parents=True, exist_ok=True)
    written = [json.loads(p.read_text())
               for p in sorted(work.glob("run_*.json"), key=lambda p: int(p.stem.split("_")[1]))]
    meta = {**meta, "t1_native_mm": float(max(nib.load(t1_raw).header.get_zooms()[:3])),
            "t1_mm": T1_MM, "bold_mm": BOLD_MM}
    (sub / f"sub-{label}_desc-fmridata.json").write_text(json.dumps(
        {"key": key, "label": label, "spaces": [SPACE], "runs": written, "meta": meta}, indent=2))


def _preproc(t1_raw, run_specs, out_bids, work, key, meta):
    """The whole subject in one process -- the unstaged path."""
    _anat(t1_raw, out_bids, work, key)
    for spec in run_specs:
        if not spec["path"].exists():
            continue
        try:
            _one_run(spec, out_bids, work, key)
        except Exception as exc:
            # Keep going on a bad RUN, but say so on stdout: the driver logs a zero-exit subject
            # that printed anything at WARNING, because the sidecar records only the runs that
            # SUCCEEDED and nothing else could reconstruct the loss.
            print(f"WARN {key} run {spec['idx']} ({spec['path'].name}) failed ({exc}) -- skipping run")
    _finish(t1_raw, out_bids, work, key, meta)


def process(item: dict, raw_dir: str, out_bids: str, cfg: dict, stage="all", run_idx=None) -> None:
    src = item["sources"]
    t1_raw = Path(raw_dir) / src["t1"]["name"]
    run_specs = [
        {"path": Path(raw_dir) / r["name"], "idx": r["idx"], "run": r["run"], "tr_acq": r.get("tr_acq")}
        for r in src["runs"]
    ]
    work, key, meta = Path(raw_dir) / "work", item["key"], src.get("meta", {})
    if stage == "anat":
        _anat(t1_raw, out_bids, work, key)
    elif stage == "run":
        spec = next(s for s in run_specs if s["idx"] == int(run_idx))
        _one_run(spec, out_bids, work, key)
    elif stage == "finish":
        _finish(t1_raw, out_bids, work, key, meta)
    else:
        _preproc(t1_raw, run_specs, out_bids, work, key, meta)


def bids_images(raw_dir: Path, suffix: str) -> list[Path]:
    return sorted(
        p for p in raw_dir.glob(f"**/*_{suffix}.nii*")
        if p.is_file() and "/derivatives/" not in str(p).lower() and "/sourcedata/" not in str(p).lower()
    )


def task_of(bold: Path) -> str:
    m = re.search(r"_task-([A-Za-z0-9]+)", bold.name)
    return m.group(1) if m else "bold"


def run_tr(bold: Path, dataset_root: Path) -> float | None:
    own = bold.with_name(re.sub(r"\.nii(\.gz)?$", ".json", bold.name))
    for sidecar in (own, dataset_root / f"task-{task_of(bold)}_bold.json", dataset_root / "bold.json"):
        try:
            tr = json.loads(sidecar.read_text()).get("RepetitionTime")
        except Exception:
            continue
        if tr:
            return float(tr)
    return None


def participants_meta(dataset_root: Path, sub_label: str) -> dict:
    tsv = dataset_root / "participants.tsv"
    if not tsv.exists():
        return {}
    lines = [ln for ln in tsv.read_text(errors="replace").splitlines() if ln.strip()]
    if not lines:
        return {}
    col = {name: i for i, name in enumerate(lines[0].split("\t"))}
    for line in lines[1:]:
        row = line.split("\t")
        if row and row[0] == f"sub-{sub_label}":
            return {
                k: row[col[k]] for k in ("age", "sex")
                if k in col and col[k] < len(row) and row[col[k]] not in ("", "n/a", "N/A")
            }
    return {}


def subject_label(raw_dir: Path) -> str:
    if raw_dir.name.startswith("sub-"):
        return raw_dir.name[len("sub-"):]
    subs = sorted(p.name[len("sub-"):] for p in raw_dir.glob("sub-*") if p.is_dir())
    if len(subs) != 1:
        raise RuntimeError(f"expected exactly one sub-* under {raw_dir}, found {subs}")
    return subs[0]


def main(argv=None) -> None:
    """Two entry points: the driver's (``--item``/``--cfg``, one work item), and a standalone one
    that walks a raw-BIDS subject directory itself -- which is how this file stays liftable."""
    ap = argparse.ArgumentParser(
        description="raw BIDS subject -> preprocessed BIDS subject on the shared 2 mm grid")
    ap.add_argument("raw_dir", type=Path, nargs="?")
    ap.add_argument("out_dir", type=Path, nargs="?")
    ap.add_argument("--accession", default=None)
    ap.add_argument("--key", default=None)
    ap.add_argument("--max-runs", type=int, default=None)
    ap.add_argument("--tr", type=float, default=None)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--item", default=None)
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
    if args.item is not None:
        process(json.loads(Path(args.item).read_text()), str(args.raw_dir), str(args.out_dir), cfg,
                stage=args.stage, run_idx=args.run_idx)
        return

    preflight()
    sub_label = subject_label(args.raw_dir)
    dataset_root = args.raw_dir.parent if args.raw_dir.name.startswith("sub-") else args.raw_dir
    accession = args.accession or dataset_root.name
    key = args.key or f"{accession}__{sub_label}"
    runs = bids_images(args.raw_dir, "bold")
    if args.max_runs:
        runs = runs[: int(args.max_runs)]
    run_specs = [
        {"path": b, "idx": i, "run": f"func_{task_of(b)}_{i}", "tr_acq": args.tr or run_tr(b, dataset_root)}
        for i, b in enumerate(runs)
    ]
    meta = {"accession": accession, **participants_meta(dataset_root, sub_label)}
    work = (Path(args.work_dir) if args.work_dir else args.out_dir / "work") / sanitize_label(key)
    _preproc(bids_images(args.raw_dir, "T1w")[0], run_specs, args.out_dir, work, key, meta)


if __name__ == "__main__":
    main()
