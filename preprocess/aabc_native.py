#!/usr/bin/env python3
"""AABC native-from-raw core: the shared from-raw kernel driven over AABC's RAW scanner-space REST.

Reuses ``openneuro`` verbatim for the anat and for the per-run geometry primitives. Three AABC
specifics live here:

1. **Zip-member I/O.** AABC ships per-session zips, not loose NIfTIs. The STRUCT zip is staged
   locally by the driver and one member -- ``T1w/T1w_acpc_dc.nii.gz``, AC-PC and gradient-distortion
   corrected but NOT bias-corrected, so N4 still does real work -- is extracted for the anat stage.
   The RAW REST members are streamed straight out of the ~2.34 GB remote session zip (a central
   directory read + one ranged member read), so a subject costs ~1.06 GB of transfer, not 2.34 GB.

2. **The eval window spans two runs.** The benchmark windows the REST concat at ``[0:500]``, and that
   concat is in acquisition order REST1_AP, REST1_PA, ... at 478 frames each -- so the window is all
   478 frames of REST1_AP followed by the first 22 of REST1_PA. `_rest_window` motion-corrects and
   registers EACH raw run independently (different phase-encode directions, hence different mean
   EPIs), then streams both into ONE 500-frame series on the shared func grid. That is why this core
   is not run-staged: the two runs are not independent DAG nodes.

3. **Per-run intensity normalisation.** The vendor pipeline scales each run to a common global mean
   before concatenating; from raw there is no such shared scaling, so a bare concat would put a step
   discontinuity at frame 478 that the finalizer's per-voxel temporal z-score cannot remove (it is a
   SCALE change, not an offset the z-score absorbs). `_brain_mean` measures each run's in-brain mean
   on the resampled grid and `_rest_window` rescales every later run onto run 0's.

Deliberately NO susceptibility-distortion correction, even though the raw zips ship spin-echo AP/PA
fieldmaps: every from-raw build here drops SDC, and that gap versus the vendor product is exactly
what the native-vs-vendor comparison measures.

Standalone-but-composing: imports nothing from the package, and sibling-imports ``openneuro`` (also
package-free) -- legal because the driver invokes these scripts BY PATH.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.affines import apply_affine
from scipy.ndimage import map_coordinates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openneuro  # noqa: E402  -- the shared from-raw kernel

#: The benchmark's single REST window, in RAW frames at the 0.8 s acquisition TR (-> 400 after the
#: finalizer resamples to 1.0 s). The vendor AABC core windows the same 500 concat frames.
REST_WINDOW = 500
T1_LOCAL = "t1_acpc.nii.gz"  # the T1 member extracted out of the staged struct zip, in `work/`


def preflight(cfg=None) -> None:
    openneuro.preflight(cfg)


# --- zip member I/O ----------------------------------------------------------
def _open_zip(uri: str, storage_options: dict | None):
    """A ``ZipFile`` over a local path or an ``s3://`` object. For a remote zip this reads only the
    end-of-central-directory + central directory (a couple of ranged GETs); member reads are ranged too,
    so extracting one run never transfers the whole archive."""
    if str(uri).startswith("s3://"):
        import s3fs

        fh = s3fs.S3FileSystem(**(storage_options or {})).open(uri, "rb", block_size=16 * 2**20)
    else:
        fh = open(uri, "rb")
    return zipfile.ZipFile(fh), fh


def _extract_member(zip_uri: str, member: str, dest: Path, storage_options: dict | None = None,
                    *, gunzip: bool = False) -> Path:
    """Copy one zip member to ``dest``. ``gunzip`` decompresses a ``.nii.gz`` member to a plain ``.nii``.

    That matters for the BOLD: `openneuro.resample_bold_to_file` reads ``dataobj[..., t]`` frame by
    frame, and without ``indexed_gzip`` nibabel restarts the gzip stream for every random access -- so a
    478-frame run on a ``.nii.gz`` is quadratic. We are already decompressing the member as we copy it,
    so writing plain ``.nii`` (memory-mappable, O(1) per frame) is free."""
    z, fh = _open_zip(zip_uri, storage_options)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with z.open(member) as src, open(dest, "wb") as out:
            if gunzip:
                with gzip.open(src) as gz:
                    shutil.copyfileobj(gz, out, 16 * 2**20)
            else:
                shutil.copyfileobj(src, out, 16 * 2**20)
    finally:
        z.close()
        fh.close()
    return dest


def _stage_t1(item, raw_dir, work: Path) -> Path:
    """Extract the AC-PC T1 out of the locally staged struct zip (idempotent across stages)."""
    t1 = work / T1_LOCAL
    if not t1.exists():
        src = item["sources"]["t1"]
        work.mkdir(parents=True, exist_ok=True)
        _extract_member(str(Path(raw_dir) / src["name"]), src["member"], t1)
    return t1


# --- per-run motion + registration (openneuro primitives) --------------------
def _run_transforms(bold_raw: Path, work: Path, idx: int, struct, mni_to_t1):
    """mcflirt + rigid mean-EPI->T1, composed with the anat's T1->template rigid.

    Returns ``(raw_bold_img, mean_epi_img, per_volume, mni_to_mean)``: ``per_volume[t]`` maps a func-grid
    world point back into raw-BOLD world space for frame ``t``, exactly as `openneuro._one_run` builds
    it, so the resample is a SINGLE interpolation of the raw data (motion + registration composed)."""
    moco = work / f"moco{idx:02d}"
    openneuro.run(f"mcflirt -in {bold_raw} -out {moco} -meanvol -mats")

    epi_lta, epi_mat = work / f"epi2t1_{idx:02d}.lta", work / f"epi2t1_{idx:02d}.mat"
    mean_epi = Path(f"{moco}_mean_reg.nii.gz")
    openneuro.rigid_register(mean_epi, work / "t1_brain.nii.gz", epi_lta, epi_mat)

    mean_img, raw_bold = nib.load(mean_epi), nib.load(bold_raw)
    mni_to_mean = openneuro.fsl_to_ras(epi_mat, mean_img, struct) @ mni_to_t1
    motion_mats = sorted(Path(f"{moco}.mat").glob("MAT_*"), key=lambda p: int(p.name.split("_")[1]))
    per_volume = [openneuro.fsl_to_ras(m, mean_img, mean_img) @ mni_to_mean for m in motion_mats]
    return raw_bold, mean_img, per_volume, mni_to_mean


def _brain_mean(mean_img, world: np.ndarray, mni_to_mean) -> float:
    """The run's mean in-brain intensity, sampled from its mean EPI at the func grid's in-brain world
    points. Used only as a RATIO between runs (see the module doc on the concat discontinuity), so the
    absolute units do not matter -- only that both runs are measured the same way."""
    inv = np.linalg.inv(np.asarray(mean_img.affine, float))
    coords = apply_affine(inv @ np.asarray(mni_to_mean, float), world).T
    vals = map_coordinates(np.asarray(mean_img.dataobj, np.float32), coords, order=1,
                           mode="constant", cval=0.0)
    inside = vals > 0
    return float(vals[inside].mean()) if inside.any() else 0.0


# --- the concatenated window -------------------------------------------------
def _resample_concat_to_file(segments, reference, sel, out_path: Path, tr, order: int = 1) -> np.ndarray:
    """Stream several raw runs' resampled frames into ONE ``.nii`` on ``reference``'s grid.

    The multi-run twin of `openneuro.resample_bold_to_file`, byte-compatible with it for a single
    unscaled segment: same header, same Fortran-order frame blocks, same ``in_fov`` accumulation.
    ``segments`` is ``[(bold_img, per_volume, scale), ...]`` written in list order, so the on-disk
    time axis is the concatenation."""
    shape = tuple(reference.shape[:3])
    n_frames = sum(len(pv) for _, pv, _ in segments)
    n_vox = int(np.prod(shape))
    sel_idx = np.flatnonzero(np.asarray(sel, bool).reshape(-1))
    ijk = np.stack(np.unravel_index(sel_idx, shape), axis=1).astype(np.float64)
    world = apply_affine(np.asarray(reference.affine, float), ijk)

    absmax = np.zeros(len(sel_idx), np.float32)  # running max|value| over time, at the sel voxels
    frame = np.zeros(n_vox, np.float32)          # ONE reused dense frame buffer
    with open(out_path, "wb") as f:
        f.write(openneuro._preproc_bold_header(shape, n_frames, reference.affine, tr))
        for bold, per_volume, scale in segments:
            inv_moving = np.linalg.inv(np.asarray(bold.affine, float))
            for t, mat in enumerate(per_volume):
                coords = apply_affine(inv_moving @ np.asarray(mat, float), world).T
                vals = map_coordinates(np.asarray(bold.dataobj[..., t], np.float32), coords,
                                       order=order, mode="constant", cval=0.0).astype(np.float32)
                if scale != 1.0:
                    vals *= np.float32(scale)
                frame[sel_idx] = vals
                f.write(frame.reshape(shape).tobytes(order="F"))  # F-order == the on-disk voxel order
                frame[sel_idx] = 0.0
                np.maximum(absmax, np.abs(vals), out=absmax)
    in_fov = np.zeros(n_vox, bool)
    in_fov[sel_idx] = absmax > 0
    return in_fov.reshape(shape)


def _rest_window(item, raw_dir, out_bids, cfg) -> dict:
    """Build the single benchmark REST window (``[0:REST_WINDOW]`` of the REST concat) as run 00.

    Streams each raw REST run in turn, motion-corrects and registers it, and stops as soon as
    ``REST_WINDOW`` frames have been collected -- so REST1_PA contributes only its first 22 frames
    and the remaining REST runs are never fetched at all."""
    label = openneuro.sanitize_label(item["key"])
    work = Path(raw_dir) / "work"
    func = Path(out_bids) / f"sub-{label}" / "func"
    func.mkdir(parents=True, exist_ok=True)
    so = cfg.get("input_storage_options")

    struct, mni_bold, mni_to_t1, seg_bold = openneuro.anat_refs(work)
    brain_bold = seg_bold != 0
    sel_idx = np.flatnonzero(brain_bold.reshape(-1))
    ijk = np.stack(np.unravel_index(sel_idx, tuple(mni_bold.shape[:3])), axis=1).astype(np.float64)
    world = apply_affine(np.asarray(mni_bold.affine, float), ijk)

    segments, staged, sources, reference_mean = [], [], [], None
    remaining = REST_WINDOW
    for r in item["sources"]["eval_runs"]:
        if remaining <= 0:
            break
        # Plain `.nii`: the per-frame resample below needs O(1) random access (see `_extract_member`).
        tmp = work / f"raw_run_{int(r['idx']):02d}.nii"
        try:
            _extract_member(r["uri"], r["member"], tmp, so, gunzip=True)
        except KeyError:
            # A session need not have all four REST runs (2 of 584 eval sessions have REST1_AP +
            # REST2_AP and no REST1_PA). The vendor concat also takes whatever exists in this order,
            # so falling through to the next run reproduces its window rather than degrading it.
            print(f"WARN {item['key']}: {r['run']} not in the archive -- falling through to the next REST run")
            continue
        staged.append(tmp)
        raw_bold, mean_img, per_volume, mni_to_mean = _run_transforms(tmp, work, int(r["idx"]),
                                                                     struct, mni_to_t1)
        take = min(remaining, len(per_volume))
        mean_intensity = _brain_mean(mean_img, world, mni_to_mean)
        if reference_mean is None:
            reference_mean = mean_intensity
        # Rescale every later run onto the first run's intensity scale (the vendor pipeline does
        # this before it concatenates; from raw nothing else would). A degenerate measurement leaves
        # the run untouched.
        scale = 1.0
        if reference_mean and mean_intensity:
            scale = float(reference_mean / mean_intensity)
        segments.append((raw_bold, per_volume[:take], scale))
        sources.append({"run": r["run"], "frames": int(take), "scale": scale})
        remaining -= take

    if not segments:
        raise RuntimeError(f"{item['key']}: no raw REST run yielded frames -- refusing to write an empty window")

    bold_path = func / f"sub-{label}_run-00_space-{openneuro.SPACE}_desc-preproc_bold.nii"
    tr_acq = float(item["sources"]["eval_runs"][0]["tr_acq"])
    try:
        in_fov = _resample_concat_to_file(segments, mni_bold, brain_bold, bold_path, tr_acq)
    finally:
        for p in staged:  # free ~1.5 GB of scratch as soon as the resample has read it
            p.unlink(missing_ok=True)

    gm_bold = np.isin(seg_bold, openneuro.GM_LABELS)
    cortical_bold = np.isin(seg_bold, openneuro.CORTICAL_GM_LABELS)
    openneuro.save_mask(brain_bold & in_fov, mni_bold.affine,
                        func / f"sub-{label}_run-00_space-{openneuro.SPACE}_desc-brain_mask.nii.gz")
    openneuro.save_mask(gm_bold & in_fov, mni_bold.affine,
                        func / f"sub-{label}_run-00_space-{openneuro.SPACE}_label-GM_mask.nii.gz")
    openneuro.save_mask(cortical_bold & in_fov, mni_bold.affine,
                        func / f"sub-{label}_run-00_space-{openneuro.SPACE}_label-corticalGM_mask.nii.gz")

    n_frames = sum(len(pv) for _, pv, _ in segments)
    entry = {"idx": 0, "run": "REST", "tr_acq": tr_acq, "n_frames": int(n_frames),
             "native_mm": float(max(segments[0][0].header.get_zooms()[:3])), "sources": sources}
    (work / "run_00000.json").write_text(json.dumps(entry))
    return entry


# --- stages ------------------------------------------------------------------
def _anat(item, raw_dir, out_bids, cfg) -> None:
    work = Path(raw_dir) / "work"
    t1_raw = _stage_t1(item, raw_dir, work)
    openneuro._anat(t1_raw, out_bids, work, item["key"])


def _finish(item, raw_dir, out_bids, cfg) -> None:
    work = Path(raw_dir) / "work"
    t1_raw = _stage_t1(item, raw_dir, work)
    meta = {**(item["sources"].get("meta") or {}), "session": item["sources"].get("session")}
    openneuro._finish(t1_raw, out_bids, work, item["key"], meta)


def process(item: dict, raw_dir: str, out_bids: str, cfg: dict, stage="all", run_idx=None) -> None:
    if stage == "anat":
        _anat(item, raw_dir, out_bids, cfg)
    elif stage == "run":
        _rest_window(item, raw_dir, out_bids, cfg)
    elif stage == "finish":
        _finish(item, raw_dir, out_bids, cfg)
    else:
        _anat(item, raw_dir, out_bids, cfg)
        _rest_window(item, raw_dir, out_bids, cfg)
        _finish(item, raw_dir, out_bids, cfg)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="AABC native-from-raw core (zip-streaming)")
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
    process(json.loads(Path(args.item).read_text()), args.raw, args.out, cfg,
            stage=args.stage, run_idx=args.run_idx)


if __name__ == "__main__":
    main()
