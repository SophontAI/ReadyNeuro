"""The pretraining finalizer + the front half BOTH finalizers share.

One subject's preprocessed BIDS -> a self-contained `BundleTar` of int8 per-run blocks.
Dataset-agnostic. `finalize_eval` imports the front half FROM HERE, so the two products share ONE
loader with no drift and the dependency stays one-directional and acyclic.

The preprocessing script already did ALL the geometry: BOLD is on the fixed func grid, the T1 on its
1 mm grid, and both carry brain + gray-matter masks. So the finalizer does NO geometry -- it reads
the masks + BOLD, gathers the in-mask voxels by plain indexing (C order, exactly the codec's row
order), z-scores the 1 mm T1 over its brain mask, and then does the two things that are ours: the
temporal z-score / TR-resample, and the pack.

THE MASK CONTRACT: the store keeps three masks per layer -- ``brain`` (implicit: every stored row is
a brain voxel, since ``idx`` IS the brain mask) plus ``gm`` and ``cortical_gm`` as uint8 rows sampled
at that ``idx``. The core ships the three mask images; the codec enforces the nesting at pack time.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import nibabel as nib
import numpy as np

from readyneuro.io import BidsLayout
from readyneuro.tools import codec, shards
from readyneuro.tools.value_preproc import global_zscore_mask, zscore_resample
from readyneuro.io import ValidVolume

log = logging.getLogger("readyneuro.builder.finalize")


def _union_func_masks(bids_root: str, label: str, runs: list[dict], space: str):
    """The subject's func-grid brain + GM + cortical-GM masks = the per-mask UNION over runs (each
    per-run mask is that tissue ∩ the run's FOV).

    Union, not a single run: a voxel ANY run reached is worth storing, and a run that missed it
    stores a temporally-flat (z=0) row there, which the store tolerates. All runs share the fixed
    func grid, so this is a plain OR."""
    layout = BidsLayout(bids_root)
    brain = gm = cortical = None
    aff = None
    for r in runs:
        bp = layout.func_brain_mask_path(label, r["idx"], space)
        gp = layout.func_gm_mask_path(label, r["idx"], space)
        cp = layout.func_cortical_gm_mask_path(label, r["idx"], space)
        if not (bp.exists() and gp.exists() and cp.exists()):
            continue
        b = np.asanyarray(nib.load(bp).dataobj) != 0
        g = np.asanyarray(nib.load(gp).dataobj) != 0
        c = np.asanyarray(nib.load(cp).dataobj) != 0
        brain = b if brain is None else (brain | b)
        gm = g if gm is None else (gm | g)
        cortical = c if cortical is None else (cortical | c)
        aff = nib.load(bp).affine
    if brain is None:
        return None, None, None
    return (nib.Nifti1Image(brain.astype(np.uint8), aff),
            nib.Nifti1Image(gm.astype(np.uint8), aff),
            nib.Nifti1Image(cortical.astype(np.uint8), aff))


def _acq_tr(run: dict, bold: nib.Nifti1Image) -> float:
    """The acquisition TR to resample FROM: the sidecar's ``tr_acq``, else the BOLD header's
    pixdim[4]. The cores stamp the real TR into the header, so the fallback is meaningful -- but an
    UNSTAMPED nibabel header defaults to 1.0 s, which is a silent wrong TR, hence the warning. A
    non-positive TR (some OpenNeuro headers ship 0) is refused outright."""
    tr = run.get("tr_acq")
    if not tr:
        hdr_tr = float(bold.header.get_zooms()[3])
        log.warning("run %s: no manifest tr_acq -- falling back to BOLD header pixdim[4]=%.4g s. A core "
                    "should set tr_acq (and stamp the header via openneuro.set_tr); an unstamped header "
                    "defaults to 1.0 s (a silent wrong TR).", run.get("run"), hdr_tr)
        tr = hdr_tr
    if not tr or tr <= 0:
        raise RuntimeError(f"run {run.get('run')}: no usable acquisition TR (tr_acq={run.get('tr_acq')}, "
                           f"header={bold.header.get_zooms()[3]}) -- cannot temporally resample")
    return float(tr)


# --- BIDS-completeness gate (fail loud, never silently zero a dataset) ---------------------
def _require_file(path, label: str, space: str, what: str) -> None:
    """Fail loud if a CLAIMED space's BIDS file is absent -- the "a dropped mask save silently zeroes
    a dataset" trap. The space being in the index means the core was ASKED to produce it, so a
    missing file is a bug worth surfacing, not a subject to skip."""
    if not path.exists():
        raise RuntimeError(
            f"{label}/{space}: BIDS incomplete -- missing {what} ({path.name}). The space is CLAIMED "
            f"(in the index the core wrote) but a required file is absent -- likely a dropped save; "
            f"refusing to ship a half-formed shard."
        )


def _require_nested(brain_p, gm_p, cortical_p, label: str, space: str, where: str) -> None:
    """Assert the three masks are non-empty and nested (``cortical ⊆ gm ⊆ brain``) -- the invariant
    the codec re-enforces at pack time, checked here so a core bug fails at the boundary rather than
    shipping a degenerate mask set."""
    b = np.asanyarray(nib.load(brain_p).dataobj) != 0
    g = np.asanyarray(nib.load(gm_p).dataobj) != 0
    c = np.asanyarray(nib.load(cortical_p).dataobj) != 0
    if not b.any():
        raise RuntimeError(f"{label}/{space}: {where} brain mask is empty -- nothing to pack")
    if not (g & b == g).all() or not (c & g == c).all():
        raise RuntimeError(f"{label}/{space}: {where} masks are not nested (require cortical ⊆ gm ⊆ brain)")


def _require_subject_complete(bids_root: str, label: str, idx: dict, space: str) -> bool:
    """The completeness gate. Returns ``False`` (soft-skip: ship nothing) ONLY when ``space`` was
    never CLAIMED -- a genuinely modality-less subject, which the driver tolerates. When ``space`` IS
    claimed but any required piece is missing or inconsistent, raises so the build fails LOUDLY on
    that subject (keep-going logs and skips it, no ``.done``) rather than shipping a half-formed
    shard. The discriminator is ``space in idx['spaces']``, the CLAIMED set the core wrote."""
    if space not in (idx.get("spaces") or []):
        return False  # the subject simply lacks this space -- soft-skip, nothing to ship
    layout = BidsLayout(bids_root)
    _require_file(layout.t1_path(label, space), label, space, "1 mm T1")
    _require_file(layout.brain_mask_path(label, space), label, space, "1 mm brain mask")
    _require_file(layout.gm_mask_path(label, space), label, space, "1 mm GM mask")
    _require_file(layout.cortical_gm_mask_path(label, space), label, space, "1 mm cortical-GM mask")
    _require_nested(layout.brain_mask_path(label, space), layout.gm_mask_path(label, space),
                    layout.cortical_gm_mask_path(label, space), label, space, "1 mm anat")

    present = [r for r in idx["runs"] if layout.bold_path(label, r["idx"], space).exists()]
    if not present:
        raise RuntimeError(f"{label}/{space}: claimed but no BOLD run present (all dropped?) -- "
                           "refusing to ship an anat-only shard")
    faff = fshape = None
    for r in present:
        _require_file(layout.func_brain_mask_path(label, r["idx"], space), label, space, f"run {r['idx']} func brain mask")
        _require_file(layout.func_gm_mask_path(label, r["idx"], space), label, space, f"run {r['idx']} func GM mask")
        _require_file(layout.func_cortical_gm_mask_path(label, r["idx"], space), label, space, f"run {r['idx']} func cortical-GM mask")
        fb = nib.load(layout.func_brain_mask_path(label, r["idx"], space))
        if faff is None:
            faff, fshape = np.asarray(fb.affine), tuple(fb.shape)
        elif tuple(fb.shape) != fshape or not np.allclose(np.asarray(fb.affine), faff, atol=1e-3):
            raise RuntimeError(f"{label}/{space}: run {r['idx']} func grid differs from the subject's "
                               "func grid -- the per-run masks must share one grid to union")
        _acq_tr(r, nib.load(layout.bold_path(label, r["idx"], space)))  # raises on tr <= 0
    return True


class _Subject(NamedTuple):
    """One subject's preprocessed BIDS, loaded once -- the shared front half of BOTH finalizers.

    Carries the three mask images per layer (brain implicit in the packed rows; the codec samples
    gm/cortical at the stored voxels, `codec._tissue_rows`) + the z-scored 1 mm T1; each finalizer
    then differs only in the tail (pretrain packs a `BundleTar`; eval slices clips into Arrow)."""
    label: str                       # BIDS subject label (on-disk dir name)
    key: str                         # the true work key (== the store `sub`); may differ from `label`
    sidecar: dict                    # the BIDS index sidecar (runs / spaces / meta)
    runs: list                       # sidecar runs whose BOLD exists for this space
    t1_vv: object                    # z-scored 1 mm T1 ValidVolume (over the brain mask == its idx)
    gm1: nib.Nifti1Image             # 1 mm GM mask (T1 layer)
    cortical1: nib.Nifti1Image       # 1 mm cortical-GM mask (T1 layer)
    seg1: nib.Nifti1Image | None     # 1 mm aseg label vol (T1 layer seg); None if the core wrote none
    fbrain: nib.Nifti1Image          # func-grid brain mask (== bold idx)
    fgm: nib.Nifti1Image             # func-grid GM mask (BOLD layer)
    fcortical: nib.Nifti1Image       # func-grid cortical-GM mask (BOLD layer)
    fseg: nib.Nifti1Image | None     # func-grid aseg label vol (BOLD layer seg); None if none written
    mask: np.ndarray                 # func brain mask (bool, == bold_idx voxels)
    affine: np.ndarray               # func grid affine
    meta: dict                       # bundle meta (subject / space / demographics)


def _load_subject(bids_root: str, space: str) -> _Subject | None:
    """The front half BOTH finalizers share: read the core's outputs (the 1 mm T1 + its masks, the
    per-run func-grid masks) and z-score the T1. ``None`` when the subject has nothing to ship."""
    layout = BidsLayout(bids_root)
    label = layout.find_subject()
    idx = layout.read_index(label)
    if not _require_subject_complete(bids_root, label, idx, space):  # soft-skip iff unclaimed, else fail loud
        return None
    runs = [r for r in idx["runs"] if layout.bold_path(label, r["idx"], space).exists()]

    t1 = nib.load(layout.t1_path(label, space))                       # 1 mm brain-extracted T1 (this space's frame)
    brain1 = nib.load(layout.brain_mask_path(label, space))           # 1 mm brain mask
    gm1 = nib.load(layout.gm_mask_path(label, space))                 # 1 mm GM mask
    cortical1 = nib.load(layout.cortical_gm_mask_path(label, space))  # 1 mm cortical-GM mask
    # The full aseg label volumes the masks derive from -- SEPARATE store `seg` rows. OPTIONAL: the
    # from-raw native cores write them, the vendor MNI cores do not (`codec._seg_row` -> all-zero seg).
    seg1 = nib.load(layout.dseg_path(label, space)) if layout.dseg_path(label, space).exists() else None
    fseg = nib.load(layout.func_dseg_path(label, space)) if layout.func_dseg_path(label, space).exists() else None

    fbrain, fgm, fcortical = _union_func_masks(bids_root, label, runs, space)   # func-grid mask union (gate ensured masks present)

    return _Subject(
        label=label,
        key=idx["key"],
        sidecar=idx,
        runs=runs,
        t1_vv=global_zscore_mask(t1, brain1),  # z-score the 1 mm T1 over the brain mask (the one value op)
        gm1=gm1,
        cortical1=cortical1,
        seg1=seg1,
        fbrain=fbrain,
        fgm=fgm,
        fcortical=fcortical,
        fseg=fseg,
        mask=np.asanyarray(fbrain.dataobj) != 0,  # func brain mask (== bold_idx voxels)
        affine=np.asarray(fbrain.affine),
        meta={"subject": idx["key"], "space": space, **(idx.get("meta") or {})},
    )


# --- the GM-viability gate: reject a degenerate subject BEFORE any shard is opened -------------
def _passes_gm_gate(sub: _Subject, cfg: dict, space: str) -> bool:
    """Ship this subject only if its func-grid GM is not ABNORMALLY small. Both finalizers call this
    BEFORE opening a shard writer, so a rejection ships NOTHING and the driver rebuilds it on resume.

    A degenerate subject (failed registration / empty-FOV BOLD / bad segmentation) ships with ~0 GM
    voxels, and a near-empty ``fgm`` later crashes the LOADER selecting the GM stream. The gate is on
    the func GM count -- the exact tissue the loader selects -- so the count checked here is the
    count that would break downstream. The default 500 is conservative: a real cortex on a 2 mm grid
    is many thousands of GM voxels, a degenerate subject a few hundred at most."""
    gm_n = int(np.asanyarray(sub.fgm.dataobj).sum())
    brain_n = int(np.asanyarray(sub.fbrain.dataobj).sum())
    min_gm = int(cfg.get("min_func_gm_voxels", 500))
    ship = gm_n >= min_gm
    log.info("%s/%s: func gm=%d brain=%d -> %s", sub.key, space, gm_n, brain_n,
             "ship" if ship else f"reject <{min_gm}")
    return ship


def _run_vol(bids_root: str, sub: _Subject, run: dict, space: str, *, target_tr: float, common_tr):
    """One run's BOLD gathered at the func brain mask (a pure C-order index -- same grid) and
    temporally z-scored / resampled from its acquisition TR to ``target_tr``."""
    layout = BidsLayout(bids_root)
    bold = nib.load(layout.bold_path(sub.label, run["idx"], space))
    data = np.asarray(bold.dataobj, dtype=np.float32)                   # (X, Y, Z, T) on the func grid
    vol = ValidVolume(data[sub.mask], sub.mask, sub.affine, tr=_acq_tr(run, bold))  # (M, T) C-order gather
    return zscore_resample(vol, target_tr, common_tr=common_tr)


# --- fMRI-coverage gate: reject a run whose BOLD covers too little of the anatomical brain ---
def _coverage_ok(bids_root: str, sub: _Subject, run: dict, cfg: dict, space: str) -> bool:
    """Keep (``True``) or reject (``False``) one run on its fMRI coverage of the anatomical brain.

        coverage = |run_brain_fov ∩ anat_brain| / |anat_brain|

    ``run_brain_fov`` is the per-run func brain mask (already brain ∩ FOV; the ∩ below is a guard).
    ``anat_brain`` is ``sub.fseg != 0``, the FOV-INDEPENDENT aseg on the func grid. When ``sub.fseg``
    is None the check is SKIPPED and the run kept: only the from-raw cores ship a func dseg, so the
    vendor MNI cores reject nothing here.

    On rejection: warn and return ``False``, so the caller skips the run BEFORE the expensive
    gather. A subject whose every run is rejected ends with no runs."""
    if sub.fseg is None:
        return True  # no FOV-independent brain reference -> cannot check -> keep (non-regression)
    layout = BidsLayout(bids_root)
    bp = layout.func_brain_mask_path(sub.label, run["idx"], space)
    anat_brain = np.asanyarray(sub.fseg.dataobj) != 0
    anat_n = int(anat_brain.sum())
    if anat_n == 0:
        return True  # degenerate seg (empty brain) -- the GM gate handles that; nothing to divide by
    run_fov = np.asanyarray(nib.load(bp).dataobj) != 0
    inter = int((run_fov & anat_brain).sum())
    coverage = inter / anat_n
    thr = float(cfg.get("min_func_coverage", 0.60))
    if coverage >= thr:
        return True
    log.warning("%s/%s run %s: fMRI coverage %.1f%% < %.0f%% of the anatomical brain (likely bad "
                "registration) -- REJECTING the run", sub.key, space, run.get("run"),
                coverage * 100, thr * 100)
    return False


def finalize_pretrain(bids_root: str, cfg: dict, *, split: str, space: str) -> str | None:
    """Pack one subject's preprocessed BIDS into a `BundleTar` for ``(split, space)``. Returns the
    tar uri, or ``None`` if the subject has no usable runs.

    Builds the bundle (BOLD geometry from the func mask union + the z-scored 1 mm T1 layer), then for
    each run gathers ``bold[mask]`` and temporally z-scores / resamples it to ``target_tr``."""
    sub = _load_subject(bids_root, space)
    if sub is None:
        return None
    if not _passes_gm_gate(sub, cfg, space):  # reject a degenerate subject BEFORE opening the tar
        return None

    bundle = codec.pack_bundle(sub.fbrain, sub.fgm, sub.fcortical, sub.t1_vv, sub.gm1,
                               sub.cortical1, sub.meta, quant="f16",
                               seg_f_img=sub.fseg, seg1_img=sub.seg1)
    target_tr = float(cfg.get("target_tr", 1.0))
    common_tr = cfg.get("common_tr")

    root = cfg["pretrain_root"]
    with shards.BundleTar(root, split, space, sub.key, bundle, dataset=cfg["dataset"],
                          storage_options=cfg.get("out_storage_options")) as tar:
        n = 0
        for r in sub.runs:
            if not _coverage_ok(bids_root, sub, r, cfg, space):  # reject low-fMRI-coverage run before the gather
                continue
            vol = _run_vol(bids_root, sub, r, space, target_tr=target_tr, common_tr=common_tr)
            tar.add_run(r["run"], vol)
            n += 1
    # All runs dropped or coverage-rejected -> ship nothing. That is an EXPECTED bad-registration
    # outcome, not a machine failure, so it logs benignly; `BundleTar` already left no `.done` and
    # swept the partial tar on the way out.
    if not n:
        log.warning("%s/%s: no usable runs (all dropped/coverage-rejected) -- shipping nothing",
                    sub.key, space)
        return None
    return shards.tar_uri(root, split, space, sub.key)
