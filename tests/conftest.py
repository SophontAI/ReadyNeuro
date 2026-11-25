"""One synthetic mask-based BIDS subject -- exactly what a preprocessing core writes.

Lets the finalizer tests run with no FreeSurfer / FSL / ANTs and no real data. The geometry is
deliberately tiny (a 12 mm anat cube, its nested 2 mm func grid) and the per-run fMRI coverage is
controllable, because the gates under test are all thresholds on those counts.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

from readyneuro.io import BidsLayout, sanitize_label

RAD = np.array([[-1, 0, 0, 8], [0, 1, 0, -10], [0, 0, 1, -8], [0, 0, 0, 1]], float)


def _func_grid(ref, mm=2.0):
    """The func grid nested in ``ref``, never finer than ref's zooms -- the same rule the cores'
    inlined ``iso_grid`` applies, replicated so this fixture needs no geometry kernel."""
    zooms = np.sqrt((np.asarray(ref.affine, float)[:3, :3] ** 2).sum(axis=0))
    mm = np.maximum(np.broadcast_to(float(mm), (3,)), zooms)
    affine = np.asarray(ref.affine, float).copy()
    affine[:3, :3] = (affine[:3, :3] / zooms) * mm
    shape = tuple(int(np.ceil(s * z / m - 1e-4)) for s, z, m in zip(ref.shape[:3], zooms, mm))
    return nib.Nifti1Image(np.zeros(shape, np.float32), affine)


def write_bids(out_bids, key: str, *, runs, n_t: int = 8, space: str = "native",
               with_dseg: bool = True) -> None:
    """Write one subject's BIDS tree.

    ``runs`` is a list of ``{"idx", "run", "tr_acq"?, "cov_slabs"?}``. The func-grid anatomical
    brain (the coverage reference) is the block ``[1:5, 1:5, 1:5]`` = 64 voxels; a run with
    ``cov_slabs=k`` covers ``[1:5, 1:5, 1:1+k]`` = ``16k`` voxels, so its coverage is ``0.25k``
    (k=2 -> 50%, k=3 -> 75%, k=4 -> 100%, the default). ``with_dseg=False`` omits the reference
    entirely, which is how a vendor core that ships no segmentation skips the coverage check.
    """
    label = sanitize_label(key)
    layout = BidsLayout(out_bids)
    layout.ensure_layout(label)
    rng = np.random.default_rng(abs(hash(key)) % (2**32))

    # 1 mm anat, nested cortical < gm < brain -- the invariant the completeness gate checks.
    brain = np.zeros((12, 12, 12), np.uint8); brain[3:9, 3:9, 3:9] = 1; brain[2:10, 2:10, 2] = 1
    gm = np.zeros((12, 12, 12), np.uint8); gm[2:10, 2:10, 2] = 1
    cortical = np.zeros((12, 12, 12), np.uint8); cortical[3:9, 3:9, 2] = 1
    nib.save(nib.Nifti1Image(rng.random((12, 12, 12)).astype(np.float32) * 100, RAD),
             layout.t1_path(label, space))
    nib.save(nib.Nifti1Image(brain, RAD), layout.brain_mask_path(label, space))
    nib.save(nib.Nifti1Image(gm, RAD), layout.gm_mask_path(label, space))
    nib.save(nib.Nifti1Image(cortical, RAD), layout.cortical_gm_mask_path(label, space))

    func_ref = _func_grid(nib.Nifti1Image(brain, RAD))
    shape, affine = tuple(func_ref.shape), func_ref.affine

    if with_dseg:
        anat_labels = np.zeros(shape, np.int16); anat_labels[1:5, 1:5, 1:5] = 7
        nib.save(nib.Nifti1Image(anat_labels, affine), layout.func_dseg_path(label, space))

    fgm = np.zeros(shape, np.uint8); fgm[1, 1, 1] = 1   # inside every run's covered slab
    for run in runs:
        covered = np.zeros(shape, np.uint8)
        covered[1:5, 1:5, 1:1 + int(run.get("cov_slabs", 4))] = 1
        nib.save(nib.Nifti1Image(covered, affine), layout.func_brain_mask_path(label, run["idx"], space))
        nib.save(nib.Nifti1Image(fgm, affine), layout.func_gm_mask_path(label, run["idx"], space))
        nib.save(nib.Nifti1Image(fgm, affine), layout.func_cortical_gm_mask_path(label, run["idx"], space))
        nib.save(nib.Nifti1Image(rng.random(shape + (n_t,)).astype(np.float32), affine),
                 layout.bold_path(label, run["idx"], space))

    layout.write_index(
        label,
        key=key,
        spaces=[space],
        runs=[{"idx": r["idx"], "run": r["run"], "tr_acq": r.get("tr_acq", 1.0)} for r in runs],
        meta={"cohort": "fake"},
    )


@pytest.fixture
def bids(tmp_path):
    """``bids(key, runs=[...], **kw) -> out_bids`` -- one subject under ``tmp_path``."""
    def _make(key="subA", *, runs=None, **kw):
        out_bids = str(tmp_path / "bids" / key)
        write_bids(out_bids, key, runs=runs or [{"idx": 0, "run": "rest", "tr_acq": 1.0}], **kw)
        return out_bids
    return _make
