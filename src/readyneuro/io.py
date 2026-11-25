"""The pipeline's third-party-only io leaf: S3 storage options, the packed-volume container,
the benchmark-metadata fetcher, and the BIDS-derivatives path contract.

It imports NOTHING from ``readyneuro``, so both the storage layer (``tools/*``) and the build layer
(``builder/*``) import it downward with no layer inversion.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import nibabel as nib
import numpy as np
import s3fs

@dataclass(frozen=True)
class ValidVolume:
    """A grid's in-mask voxels, packed -- never the dense X*Y*Z*T array (~95% empty).

    ``values`` is ``(M,)`` (3D struct) or ``(M, T)`` (4D BOLD) float32, aligned to ``mask``'s True
    voxels in C order -- the same order `codec` gathers by, so packing is a no-op.
    """

    values: np.ndarray  # (M,) or (M, T) float32
    mask: np.ndarray  # (X, Y, Z) bool
    affine: np.ndarray  # (4, 4)
    tr: float | None = None  # seconds per STORED frame, after any resample; None for 3D
    #: The run's LIMITING TR period (s): ``max(acquisition_tr, harmonization_tr, tr)`` -- the
    #: COARSEST sampling in the resample chain, so its Nyquist is the true temporal bandwidth the
    #: model conditions on. NOT the acquisition TR alone: a run acquired FINER than the stored rate
    #: (HCP 0.72 s / AABC 0.8 s downsampled to 1.0 s) is bandwidth-limited by ``tr``. None = unknown.
    nyquist_tr: float | None = None
    #: Per-voxel temporal mean/std the z-score divided out, PRE-resample. `mean` is the mean EPI,
    #: which an alignment check needs and z-scoring destroys. `std` is the honest tSNR denominator
    #: (the stored series is TR-interpolated, which smooths variance).
    mean: np.ndarray | None = None  # (M,) float32
    std: np.ndarray | None = None  # (M,) float32

    def __post_init__(self) -> None:
        # Row k of `values` must BE mask voxel k; a mismatch misaligns silently.
        if self.values.shape[0] != int(self.mask.sum()):
            raise ValueError(
                f"ValidVolume: {self.values.shape[0]} value rows != "
                f"{int(self.mask.sum())} mask voxels"
            )

    @property
    def n_frames(self) -> int:
        return self.values.shape[1] if self.values.ndim == 2 else 1


def aws_config_endpoint(profile: str) -> str | None:
    """``endpoint_url`` of an ``~/.aws/config`` profile. R2's endpoint lives there, but s3fs
    does not auto-read it from the profile, so we pass it via ``client_kwargs``."""
    import configparser

    cfg = configparser.ConfigParser()
    cfg.read(os.path.expanduser("~/.aws/config"))
    section = profile if profile == "default" else f"profile {profile}"
    if cfg.has_section(section) and cfg.has_option(section, "endpoint_url"):
        return cfg.get(section, "endpoint_url")
    return None


def profile_storage_options(
    profile: str | None = None, endpoint_url: str | None = None
) -> dict:
    """s3fs ``storage_options`` for an R2/AWS **input** bucket: the profile + its endpoint,
    auto-read from ``~/.aws/config`` when ``endpoint_url`` is not given."""
    if endpoint_url is None and profile:
        endpoint_url = aws_config_endpoint(profile)
    opts: dict = {}
    if profile:
        opts["profile"] = profile
    if endpoint_url:
        opts["client_kwargs"] = {"endpoint_url": endpoint_url}
    return opts


def output_storage_options(
    profile: str | None = None, endpoint_url: str | None = None
) -> dict:
    """`profile_storage_options` plus ``fixed_upload_size``, for an **output** bucket. R2 rejects
    multipart uploads whose non-trailing parts differ in size (s3fs streams variable-sized parts by
    default), so any object over the 50 MiB block size fails on commit -- pretrain tars routinely
    exceed that."""
    opts = profile_storage_options(profile, endpoint_url)
    opts["fixed_upload_size"] = True
    return opts


def is_s3(uri) -> bool:
    return str(uri).startswith("s3://")


def s3_filesystem(storage_options=None) -> s3fs.S3FileSystem:
    return s3fs.S3FileSystem(**(storage_options or {}))


def path_exists(uri, storage_options=None) -> bool:
    """Existence check for a local path or an ``s3://`` uri."""
    return (
        s3_filesystem(storage_options).exists(uri) if is_s3(uri) else Path(uri).exists()
    )


#: gzip's magic number -- the first two bytes of any ``.gz`` stream.
GZIP_MAGIC = b"\x1f\x8b"


#: Root of the benchmark suite's per-dataset metadata (curated splits, phenotype target maps, event
#: onsets). `fetch` appends ``<DATASET>/metadata/<file>``. Defaults to the raw-GitHub ``datasets/``
#: tree of the pinned Brainmarks commit (its published package omits that folder);
#: $BRAINMARKS_METADATA_URL overrides it with any local directory
#: or a raw HTTP base (`fetch` reads a local base off disk and everything else over HTTP -- an
#: ``s3://`` base would reach `urlopen`, which cannot serve it).
BRAINMARKS_METADATA_DEFAULT = (
    "https://raw.githubusercontent.com/Aceticia/Brainmarks/d4640074168f1c6dcb0a94ec78801577792be3f4/datasets"
)
BENCHMARK_METADATA_ROOT = (
    os.environ.get("BRAINMARKS_METADATA_URL") or BRAINMARKS_METADATA_DEFAULT
).rstrip("/")

_RETRY_CODES = (429, 500, 502, 503, 504)
_ATTEMPTS = 6


@lru_cache
def fetch(base_url: str, name: str) -> bytes:
    """``{base_url}/{name}`` as bytes, preferring the local stage and retrying throttling.

    A local base (a Brainmarks checkout's datasets/ tree) is read straight off disk. Over HTTP a build
    runs a worker per subject, so that is one request per worker per file and the host answers with
    429 -- hence the stage dir and the backoff. Only 429/5xx retry; others raise at once."""
    if not base_url:
        raise RuntimeError(
            f"no source for {name}: set $BRAINMARKS_METADATA_URL to a Brainmarks datasets/ tree"
        )
    if "://" not in base_url:  # a local checkout
        return (Path(base_url) / name).read_bytes()
    stage = os.environ.get("FMRI_META_DIR")  # optional local stage: sibling workers read disk
    if stage and (p := Path(stage) / name).exists():
        return p.read_bytes()
    last: Exception | None = None
    for attempt in range(_ATTEMPTS):
        try:
            with urlopen(f"{base_url}/{name}") as r:
                data = r.read()
            if stage:
                dest = Path(stage) / name
                dest.parent.mkdir(parents=True, exist_ok=True)  # `name` may hold a subdir
                dest.write_bytes(data)
            return data
        except HTTPError as e:
            last = e
            if e.code not in _RETRY_CODES:
                raise
        except URLError as e:
            last = e
        time.sleep(min(2**attempt, 30))
    raise last  # exhausted retries


INDEX_SUFFIX = "desc-fmridata.json"


def sanitize_label(key: str) -> str:
    """A BIDS-valid subject label from a work ``key`` -- OpenNeuro ``ds000030__10159`` ->
    ``ds00003010159``. Lossy but deterministic; the reversible truth is the sidecar's ``key``."""
    return re.sub(r"[^A-Za-z0-9]", "", key)


class BidsLayout:
    """The BIDS-derivatives layout that is the core->finalizer contract, bound to one subject tree.

    The preprocessing scripts under ``preprocess/`` write these paths BY HAND (they import nothing
    from this package, deliberately), and the finalizers read them through here. Rename anything
    below and BOTH sides must change.

    Layout::

        <root>/
          sub-<label>/
            anat/                                        # 1 mm, per space (native and/or mni)
              sub-<label>_space-<space>_desc-preproc_T1w.nii.gz    # brain-extracted T1 (raw intensities)
              sub-<label>_space-<space>_desc-brain_mask.nii.gz     # brain mask (aseg label != 0)
              sub-<label>_space-<space>_label-GM_mask.nii.gz       # gray-matter mask (aseg GM labels)
              sub-<label>_space-<space>_label-corticalGM_mask.nii.gz  # cortical-GM subset
              sub-<label>_space-<space>_desc-aseg_dseg.nii.gz      # full SynthSeg aseg label vol (OPTIONAL)
            func/                                        # all on the fixed 2 mm MNI grid (nests with anat)
              sub-<label>_run-<NN>_space-<space>_desc-preproc_bold.nii   # BOLD, per run x space (UNZIPPED)
              sub-<label>_run-<NN>_space-<space>_desc-brain_mask.nii.gz  # brain mask n this run's FOV
              sub-<label>_run-<NN>_space-<space>_label-GM_mask.nii.gz    # GM mask n this run's FOV
              sub-<label>_run-<NN>_space-<space>_label-corticalGM_mask.nii.gz
              sub-<label>_space-<space>_desc-aseg_dseg.nii.gz            # func-grid aseg label vol, subject-level (OPTIONAL)
            sub-<label>_desc-fmridata.json                  # index sidecar: run map + TRs + spaces + meta

    The cores are MASK-based: they write brain + GM + cortical-GM masks (from the SynthSeg aseg). The
    from-raw native cores ALSO write the full aseg label volume (``desc-aseg_dseg``, OPTIONAL -- the
    vendor MNI cores omit it), which the finalizer packs as the store's separate ``seg`` row. Runs are
    a zero-padded ``run-<NN>`` index the sidecar maps back to the real run key.
    """

    def __init__(self, root: str) -> None:
        self.root = root

    def _sub_dir(self, label: str) -> Path:
        return Path(self.root) / f"sub-{label}"

    # Every path is space-qualified: the store's T1 layer must share the func space's frame.
    def t1_path(self, label: str, space: str) -> Path:
        return self._sub_dir(label) / "anat" / f"sub-{label}_space-{space}_desc-preproc_T1w.nii.gz"

    def brain_mask_path(self, label: str, space: str) -> Path:
        return self._sub_dir(label) / "anat" / f"sub-{label}_space-{space}_desc-brain_mask.nii.gz"

    def gm_mask_path(self, label: str, space: str) -> Path:
        """Anatomical gray-matter mask (cortex + cerebellar cortex + subcortical GM), on the T1 grid."""
        return self._sub_dir(label) / "anat" / f"sub-{label}_space-{space}_label-GM_mask.nii.gz"

    def cortical_gm_mask_path(self, label: str, space: str) -> Path:
        """Anatomical cortical gray-matter mask (cerebral cortex only), on the T1 grid -- a GM subset."""
        return self._sub_dir(label) / "anat" / f"sub-{label}_space-{space}_label-corticalGM_mask.nii.gz"

    def dseg_path(self, label: str, space: str) -> Path:
        """Full SynthSeg aseg LABEL volume (int16) on the 1 mm T1 grid. OPTIONAL: the from-raw native
        cores write it (the seg the gm/cortical masks are derived from), the vendor MNI cores do not --
        the finalizer packs an all-zero ``seg`` row when it is absent (`codec._seg_row`)."""
        return self._sub_dir(label) / "anat" / f"sub-{label}_space-{space}_desc-aseg_dseg.nii.gz"

    # --- func paths ---------------------------------------------------------
    def bold_path(self, label: str, run_idx: int, space: str) -> Path:
        """One run's preprocessed BOLD on the func grid.

        UNCOMPRESSED (``.nii``): the BIDS tree is scratch and the only reader is the finalizer, so
        gzip buys nothing while its CPU cost dominates -- HCP's ~600 MB float32 volumes
        gzip-bottlenecked the build at ~2 min/run. The masks + T1 stay gzipped (tiny)."""
        name = f"sub-{label}_run-{run_idx:02d}_space-{space}_desc-preproc_bold.nii"
        return self._sub_dir(label) / "func" / name

    def func_brain_mask_path(self, label: str, run_idx: int, space: str) -> Path:
        """One run's brain mask on the BOLD grid, zeroed outside the run's field of view (brain n FOV)."""
        return self._sub_dir(label) / "func" / f"sub-{label}_run-{run_idx:02d}_space-{space}_desc-brain_mask.nii.gz"

    def func_gm_mask_path(self, label: str, run_idx: int, space: str) -> Path:
        """One run's gray-matter mask on the BOLD grid, zeroed outside the run's field of view (GM n FOV)."""
        return self._sub_dir(label) / "func" / f"sub-{label}_run-{run_idx:02d}_space-{space}_label-GM_mask.nii.gz"

    def func_cortical_gm_mask_path(self, label: str, run_idx: int, space: str) -> Path:
        """One run's cortical gray-matter mask on the BOLD grid, zeroed outside the run's FOV (corticalGM n FOV)."""
        return self._sub_dir(label) / "func" / f"sub-{label}_run-{run_idx:02d}_space-{space}_label-corticalGM_mask.nii.gz"

    def func_dseg_path(self, label: str, space: str) -> Path:
        """Full SynthSeg aseg LABEL volume (int16) on the func (2 mm) grid, SUBJECT-LEVEL (not per run):
        the labels are anatomy, FOV-independent, so one image covers every run; the finalizer samples it
        at the func brain-mask union ``idx``. OPTIONAL, same as `dseg_path` -- absent -> an all-zero seg."""
        return self._sub_dir(label) / "func" / f"sub-{label}_space-{space}_desc-aseg_dseg.nii.gz"

    # --- index sidecar ------------------------------------------------------
    def index_path(self, label: str) -> Path:
        return self._sub_dir(label) / f"sub-{label}_{INDEX_SUFFIX}"

    def write_index(self, label: str, *, key: str, spaces: list[str], runs: list[dict],
                    meta: dict | None = None) -> None:
        """Write the sidecar the finalizer reads. ``runs`` is a list of ``{idx, run, tr_acq}``:
        ``idx`` is the ``run-<NN>`` in the BOLD filename, ``run`` the true run key, ``tr_acq`` the
        acquisition TR the finalizer resamples FROM.

        CONTRACT -- ``spaces`` is the CLAIMED deliverable spaces (what the core was ASKED to produce),
        never the achieved set. The finalizer's completeness gate keys on ``space in spaces`` to tell a
        genuinely modality-less subject (soft-skip) from a dropped mask save (fail loud); recording
        only achieved spaces would silently turn every dropped save into a skip."""
        p = self.index_path(label)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(
            {"key": key, "label": label, "spaces": list(spaces), "runs": runs, "meta": meta or {}},
            indent=2,
        ))

    def read_index(self, label: str) -> dict:
        return json.loads(self.index_path(label).read_text())

    def find_subject(self) -> str:
        """The single ``sub-<label>`` a per-subject BIDS tree contains -> its label. Raises if zero or
        many (the driver writes one subject per tree, so either is a bug worth surfacing)."""
        subs = sorted(p.name[len("sub-"):] for p in Path(self.root).glob("sub-*") if p.is_dir())
        if len(subs) != 1:
            raise ValueError(f"expected exactly one sub-* under {self.root}, found {subs}")
        return subs[0]

    def ensure_layout(self, label: str) -> None:
        """Create the anat/ + func/ dirs for a subject (idempotent)."""
        for sub in ("anat", "func"):
            (self._sub_dir(label) / sub).mkdir(parents=True, exist_ok=True)
