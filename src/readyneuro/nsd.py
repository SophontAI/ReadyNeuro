"""NSD: the cococlip metadata (subjects, runs, trial onsets) and the vendor func1mm S3 paths."""

from __future__ import annotations

import io
from functools import lru_cache, partial

import pandas as pd

from readyneuro.io import BENCHMARK_METADATA_ROOT, fetch

_META = f"{BENCHMARK_METADATA_ROOT}/NSD/metadata"
NSD_S3 = "s3://natural-scenes-dataset"
NSD_STORAGE_OPTIONS = {"anon": True}  # public bucket, anonymous reads

# Held-out-subject scheme; subject-disjoint, so a run maps to exactly one split.
EVAL_SPLITS = ("train", "validation", "test")
# NSD's func1mm timeseries are already upsampled to 1.0 s, so the model's target TR is a no-op
# resample for the vendor build (the from-raw build acquires at 1.6 s and does resample).
TARGET_TR = 1.0


#: The benchmark-metadata fetcher (local stage + 429 backoff), bound to NSD's metadata dir.
_fetch = partial(fetch, _META)


@lru_cache
def cococlip_metadata() -> pd.DataFrame:
    """The benchmark suite's ``nsd_cococlip_metadata.csv``: one row per kept trial with
    ``sub, ses, run, trial_id, onset, nsd_id, split, category_id``."""
    return pd.read_csv(io.BytesIO(_fetch("nsd_cococlip_metadata.csv")))


def runs_by_subject(
    meta_df: pd.DataFrame | None = None,
) -> dict[str, list[tuple[int, int, str]]]:
    """``sub -> sorted [(ses, run, split), ...]`` over the eval splits.

    Pure function of the metadata (inject ``meta_df`` to stay offline). Each
    subject's runs all carry the same split (the scheme is subject-disjoint).
    """
    df = meta_df if meta_df is not None else cococlip_metadata()
    df = df[df["split"].isin(EVAL_SPLITS)]
    out: dict[str, set[tuple[int, int, str]]] = {}
    for sub, ses, run, split in (
        df[["sub", "ses", "run", "split"]].drop_duplicates().itertuples(index=False)
    ):
        out.setdefault(sub, set()).add((int(ses), int(run), str(split)))
    return {sub: sorted(v) for sub, v in out.items()}


# --- native func1mm S3 paths (all on one shared 1mm grid) -------------------
def t1_uri(sub: str) -> str:
    """T1 resampled into native func space (the struct input)."""
    return f"{NSD_S3}/nsddata/ppdata/{sub}/func1mm/T1_to_func1mm.nii.gz"


def bold_uri(sub: str, ses: int, run: int) -> str:
    return (
        f"{NSD_S3}/nsddata_timeseries/ppdata/{sub}/func1mm/timeseries/"
        f"timeseries_session{ses:02d}_run{run:02d}.nii.gz"
    )


def mni_transform_uri(sub: str) -> str:
    """NSD's ``func1pt0 -> MNI`` transform volume (per-MNI-voxel func1pt0 coords) -- the vendor
    warp that puts the func1mm BOLD/struct on the canonical MNI 1 mm grid."""
    return f"{NSD_S3}/nsddata/ppdata/{sub}/transforms/func1pt0-to-MNI.nii.gz"


# --- the cococlip index -----------------------------------------------------
NSD_TR = 1.0
NUM_FRAMES = 16  # 16s clip at 1.0s TR


def run_key(ses: int, run: int) -> str:
    return f"{int(ses):02d}_{int(run):02d}"


def run_clips(trials, t_run: int | None) -> list[dict]:
    """``trials`` = iterable of ``(onset, category_id)`` -> ``[{category_id, cond_id,
    start, end}]``. With ``t_run`` set (the producer's actual run length) windows
    past the end are dropped; ``None`` keeps every trial (the finalizer then bounds them)."""
    out: list[dict] = []
    for onset, category_id in trials:
        start = int(onset / NSD_TR)
        end = start + NUM_FRAMES
        if t_run is not None and end > t_run:
            continue
        out.append(
            {
                "category_id": int(category_id),
                "cond_id": int(category_id),
                "start": start,
                "end": end,
            }
        )
    return out


def subject_run_clips(meta_df, sub: str, ses: int, run: int, t_run: int) -> list[dict]:
    """Clips of one ``(sub, ses, run)`` within ``t_run`` (the producer's per-run
    join; ``t_run`` = the actual resampled run length)."""
    df = meta_df[
        (meta_df["sub"] == sub)
        & (meta_df["ses"] == ses)
        & (meta_df["run"] == run)
    ]
    trials = [(r.onset, r.category_id) for r in df.itertuples(index=False)]
    return run_clips(trials, t_run)
