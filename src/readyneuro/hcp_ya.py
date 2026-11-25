"""HCP-YA: the shipped-URI resolver, the split / task21 logic, and the task21 event onsets.

One module because the three are one dataset's facts; the build layer (``builder/discover/hcp_ya*``)
imports all of them.
"""

from __future__ import annotations

import io
import json
import re
from functools import lru_cache, partial

import pandas as pd
import s3fs

from readyneuro.io import BENCHMARK_METADATA_ROOT, fetch

_META = f"{BENCHMARK_METADATA_ROOT}/HCP-YA/metadata"
HCP_S3 = "s3://hcp-openaccess/HCP_1200"

# 3T runs only: rest REST1/2 LR/RL + the 7 tasks LR/RL. 7T runs are AP/PA and so
# never match the (LR|RL) anchor.
RUN_3T = re.compile(r"^(rfMRI_REST[12]|tfMRI_[A-Z]+)_(LR|RL)$")

#: The benchmark-metadata fetcher (local stage + 429 backoff), bound to HCP-YA's metadata dir.
_fetch = partial(fetch, _META)


@lru_cache
def subject_batch_splits() -> dict[str, list[str]]:
    """Raw `hcpya_subject_batch_splits.json`: ``{"batch-NN": [sub, ...]}``."""
    return json.loads(_fetch("hcpya_subject_batch_splits.json"))


@lru_cache
def subject_to_batch() -> dict[str, int]:
    """sub -> batch index, from Brainmarks `hcpya_subject_batch_splits.json`."""
    return {
        s: int(b.split("-")[1])
        for b, subs in subject_batch_splits().items()
        for s in subs
    }


@lru_cache
def pheno_targets() -> pd.DataFrame:
    """Brainmarks `hcpya_pheno_targets.csv` (demographics + behavioral targets)."""
    return pd.read_csv(io.BytesIO(_fetch("hcpya_pheno_targets.csv")))


@lru_cache
def metadata() -> pd.DataFrame:
    """Brainmarks `hcpya_metadata.parquet` (per-run events + dims; the task21
    clip index joins onsets from here)."""
    return pd.read_parquet(io.BytesIO(_fetch("hcpya_metadata.parquet")))


def list_runs(sub: str, *, storage_options: dict | None = None) -> list[str]:
    """3T run dirs under the subject's MNINonLinear/Results with a volumetric
    ``{run}/{run}.nii.gz`` (the surface-only dtseries is ignored). `storage_options`
    selects the HCP credentials (e.g. ``{"profile": "hcp"}``)."""
    fs = s3fs.S3FileSystem(**(storage_options or {}))
    base = f"{HCP_S3.removeprefix('s3://')}/{sub}/MNINonLinear/Results"
    runs = []
    for p in fs.ls(base, detail=False):
        run = p.rstrip("/").rsplit("/", 1)[-1]
        if RUN_3T.match(run) and fs.exists(f"{base}/{run}/{run}.nii.gz"):
            runs.append(run)
    return sorted(runs)


def bold_uri(sub: str, run: str) -> str:
    return f"{HCP_S3}/{sub}/MNINonLinear/Results/{run}/{run}.nii.gz"


def t1_uri(sub: str, space: str) -> str:
    # "native" = the subject's AC-PC-aligned space (HCP names that file `T1w_acpc_dc`)
    name = "T1w/T1w_acpc_dc_restore" if space == "native" else "MNINonLinear/T1w_restore"
    return f"{HCP_S3}/{sub}/{name}.nii.gz"


def warp_uri(sub: str, kind: str) -> str:
    # HCP MNINonLinear/xfms warp names:
    #   "standard2acpc_dc": resample MNI into acpc (the acpc BOLD warp)
    #   "acpc_dc2standard": resample acpc into MNI (the parc->MNI label warp)
    return f"{HCP_S3}/{sub}/MNINonLinear/xfms/{kind}.nii.gz"


# --- splits + the task21 gate -----------------------------------------------
# Eval splits mirror the benchmark suite's subject batches exactly; pretrain keeps the wide train.
PRETRAIN_SPLITS = {"train": set(range(16)), "val": {16, 17}, "test": {18, 19}}
EVAL_SPLITS = {"train": set(range(8)), "validation": {16, 17}, "test": {18, 19}}
TASK21_TASKS = frozenset({"EMOTION", "LANGUAGE", "MOTOR", "RELATIONAL", "SOCIAL", "WM"})


def _split_of(batch: int, split_map: dict[str, set[int]]) -> str | None:
    return next((name for name, b in split_map.items() if batch in b), None)


def assign_splits(sub: str) -> tuple[str | None, str | None]:
    batch = subject_to_batch()[sub]
    return _split_of(batch, PRETRAIN_SPLITS), _split_of(batch, EVAL_SPLITS)


def _task_of(run: str) -> str:
    return run.split("_")[1]  # "tfMRI_EMOTION_LR" -> "EMOTION"


def task21_lr_runs(runs: list[str]) -> list[str]:
    return [
        r
        for r in runs
        if r.startswith("tfMRI_") and r.endswith("_LR") and _task_of(r) in TASK21_TASKS
    ]


def has_all_task21(runs: list[str]) -> bool:
    return {_task_of(r) for r in task21_lr_runs(runs)} == TASK21_TASKS


def discover_subjects(n_subjects: int | None, only: list[str] | None) -> list[str]:
    subs = sorted(subject_to_batch())
    if only:
        subs = [s for s in subs if s in set(only)]
    return subs[:n_subjects] if n_subjects else subs


# --- the task21 clip index --------------------------------------------------
NUM_FRAMES = 16
TARGET_TR = 1.0
_NATIVE_TR_3T = 0.72

HCP_TASK21_TRIAL_TYPES = {
    "fear": 0,
    "neut": 1,
    "math": 2,
    "story": 3,
    "lf": 4,
    "lh": 5,
    "rf": 6,
    "rh": 7,
    "t": 8,
    "match": 9,
    "relation": 10,
    "mental": 11,
    "rnd": 12,
    "0bk_body": 13,
    "2bk_body": 14,
    "0bk_faces": 15,
    "2bk_faces": 16,
    "0bk_places": 17,
    "2bk_places": 18,
    "0bk_tools": 19,
    "2bk_tools": 20,
}


def _ref_T_run(n_frames_native: int) -> int:
    """The post-resample run length the benchmark suite's clip index assumes. Polyphase rounds up
    by 1; we deliberately take the SMALLER value, so the boundary skip matches the reference set and
    never indexes past it (the producer's resampled run is at least this long)."""
    return int(_NATIVE_TR_3T * n_frames_native / TARGET_TR)


def run_clips(events, t_run: int) -> list[dict]:
    """One run's events -> ``[{cond, cond_id, start, end}]``, windows within
    ``t_run``. Pure; both callers feed ``t_run = _ref_T_run(n_frames_native)``."""
    out: list[dict] = []
    for ev in events:
        cond = ev["trial_type"]
        if cond not in HCP_TASK21_TRIAL_TYPES:
            continue
        start = int(ev["onset"] / TARGET_TR)
        end = start + NUM_FRAMES
        if end > t_run:
            continue
        out.append(
            {
                "cond": cond,
                "cond_id": HCP_TASK21_TRIAL_TYPES[cond],
                "start": start,
                "end": end,
            }
        )
    return out


def subject_run_clips(sub: str, task: str, *, meta_df) -> list[dict]:
    """Clips of one subject's 3T/LR run for ``task`` (the producer's per-run join).
    Empty if the metadata has no such run."""
    rows = meta_df[
        (meta_df["sub"] == sub)
        & (meta_df["task"] == task)
        & (meta_df["mag"] == "3T")
        & (meta_df["dir"] == "LR")
    ]
    if rows.empty:
        return []
    row = rows.iloc[0]
    return run_clips(row["events"], _ref_T_run(int(row["n_frames"])))
