"""AABC age-bracket and sex readers for the packed volumetric eval store.

One resting-state row per subject: the store already holds the 500-raw-frame window, which is 400
frames after the 0.8 -> 1.0 s resample. The label is joined from a Brainmarks per-subject target map.
"""

from __future__ import annotations

import json

import numpy as np

from brainmarks.datasets._base import (
    VolumeClipDataset, read_targets, resolve_eval_root, targets_uri)
from brainmarks.datasets.registry import register_dataset


_DEFAULT_EVAL_ROOT = "s3://<RESTRICTED_ROOT>/aabc-eval"
_REST_RESAMPLED_FRAMES = 400
TARGET_MAP_FILES = {
    "age": "aabc_target_map_age_open.json",
    "sex": "aabc_target_map_sex.json",
}


def load_target_map(target: str, targets_root=None) -> dict[str, int]:
    if target not in TARGET_MAP_FILES:
        raise ValueError(f"unknown AABC target {target!r}; known: {sorted(TARGET_MAP_FILES)}")
    name = TARGET_MAP_FILES[target]
    raw = json.loads(read_targets(targets_uri(targets_root, name, f"AABC/metadata/targets/{name}")))
    return {str(sub): int(value) for sub, value in raw.items() if value is not None}


class AABCVolDataset(VolumeClipDataset):
    _name = "AABCVolDataset"
    _cache_namespace = "AABC"
    _num_frames = _REST_RESAMPLED_FRAMES
    _tr = 1.0

    def __init__(self, func_ds, anat_ds, space: str, target_map: dict[str, int]):
        super().__init__(func_ds, anat_ds, space)
        self.samples = [sample for sample in self.samples if sample["sub"] in target_map]
        targets = np.asarray([target_map[sample["sub"]] for sample in self.samples], dtype=np.int64)
        labels = np.asarray(sorted(set(target_map.values())), dtype=np.int64)
        label_to_id = {int(label): index for index, label in enumerate(labels)}
        self._labels_raw = targets
        self.labels = labels
        self.target_ids = np.asarray([label_to_id[int(value)] for value in targets], dtype=np.int64)
        self.label_counts = np.bincount(self.target_ids, minlength=len(self.labels))
        self.num_classes = int(len(self.labels))


def _load_aabc(
    space: str,
    target: str,
    *,
    eval_root=None,
    targets_root=None,
    storage_options=None,
) -> dict:
    return AABCVolDataset.load_splits(
        resolve_eval_root(eval_root, "AABC_FMRI_EVAL_ROOT", _DEFAULT_EVAL_ROOT),
        space,
        extra=load_target_map(target, targets_root),
        storage_options=storage_options,
    )


@register_dataset
def aabc_age_readyneuro(
    space: str,
    eval_root=None,
    targets_root=None,
    storage_options=None,
    **kwargs,
):
    return _load_aabc(
        space,
        "age",
        eval_root=eval_root,
        targets_root=targets_root,
        storage_options=storage_options,
    )


@register_dataset
def aabc_sex_readyneuro(
    space: str,
    eval_root=None,
    targets_root=None,
    storage_options=None,
    **kwargs,
):
    return _load_aabc(
        space,
        "sex",
        eval_root=eval_root,
        targets_root=targets_root,
        storage_options=storage_options,
    )
