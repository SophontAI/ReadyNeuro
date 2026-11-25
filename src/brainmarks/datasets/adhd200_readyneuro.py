"""ADHD-200 diagnosis reader for the packed volumetric eval store.

Unlike PPMI and ABIDE there is no target-map JSON upstream: the diagnosis is the ``dx`` column of
``ADHD200_curated.csv``, so that CSV is the label source when no ``targets_root`` is given. ``sub``
is a zero-padded string id and must never be int-cast.
"""

from __future__ import annotations

import csv
import io
import json

from brainmarks.datasets._base import (
    DiagnosisVolumeDataset,
    read_targets,
    resolve_eval_root,
    targets_uri,
)
from brainmarks.datasets.registry import register_dataset


_DEFAULT_EVAL_ROOT = "s3://<RESTRICTED_ROOT>/adhd200-eval"
_DX_CLASSES = ("ADHD", "Control")
_TARGETS_FILE = "adhd200_target_map_dx.json"
_REMOTE_PATH = "ADHD200/metadata/ADHD200_curated.csv"


def load_dx_map(targets_root=None) -> dict[str, str]:
    text = read_targets(targets_uri(targets_root, _TARGETS_FILE, _REMOTE_PATH))
    if targets_root:
        raw = json.loads(text)
        return {str(sub): str(dx) for sub, dx in raw.items() if dx is not None}
    rows = csv.DictReader(io.StringIO(text))
    return {row["sub"].strip(): row["dx"].strip() for row in rows if row.get("dx", "").strip()}


class Adhd200VolDataset(DiagnosisVolumeDataset):
    _name = "Adhd200VolDataset"
    _cache_namespace = "ADHD200"
    _num_frames = 300
    _tr = 1.0
    _classes = _DX_CLASSES


@register_dataset
def adhd200_dx_readyneuro(
    space: str,
    eval_root=None,
    targets_root=None,
    storage_options=None,
    **kwargs,
):
    return Adhd200VolDataset.load_splits(
        resolve_eval_root(eval_root, "ADHD200_FMRI_EVAL_ROOT", _DEFAULT_EVAL_ROOT),
        space,
        extra=load_dx_map(targets_root),
        storage_options=storage_options,
    )
