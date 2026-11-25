"""PPMI diagnosis reader for the packed volumetric eval store.

One 300-frame resting-state row per subject with ``cond_id == -1``, so the label is joined from a
per-subject Brainmarks target map and encoded against a FIXED class order -- never a per-split
``np.unique``, which would silently remap if a split happened to miss a class.
"""

from __future__ import annotations

import json

from brainmarks.datasets._base import (
    DiagnosisVolumeDataset,
    read_targets,
    resolve_eval_root,
    targets_uri,
)
from brainmarks.datasets.registry import register_dataset


_DEFAULT_EVAL_ROOT = "s3://<RESTRICTED_ROOT>/ppmi-eval"
_DX_CLASSES = ("PD", "Prodromal")
_TARGETS_FILE = "ppmi_target_map_dx.json"
_REMOTE_PATH = f"PPMI/metadata/targets/{_TARGETS_FILE}"


def load_dx_map(targets_root=None) -> dict[str, str]:
    raw = json.loads(read_targets(targets_uri(targets_root, _TARGETS_FILE, _REMOTE_PATH)))
    return {str(sub): str(dx) for sub, dx in raw.items() if dx is not None}


class PPMIVolDataset(DiagnosisVolumeDataset):
    _name = "PPMIVolDataset"
    _cache_namespace = "PPMI"
    _num_frames = 300
    _tr = 1.0
    _classes = _DX_CLASSES


@register_dataset
def ppmi_dx_readyneuro(
    space: str,
    eval_root=None,
    targets_root=None,
    storage_options=None,
    **kwargs,
):
    return PPMIVolDataset.load_splits(
        resolve_eval_root(eval_root, "PPMI_FMRI_EVAL_ROOT", _DEFAULT_EVAL_ROOT),
        space,
        extra=load_dx_map(targets_root),
        storage_options=storage_options,
    )
