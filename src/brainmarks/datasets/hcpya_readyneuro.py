"""HCP-YA Task21 reader for the packed volumetric eval store."""

from __future__ import annotations

import logging

from brainmarks.datasets._base import VolumeClipDataset, resolve_eval_root
from brainmarks.datasets.registry import register_dataset


log = logging.getLogger(__name__)
NUM_FRAMES = 16
TARGET_TR = 1.0
_DEFAULT_EVAL_ROOT = "s3://<RESTRICTED_ROOT>/hcp-ya-task21-eval"


class HCPYATask21Vol(VolumeClipDataset):
    _name = "HCPYATask21Vol"
    _cache_namespace = "HCP-YA"
    _num_frames = NUM_FRAMES
    _tr = TARGET_TR
    _extra_cols = ("cond",)

    def _extra(self, columns: dict, index: int) -> dict:
        return {"cond": str(columns["cond"][index])}


@register_dataset
def hcpya_task21_readyneuro(
    space: str,
    eval_root=None,
    storage_options=None,
    **kwargs,
):
    root = resolve_eval_root(eval_root, "HCPYA_FMRI_EVAL_ROOT", _DEFAULT_EVAL_ROOT)
    log.info("hcpya_task21_readyneuro space=%s eval_root=%s", space, root)
    return HCPYATask21Vol.load_splits(
        root,
        space,
        storage_options=storage_options,
    )
