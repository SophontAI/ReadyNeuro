"""NSD COCO-category clip reader for the packed volumetric eval store.

The native and MNI arms live in separate roots here (unlike the other benchmarks, whose two spaces
share one root), so the native arm must be given an explicit ``eval_root``.
"""

from __future__ import annotations

import logging

from brainmarks.datasets._base import VolumeClipDataset, resolve_eval_root
from brainmarks.datasets.registry import register_dataset


log = logging.getLogger(__name__)
NUM_FRAMES = 16
NSD_TR = 1.0
_DEFAULT_EVAL_ROOT = "s3://<RESTRICTED_ROOT>/nsd-cococlip-eval-mni"


class NSDCocoClipVol(VolumeClipDataset):
    _name = "NSDCocoClipVol"
    _cache_namespace = "NSD"
    _num_frames = NUM_FRAMES
    _tr = NSD_TR
    _extra_cols = ("ses", "run")

    def _extra(self, columns: dict, index: int) -> dict:
        return {
            "ses": int(columns["ses"][index]),
            "run": int(columns["run"][index]),
            "category_id": int(columns["cond_id"][index]),
        }


@register_dataset
def nsd_cococlip_readyneuro(
    space: str,
    eval_root=None,
    storage_options=None,
    **kwargs,
):
    root = resolve_eval_root(eval_root, "NSD_FMRI_EVAL_ROOT", _DEFAULT_EVAL_ROOT)
    log.info("nsd_cococlip_readyneuro space=%s eval_root=%s", space, root)
    return NSDCocoClipVol.load_splits(
        root,
        space,
        storage_options=storage_options,
    )
