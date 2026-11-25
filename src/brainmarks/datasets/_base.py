"""Shared Brainmarks dataset surface over the packed volumetric eval stores."""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from urllib.request import urlopen

import fsspec
import numpy as np
import torch

from brainmarks.datasets import eval_store
from brainmarks.datasets.eval_store import _optional_float
from readyneuro.io import BENCHMARK_METADATA_ROOT


log = logging.getLogger(__name__)
_GEOMETRY_FIELDS = ("idx", "shape", "affine", "gm", "cortical_gm")


def resolve_eval_root(override, env_var: str, default: str) -> str:
    return str(override or os.getenv(env_var) or default)


def targets_uri(targets_root, filename: str, remote_path: str) -> str:
    """Locate a Brainmarks per-subject target map.

    Without ``targets_root`` the map comes from ``$BRAINMARKS_METADATA_URL``, or from the
    raw-GitHub ``datasets/`` tree of the pinned Brainmarks commit, which is the default. Point
    the variable at an unpinned raw HTTP base and a label map can change under you.
    """
    if targets_root:
        return f"{str(targets_root).rstrip('/')}/{filename}"
    return f"{BENCHMARK_METADATA_ROOT}/{remote_path}"


def read_targets(uri: str) -> str:
    if uri.startswith(("http://", "https://")):
        with urlopen(uri) as response:
            return response.read().decode()
    with fsspec.open(uri, "r") as handle:
        return handle.read()


class VolumeClipDataset(torch.utils.data.Dataset):
    """One sample per pre-sliced functional row of a packed eval store."""

    _name = "VolumeClipDataset"
    _valid_spaces = ("native", "mni")
    _cache_namespace = ""
    _num_frames: int
    _tr: float
    _extra_cols: tuple[str, ...] = ()

    def __init__(self, func_ds, anat_ds, space: str = "native") -> None:
        self.func_ds = func_ds.with_format("numpy")
        self.anat_ds = anat_ds.with_format("numpy")
        self.space = space

        anat_subjects = [str(sub) for sub in self.anat_ds["sub"]]
        have_anat = set(anat_subjects)
        self._anat_row = {sub: index for index, sub in enumerate(anat_subjects)}
        self._anat_cache: OrderedDict[str, dict] = OrderedDict()

        columns = {
            name: self.func_ds[name]
            for name in (
                "sub",
                "task",
                "cond_id",
                "start",
                "end",
                *self._extra_cols,
            )
        }
        self.samples: list[dict] = []
        self._geometry_rows: dict[tuple[str, str], int] = {}
        for row_index, raw_sub in enumerate(columns["sub"]):
            sub = str(raw_sub)
            if sub not in have_anat:
                continue
            task = str(columns["task"][row_index])
            self._geometry_rows.setdefault((sub, task), row_index)
            self.samples.append(
                {
                    "_row": row_index,
                    "sub": sub,
                    "task": task,
                    "cond_id": int(columns["cond_id"][row_index]),
                    "start": int(columns["start"][row_index]),
                    "end": int(columns["end"][row_index]),
                    **self._extra(columns, row_index),
                }
            )
        dropped = len(columns["sub"]) - len(self.samples)
        if dropped:
            log.warning("dropped %d clips with no matching anatomy row", dropped)

        self.labels, self.target_ids, self.label_counts = np.unique(
            np.asarray([sample["cond_id"] for sample in self.samples], dtype=np.int64),
            return_inverse=True, return_counts=True,
        )
        self.num_classes = int(len(self.labels))
        self._transforms: list = []
        self._geom_cache: dict[int, dict] = {}

    def _extra(self, columns: dict, index: int) -> dict:
        return {}

    def __len__(self) -> int:
        return len(self.samples)

    def _func_geom_row(self, row_index: int) -> dict:
        if row_index in self._geom_cache:
            return self._geom_cache[row_index]
        decoded = eval_store.decode_func_row(self.func_ds[row_index])
        geometry = {f"bold_{field}": decoded[field] for field in _GEOMETRY_FIELDS}
        if "seg" in decoded:
            geometry["bold_seg"] = decoded["seg"]
        self._geom_cache[row_index] = geometry
        return geometry

    def func_geometries(self) -> list[dict]:
        keys = dict.fromkeys((sample["sub"], sample["task"]) for sample in self.samples)
        return [self._func_geom_row(self._geometry_rows[key]) for key in keys]

    def anat(self, sub: str) -> dict:
        if sub in self._anat_cache:
            self._anat_cache.move_to_end(sub)
            return self._anat_cache[sub]
        decoded = eval_store.decode_anat_row(self.anat_ds[self._anat_row[sub]])
        self._anat_cache[sub] = decoded
        if len(self._anat_cache) > 8:
            self._anat_cache.popitem(last=False)
        return decoded

    def __getitem__(self, index: int) -> dict:
        identity = self.samples[index]
        row = self.func_ds[identity["_row"]]
        bold = eval_store.decode_func_row(row)
        anat = self.anat(identity["sub"])
        sample = {key: value for key, value in identity.items() if key != "_row"}
        sample.update(
            {
                "dataset": self._cache_namespace,
                "space": self.space,
                "n_frames": self._num_frames,
                "tr": self._tr,
                "nyquist_tr": _optional_float(row, "nyquist_tr"),
                "bold_values": np.ascontiguousarray(bold["values"], dtype=np.float16),
                "bold_idx": bold["idx"],
                "bold_shape": bold["shape"],
                "bold_affine": bold["affine"].astype(np.float64),
                "bold_gm": bold["gm"],
                "bold_cortical_gm": bold["cortical_gm"],
                "T1_values": anat["values"],
                "T1_idx": anat["idx"],
                "T1_shape": anat["shape"],
                "T1_affine": anat["affine"],
                "T1_gm": anat["gm"],
                "T1_cortical_gm": anat["cortical_gm"],
                "target": int(self.target_ids[index]),
            }
        )
        if "seg" in bold:
            sample["bold_seg"] = bold["seg"]
        if "seg" in anat:
            sample["T1_seg"] = anat["seg"]
        for transform in self._transforms:
            sample = transform(sample)
        return sample

    def compose(self, transform):
        self._transforms.append(transform)
        return self

    def __repr__(self) -> str:
        return f"{self._name}(space={self.space!r}, n={len(self)}, num_classes={self.num_classes})"

    @classmethod
    def load_splits(
        cls,
        root,
        space: str,
        splits=("train", "validation", "test"),
        *,
        extra=None,
        storage_options=None,
    ) -> dict:
        """Every available split of one store. ``extra`` is the subclass's trailing constructor
        argument (a dx map / target map); omitted for the plain clip datasets.

        ``space`` is validated HERE, the one entry point every plugin routes through, so a typo
        fails before any object-store round trip rather than as a missing-shard error."""
        if space not in cls._valid_spaces:
            raise ValueError(f"space must be one of {cls._valid_spaces}, got {space!r}")
        tail = () if extra is None else (extra,)
        result = {}
        for split in splits:
            try:
                func_ds = eval_store.load_arrow_dir(
                    eval_store.func_dir(root, space, split), storage_options
                )
                anat_ds = eval_store.load_arrow_dir(
                    eval_store.anat_dir(root, space, split), storage_options
                )
            except FileNotFoundError:
                log.warning("%s: no shards under %s; skipping", split, root)
                continue
            result[split] = cls(func_ds, anat_ds, space, *tail)
        if not result:  # a wrong root must not read back as an empty benchmark
            raise FileNotFoundError(f"no shards for any split under {root}/{space}")
        return result


class DiagnosisVolumeDataset(VolumeClipDataset):
    """Subject-level dataset relabeled from a fixed diagnosis vocabulary."""

    _classes: tuple[str, ...]

    def __init__(self, func_ds, anat_ds, space: str, dx_map: dict[str, str]) -> None:
        super().__init__(func_ds, anat_ds, space)
        self.samples = [sample for sample in self.samples if sample["sub"] in dx_map]
        labels = [dx_map[sample["sub"]] for sample in self.samples]
        class_to_id = {label: index for index, label in enumerate(self._classes)}
        unknown = sorted(set(labels) - set(class_to_id))
        if unknown:
            raise ValueError(f"{self._name} target map contains unknown labels: {unknown}")
        self.target_ids = np.asarray([class_to_id[label] for label in labels], dtype=np.int64)
        self.labels = np.asarray(self._classes)
        self.label_counts = np.asarray([labels.count(label) for label in self._classes], dtype=np.int64)
        self.num_classes = len(self._classes)
