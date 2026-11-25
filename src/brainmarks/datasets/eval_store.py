"""Read the pipeline's packed Arrow IPC evaluation stores.

Layout under an eval root, one directory per space (``native`` | ``mni``) and split
(``train`` | ``validation`` | ``test``)::

    <root>/func.<space>.arrow/<split>/*.arrow
    <root>/anat.<space>.arrow/<split>/*.arrow

A **functional** row is one pre-sliced clip:

===============  =========================================================================
``sub``          subject id (string), the join key to the anat row
``task``         run key (``"REST"`` for the resting-state benchmarks)
``cond_id``      integer class label, ``-1`` when the clip carries no condition
``start``/``end``  frame bounds of the clip within its resampled run
``m``            number of stored voxels
``t``            number of frames
``values``       flat float16, reshaped to ``[m, t]``
``idx``          int32 flat C-order voxel indices into ``shape`` (== ``flatnonzero(mask)``)
``gm``           uint8 per stored voxel: grey matter
``cortical_gm``  uint8 per stored voxel: cortical grey matter (a subset of ``gm``)
``affine``       flat 16 floats, reshaped to the 4x4 voxel-to-world affine
``shape``        the dense grid shape ``idx`` indexes into
``seg``          optional int16 aseg labels per stored voxel
``nyquist_tr``   optional float: the effective sampling period after any resampling
===============  =========================================================================

``nyquist_tr`` is written as NaN when unknown (Arrow float32 has no null) and surfaces as
``None``. Per-dataset extras ride alongside: ``cond`` (HCP-YA trial type),
``ses`` and ``run`` (NSD).

An **anatomy** row is the same schema minus ``t``: one 1 mm T1 layer per subject, with
``values`` a flat ``[m]`` vector.
"""

from __future__ import annotations

import itertools
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import datasets as hfds
import numpy as np
import pyarrow as pa
import s3fs


def func_dir(root, space: str, split: str) -> str:
    return f"{str(root).rstrip('/')}/func.{space}.arrow/{split}"


def anat_dir(root, space: str, split: str) -> str:
    return f"{str(root).rstrip('/')}/anat.{space}.arrow/{split}"


def decode_func_row(row: dict) -> dict:
    """Decode one functional row into sparse geometry plus ``[m, t]`` values."""
    m, frames = int(row["m"]), int(row["t"])
    decoded = {
        "values": np.asarray(row["values"], np.float16).reshape(m, frames),
        "idx": np.asarray(row["idx"], np.int32),
        "gm": np.asarray(row["gm"], np.uint8),
        "cortical_gm": np.asarray(row["cortical_gm"], np.uint8),
        "affine": np.asarray(row["affine"], np.float32).reshape(4, 4),
        "shape": tuple(int(size) for size in row["shape"]),
    }
    if "seg" in row:
        decoded["seg"] = np.asarray(row["seg"], np.int16)
    return decoded


def decode_anat_row(row: dict) -> dict:
    """Decode one per-subject anatomy row."""
    decoded = {
        "values": np.asarray(row["values"], np.float16).reshape(int(row["m"])),
        "idx": np.asarray(row["idx"], np.int32),
        "gm": np.asarray(row["gm"], np.uint8),
        "cortical_gm": np.asarray(row["cortical_gm"], np.uint8),
        "affine": np.asarray(row["affine"], np.float32).reshape(4, 4),
        "shape": tuple(int(size) for size in row["shape"]),
    }
    if "seg" in row:
        decoded["seg"] = np.asarray(row["seg"], np.int16)
    return decoded


def _optional_float(row: dict, key: str) -> float | None:
    """An optional store float: NaN (Arrow float32 has no null) reads back as ``None``. A NaN riding
    into the sampler's ``max`` over periods NaNs the conditioning for a whole batch, silently."""
    value = row.get(key)
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _is_s3(path) -> bool:
    return str(path).startswith("s3://")


def _filesystem(storage_options=None) -> s3fs.S3FileSystem:
    return s3fs.S3FileSystem(**(storage_options or {}))


def list_arrow_shards(directory, storage_options=None) -> list[str]:
    """Return sorted ``*.arrow`` shards or raise when a split is absent."""
    directory = str(directory)
    if _is_s3(directory):
        fs = _filesystem(storage_options)
        hits = sorted(fs.glob(f"{directory}/*.arrow")) if fs.exists(directory) else []
        shards = [hit if _is_s3(hit) else f"s3://{hit}" for hit in hits]
    else:
        path = Path(directory)
        shards = sorted(str(item) for item in path.glob("*.arrow")) if path.exists() else []
    if not shards:
        raise FileNotFoundError(f"no .arrow shards under {directory}")
    return shards


def _large_list_schema(schema: pa.Schema) -> pa.Schema:
    # Large eval concatenations can exceed the signed int32 offset limit.
    return pa.schema(
        [
            pa.field(
                field.name,
                pa.large_list(field.type.value_type) if pa.types.is_list(field.type) else field.type,
            )
            for field in schema
        ]
    )


def _read_workers(n_shards: int) -> int:
    requested = int(os.environ.get("FMRI_ARROW_READ_WORKERS", "16"))
    return max(1, min(requested, max(1, n_shards)))


def _ranged_chunks(
    fs,
    path: str,
    total: int,
    *,
    chunk: int = 256 * 2**20,
    retries: int = 6,
):
    offset = 0
    while offset < total:
        end = min(offset + chunk, total)
        for attempt in range(retries):
            try:
                data = fs.cat_file(path, start=offset, end=end)
                break
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(min(2**attempt, 30))
        yield data
        offset = end


def _read_shard_bytes(uri: str, storage_options=None) -> bytes:
    if not _is_s3(uri):
        return Path(uri).read_bytes()
    fs = _filesystem(storage_options)
    bare_path = uri.removeprefix("s3://")
    total = int(fs.info(uri)["size"])
    data = bytearray()
    for chunk in _ranged_chunks(fs, bare_path, total):
        data += chunk
    if len(data) != total:
        raise OSError(f"{uri}: read {len(data)} of {total} bytes")
    return bytes(data)


def _iter_shard_bytes(shards, storage_options=None):
    workers = _read_workers(len(shards))
    if workers == 1 or len(shards) == 1:
        for shard in shards:
            yield _read_shard_bytes(shard, storage_options)
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        iterator = iter(shards)
        window = deque(
            executor.submit(_read_shard_bytes, shard, storage_options)
            for shard in itertools.islice(iterator, workers)
        )
        while window:
            yield window.popleft().result()
            if (next_shard := next(iterator, None)) is not None:
                window.append(executor.submit(_read_shard_bytes, next_shard, storage_options))


def _empty_dataset(schema: pa.Schema) -> hfds.Dataset:
    features = hfds.Features.from_arrow_schema(schema)
    return hfds.Dataset.from_dict({name: [] for name in features}, features=features)


def _schema_for_shard(uri: str, storage_options=None) -> pa.Schema:
    data = _read_shard_bytes(uri, storage_options)
    return pa.ipc.open_file(pa.BufferReader(data)).schema


def _load_in_memory(directory, storage_options=None) -> hfds.Dataset:
    shards = list_arrow_shards(directory, storage_options)
    schema = None
    tables = []
    for data in _iter_shard_bytes(shards, storage_options):
        reader = pa.ipc.open_file(pa.BufferReader(data))
        if reader.num_record_batches == 0:
            continue
        if schema is None:
            schema = _large_list_schema(reader.schema)
        tables.append(reader.read_all().cast(schema))
    if schema is None:
        return _empty_dataset(_large_list_schema(_schema_for_shard(shards[0], storage_options)))
    features = hfds.Features.from_arrow_schema(schema)
    table = pa.concat_tables(tables)
    return hfds.Dataset(table, info=hfds.DatasetInfo(features=features))


def load_arrow_dir(directory, storage_options=None) -> hfds.Dataset:
    """Load one split."""
    return _load_in_memory(directory, storage_options)
