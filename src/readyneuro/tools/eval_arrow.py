"""The evaluation deliverable: per-clip Arrow shards -- one func row per clip, one anat row per
subject.

WRITE side only. The read side is deliberately NOT here: it lives with the benchmark datasets
(``brainmarks.datasets.eval_store``), so the evaluation code can be lifted out without this package.
"""

from __future__ import annotations

import itertools
import logging
import os
from pathlib import Path

import datasets as hfds
import numpy as np
import pyarrow as pa

from readyneuro.io import is_s3, path_exists, s3_filesystem
from readyneuro.tools.shards import done_uri, write_done

log = logging.getLogger("readyneuro.datasets.eval_arrow")


# --- schema -----------------------------------------------------------------
def func_features() -> hfds.Features:
    return hfds.Features(
        {
            "sub": hfds.Value("string"),
            "task": hfds.Value("string"),  # HCP: run's task; NSD: run-key "ses_run"
            "ses": hfds.Value("int32"),  # per-run identity (NSD); -1 when unused (HCP)
            "run": hfds.Value("int32"),
            "cond": hfds.Value("string"),  # HCP trial_type; "" for NSD
            "cond_id": hfds.Value("int32"),  # class id (HCP 0..20 / NSD category_id)
            "start": hfds.Value("int32"),  # clip onset frame in the run
            "end": hfds.Value("int32"),  # start + NUM_FRAMES
            "tr": hfds.Value("float32"),
            # The run's limiting TR period (`codec.pack_bold`), precomputed at build:
            # max(acquisition, harmonization band, stored tr). NaN = UNKNOWN -- Arrow float32 has no
            # null, so a `None` source is written as NaN and a reader falls back to `tr`. This is the
            # model's nyquist conditioning (`samplers._nyquist_tr`), read directly, not re-maxed.
            "nyquist_tr": hfds.Value("float32"),
            "n_frames": hfds.Value("int32"),  # clip length (== t)
            "m": hfds.Value("int32"),
            "t": hfds.Value("int32"),
            "values": hfds.LargeList(hfds.Value("float16")),  # (m * t,) flattened
            "idx": hfds.Sequence(hfds.Value("int32")),  # (m,)
            "gm": hfds.Sequence(hfds.Value("uint8")),  # (m,) 1 = gray matter
            "cortical_gm": hfds.Sequence(hfds.Value("uint8")),  # (m,) 1 = cortical GM
            "seg": hfds.Sequence(hfds.Value("int16")),  # (m,) SynthSeg aseg label (0 = unlabeled)
            "affine": hfds.Sequence(hfds.Value("float32")),  # (16,)
            "shape": hfds.Sequence(hfds.Value("int32")),  # (3,)
        }
    )


def anat_features() -> hfds.Features:
    return hfds.Features(
        {
            "sub": hfds.Value("string"),
            "m": hfds.Value("int32"),
            "values": hfds.LargeList(hfds.Value("float16")),
            "idx": hfds.Sequence(hfds.Value("int32")),
            "gm": hfds.Sequence(hfds.Value("uint8")),
            "cortical_gm": hfds.Sequence(hfds.Value("uint8")),
            "seg": hfds.Sequence(hfds.Value("int16")),  # (m,) SynthSeg aseg label (0 = unlabeled)
            "affine": hfds.Sequence(hfds.Value("float32")),
            "shape": hfds.Sequence(hfds.Value("int32")),
        }
    )


# --- records (from codec.pack, quant="f16") -> rows -------------------------
def func_row(
    rec: dict,
    *,
    sub: str,
    task: str = "",
    ses: int = -1,
    run: int = -1,
    cond: str = "",
    cond_id: int = -1,
    start: int = -1,
    end: int = -1,
    tr: float,
    nyquist_tr: float | None = None,
    n_frames: int,
) -> dict:
    values = np.asarray(rec["values"], dtype=np.float16)  # (M, T) -- one clip's window
    m, t = values.shape
    return {
        "sub": sub,
        "task": task,
        "ses": int(ses),
        "run": int(run),
        "cond": cond,
        "cond_id": int(cond_id),
        "start": int(start),
        "end": int(end),
        "tr": float(tr),
        # None -> NaN: Arrow float32 has no null; a reader treats NaN as UNKNOWN and falls back to
        # `tr`. Mirrors `codec.pack_bold`'s optional-TR rule on the eval side.
        "nyquist_tr": float(nyquist_tr) if nyquist_tr is not None else float("nan"),
        "n_frames": int(n_frames),
        "m": int(m),
        "t": int(t),
        "values": values.reshape(-1),
        "idx": rec["idx"],
        "gm": rec["gm"],
        "cortical_gm": rec["cortical_gm"],
        "seg": rec["seg"],
        "affine": np.asarray(rec["affine"], np.float32).reshape(-1),
        "shape": list(rec["shape"]),
    }


def run_key(ses: int, run: int) -> str:
    """The stringified per-run identity written into the ``task`` column: ``"{ses:02d}_{run:02d}"``.

    ONE definition, byte-identical to `readyneuro.nsd.run_key`, which the NSD reader and the curated
    cococlip reference both key on: a store whose ``task`` disagrees with the metadata's run key by
    so much as a zero-pad is a store no audit can join back to its source."""
    return f"{int(ses):02d}_{int(run):02d}"


def _warn_unidentified(rows: list[dict], sub: str, where: str) -> None:
    """Loud guard against shipping ANONYMOUS clip rows -- ``task == ""`` AND ``ses == run == -1``.

    A warning and not an assert: ``ses``/``run`` legitimately stay -1 for cores whose runs have no
    numeric session/run number, so -1 alone is not an error; only the total absence of identity is
    (see `finalize_eval`'s module docstring for why)."""
    n = sum(1 for r in rows if not r["task"] and r["ses"] < 0 and r["run"] < 0)
    if n:
        log.warning(
            "%s: %d/%d clip rows carry NO run identity (task='' ses=-1 run=-1) -- %s will be "
            "unauditable (no row can be joined back to the run it came from). A producing path is "
            "not stamping task/ses/run; see eval_arrow._warn_unidentified.",
            sub, n, len(rows), where or "this shard",
        )


def _func_clip_rows(clip_recs: list[dict], sub: str, *, where: str = "") -> list[dict]:
    """A list of clip-rec dicts -> func rows. Each rec carries its packed ``rec`` (from
    ``codec.pack(quant="f16")`` on the sliced window) + the clip's identity/label; missing keys
    default (HCP omits ``ses``/``run``, NSD ``cond``)."""
    rows = [
        func_row(
            r["rec"],
            sub=sub,
            task=r.get("task", ""),
            ses=r.get("ses", -1),
            run=r.get("run", -1),
            cond=r.get("cond", ""),
            cond_id=r.get("cond_id", -1),
            start=r.get("start", -1),
            end=r.get("end", -1),
            tr=r["tr"],
            # Optional; absent in a rec predating the field -> None -> NaN (unknown). `func_row`
            # maps None -> NaN, so `.get` (not `r[...]`) keeps such recs writable.
            nyquist_tr=r.get("nyquist_tr"),
            n_frames=r["n_frames"],
        )
        for r in clip_recs
    ]
    _warn_unidentified(rows, sub, where)
    return rows


def anat_row(rec: dict, *, sub: str) -> dict:
    values = np.asarray(rec["values"], dtype=np.float16)  # (M,)
    return {
        "sub": sub,
        "m": int(values.shape[0]),
        "values": values,
        "idx": rec["idx"],
        "gm": rec["gm"],
        "cortical_gm": rec["cortical_gm"],
        "seg": rec["seg"],
        "affine": np.asarray(rec["affine"], np.float32).reshape(-1),
        "shape": list(rec["shape"]),
    }


# --- io ---------------------------------------------------------------------
def _write_row_batches(
    row_batches, features: hfds.Features, uri: str, storage_options=None
) -> bool:
    """Stream per-run row lists into ONE IPC file, a record batch at a time, so peak
    memory is a single run's clips -- not the whole subject's. Materializing
    ``recs -> rows -> table -> ipc bytes`` at once instead OOMs a 386 GB node on an uncapped
    NSD subject, whose clips reach hundreds of GB. The record-batch boundaries are transparent
    to every reader, which ``read_all()``s the file.

    Returns True iff a row was written. When every batch is empty it writes NO file at all,
    leaving the "no rows -> ship nothing" decision to the caller."""
    schema = hfds.Dataset.from_dict(
        {k: [] for k in features}, features=features
    ).data.table.schema
    it = iter(row_batches)
    first = None
    for rows in it:
        if rows:
            first = rows
            break
    if first is None:
        return False

    def _stream(sink) -> None:
        with pa.ipc.new_file(sink, schema) as writer:
            for rows in itertools.chain([first], it):
                if not rows:
                    continue
                # `.cast(schema)` pins each batch to the file's exact schema; `from_list`
                # re-derives that same schema from `features`, so this is a no-op guard.
                table = hfds.Dataset.from_list(
                    rows, features=features
                ).data.table.cast(schema)
                writer.write_table(table)

    if is_s3(uri):
        # S3/R2 has no atomic rename, so this streams into the final key; the caller writes
        # ``.done`` only after a clean return, so a killed write leaves a partial object no
        # resume will trust.
        with s3_filesystem(storage_options).open(uri, "wb") as f:
            _stream(f)
    else:
        p = Path(uri)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f".{p.name}.tmp.{os.getpid()}")
        try:
            with open(tmp, "wb") as f:
                _stream(f)
            os.replace(tmp, p)  # atomic on the same filesystem
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
    return True


def func_dir(root, space, split) -> str:
    """Directory of a split's ``func`` shards (globbed by the eval loaders)."""
    return f"{root}/func.{space}.arrow/{split}"


def anat_dir(root, space, split) -> str:
    return f"{root}/anat.{space}.arrow/{split}"


def func_uri(root, space, split, sub) -> str:
    return f"{func_dir(root, space, split)}/{sub}.arrow"


def anat_uri(root, space, split, sub) -> str:
    return f"{anat_dir(root, space, split)}/{sub}.arrow"


# --- resume: completion sentinels (skip done work before any read) ----------
# Like the pretrain tars, eval Arrow streams into the final key, so resume trusts a ``{uri}.done``
# sentinel written only after a clean write, never bare existence -- a killed write leaves a
# partial shard with no sentinel.


def subject_eval_exists(root, space, split, sub, *, storage_options=None) -> bool:
    """A subject's func AND anat shards both fully written. BOTH, because `write_subject_streamed`
    writes them separately: a crash in between leaves a func sentinel with no anat, which a
    single-uri check would read as shipped."""
    return path_exists(done_uri(func_uri(root, space, split, sub)), storage_options) and path_exists(
        done_uri(anat_uri(root, space, split, sub)), storage_options)


def write_subject_streamed(
    root,
    space,
    split,
    sub,
    func_clip_rec_batches,
    anat_rec: dict,
    *,
    storage_options=None,
) -> list[str]:
    """Write one subject's func (a row per clip) + anat (one row) shards.

    ``func_clip_rec_batches`` is an ITERABLE of per-run clip-rec lists (each the ``codec.slice_clips``
    output for one run), consumed lazily so the whole subject's clips never coexist in memory -- an
    uncapped NSD subject's clips reach hundreds of GB, which OOMs a 386 GB node.

    Returns the produced uris, or ``[]`` when the subject yields no rows -- then NO func shard, anat,
    or ``.done`` is written, the "ship nothing" signal `finalize_eval` turns into a dropped subject.

    IDENTITY: this writer packs a whole subject into ONE shard and cannot know which run a batch came
    from, so the per-run ``task``/``ses``/``run`` must arrive ON the recs, stamped by the caller."""
    f_uri, a_uri = func_uri(root, space, split, sub), anat_uri(root, space, split, sub)
    row_batches = (
        _func_clip_rows(recs, sub, where=f_uri) if recs else []
        for recs in func_clip_rec_batches
    )
    if not _write_row_batches(row_batches, func_features(), f_uri, storage_options):
        return []
    write_done(f_uri, storage_options=storage_options)
    _write_row_batches(iter([[anat_row(anat_rec, sub=sub)]]), anat_features(), a_uri, storage_options)
    write_done(a_uri, storage_options=storage_options)
    return [f_uri, a_uri]
