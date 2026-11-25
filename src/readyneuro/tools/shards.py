"""The pretraining deliverable: one subject's WebDataset tar, plus the ``.done`` resume sentinel.
"""

from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path

from readyneuro.tools import codec
from readyneuro.io import is_s3, path_exists, s3_filesystem


# --- uri helpers ------------------------------------------------------------
def tar_uri(root, split, space, sub) -> str:
    return f"{root}/{split}/{space}/{sub}.tar"


def done_uri(uri) -> str:
    """Completion sentinel for a streamed object. The tar/Arrow writers stream into the
    final key with no atomic rename on S3/R2, so a hard kill leaves a *partial* object at
    the real uri; resume must trust this sentinel (written only after a full, clean close),
    not bare existence."""
    return f"{uri}.done"


def write_done(uri, *, storage_options=None) -> None:
    d = done_uri(uri)
    if is_s3(d):
        with s3_filesystem(storage_options).open(d, "wb") as f:
            f.write(b"")
    else:
        Path(d).write_bytes(b"")


def clear_done(uri, *, storage_options=None) -> None:
    """Drop a stale sentinel before a (re)write, so a crash mid-rewrite isn't seen as done."""
    d = done_uri(uri)
    if not path_exists(d, storage_options):
        return
    s3_filesystem(storage_options).rm(d) if is_s3(d) else Path(d).unlink(missing_ok=True)


# --- self-contained tar: bundle-first + per-run BOLD blocks ------------------------
class BundleTar:
    """One SELF-CONTAINED ``{root}/{split}/{space}/{sub}.tar``: the subject BUNDLE as the FIRST
    sample (``{sub}`` key -- BOLD geometry + tissue rows, the packed 1 mm T1 layer, meta), then one
    BOLD block per run (``{sub}__{run}`` key -- values + scale + temporal meta, no geometry). A
    loader re-attaches the bundle to each block, so the tar needs no external cache. Bundle-first
    (``.`` sorts before ``_``) so a streaming loader sees it ahead of the runs. Local writes are
    atomic (temp + rename); ``.done`` only after a clean close.

    WebDataset splits a member name on the FIRST ``.`` to get its key -- which is why a run member
    joins subject and run with a DOUBLE UNDERSCORE: a dot there would split the key into a field.
    """

    def __init__(self, root, split, space, sub, bundle, *, storage_options=None, dataset=None):
        self.uri = tar_uri(root, split, space, sub)
        self.sub, self.space, self.split = sub, space, split
        self._so = storage_options
        # Two distinct identities: ``work_key`` is the tar key (per-session for AABC), ``subject``
        # the id a leave-one-subject-out probe groups on. The core's meta is spread LAST on purpose,
        # so it can override ``subject`` with the bare subject and keep a person's visits together;
        # reordering would silently regroup folds by session and leak visits across the fold.
        meta = {"subject": sub, "work_key": sub, "space": space, "split": split, **bundle["meta"]}
        if dataset is not None:
            meta["dataset"] = dataset
        self._bundle = {**codec.requantize_bundle_t1(bundle, "int8"), "meta": meta}
        self._fh = self._tar = self._tmp = None
        self._n_runs = 0

    def __enter__(self) -> "BundleTar":
        clear_done(self.uri, storage_options=self._so)
        if is_s3(self.uri):
            self._fh = s3_filesystem(self._so).open(self.uri, "wb")
        else:
            Path(self.uri).parent.mkdir(parents=True, exist_ok=True)
            self._tmp = f"{self.uri}.tmp"
            self._fh = open(self._tmp, "wb")
        self._tar = tarfile.open(fileobj=self._fh, mode="w")
        for name, data in codec.bundle_to_tar_members(self.sub, self._bundle).items():
            self._add(name, data)  # BUNDLE FIRST
        return self

    def add_run(self, run, vol) -> None:
        """Append one run's BOLD block, packed at the tar's quant.

        RUN KEYS CARRY MEANING POSITIONALLY: ``run`` is ``mod_task_..._phase``, so ``task`` is
        ``parts[1]``, ``phase`` is ``parts[-1]``, and the run is REST iff ``parts[0] == "rfMRI"``.
        A core that renames a run breaks these labels silently."""
        block = codec.pack_bold(vol, quant="int8")
        parts = run.split("_")
        meta = {
            "task": parts[1] if len(parts) > 1 else run,
            "phase": parts[-1],
            "task_group": "rest" if parts[0] == "rfMRI" else (parts[1] if len(parts) > 1 else run).lower(),
        }
        for name, data in codec.bold_to_tar_members(f"{self.sub}__{run}", block, meta).items():
            self._add(name, data)
        self._n_runs += 1

    def _add(self, name, data) -> None:
        info = tarfile.TarInfo(name)
        info.size = len(data)
        self._tar.addfile(info, io.BytesIO(data))

    def __exit__(self, exc_type, *_) -> None:
        # A produced tar with zero runs is a non-deliverable (bundle only) -> don't mark done.
        self._finish(ok=exc_type is None and self._n_runs > 0)

    def _finish(self, *, ok) -> None:
        if self._tar is not None:
            self._tar.close()
        if self._fh is not None:
            self._fh.close()
        wrote = False
        if self._tmp is not None:  # local: atomic rename, drop partial on error
            if ok:
                os.replace(self._tmp, self.uri)
                wrote = True
            else:
                Path(self._tmp).unlink(missing_ok=True)
        elif self._tar is not None:  # s3: streamed into the final key
            if ok:
                wrote = True
            elif is_s3(self.uri):
                try:
                    s3_filesystem(self._so).rm(self.uri)  # best-effort: drop partial s3 tar
                except FileNotFoundError:
                    pass
        if wrote:  # mark complete only after a full, clean close (resume trusts this)
            write_done(self.uri, storage_options=self._so)
        self._fh = self._tar = self._tmp = None
