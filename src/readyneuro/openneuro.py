"""OpenNeuro as a sampling universe: the frozen subject index, the disjoint seeded batches, and
the raw-BIDS file resolution (T1 + runs + TR) each batch draws from.

Everything here reads the public bucket ANONYMOUSLY and pays only ranged header GETs: a NIfTI
header is the first 348 bytes, so frame counts and FOV extents cost one ~8 KB GET instead of a
30-150 MB download, and a batch of 1000 buildable subjects is chosen before a single image is
downloaded.
"""

from __future__ import annotations

import io
import json
import logging
import os
import posixpath
import random
import re
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor

import nibabel as nib

from readyneuro.io import GZIP_MAGIC, s3_filesystem

log = logging.getLogger("readyneuro.openneuro")

BUCKET = "openneuro.org"
#: OpenNeuro's bucket is public -> read with no credentials. The timeouts are NOT optional: botocore
#: defaults to no read timeout, so a stalled anonymous GET hangs the calling rule *forever* (observed:
#: register jobs idle 4-5 h mid-BOLD-read, blocking a batch until its wall clock). `read_timeout` turns
#: a stall into a retryable error, and `retries` recovers the transient ones in-band.
ANON = {
    "anon": True,
    "config_kwargs": {
        "connect_timeout": 15,
        "read_timeout": 60,
        "retries": {"max_attempts": 5, "mode": "standard"},
        # botocore defaults to 10 connections, which silently caps `sample_batch`'s thread pool --
        # 16 scan workers measured only 3.4x because 6 of them were always queued on the pool.
        "max_pool_connections": 64,
    },
}

_ACCESSION = re.compile(r"^ds\d{6,}$")  # dsNNNNNN accession dirs (skip derivatives/ etc.)
_LABEL = re.compile(r"^[A-Za-z0-9]+$")  # a BIDS entity label (keeps the work key wildcard-safe)


# --- s3 helpers -------------------------------------------------------------
def _fs():
    """The anonymous OpenNeuro s3fs filesystem."""
    return s3_filesystem(ANON)


def _basename(path: str) -> str:
    return posixpath.basename(path.rstrip("/"))


def _strip_scheme(uri: str) -> str:
    """``s3://bucket/key`` -> ``bucket/key`` (s3fs takes the scheme-less form)."""
    return uri[len("s3://"):] if uri.startswith("s3://") else uri


def retry(fn, *, what: str, attempts: int = 3, fatal: tuple = ()):
    """Call ``fn``, re-trying a transient failure; raise once the attempts are exhausted.

    A transient error must never be absorbed into a *data* answer: "unreadable" and "absent/short"
    are different facts, and swallowing the first into the second alters a subject's validity
    undetectably. ``fatal`` types are DETERMINISTIC answers (e.g. ``AccessDenied``) and re-raise at
    once rather than burning retries."""
    for attempt in range(attempts):
        try:
            return fn()
        except fatal:
            raise
        except Exception as exc:  # noqa: BLE001 -- any transient failure is worth a re-read
            if attempt == attempts - 1:
                raise
            log.warning("%s failed (%s) -- retry %d/%d", what, exc, attempt + 1, attempts - 1)


def _is_bids_file(key: str) -> bool:
    """A raw-BIDS image: not under ``derivatives/`` or ``sourcedata/`` -- a guard against ever
    picking up fMRIPrep output or a raw copy as if it were an acquisition."""
    low = key.lower()
    return "/derivatives/" not in low and "/sourcedata/" not in low


# --- 1. the sampling universe ----------------------------------------------
def list_accessions(fs) -> list[str]:
    """Every ``dsNNNNNN`` accession at the bucket root (sorted, deterministic)."""
    names = (_basename(p) for p in fs.ls(BUCKET, detail=False))
    return sorted(n for n in names if _ACCESSION.match(n))


def list_subject_labels(fs, ds: str) -> list[str]:
    """The dataset's top-level ``sub-<label>`` dirs -> the bare labels (sessions resolved later)."""
    out = []
    for p in fs.ls(f"{BUCKET}/{ds}", detail=False):
        name = _basename(p)
        if name.startswith("sub-") and _LABEL.match(name[4:]):
            out.append(name[4:])
    return sorted(out)


#: OpenNeuro's public GraphQL endpoint (reads need no auth).
OPENNEURO_GRAPHQL = "https://openneuro.org/crn/graphql"

#: ``metadata.species`` values (case/space-normalised) that count as HUMAN. Undeclared
#: (``None``/``""``) is human too -- OpenNeuro's default, and how ~450 human datasets leave it.
#: This is an ALLOWLIST: every OTHER declared value (Mouse, Rat, Macaque, Sheep, Pigeon, the
#: phantoms, ...) is non-human and drops, so a species label OpenNeuro adds later drops by default
#: rather than slipping in unrecognised.
_HUMAN_SPECIES = frozenset({"", "human", "humans", "homo sapiens", "homo sapien"})


def is_human_species(species: str | None) -> bool:
    """Whether a dataset's OpenNeuro ``metadata.species`` denotes a human cohort. Matches the WHOLE
    normalised string, not a substring, so ``"Phantom Human"`` (a scanner phantom) is NOT human."""
    return species is None or species.strip().lower() in _HUMAN_SPECIES


def _graphql(query: str, *, endpoint: str = OPENNEURO_GRAPHQL) -> dict:
    """POST one GraphQL query to OpenNeuro; parsed JSON (``{data, errors?}``). Stdlib ``urllib`` --
    off the s3fs path, no added dependency. Reads are public, so no auth."""
    req = urllib.request.Request(
        endpoint,
        data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as fh:  # noqa: S310 -- fixed https endpoint
        return json.load(fh)


def nonhuman_accessions(*, gql=_graphql) -> set[str]:
    """Every OpenNeuro accession whose curated ``metadata.species`` is declared non-human (or a
    phantom), by paging the ``datasets`` GraphQL connection. A dataset that declares no species is
    human by default and is NOT returned.

    A handful of datasets ship broken ``metadata`` and error that ONE field: GraphQL returns the
    rest of the page plus an ``errors`` entry, so the partial ``data`` is used and the broken node
    reads as unknown -> kept. Same rule as :func:`bold_frames`: never turn a partial/transient
    failure into a data claim that silently drops a subject. ``gql`` is injected for tests."""
    drop: set[str] = set()
    after: str | None = None
    while True:
        aft = f', after: "{after}"' if after else ""
        query = (
            f"{{ datasets(first: 100{aft}) {{ pageInfo {{ hasNextPage endCursor }} "
            f"edges {{ node {{ id metadata {{ species }} }} }} }} }}"
        )
        conn = retry(
            lambda q=query: (gql(q).get("data") or {}).get("datasets"),
            what="openneuro species page",
        )
        if conn is None:
            raise RuntimeError("openneuro species query returned no `datasets` data")
        for edge in conn["edges"]:
            node = edge.get("node")
            if node is None:  # a null edge (deleted/private) carries no species -> not our concern
                continue
            if not is_human_species((node.get("metadata") or {}).get("species")):
                drop.add(node["id"])
        page = conn["pageInfo"]
        if not page["hasNextPage"]:
            break
        after = page["endCursor"]
    return drop


def build_index(cfg: dict) -> list[list[str]]:
    """The flat ``[[ds, sub_label], ...]`` universe, cached to ``cfg['openneuro_index']`` -- a path
    every batch SHARES so they all sample one frozen archive. Built once by listing every
    accession's top-level subjects (thread-pooled); modality is NOT checked here (too costly at this
    scale) -- :func:`sample_batch` verifies lazily as it draws.

    ``filter_species`` drops every DECLARED-non-human accession up front, from OpenNeuro's curated
    ``metadata.species`` via GraphQL, so Mouse / Rat / Macaque / phantom datasets never enter the
    frozen universe and SynthSeg is never handed a 17 mm brain. Undeclared species stay (human by
    default); an UNDECLARED non-human is a geometry-side concern `is_wholebrain_t1` covers."""
    cache = cfg.get("openneuro_index") or os.path.join(cfg["results_dir"], "openneuro_index.json")
    if os.path.exists(cache):
        with open(cache) as fh:
            return json.load(fh)

    fs = _fs()
    accessions = list_accessions(fs)
    if cfg.get("filter_species", True):
        drop = nonhuman_accessions()
        kept = [a for a in accessions if a not in drop]
        if len(kept) != len(accessions):
            log.info(
                "species filter: dropped %d non-human accession(s), %d remain",
                len(accessions) - len(kept),
                len(kept),
            )
        accessions = kept
    with ThreadPoolExecutor(max_workers=int(cfg.get("openneuro_list_workers", 16))) as pool:
        per_ds = pool.map(lambda ds: [[ds, s] for s in list_subject_labels(fs, ds)], accessions)
    index = [pair for rows in per_ds for pair in rows]

    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    tmp = f"{cache}.tmp"  # atomic: never leave a partial index a later batch would trust
    with open(tmp, "w") as fh:
        json.dump(index, fh)
    os.replace(tmp, cache)
    return index


# --- 2. deterministic, disjoint batches ------------------------------------
def work_key(ds: str, sub_label: str) -> str:
    return f"{ds}__{sub_label}"


def parse_key(key: str) -> tuple[str, str]:
    """``{ds}__{sub_label}`` -> ``(ds, sub_label)``."""
    ds, sub = key.split("__", 1)
    return ds, sub


#: Memoized ``dim[4]`` (frame count) per BOLD uri. Successful reads ONLY -- a failure must never be
#: cached, or one transient error permanently fixes a run's apparent length. See :func:`bold_frames`.
_BOLD_FRAMES: dict[str, int] = {}


class UnreadableHeader(ValueError):
    """A structurally unparseable NIfTI header -- truncated, empty, or a DataLad pointer file. A
    DETERMINISTIC property, so callers DROP the run/subject rather than let one malformed object
    abort a whole batch's discovery."""


def _is_invalid_range(exc: Exception) -> bool:
    """True for the S3 "range not satisfiable" error (HTTP 416) that a 0-byte object raises on a
    ranged GET -- the object is deterministically empty, not a transient failure."""
    msg = str(exc)
    return "InvalidRange" in msg or "not satisfiable" in msg


def _ranged_header(uri: str, fs) -> "nib.Nifti1Header":
    """The NIfTI header from a ~8 KB ranged read, or `UnreadableHeader` if the object is not one."""
    try:
        raw = fs.cat_file(_strip_scheme(uri), start=0, end=8192)
    except Exception as exc:  # noqa: BLE001
        # A 0-byte object makes the ranged GET itself fail with HTTP 416 (InvalidRange) BEFORE the
        # <348-byte check below runs -- the same deterministic "not a readable header" fact. DROP the
        # run (via UnreadableHeader) rather than let retry() re-raise and abort a whole 1000-subject
        # discovery. A real transient still propagates.
        if _is_invalid_range(exc):
            raise UnreadableHeader(f"{uri}: empty object -- ranged read returned InvalidRange") from exc
        raise
    if raw[:2] == GZIP_MAGIC:
        try:
            raw = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(raw, 400)
        except zlib.error as exc:
            raise UnreadableHeader(f"{uri}: not a readable gzip stream ({exc})") from exc
    if len(raw) < 348:
        raise UnreadableHeader(f"{uri}: {len(raw)} bytes, shorter than a 348-byte NIfTI header")
    try:
        return nib.Nifti1Header.from_fileobj(io.BytesIO(raw[:348]), check=False)
    except Exception as exc:
        raise UnreadableHeader(f"{uri}: unparseable NIfTI header ({exc})") from exc


def _header_tr(bold_uri: str, fs) -> float | None:
    """The NIfTI header's ``pixdim[4]`` (TR, seconds) from a ~8 KB ranged read -- the sidecar-less
    fallback. dcm2niix-converted OpenNeuro raw carries the true acquisition TR here, so a run whose BIDS
    omits a ``RepetitionTime`` sidecar is still recoverable (without this, ``run_tr`` returned ``None`` and
    the finalizer silently resampled from the header-default 1.0 s -- corrupting genuine 2-3 s runs).
    Some headers ship ``0`` (ds000214) or a value in the wrong unit, so accept ONLY a plausible fMRI TR
    (0.1-20 s); otherwise ``None``, so a truly-unknowable TR stays a loud finalizer failure, not a silent 1.0."""
    try:
        z = _ranged_header(bold_uri, fs).get_zooms()
    except Exception:
        return None
    tr = float(z[3]) if len(z) > 3 else 0.0
    return tr if 0.1 <= tr <= 20.0 else None


def bold_frames(bold_uri: str, *, fs=None) -> int:
    """Frames (``dim[4]``) of a BOLD, from a ranged header read.

    Raises if the header cannot be read after :func:`retry`. It must NOT answer "0 frames" for an
    unreadable run: that is a data claim the caller acts on, indistinguishable from a genuinely
    degenerate run."""
    if bold_uri in _BOLD_FRAMES:
        return _BOLD_FRAMES[bold_uri]
    fs = fs or _fs()

    def _read() -> int:
        hdr = _ranged_header(bold_uri, fs)
        try:
            dim = hdr.get_data_shape()
        except Exception as exc:
            raise UnreadableHeader(f"{bold_uri}: unusable header fields ({exc})") from exc
        return int(dim[3]) if len(dim) > 3 else 1

    frames = retry(
        _read,
        what=f"header read {bold_uri}",
        fatal=(PermissionError, FileNotFoundError, UnreadableHeader),
    )
    _BOLD_FRAMES[bold_uri] = frames
    return frames


#: Memoized per-axis FOV extent (mm) per T1 uri. Successful reads ONLY -- like :data:`_BOLD_FRAMES`,
#: a failure is transient and must never be cached, or one hiccup permanently fixes a T1's apparent FOV.
_T1_EXTENT: dict[str, tuple[float, float, float]] = {}


def t1_extent(t1_uri: str, *, fs=None) -> tuple[float, float, float]:
    """The T1's physical FOV extent per axis (mm) = ``shape[i] * zoom[i]``, from a ranged header
    read. A missing axis (a 2-D volume) reads as extent 0, so it fails any positive threshold.
    Raises rather than answer a fake extent, same data-claim contract as `bold_frames`."""
    if t1_uri in _T1_EXTENT:
        return _T1_EXTENT[t1_uri]
    fs = fs or _fs()

    def _read() -> tuple[float, float, float]:
        hdr = _ranged_header(t1_uri, fs)
        try:
            dims, zooms = hdr.get_data_shape(), hdr.get_zooms()
        except Exception as exc:
            raise UnreadableHeader(f"{t1_uri}: unusable header fields ({exc})") from exc
        return tuple(
            float(dims[i]) * float(zooms[i]) if i < len(dims) and i < len(zooms) else 0.0
            for i in range(3)
        )

    extent = retry(_read, what=f"T1 header read {t1_uri}",
                   fatal=(PermissionError, FileNotFoundError, UnreadableHeader))
    _T1_EXTENT[t1_uri] = extent
    return extent


def is_wholebrain_t1(t1_uri: str, min_fov_mm: float, *, fs=None) -> bool:
    """Whether the T1's SMALLEST-axis FOV extent is >= ``min_fov_mm``. A whole human head spans
    >=~150 mm on every axis, so one min-axis test rejects both an undeclared non-human (rodent
    ~15-25 mm, macaque ~60-80 mm) and a partial-brain slab, before SynthSeg makes garbage of it.
    An `UnreadableHeader` answers False -- the subject is dropped either way."""
    try:
        return min(t1_extent(t1_uri, fs=fs)) >= float(min_fov_mm)
    except UnreadableHeader:
        return False


def sample_batch(
    index,
    seed: int,
    batch: int,
    n: int,
    *,
    fs=None,
    min_frames: int | None = None,
    min_fov_mm: float | None = None,
    scan_workers: int = 16,
) -> list[str]:
    """The ``batch``-th disjoint window of ``n`` **buildable** subjects, as work keys.

    One seeded shuffle of the whole universe fixes a global order; scanning it and keeping only
    subjects with a T1w AND >=1 usable BOLD yields the *valid order*; batch *b* is its slice
    ``[b*n : (b+1)*n]``. Same ``(index, seed)`` -> identical, non-overlapping batches, which is what
    makes the cumulative corpus subsets nest. Scans just far enough to fill through batch ``b``.

    ``min_frames`` drops runs shorter than that many timepoints, and with them any subject left
    with no runs, so a batch of ``n`` is ``n`` USABLE subjects (OpenNeuro ships ~13-frame runs).
    ``min_fov_mm`` is the geometry-side twin (`is_wholebrain_t1`). Both ride the ranged header
    read."""
    fs = fs or _fs()
    order = [tuple(p) for p in index]
    random.Random(seed).shuffle(order)

    need = (batch + 1) * n
    valid: list[str] = []
    workers = max(1, int(scan_workers))
    # Candidates are resolved CONCURRENTLY but consumed in shuffled-index order, so the result is
    # byte-identical to the serial scan -- which it MUST be: batch `b` is a slice of `valid`, so any
    # reordering would overlap or gap the production batches. A chunk is resolved, then drained in
    # order before the next is submitted; overshooting the chunk boundary only wastes a few reads.
    chunk = max(workers * 8, 64)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(order), chunk):
            if len(valid) >= need:
                break
            futures = [
                pool.submit(_resolve_candidate, ds, sub, fs=fs,
                            min_frames=min_frames, min_fov_mm=min_fov_mm)
                for ds, sub in order[start : start + chunk]
            ]
            for i, future in enumerate(futures):
                if len(valid) >= need:
                    # Never call .result() past the point the serial scan would have stopped: a
                    # TRANSIENT error on a candidate we did not need must not fail the run.
                    for pending in futures[i:]:
                        pending.cancel()
                    break
                key = future.result()
                if key is not None:
                    valid.append(key)
    return valid[batch * n : (batch + 1) * n]


def _resolve_candidate(ds, sub, *, fs, min_frames, min_fov_mm) -> str | None:
    """One candidate's verdict for `sample_batch`: its work key if buildable, else ``None``.

    ``min_fov_mm`` does a ranged GET of the T1 to measure its FOV, so a subject that gets this far
    is PROVEN anonymously readable -- a real subset of OpenNeuro datasets are listable and HEAD-able
    but ``GetObject``-restricted, and those raise ``PermissionError`` here and are skipped."""
    try:
        files = subject_files(ds, sub, fs=fs, min_frames=min_frames, min_fov_mm=min_fov_mm)
    except (PermissionError, FileNotFoundError):
        # A DETERMINISTIC denial: this subject is unbuildable, skip it. A TRANSIENT error still
        # propagates out of `retry` -- absorbing one into "invalid subject" would silently shift
        # every later batch's window.
        return None
    if not files["t1"] or not files["runs"]:
        return None
    return work_key(ds, sub)


# --- 3. raw-BIDS file resolution -------------------------------------------
def _usable_run(bold_uri: str, min_frames: int, fs) -> bool:
    """Whether a run is long enough to keep. An `UnreadableHeader` drops just THAT run -- the subject
    survives on its other runs, and only becomes unbuildable if none are left (the same gate an
    all-too-short subject hits). A transient error still propagates."""
    try:
        return bold_frames(bold_uri, fs=fs) >= min_frames
    except UnreadableHeader:
        return False


def _task_of(bold_key: str) -> str:
    """The BIDS ``task-<label>`` of a BOLD filename (``bold`` if unlabeled). BIDS labels are
    alphanumeric, so it is safe in the ``_``-split run key the tar derives task/phase from."""
    m = re.search(r"_task-([A-Za-z0-9]+)", _basename(bold_key))
    return m.group(1) if m else "bold"


def subject_files(
    ds: str,
    sub_label: str,
    *,
    fs=None,
    with_tr: bool = False,
    max_runs: int | None = None,
    min_frames: int | None = None,
    min_fov_mm: float | None = None,
) -> dict:
    """Resolve one subject's reusable inputs from raw BIDS (flat or ``ses-*``):

    ``{"t1": <s3 uri | None>, "runs": {run_key: <bold s3 uri>, ...}, "tr": {run_key: float|None}}``

    ``t1`` is the first ``anat/*_T1w`` found; ``runs`` is every ``func/*_bold`` across all sessions,
    keyed ``func_<task>_<idx>`` (``idx`` = position in the sorted file list, so keys are stable +
    wildcard/filename-safe + split into clean task/phase labels for the tar). ``tr`` is filled from
    each BOLD's JSON sidecar (``RepetitionTime``) only when ``with_tr`` (an extra S3 read per run).

    ``max_runs`` truncates the (sorted, hence stable) run list: ``ds005354/sub-20`` ships 121 BOLD
    runs, and one such subject holds a whole batch open."""
    fs = fs or _fs()
    base = f"{BUCKET}/{ds}/sub-{sub_label}"

    # `*_T1w`/`*_bold` only occur under anat/ resp. func/ in BIDS, so a depth-agnostic `**` glob
    # (flat or under ses-*) is enough and dodges fsspec's `**/<seg>/` flat-case quirk. The `_T1w`
    # SUFFIX also excludes deface masks for free -- BIDS names those `sub-01_mod-T1w_defacemask`,
    # where `T1w` is an entity VALUE, not the suffix.
    t1s = [k for k in fs.glob(f"{base}/**/*_T1w.nii*") if _is_bids_file(k)]
    bolds = sorted(k for k in fs.glob(f"{base}/**/*_bold.nii*") if _is_bids_file(k))
    # Drop too-short runs BEFORE the cap, so `max_runs` keeps N usable runs rather than N runs of
    # which some are unusable. Costs one ~8 KB ranged header read per run (see `bold_frames`).
    if min_frames is not None:
        bolds = [k for k in bolds if _usable_run(f"s3://{k}", int(min_frames), fs)]
    if max_runs is not None:
        bolds = bolds[: int(max_runs)]

    runs = {f"func_{_task_of(k)}_{i}": f"s3://{k}" for i, k in enumerate(bolds)}
    tr = {r: (run_tr(uri, fs=fs) if with_tr else None) for r, uri in runs.items()}

    t1 = f"s3://{sorted(t1s)[0]}" if t1s else None
    # Drop a T1 whose FOV is too small to be a whole human brain (undeclared non-human, or a
    # partial-brain / sub-mm slab) -- one ~8 KB header read, BEFORE download + SynthSeg. A dropped T1
    # makes the subject unbuildable, exactly as `min_frames` emptying `runs` does (both `sample_batch`
    # and the core's discover gate on a present t1 + runs).
    if t1 is not None and min_fov_mm is not None and not is_wholebrain_t1(t1, min_fov_mm, fs=fs):
        t1 = None
    return {"t1": t1, "runs": runs, "tr": tr}


#: Memoized ``RepetitionTime`` per sidecar path that the inheritance walk actually found. Values
#: include ``None`` -- a file that exists but declares no ``RepetitionTime`` is a genuine, stable
#: ABSENCE, so caching it is correct. (Contrast :data:`_BOLD_FRAMES`, where a failure is transient
#: and must never be cached.)
_SIDECAR_TR: dict[str, float | None] = {}

#: Memoized directory listing per level. The walk lists the same handful of dirs for every run of a
#: subject (and every subject of a dataset shares the root), so this turns O(runs x levels) S3 `ls`
#: calls into one per dir. Tests that reuse paths across filesystems must clear it.
_DIR_LS: dict[str, list[str]] = {}

_ENTITY = re.compile(r"^([A-Za-z0-9]+)-([A-Za-z0-9]+)$")


def _parse_bids_name(name: str) -> tuple[dict[str, str], str] | None:
    """``sub-01_task-rest_run-01_bold.json`` -> ``({sub: 01, task: rest, run: 01}, 'bold.json')``.

    ``None`` if the name is not ``<key-value>_..._<suffix>`` shaped (a dir, a non-BIDS file)."""
    parts = name.split("_")
    ents: dict[str, str] = {}
    for tok in parts[:-1]:
        m = _ENTITY.match(tok)
        if not m:
            return None
        ents[m.group(1)] = m.group(2)
    return ents, parts[-1]


def _ls(fs, base: str) -> list[str]:
    """Cached one-level listing of ``base`` (``[]`` when it does not exist)."""
    if base not in _DIR_LS:
        try:
            _DIR_LS[base] = list(fs.ls(base, detail=False))
        except FileNotFoundError:
            _DIR_LS[base] = []
    return _DIR_LS[base]


def _sidecar_tr(fs, path: str) -> float | None:
    """``RepetitionTime`` from one sidecar path (cached; ``None`` = the file declares none)."""
    if path not in _SIDECAR_TR:
        try:
            with fs.open(path) as fh:
                tr = json.load(fh).get("RepetitionTime")
            _SIDECAR_TR[path] = float(tr) if tr else None
        except Exception:  # noqa: BLE001 -- an unparseable sidecar is a miss, not a build failure
            _SIDECAR_TR[path] = None
    return _SIDECAR_TR[path]


def run_tr(bold_uri: str, *, fs=None) -> float | None:
    """``RepetitionTime`` (seconds) for a BOLD run, honouring the **BIDS inheritance principle**: a
    sidecar applies when its suffix matches and its entity set is a SUBSET of the image's with equal
    values; nearest level wins, and within a level the most specific wins. So each level is listed
    once and filtered, rather than enumerating candidate names (which also misses the legal
    task-less ``bold.json``).

    With no sidecar TR anywhere, fall back to the header's ``pixdim[4]``, guarded to a plausible
    0.1-20 s (ds000214 ships ``0`` and is rejected). ``None`` only when neither exists -- the
    finalizer then fails loud rather than silently resampling a 2-3 s run as if it were 1.0 s."""
    fs = fs or _fs()
    parts = _strip_scheme(bold_uri).split("/")
    parsed = _parse_bids_name(re.sub(r"\.nii(\.gz)?$", ".json", parts[-1]))
    if parsed is None:
        return None
    img_ents, suffix = parsed

    # Walk upward, nearest first; stop above the dataset root (depth 1 is the bucket).
    for depth in range(len(parts) - 1, 1, -1):
        base = "/".join(parts[:depth])
        applicable = []
        for entry in _ls(fs, base):
            name = _basename(entry)
            cand = _parse_bids_name(name)
            if cand is None or cand[1] != suffix:
                continue
            ents = cand[0]
            if all(img_ents.get(k) == v for k, v in ents.items()):
                applicable.append((len(ents), f"{base}/{name}"))
        # Most specific first; a sidecar that applies but declares no TR falls through to the next
        # candidate, then to the parent level -- BIDS merges metadata per-key across levels.
        for _, path in sorted(applicable, reverse=True):
            tr = _sidecar_tr(fs, path)
            if tr:
                return tr
    return _header_tr(bold_uri, fs)  # sidecar-less: the raw header's pixdim[4] (guarded), else None


def participants_meta(ds: str, sub_label: str, *, fs=None) -> dict:
    """Best-effort ``{age, sex}`` for a subject from the dataset ``participants.tsv`` (BIDS).
    Empty when the file/columns/row are absent — demographics are optional in the bundle."""
    fs = fs or _fs()
    try:
        with fs.open(f"{BUCKET}/{ds}/participants.tsv") as fh:
            lines = fh.read().decode("utf-8", "replace").splitlines()
        header = lines[0].split("\t")
        rows = {r.split("\t")[0]: r.split("\t") for r in lines[1:] if r.strip()}
        row = rows.get(f"sub-{sub_label}")
        if not row:
            return {}
        col = {name: i for i, name in enumerate(header)}
        out = {}
        for key in ("age", "sex"):
            i = col.get(key)
            if i is not None and i < len(row) and row[i] not in ("", "n/a", "N/A"):
                out[key] = row[i]
        return out
    except Exception:
        return {}
