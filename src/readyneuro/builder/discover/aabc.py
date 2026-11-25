"""AABC (Aging Adult Brain Connectome) discovery + its source layer.

EVAL-ONLY -- the age-bracket and sex benchmarks. One item per eval subject (batches 10-19), keyed by
the BARE subject, contributing its sampled visit's 500-TR REST window. AABC ships each subject-visit
as a pair of zips rather than an S3 tree, so the whole zips are staged and the core reads the members
it needs out of them.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path

import s3fs

from readyneuro.io import BENCHMARK_METADATA_ROOT, fetch, s3_filesystem

log = logging.getLogger(__name__)

#: ``aabc_pheno_targets_root`` must be declared: `discover` bakes `subject_meta` (age band + sex,
#: read from that root) into every item's ``sources.meta``, so the root is discovery-affecting. Left
#: undeclared, re-pointing the demographics mirror would silently REPLAY the manifest cached under
#: the OLD root -- an eval store whose labels came from the wrong mirror, with nothing to detect it.
#: ``aabc_eval_split`` (see `eval_split_map`) picks the items and their splits, so it keys it too.
DISCOVERY_KEYS = ("aabc_pheno_targets_root", "aabc_eval_split")

NATIVE_TR = 0.8            # HCP-Aging 3T acquisition TR (the finalizer resamples FROM this to target_tr)
STRUCT_ZIP = "struct.zip"  # the whole StructuralRecommended zip (staged by the driver as sources.t1)
FUNC_ZIP = "func.zip"      # the whole ConcatFmriRecommended zip (staged as sources.aux[0])

# --- source locations -------------------------------------------------------
# AABC is access-controlled, so these prefixes are a private mirror of the released zips.
FUNC_PREFIX = "s3://<RAW_ROOT>/AABC-c2-fmri"
STRUCT_PREFIX = "s3://<RAW_ROOT>/AABC-struct-MNI"
_FUNC_SUFFIX = "_ConcatFmriRecommended.zip"
_STRUCT_SUFFIX = "_StructuralRecommended.zip"

_META = f"{BENCHMARK_METADATA_ROOT}/AABC/metadata"

# Splits: batches 0-9 pretrain, 10-19 eval (the benchmark suite's partition). AABC contributes
# EVAL ONLY -- its pretraining half is deliberately unused, so only the eval batches are read.
EVAL_BATCHES = frozenset(range(10, 20))

# --- zip uris + session parsing ---------------------------------------------
def func_zip_uri(session: str) -> str:
    return f"{FUNC_PREFIX}/{session}{_FUNC_SUFFIX}"


def struct_zip_uri(session: str) -> str:
    return f"{STRUCT_PREFIX}/{session}{_STRUCT_SUFFIX}"


def member(session: str, rel: str) -> str:
    """A member path inside ``{session}.zip`` (entries are rooted at ``SESSION/``)."""
    return f"{session}/{rel}"


def sub_visit(session: str) -> tuple[str, str]:
    """``HCA0000000_V1_MR`` -> ``("HCA0000000", "V1")``."""
    parts = session.split("_")
    return parts[0], parts[1]


def list_sessions(storage_options: dict | None = None) -> list[str]:
    """Sorted sessions (``HCA…_V#_MR``) present as **both** a struct and a func zip."""
    fs = s3fs.S3FileSystem(**(storage_options or {}))

    def names(prefix: str, suffix: str) -> set[str]:
        base = prefix.removeprefix("s3://")
        out = set()
        for p in fs.ls(base, detail=False):
            name = p.rsplit("/", 1)[-1]
            if name.endswith(suffix):
                out.add(name[: -len(suffix)])
        return out

    func = names(FUNC_PREFIX, _FUNC_SUFFIX)
    struct = names(STRUCT_PREFIX, _STRUCT_SUFFIX)
    return sorted(func & struct)


# --- splits -----------------------------------------------------------------
#: The benchmark-metadata fetcher (local stage + 429 backoff), bound to AABC's metadata dir.
_fetch = partial(fetch, _META)


@lru_cache
def subject_to_batch() -> dict[str, int]:
    """sub -> batch index, from the benchmark suite's ``aabc_subject_batch_splits.json``."""
    splits = json.loads(_fetch("aabc_subject_batch_splits.json"))
    return {s: int(b.split("-")[1]) for b, subs in splits.items() for s in subs}


@lru_cache
def eval_split_map(path: str | None) -> dict[str, tuple[str, str]]:
    """``sub -> (split, visit)`` for the AABC eval subjects, read from the local
    JSON file named by the ``aabc_eval_split`` config key: ``{"HCA...": [split, visit], ...}`` with
    ``split`` in {train, validation, test} and ``visit`` (``V1``...) the one the benchmark suite
    sampled. It lists AABC subject IDs, so it is not shipped here; derive it from your own
    Brainmarks AABC eval build (``brainmarks/datasets/AABC/scripts/make_aabc_eval.py``: one seeded
    visit per subject of batches 10-19, stratified 80/10/10)."""
    if not path or not Path(path).is_file():
        raise RuntimeError(
            f"AABC needs its eval split: set `aabc_eval_split` in the dataset config to a local JSON "
            f'file {{"<subject>": ["train"|"validation"|"test", "<visit>"], ...}} (got {path!r}). '
            f"See readyneuro.builder.discover.aabc.eval_split_map for how to derive it."
        )
    return {s: (v[0], v[1]) for s, v in json.loads(Path(path).read_text()).items()}


# --- best-effort demographics (the benchmark suite's target maps, mirrored) ------------------
# AABC demographics are access-controlled, so the target maps live on a private mirror; override
# ``aabc_pheno_targets_root`` if it moves. Each ``{name}`` has a ``_target_map_{name}.json``
# (bare-sub -> int) + a ``_target_info_{name}.json`` (classes / bin ranges we decode to labels).
PHENO_TARGETS_ROOT = "s3://<EVAL_ROOT>/targets"
_PHENO_NAME = {"age": "age_open", "sex": "sex"}  # our key -> the suite's target name
_pheno_cache: dict[str, dict] = {}


def _read_json(uri: str, storage_options: dict | None) -> dict:
    with s3_filesystem(storage_options).open(uri) as f:
        return json.load(f)


def pheno_map(
    target: str, *, root: str | None = None, storage_options: dict | None = None
) -> dict[str, str]:
    """Best-effort ``bare-sub -> decoded label`` for an AABC phenotype: ``sex`` -> ``'F'``/``'M'``
    (info ``classes``), ``age`` -> the open-age bin range ``'lo-hi'`` (info ``bin_stats``). Returns
    ``{}`` (logged once) when unreachable -- demographics are best-effort."""
    name = _PHENO_NAME[target]
    base = (root or PHENO_TARGETS_ROOT).rstrip("/")
    ck = f"{base}/{name}"
    if ck not in _pheno_cache:
        try:
            m = _read_json(f"{base}/aabc_target_map_{name}.json", storage_options)
            info = _read_json(f"{base}/aabc_target_info_{name}.json", storage_options)
            if target == "sex":
                classes = info["classes"]
                dec = {str(k): classes[int(v)] for k, v in m.items() if v is not None}
            else:  # age: decode the bin index to its 'min-max' year range
                rng = {b["bin"]: f"{int(b['min'])}-{int(b['max'])}" for b in info["bin_stats"]}
                dec = {str(k): rng[int(v)] for k, v in m.items() if v is not None and int(v) in rng}
            _pheno_cache[ck] = dec
        except Exception:  # noqa: BLE001 -- demographics are best-effort
            log.warning("AABC pheno map unreachable: %s/aabc_target_map_%s.json (best-effort skip)", base, name)
            _pheno_cache[ck] = {}
    return _pheno_cache[ck]


# --- work items -------------------------------------------------------------
@dataclass(frozen=True)
class AABCWork:
    """One subject-visit session: its 500-TR REST window and the eval split it belongs to."""

    session: str
    sub: str
    visit: str
    split: str


def discover_work(
    storage_options: dict | None = None,
    *,
    split_path: str | None = None,
    only_subjects: list[str] | None = None,
    n_sessions: int | None = None,
) -> list[AABCWork]:
    """Every eval subject paired to its split -- and to ONLY the one visit the benchmark suite
    sampled, so a subject contributes exactly one session."""
    eval_map = eval_split_map(split_path)  # first: a missing split fails before any S3 listing
    sessions = list_sessions(storage_options)
    if only_subjects:
        keep = {str(x) for x in only_subjects}  # str: a YAML-parsed numeric id still matches
        sessions = [s for s in sessions if sub_visit(s)[0] in keep]
    batch_of = subject_to_batch()
    work: list[AABCWork] = []
    for session in sessions:
        sub, visit = sub_visit(session)
        if batch_of.get(sub) not in EVAL_BATCHES:
            continue
        assign = eval_map.get(sub)
        if assign is not None and assign[1] == visit:
            work.append(AABCWork(session, sub, visit, assign[0]))
    return work[:n_sessions] if n_sessions else work


# --- the driver's discover() entrypoint -------------------------------------
def subject_meta(sub_bare: str, cfg: dict) -> dict:
    """Best-effort age band + sex from the AABC target maps, read with the AABC input's own creds.
    ``subject`` is the BARE subject, so a leave-one-subject-out probe groups a subject's visits
    together (the bundle spreads this meta LAST, over the per-session work key)."""
    so, root = cfg.get("input_storage_options"), cfg.get("aabc_pheno_targets_root")
    out = {"subject": sub_bare}
    for key in ("age", "sex"):
        v = pheno_map(key, root=root, storage_options=so).get(sub_bare)
        if v is not None:
            out[key] = v
    return out


def discover(cfg: dict) -> list[dict]:
    """One eval item per eval subject (key = the bare subject, its sampled visit's REST window). The
    whole struct + func zips are staged as ``sources.t1`` + ``sources.aux[0]``; ``sources.runs`` stays
    empty (the core reads the concat out of the func zip). ``session`` rides in ``sources`` so the
    core can build the zip member paths."""
    if list(cfg["spaces"]) != ["mni"]:
        raise ValueError(f"{__name__} builds only 'mni'; for native space use the *_native dataset (got {cfg['spaces']!r})")
    so = cfg.get("input_storage_options")
    spaces = list(cfg["spaces"])
    works = discover_work(so, split_path=cfg.get("aabc_eval_split"),
                          only_subjects=cfg.get("subjects"), n_sessions=cfg.get("n_sessions"))

    items: list[dict] = []
    for w in works:
        items.append(
            {
                "key": w.sub,
                "deliverables": [{"kind": "eval_arrow", "space": sp, "split": w.split} for sp in spaces],
                "sources": {
                    "t1": {"name": STRUCT_ZIP, "uri": struct_zip_uri(w.session)},
                    "aux": [{"name": FUNC_ZIP, "uri": func_zip_uri(w.session)}],
                    "runs": [],
                    "session": w.session,
                    "meta": subject_meta(w.sub, cfg),
                },
            }
        )
    return items
