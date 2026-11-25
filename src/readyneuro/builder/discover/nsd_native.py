"""NSD native-from-raw discovery, reusing the vendor cococlip sampling layer.

Same held-out-subject splits and same per-(ses, run) enumeration as ``discover.nsd``, but the inputs
are NSD's RAW BIDS: the raw ``ses-nsdanat`` T1w (staged once) + the raw per-run ``task-nsdcore`` BOLD
(streamed via ``eval_runs``). The acquisition TR is the RAW 1.6 s, not the vendor func1mm's
already-upsampled 1.0 s, so this build's sidecar carries 1.6 and the finalizer resamples to 1.0.

THE RUN-NUMBER NAMESPACE. The cococlip metadata's ``run`` counts ALL functional runs in a session
(NSD's later sessions open with ``task-rest``, so the 12 nsdcore runs are numbered 2..13 there); the
raw BIDS numbers each task independently, so ``task-nsdcore_run-NN`` is a different namespace. The
map is ``nsdcore_idx = run_num - min(run_num over that sub/ses) + 1``, derived per (sub, ses). No
function of ``ses`` alone is correct: subj02/subj07 have offset 0 in ses 31-35 and 1 in 21-30.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from functools import lru_cache

from readyneuro.builder.discover import nsd as _vendor
from readyneuro import nsd

#: The RAW BIDS root (public-anon), and the RAW acquisition TR (NSD acquires nsdcore at 1.6 s; the vendor
#: func1mm timeseries are upsampled to 1.0 s, but from raw we keep the true 1.6 s and let the finalizer
#: resample down to the model's target 1.0 s).
RAW_ROOT = f"{nsd.NSD_S3}/nsddata_rawdata"
RAW_TR = 1.6

#: `eval_max_runs` changes which runs a manifest contains -- it must key the cache.
DISCOVERY_KEYS = ("eval_max_runs",)

eval_clips = _vendor.eval_clips  # the same cococlip planner (it keys on ses/run_num)

#: Pulls the nsdcore index out of a raw BIDS BOLD basename (``..._task-nsdcore_run-07_bold.nii.gz``).
#: Anchored on ``_bold.nii.gz`` so the co-located ``_sbref.nii.gz`` / ``_events.tsv`` / physio siblings
#: (a session's ``func/`` listing is ~96 objects for 12 runs) can never be mistaken for a BOLD run.
_NSDCORE_BOLD_RE = re.compile(r"_task-nsdcore_run-(\d+)_bold\.nii\.gz$")


def _bids_sub(sub: str) -> str:
    """NSD's cococlip metadata / func1mm derivatives key subjects as ``subj01``; the RAW BIDS
    (``nsddata_rawdata``) uses the BIDS form ``sub-01``. The store key stays ``subj01`` -- only the raw
    S3 paths need this mapping."""
    return f"sub-{sub.removeprefix('subj')}"


def raw_t1_uri(sub: str) -> str:
    """The raw structural T1w (first anat run) from the dedicated anat session."""
    b = _bids_sub(sub)
    return f"{RAW_ROOT}/{b}/ses-nsdanat/anat/{b}_ses-nsdanat_run-01_T1w.nii.gz"


def raw_bold_uri(sub: str, ses: int, nsdcore_run: int) -> str:
    """Path of one raw ``task-nsdcore`` BOLD run. ``nsdcore_run`` is the RAW BIDS index (1..12),
    NOT the cococlip metadata's ``run`` -- see the module docstring."""
    b = _bids_sub(sub)
    return (f"{RAW_ROOT}/{b}/ses-nsd{ses:02d}/func/"
            f"{b}_ses-nsd{ses:02d}_task-nsdcore_run-{nsdcore_run:02d}_bold.nii.gz")


def nsdcore_offsets(runs: Iterable[tuple[int, int, str]]) -> dict[int, int]:
    """``ses -> offset`` for one subject's ``runs_by_subject`` list (``[(ses, run, split), ...]``).

    The offset is ``min(run_num in that session) - 1``: the count of non-nsdcore functional runs the
    metadata numbered ahead of the session's first nsdcore run (1 where the session opens with
    ``task-rest``, 0 otherwise). Computed ONCE per (sub, ses) from the metadata's own numbering -- never
    guessed per run and never derived from ``ses`` (see the module docstring's subj02/subj07 trap).

    Pass the subject's FULL run list, not an ``eval_max_runs``-capped one: the offset is a property of
    the session's acquisition, not of how many runs this build happens to keep."""
    lowest: dict[int, int] = {}
    for ses, run, *_ in runs:
        ses, run = int(ses), int(run)
        lowest[ses] = run if ses not in lowest else min(lowest[ses], run)
    return {ses: mn - 1 for ses, mn in lowest.items()}


@lru_cache(maxsize=None)
def _list_nsdcore_runs(sub: str, ses: int) -> frozenset[int]:
    """The ``task-nsdcore`` BOLD run indices actually PRESENT in one raw session, from ONE listing.

    A batched ``ls`` of the session's ``func/`` dir costs a single request for all 12 runs, where a
    HEAD per run would cost 12 -- ~2900 HEADs over a full build. Memoized because a session is listed
    once per subject and the raw bucket is immutable. ``anon`` per `nsd.NSD_STORAGE_OPTIONS`: the
    raw NSD bucket is public, so this never depends on the build's output credentials."""
    import s3fs

    fs = s3fs.S3FileSystem(**nsd.NSD_STORAGE_OPTIONS)
    d = f"{RAW_ROOT}/{_bids_sub(sub)}/ses-nsd{ses:02d}/func"
    found = set()
    for p in fs.ls(d, detail=False):
        m = _NSDCORE_BOLD_RE.search(p.rsplit("/", 1)[-1])
        if m:
            found.add(int(m.group(1)))
    return frozenset(found)


def _assert_raw_runs_exist(sub: str, plan: list[tuple[int, int, int, bool]]) -> None:
    """FAIL LOUD at discovery on a derived URI with no file, rather than let the driver's per-run
    keep-going path swallow it -- a missing raw file here is a BUG in the mapping, never a data fact.
    ``plan`` is ``[(ses, run_num, nsdcore_idx, session_is_whole), ...]``.

    The second check exists because existence alone is not sufficient: anchoring on the session's
    MINIMUM run_num assumes the metadata covers that session's nsdcore runs whole, and a dropped
    FIRST run would shift every later run by one onto files that all exist. So an untruncated
    session must have as many metadata runs as the raw session has nsdcore BOLDs (today 2928 ==
    244 x 12)."""
    by_ses: dict[int, list[tuple[int, int]]] = {}
    whole: dict[int, bool] = {}
    for ses, run_num, idx, is_whole in plan:
        by_ses.setdefault(int(ses), []).append((int(run_num), int(idx)))
        whole[int(ses)] = bool(is_whole)
    missing, uncovered = [], []
    for ses in sorted(by_ses):
        present = _list_nsdcore_runs(sub, ses)  # ONE listing serves the session's ~12 runs
        for run_num, idx in by_ses[ses]:
            if idx not in present:
                missing.append(f"ses-nsd{ses:02d} run_num={run_num} -> task-nsdcore_run-{idx:02d}")
        if whole[ses] and len(by_ses[ses]) != len(present):
            uncovered.append(
                f"ses-nsd{ses:02d}: {len(by_ses[ses])} metadata runs vs {len(present)} raw nsdcore BOLDs")
    if missing:
        raise FileNotFoundError(
            f"{sub}: {len(missing)} cococlip run(s) have no raw task-nsdcore BOLD -- the metadata "
            f"run_num -> nsdcore index mapping is wrong for this subject, NOT a missing-data fact. "
            f"Offending: {missing[:10]}{' ...' if len(missing) > 10 else ''}"
        )
    if uncovered:
        raise AssertionError(
            f"{sub}: {len(uncovered)} session(s) where the cococlip metadata does not cover the raw "
            f"nsdcore runs one-for-one. The per-session minimum run_num can then be the WRONG anchor "
            f"(a dropped first run shifts every later run onto an existing but wrong file -- the v6 "
            f"failure mode again). Re-derive the offset before building. "
            f"Offending: {uncovered[:10]}{' ...' if len(uncovered) > 10 else ''}"
        )


def discover(cfg: dict) -> list[dict]:
    """One ``native``-only eval item per subject: the raw T1 staged in ``sources.t1`` + the raw
    per-run BOLD URIs in ``sources.eval_runs`` (streamed by the core, never pre-staged).
    ``eval_max_runs`` bounds a subject's runs -- uncapped is hundreds of ~235 MB raw runs.

    ``run_num`` in each emitted entry stays the METADATA number -- the cococlip planner (`eval_clips`)
    looks its trial onsets up by ``(ses, run_num)``, so it must NOT be rewritten. Only the ``uri``'s
    nsdcore index is translated (plus the derived index echoed as ``nsdcore_run`` for auditability)."""
    by_sub = nsd.runs_by_subject()
    subs = sorted(by_sub)
    if cfg.get("subjects"):
        subs = [s for s in subs if s in set(cfg["subjects"])]
    if cfg.get("n_subjects"):
        subs = subs[: int(cfg["n_subjects"])]

    items: list[dict] = []
    for sub in subs:
        runs = by_sub[sub]
        split = runs[0][2]  # subject-disjoint: all of a subject's runs share one split
        offsets = nsdcore_offsets(runs)  # from the FULL list, before any cap
        # How many runs each session contributes UNCAPPED -- so the coverage check below can tell a
        # session the cap truncated (expected to be short) from one the metadata itself covers partly
        # (which would invalidate the min-anchored offset).
        full_n: dict[int, int] = {}
        for ses, _, _ in runs:
            full_n[int(ses)] = full_n.get(int(ses), 0) + 1
        if cfg.get("eval_max_runs"):
            runs = runs[: int(cfg["eval_max_runs"])]
        kept_n: dict[int, int] = {}
        for ses, _, _ in runs:
            kept_n[int(ses)] = kept_n.get(int(ses), 0) + 1
        eval_runs, plan = [], []
        for i, (ses, run, _) in enumerate(runs):
            idx = int(run) - offsets[int(ses)]  # metadata run_num -> raw nsdcore index
            plan.append((int(ses), int(run), idx, kept_n[int(ses)] == full_n[int(ses)]))
            eval_runs.append(
                {"idx": i, "run": nsd.run_key(ses, run), "ses": int(ses), "run_num": int(run),
                 "nsdcore_run": idx, "tr_acq": RAW_TR, "uri": raw_bold_uri(sub, ses, idx)}
            )
        _assert_raw_runs_exist(sub, plan)
        items.append(
            {
                "key": sub,
                "deliverables": [{"kind": "eval_arrow", "space": "native", "split": split}],
                "sources": {"t1": {"name": "t1w.nii.gz", "uri": raw_t1_uri(sub)}, "aux": [],
                            "runs": [], "eval_runs": eval_runs, "meta": {}},
            }
        )
    return items
