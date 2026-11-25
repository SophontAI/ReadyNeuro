"""The common driver: discover -> resume -> stream(download, preprocess, finalize, ship).

Dataset-agnostic. It looks the dataset up in `wrappers.CORES`, calls its ``discover``, INVOKES the
named preprocessing script (``preprocess/<preproc>.py``) AS A SUBPROCESS CLI, then runs the shared
`finalize_pretrain` / `finalize_eval` and ships the result. There is deliberately NO retry: ``_fetch``
makes one attempt, and a dropped subject is redrawn by the next run's resume.

The core<->driver contract is plain JSON (no shared type module): ``discover`` returns work items

    {"key": str,
     "deliverables": [{"kind": "pretrain_tar"|"eval_arrow", "space": str, "split": str}, ...],
     "sources": {...}}          # opaque to the driver except the URIs it downloads

which the driver caches (with a signature) as ``subjects.json`` and narrows to the not-yet-done
subset before preprocessing anything.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import (FIRST_COMPLETED, Future, ProcessPoolExecutor,
                                ThreadPoolExecutor, wait)
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

from readyneuro.builder import wrappers

log = logging.getLogger("readyneuro.builder.driver")

PRETRAIN = "pretrain_tar"
EVAL = "eval_arrow"

#: The self-contained preprocessing scripts, at the repo root beside ``src/``. The driver SHELLS OUT
#: to one rather than importing it, so a script keeps its own interpreter/deps and stays liftable --
#: which is also why they are located by PATH, walking up from this file.
PREPROC_DIR = Path(__file__).resolve().parents[3] / "preprocess"

#: Config keys every ``discover`` may read -- what the manifest cache is keyed on. A dataset with
#: extra discovery-affecting keys declares them in its `wrappers.Core`.
BASE_DISCOVERY_KEYS = ("dataset", "subjects", "n_subjects", "n_sessions", "spaces", "split")

MANIFEST = "subjects.json"


def preproc_script(core) -> Path:
    """The path of a dataset's preprocessing script, checked. Missing means the package was
    installed away from the repo's ``preprocess/`` directory (or a `wrappers.Core` names a script
    that does not exist) -- fail here, loudly, rather than as a bare "No such file" from
    ``subprocess`` after a discovery pass has already run."""
    script = PREPROC_DIR / f"{core.preproc}.py"
    if not script.is_file():
        raise RuntimeError(
            f"preprocessing script not found: {script}. The driver locates `preprocess/` by walking "
            f"up from its own file, so this means the package is installed away from the repo. Run "
            f"from a checkout (an editable install is fine)."
        )
    return script


class CoreFailed(RuntimeError):
    """The core script exited nonzero: a fact about this subject's DATA, not about the machine.
    A data failure drops the run and ships the rest; every other exception drops the subject,
    because a machine failure says nothing about whether the run was usable."""


def _run_preproc(core, item: dict, raw_dir: str, out_bids: str, cfg: dict,
                 stage: str = "all", run_idx=None) -> None:
    """Preprocess one subject by SHELLING OUT to ``preprocess/{core.preproc}.py`` -- the work item +
    config cross as json in a throwaway temp dir (``default=str`` so Paths survive), never as a shared
    import. That keeps each script liftable and contains a segfaulting CLI to the child. A nonzero
    exit raises naming the script + the stderr tail, so the caller's keep-going log carries the real
    error, not a bare returncode.

    ``stage`` addresses one node of the subject's DAG (``anat`` / ``run`` + ``run_idx`` / ``finish``);
    ``all`` is the one-shot path an unstaged core gets."""
    extra: list[str] = []
    if stage != "all":
        extra = ["--stage", stage] + (["--run-idx", str(run_idx)] if run_idx is not None else [])
    with tempfile.TemporaryDirectory() as td:
        ip, cp = str(Path(td) / "item.json"), str(Path(td) / "cfg.json")
        Path(ip).write_text(json.dumps(item, default=str))
        Path(cp).write_text(json.dumps(cfg, default=str))
        proc = subprocess.run(
            [sys.executable, str(preproc_script(core)), raw_dir, out_bids,
             "--item", ip, "--cfg", cp, *extra],
            capture_output=True, text=True)
    where = f"{core.preproc}.py [{stage}{'' if run_idx is None else f' {run_idx}'}]"
    if proc.returncode == 0:
        # A core prints `WARN ... skipping run` and exits 0 when it drops a run, and the sidecar
        # records only the runs that SUCCEEDED -- so nothing else can reconstruct what was lost.
        if proc.stdout and proc.stdout.strip():
            log.warning("%s exited 0 but reported: %s", where, proc.stdout.strip()[-2000:])
        return
    if proc.returncode < 0:
        # Killed by a SIGNAL (the OOM killer picks the core subprocess, not the pool worker), which
        # says nothing about the data -- a MACHINE failure.
        raise RuntimeError(f"{where} killed by signal {-proc.returncode} -- machine failure, "
                           f"not a verdict on the data:\n{proc.stderr[-2000:]}")
    raise CoreFailed(f"{where} failed ({proc.returncode}):\n{proc.stderr[-4000:]}")


def _run_preflight(core, cfg: dict) -> None:
    """Validate the tooling ONCE per build -- ``{core.preproc}.py --preflight`` -- before anything is
    downloaded, so a misconfigured environment (a missing FreeSurfer/FSL/ANTs binary, no FS license)
    fails the build loudly rather than after every subject has died the same per-run death."""
    with tempfile.TemporaryDirectory() as td:
        cp = str(Path(td) / "cfg.json")
        Path(cp).write_text(json.dumps(cfg, default=str))
        proc = subprocess.run([sys.executable, str(preproc_script(core)), "--preflight",
                               "--cfg", cp], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{core.preproc}.py --preflight failed ({proc.returncode}):\n{proc.stderr[-4000:]}")


# --- pure planning ----------------------------------------------------------
def discovery_sig(cfg: dict, core) -> str:
    """A 16-hex digest of the discovery-affecting config -- the base keys plus the dataset's own.
    The same ``results_dir`` under a different ``n_subjects`` / ``subjects`` is a DIFFERENT
    deliverable set; the cache guards against replaying the wrong one."""
    keys = set(BASE_DISCOVERY_KEYS) | set(core.discovery_keys)
    payload = json.dumps({k: cfg.get(k) for k in sorted(keys)}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_or_discover(cfg: dict, core, results_dir: str) -> list[dict]:
    """Read the cached ``subjects.json`` when its signature matches (resume -- no S3 re-listing),
    else discover and cache it atomically. A cache built from a DIFFERENT discovery config against a
    populated ``results_dir`` fails loudly rather than silently mixing two deliverable sets."""
    path = Path(results_dir) / MANIFEST
    sig = discovery_sig(cfg, core)
    if path.exists():
        doc = json.loads(path.read_text())
        if doc.get("sig") == sig:
            return doc["items"]
        raise RuntimeError(
            f"{path} was built from a different discovery config (expected sig={sig}). Use a "
            f"per-batch results_dir, or delete {MANIFEST} to re-discover."
        )
    items = core.discover(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")  # atomic: never leave a partial manifest a later run trusts
    tmp.write_text(json.dumps({"sig": sig, "items": items}))
    os.replace(tmp, path)
    return items


def deliverable_uri(cfg: dict, key: str, d: dict) -> str:
    """The remote uri a deliverable ships to. Resume checks this uri's ``.done`` sentinel."""
    from readyneuro.tools import shards
    if d["kind"] == PRETRAIN:
        return shards.tar_uri(cfg["pretrain_root"], d["split"], d["space"], key)
    from readyneuro.tools import eval_arrow
    return eval_arrow.func_uri(cfg["eval_root"], d["space"], d["split"], key)


def pending(cfg: dict, key: str, deliverables: list[dict], done) -> list[dict]:
    """The deliverables NOT already shipped. Pretrain: ``done(uri)`` tests the tar's remote ``.done``
    sentinel (never bare existence -- a streamed object has no atomic rename). Eval: BOTH the func AND
    anat sentinels (`eval_arrow.subject_eval_exists`) -- `finalize_eval` writes them separately, so a
    crash between could leave func.done without anat, which the tar-style single-uri check would read
    as shipped."""
    def shipped(d: dict) -> bool:
        if d["kind"] == EVAL:
            from readyneuro.tools import eval_arrow
            return eval_arrow.subject_eval_exists(cfg["eval_root"], d["space"], d["split"], key,
                                                  storage_options=cfg.get("out_storage_options"))
        return done(deliverable_uri(cfg, key, d))

    return [d for d in deliverables if not shipped(d)]


# --- execution: download batch -> (core -> finalize) pool -> evict, prefetching ------
def _fetch(uri: str, dest: str, storage_options: dict | None) -> None:
    """Stage one raw input to ``dest`` -- a streamed block read for an ``s3://`` uri, else a local copy.

    A streaming ``fs.open().read()`` in large blocks measures ~40% faster than ``fs.get()`` for the
    ~1 GB BOLD objects a subject is mostly made of (124 vs 80 MB/s at 8 concurrent streams),
    and downloads are the whole build's critical path (~20 GB/subject for HCP). Byte-identical: both
    copy the full object, and a partial write leaves no ``.done`` so a resume redraws the subject."""
    from readyneuro.io import is_s3, s3_filesystem
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    if not is_s3(uri):
        shutil.copyfile(uri, dest)
        return
    block = 16 * 2**20
    fs = s3_filesystem(storage_options)
    with fs.open(uri, "rb", block_size=block) as src, open(dest, "wb") as out:
        shutil.copyfileobj(src, out, block)


def download_item(item: dict, scratch: str, cfg: dict) -> None:
    """Stage a work item's raw inputs (``sources.t1`` + each ``sources.runs`` entry, plus any
    ``sources.aux`` extras) into ``{scratch}/{key}/raw/{name}`` -- the layout the core ``process``
    reads. Dataset-agnostic: the driver only downloads the URIs ``discover`` named, with the input
    creds from ``cfg``. ``aux`` is a flat ``[{name, uri}, ...]`` list for cores that need per-space
    anatomical inputs beyond the single T1 (HCP-YA's MNI T1 + shipped acpc<->MNI warps).

    A subject's files are independent, so they are fetched concurrently: a single anonymous S3
    stream measures ~20 MB/s, which made downloads the build's critical path (~19 h per 1000
    subjects against ~16.5 h of compute). ``download_workers`` bounds it."""
    so = cfg.get("input_storage_options")
    raw = Path(scratch) / item["key"] / "raw"
    src = item["sources"]
    jobs = [(src["t1"]["uri"], raw / src["t1"]["name"])]
    jobs += [(a["uri"], raw / a["name"]) for a in src.get("aux", [])]
    jobs += [(r["uri"], raw / r["name"]) for r in src["runs"]]

    workers = min(max(1, int(cfg.get("download_workers", 8))), len(jobs))
    if workers == 1:
        for uri, dest in jobs:
            _fetch(uri, dest, so)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_fetch, uri, dest, so) for uri, dest in jobs]
        for future in futures:
            # The first failure propagates; the caller drops that subject, and a partial staging is
            # harmless because it wrote no `.done`.
            future.result()


def _done(cfg: dict):
    """A resume predicate: True iff the deliverable's remote ``.done`` sentinel exists (never bare
    existence -- a streamed object has no atomic rename)."""
    from readyneuro.io import path_exists
    from readyneuro.tools.shards import done_uri
    so = cfg.get("out_storage_options")
    return lambda uri: path_exists(done_uri(uri), so)


def shard(items: list[dict], cfg: dict) -> list[dict]:
    """Keep only the work items belonging to this run's ``shard_index`` of ``shard_count`` -- what
    makes CONCURRENT runs over one manifest safe, since `.done` is read once at startup and N runs
    launched together would otherwise each build every subject.

    Partition is by a stable hash of the KEY (not list position, and not salted `hash()`), so a
    resumed shard picks up its own leftovers. Neither key is in ``BASE_DISCOVERY_KEYS``, so every
    shard shares one discovery signature, one ``results_dir`` and one cached manifest."""
    n = int(cfg.get("shard_count", 1) or 1)
    if n <= 1:
        return items
    i = int(cfg.get("shard_index", 0) or 0)
    if not 0 <= i < n:
        raise ValueError(f"shard_index={i} out of range for shard_count={n}")
    return [it for it in items
            if int.from_bytes(hashlib.md5(it["key"].encode()).digest()[:8], "big") % n == i]


def run(cfg: dict) -> tuple[int, int]:
    """Discover -> resume -> stream. Returns ``(shipped, failed)``, which `main` turns into
    the process exit code; ``(0, 0)`` means there was nothing left to do, which is success.

    THERE IS NO FORCE-REBUILD FLAG: the remote ``.done`` sentinels filter the work list before any
    work is planned, so re-running a finished build is a no-op and rebuilding means pointing at a new
    output root (or deleting sentinels by hand)."""
    core = wrappers.load_core(cfg["dataset"])
    _run_preflight(core, cfg)
    items = load_or_discover(cfg, core, cfg["results_dir"])
    if cfg.get("subjects") and not items:
        # A `--set subjects=[...]` that matched nothing is almost always the YAML id-format trap, and
        # a silent "0/0 work items" hides it: `yaml.safe_load` parses a bare number to int (999001) and
        # a LEADING-ZERO id to OCTAL (0050004 -> 20484), neither matching the string ids. QUOTE them.
        log.warning("subjects=%r matched 0 work items -- ids must be QUOTED in YAML so they stay strings: "
                    "--set \"subjects=['999001']\" (a bare number parses to int; a leading-zero id to octal).",
                    cfg["subjects"])

    # Shard BEFORE the resume check, so each concurrent run pays the remote `.done` lookups for its
    # own slice only.
    scoped = shard(items, cfg)
    done = _done(cfg)
    # One REMOTE sentinel probe per subject: ~7.4 min serially for a 1000-subject batch, with the
    # node idle for all of it. The probes are independent and pure, so a pool cannot change the
    # selection (`map` yields in order).
    resume_workers = max(1, int(cfg.get("resume_workers", 16)))
    if resume_workers == 1 or len(scoped) < resume_workers:
        keep = [bool(pending(cfg, it["key"], it["deliverables"], done)) for it in scoped]
    else:
        with ThreadPoolExecutor(max_workers=resume_workers) as pool:
            keep = list(pool.map(
                lambda it: bool(pending(cfg, it["key"], it["deliverables"], done)), scoped))
    todo = [it for it, unshipped in zip(scoped, keep) if unshipped]
    if int(cfg.get("shard_count", 1) or 1) > 1:
        log.info("shard %s/%s: %d of %d manifest items",
                 cfg.get("shard_index", 0), cfg["shard_count"], len(scoped), len(items))
    log.info("%s: %d/%d work items to build (rest already shipped)", cfg["dataset"], len(todo), len(scoped))
    if not todo:
        return (0, 0)

    return _pipeline(cfg, core, todo)


class _Inline:
    """``workers <= 1`` executes tasks in THIS process rather than a pool -- a pool worker is a
    separate interpreter, so anything the caller patched would not exist there."""

    def submit(self, fn, *args):
        fut: Future = Future()
        try:
            fut.set_result(fn(*args))
        except BaseException as exc:  # noqa: BLE001 -- mirrored onto the future, as a pool would
            fut.set_exception(exc)
        return fut

    def shutdown(self, wait=True):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def _stage_task(cfg: dict, item: dict, scratch: str, stage: str, run_idx) -> list[str]:
    """One node of a subject's DAG, run in a pool worker. ``finish`` also finalizes + ships, so the
    remote `.done` stays the single completion signal."""
    core = wrappers.load_core(cfg["dataset"])
    base = Path(scratch) / item["key"]
    raw, out_bids = str(base / "raw"), str(base / "bids")
    _run_preproc(core, item, raw, out_bids, cfg, stage=stage, run_idx=run_idx)
    if stage not in ("finish", "all"):  # `all` is a whole subject, so it finalizes too
        return []
    from readyneuro.builder import finalize_eval, finalize_pretrain
    produced = []
    for d in item["deliverables"]:
        if d["kind"] == PRETRAIN:
            uri = finalize_pretrain.finalize_pretrain(out_bids, cfg, split=d["split"], space=d["space"])
            if uri:
                produced.append(uri)
        elif d["kind"] == EVAL:
            produced += finalize_eval.finalize_eval(out_bids, cfg, split=d["split"], space=d["space"],
                                                    core=core)
    return produced


def _pipeline(cfg: dict, core, todo: list[dict]) -> tuple[int, int]:
    """Stream subjects through download -> compute with NO barriers between them. Returns
    ``(shipped, failed)``.

    A subject's work is a DAG -- ``anat -> run(k) x N -> finish`` for a staged core, else a single
    whole-subject task. The atomic unit is a RUN (~12 min) rather than a subject (up to ~4 h):
    under a whole-subject barrier a capped 20-run subject stalled the batch, at 17 of 20 workers
    idle.

    Downloads run on their own thread pool concurrently with compute, bounded by ``stage_ahead``
    subjects in flight -- that bound IS the disk cap (staged subjects x ~1.4 GB).

    Failure granularity follows the DAG: a failed ``anat`` drops its subject (its runs never run), a
    failed ``run`` drops only that run and the subject still ships with the rest, and a failed
    ``finish`` drops the subject. Nothing writes `.done`, so a resume redraws exactly what was lost."""
    scratch = cfg["scratch"]
    workers = max(1, int(cfg.get("workers", 1)))
    dl_workers = max(1, int(cfg.get("batch_workers", 2)))
    stage_ahead = max(1, int(cfg.get("stage_ahead", 2 * workers)))
    if stage_ahead < workers:
        log.info("stage_ahead=%d is below workers=%d -- staging, not cores, will be the limit",
                 stage_ahead, workers)

    staged = core.staged

    queue, active = list(todo), {}
    dl_futs: dict = {}
    cpu_futs: dict = {}
    shipped = failed = 0

    def evict(key: str) -> None:
        shutil.rmtree(Path(scratch) / key, ignore_errors=True)

    def new_pool():
        return (_Inline() if workers <= 1 else
                ProcessPoolExecutor(max_workers=workers, initializer=_init_pool_worker,
                                    initargs=(log.getEffectiveLevel(),)))

    def drop(key: str, why: str, exc=None) -> None:
        nonlocal failed
        if key not in active:  # already dropped (e.g. one machine failure, then a pool break)
            return
        log.warning("%s %s (%s) -- no .done, resume rebuilds it", key, why, exc)
        failed += 1
        evict(key)
        active.pop(key, None)

    pool = new_pool()
    try:
        with ThreadPoolExecutor(max_workers=dl_workers) as dl:
            while queue or dl_futs or cpu_futs:
                try:
                    while queue and len(dl_futs) + len(active) < stage_ahead:
                        it = queue.pop(0)
                        dl_futs[dl.submit(download_item, it, scratch, cfg)] = it

                    done_set, _ = wait(set(dl_futs) | set(cpu_futs), return_when=FIRST_COMPLETED)
                    for fut in done_set:
                        if fut in dl_futs:
                            it = dl_futs.pop(fut)
                            key = it["key"]
                            if fut.exception() is not None:
                                # NOT in `active` yet, so `drop()` (which guards on `key in active`)
                                # would swallow both the log and the count -- handle it inline.
                                log.warning("%s download failed (%s) -- no .done, resume rebuilds it",
                                            key, fut.exception())
                                failed += 1
                                evict(key)
                                continue
                            active[key] = {"item": it, "pending": set()}
                            first = "anat" if staged else "all"
                            cpu_futs[pool.submit(_stage_task, cfg, it, scratch, first, None)] = \
                                (key, first, None)
                            continue

                        key, kind, idx = cpu_futs.pop(fut)
                        state = active.get(key)
                        if state is None:  # its subject was already dropped (e.g. a broken pool)
                            continue
                        exc = fut.exception()
                        if kind == "anat":
                            if exc is not None:
                                drop(key, "anat failed", exc)
                                continue
                            # `runs` is what the driver PRE-STAGED, `eval_runs` what a core STREAMS
                            # itself (NSD: hundreds of runs, hundreds of GB). `download_item` reads
                            # `runs` alone, so honouring `eval_runs` here adds no downloads.
                            src = state["item"]["sources"]
                            runs = [r["idx"] for r in (src.get("eval_runs") or src["runs"])]
                            state["pending"] = set(runs)
                            for i in runs:
                                cpu_futs[pool.submit(_stage_task, cfg, state["item"], scratch, "run", i)] = \
                                    (key, "run", i)
                            if not runs:  # a subject with no runs still needs its sidecar written
                                cpu_futs[pool.submit(_stage_task, cfg, state["item"], scratch, "finish", None)] = \
                                    (key, "finish", None)
                        elif kind == "run":
                            state["pending"].discard(idx)
                            if exc is not None and not isinstance(exc, CoreFailed):
                                # A MACHINE failure: we do not know whether this run was usable, so
                                # finishing would write `.done` on a silently incomplete subject.
                                drop(key, f"run {idx} failed on infrastructure", exc)
                                continue
                            if exc is not None:  # keep-going PER RUN: ship the subject's good runs
                                log.warning("run %s[%s] unusable (%s) -- dropping the run", key, idx, exc)
                            if not state["pending"]:
                                cpu_futs[pool.submit(_stage_task, cfg, state["item"], scratch, "finish", None)] = \
                                    (key, "finish", None)
                        else:  # "finish" or the unstaged "all"
                            if exc is not None:
                                drop(key, "failed", exc)
                                continue
                            produced = fut.result()
                            if not produced:
                                # Nothing shipped, so there is no `.done`: counting it shipped would
                                # hide a subject every future run re-downloads forever.
                                drop(key, "produced no deliverable", None)
                                continue
                            shipped += 1
                            log.info("%s -> %d deliverable(s) [%d shipped, %d failed, %d in flight]",
                                     key, len(produced), shipped, failed, len(active) - 1)
                            evict(key)
                            active.pop(key, None)
                except BrokenProcessPool as exc:
                    # One abruptly-killed worker (usually an OOM) otherwise poisons `submit` and
                    # `result` forever. Drop ALL of `active`, not just the subjects in `cpu_futs`:
                    # a subject whose `submit` itself raised has no future yet, and a leaked one
                    # holds its `stage_ahead` slot until the loop spins on `wait(set())` -- a hang.
                    log.error("process pool broke (%s) -- dropping %d in-flight subject(s) and "
                              "restarting the pool", exc, len(active))
                    for key in list(active):
                        drop(key, "in flight when the pool broke", exc)
                    cpu_futs.clear()
                    pool.shutdown(wait=False)
                    pool = new_pool()
    finally:
        for key in list(active):  # never strand scratch, whatever took us out of the loop
            evict(key)
        # ...and the subjects still being STAGED: leaving the `with` block waits for every in-flight
        # fetch, which writes a full subject the `active` sweep above cannot see (~1.4 GB each, and
        # one pathological OpenNeuro subject reaches ~143 GB).
        for it in dl_futs.values():
            evict(it["key"])
        pool.shutdown(wait=False)

    log.info("%s: %d shipped, %d failed", cfg["dataset"], shipped, failed)
    return shipped, failed


#: The driver's log format, shared by `main` and every pool worker so a run reads as one stream.
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def _init_pool_worker(level: int) -> None:
    """Give a pool worker the parent's logging config. A worker is a FRESH interpreter under
    forkserver/spawn, so it never ran `main`'s `basicConfig` and its root logger falls back to
    `lastResort`, which DROPS INFO -- every per-subject success line vanished."""
    logging.basicConfig(level=level, format=LOG_FORMAT)


# --- CLI entrypoint ---------------------------------------------------------
def _print_plan(cfg: dict, config_path: str) -> None:
    """Print the resolved build parameters and return -- ``--plan``. The values that bite are the
    ones a `--set` silently fails to change: the core SCRIPT, the output roots, and the discovery
    signature (a mismatch against a cached ``subjects.json`` aborts the run). Does NO network I/O,
    so it stays safe on a login node."""
    core = wrappers.load_core(cfg["dataset"])
    script = PREPROC_DIR / f"{core.preproc}.py"
    rows = [
        ("config", config_path),
        ("dataset", cfg["dataset"]),
        ("preproc script", f"{script} {'(FOUND)' if script.is_file() else '(MISSING!)'}"),
        ("staged (run-level DAG)", str(core.staged)),
        ("spaces", str(cfg.get("spaces"))),
        ("pretrain_root", str(cfg.get("pretrain_root", "(none)"))),
        ("eval_root", str(cfg.get("eval_root", "(none)"))),
        ("results_dir", str(cfg.get("results_dir"))),
        ("scratch", str(cfg.get("scratch"))),
        ("workers / stage_ahead", f"{cfg.get('workers', 1)} / {cfg.get('stage_ahead', '(derived)')}"),
        ("shard", f"{cfg.get('shard_index', 0)} of {cfg.get('shard_count', 1)}"),
        ("discovery sig", discovery_sig(cfg, core)),
        ("discovery keys", str(sorted(set(BASE_DISCOVERY_KEYS) | set(core.discovery_keys)))),
    ]
    width = max(len(k) for k, _ in rows)
    print("\n".join(f"{k.rjust(width)} : {v}" for k, v in rows))
    manifest = Path(cfg["results_dir"]) / MANIFEST if cfg.get("results_dir") else None
    if manifest and manifest.exists():
        cached = json.loads(manifest.read_text()).get("sig")
        state = "MATCHES" if cached == discovery_sig(cfg, core) else "STALE -- run would abort"
        print(f"{'cached manifest'.rjust(width)} : {manifest} sig={cached} ({state})")



def resolve_storage_options(cfg: dict) -> None:
    """Translate the config's ``out_profile`` / ``input_profile`` into the s3fs storage-option dicts
    the finalizer and the downloads read; the profile's endpoint comes from ``~/.aws/config``, which
    s3fs does not auto-read. A public input sets ``input_storage_options`` directly
    (``{anon: true}``). Idempotent."""
    from readyneuro.io import output_storage_options, profile_storage_options
    if cfg.get("out_storage_options") is None and cfg.get("out_profile"):
        cfg["out_storage_options"] = output_storage_options(cfg["out_profile"])
    if cfg.get("input_storage_options") is None and cfg.get("input_profile"):
        cfg["input_storage_options"] = profile_storage_options(cfg["input_profile"])


def main(argv=None) -> None:
    """Run the build for one dataset config: ``python -m readyneuro.builder.driver --config
    dataset_configs/<ds>.yaml``. ``--set key=value`` overrides a config key (value parsed as YAML),
    repeatable -- for per-run scoping / throwaway destinations without editing the file."""
    import argparse

    import yaml

    ap = argparse.ArgumentParser(description="run the build driver for one dataset config")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config key (VALUE parsed as YAML); repeatable")
    ap.add_argument("--plan", action="store_true",
                    help="print what this invocation WOULD build (dataset, core script, roots, "
                         "concurrency, shard, discovery signature) and exit 0 without running")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    resolve_storage_options(cfg)

    if args.plan:
        _print_plan(cfg, args.config)
        return

    shipped, failed = run(cfg)
    # Exit NONZERO only when there was work and NONE of it shipped. Not "any failure": at scale a
    # healthy build always drops a few subjects, and failing on that trains the operator to ignore
    # the exit code. `(0, 0)` -- nothing left to do -- stays success.
    if failed and not shipped:
        log.error("%s: every one of the %d work items failed -- exiting nonzero", cfg["dataset"], failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
