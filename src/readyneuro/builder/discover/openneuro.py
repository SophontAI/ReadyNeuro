"""OpenNeuro discovery: sample one disjoint batch of subjects, emit one pretrain deliverable each.

Pretrain-only and ``native``-only -- OpenNeuro is raw BIDS, so there is no vendor template warp and
no eval product."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from readyneuro import openneuro as sources

#: Config keys beyond the driver's base set that change WHICH subjects/runs a manifest contains, so
#: they must key its cache. ``openneuro_batch`` is here, which is why each batch needs its own
#: ``results_dir``; ``shard_index``/``shard_count`` deliberately are NOT (concurrent shards of ONE
#: batch share one manifest).
DISCOVERY_KEYS = (
    "openneuro_seed",
    "openneuro_batch",
    "openneuro_index",
    "min_frames",
    "min_fov_mm",
    "max_runs_per_subject",
    "filter_species",
)


def _staged_name(stem: str, uri: str) -> str:
    """``stem`` plus the SOURCE's real NIfTI extension.

    The staged filename is the contract between discovery and the preprocessing script, which opens
    ``{raw_dir}/{name}`` verbatim, so it must not claim ``.gz`` for an uncompressed source. OpenNeuro
    is raw BIDS and ships both: hardcoding ``.nii.gz`` made nibabel try to gunzip plain NIfTI and
    kill the subject. Measured on a 1000-subject manifest: 91 subjects affected (9.1%)."""
    return f"{stem}.nii.gz" if uri.endswith(".gz") else f"{stem}.nii"


def discover(cfg: dict) -> list[dict]:
    # `discover_workers` is deliberately NOT in DISCOVERY_KEYS: it changes only how fast the scan
    # runs, never its result, so it must not perturb the manifest signature.
    workers = max(1, int(cfg.get("discover_workers", 16)))
    subjects = cfg.get("subjects") or sources.sample_batch(
        sources.build_index(cfg),
        int(cfg["openneuro_seed"]),
        int(cfg["openneuro_batch"]),
        int(cfg.get("n_subjects", 100)),
        min_frames=cfg.get("min_frames"),
        min_fov_mm=cfg.get("min_fov_mm"),
        scan_workers=workers,
    )
    split = cfg.get("split", "train")

    def resolve(subject):
        """Re-resolve an accepted subject WITH TRs + the run cap. Same swallow-vs-propagate rule as
        the sampler: a deterministic denial drops the subject, anything transient propagates."""
        accession, label = sources.parse_key(subject)
        try:
            return sources.subject_files(
                accession,
                label,
                with_tr=True,
                max_runs=cfg.get("max_runs_per_subject"),
                min_frames=cfg.get("min_frames"),
                min_fov_mm=cfg.get("min_fov_mm"),
            )
        except (PermissionError, FileNotFoundError):
            return None

    # Concurrent, but `map` yields in input order, so `items` keeps the sampler's ordering.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        resolved = list(pool.map(resolve, subjects))

    items = []
    for subject, files in zip(subjects, resolved):
        accession, label = sources.parse_key(subject)
        if files is None or not files["t1"] or not files["runs"]:
            continue
        runs = [
            {
                "idx": i,
                "run": run,
                "name": _staged_name(f"run-{i:02d}_bold", uri),
                "uri": uri,
                "tr_acq": files["tr"].get(run),
            }
            for i, (run, uri) in enumerate(files["runs"].items())
        ]
        items.append(
            {
                "key": subject,
                "deliverables": [
                    {"kind": "pretrain_tar", "space": "native", "split": split}
                ],
                "sources": {
                    "t1": {"name": _staged_name("T1w", files["t1"]), "uri": files["t1"]},
                    "runs": runs,
                    "meta": {
                        "accession": accession,
                        **sources.participants_meta(accession, label),
                    },
                },
            }
        )
    return items
