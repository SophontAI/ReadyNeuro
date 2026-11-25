"""The dataset table: what the driver needs to know about each dataset, in one place.

Each entry names the PREPROCESSING SCRIPT the driver shells out to (``preprocess/<preproc>.py``,
never imported -- see `driver`), the ``discover`` that enumerates its work items, whether the driver
may pipeline it at RUN granularity, and its eval clip planner if it has one.

The ``*_native`` twins are separate datasets, not a flag: the vendor build reads a shipped template
warp, the from-raw build runs the whole geometry itself, and they write into the same eval store side
by side. That pairing is the paper's native-vs-vendor comparison.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from readyneuro.builder.discover import (
    aabc,
    aabc_native,
    abide,
    abide_native,
    adhd200,
    adhd200_native,
    hcp_ya,
    hcp_ya_native,
    nsd,
    nsd_native,
    openneuro,
    ppmi,
    ppmi_native,
)


@dataclass(frozen=True)
class Core:
    """One dataset's build wiring."""

    #: ``preprocess/{preproc}.py`` -- the standalone script the driver invokes as a subprocess CLI.
    preproc: str
    #: ``cfg -> [work item]``. The core<->driver contract is plain JSON; see `driver`.
    discover: Callable[[dict], list[dict]]
    #: The script exposes ``--stage anat|run|finish``, so the driver can schedule a subject's runs as
    #: independent tasks. Worth it only for a many-run subject: it turns the straggler unit from a
    #: whole subject (up to ~4 h) into one run (~12 min). A dataset whose BOLD rides in ``aux``
    #: rather than ``sources.runs`` MUST stay unstaged -- the run DAG would find no runs to schedule.
    staged: bool = False
    #: ``(sidecar, run, cfg) -> {clips, n_frames, extra}``, `finalize_eval`'s per-dataset seam. None
    #: means the default planner: one unlabelled whole-window REST clip.
    eval_clips: Callable | None = None
    #: Config keys beyond `driver.BASE_DISCOVERY_KEYS` that change what ``discover`` returns, and so
    #: must key the cached manifest.
    discovery_keys: tuple[str, ...] = field(default_factory=tuple)


CORES: dict[str, Core] = {
    # --- pretraining corpora -------------------------------------------------
    "openneuro": Core("openneuro", openneuro.discover, staged=True,
                      discovery_keys=openneuro.DISCOVERY_KEYS),
    # HCP-YA is dual-product (pretraining shards + the task21 eval) in one pass, and up to 18 runs
    # per subject -- the worst straggler in the corpus if it were not staged.
    "hcp_ya": Core("hcp", hcp_ya.discover, staged=True, eval_clips=hcp_ya.eval_clips,
                   discovery_keys=hcp_ya.DISCOVERY_KEYS),
    "hcp_ya_native": Core("openneuro", hcp_ya_native.discover, staged=True,
                          eval_clips=hcp_ya_native.eval_clips,
                          discovery_keys=hcp_ya_native.DISCOVERY_KEYS),

    # --- evaluation stores ---------------------------------------------------
    # NSD carries HUNDREDS of runs per subject, so run-level staging matters more here than anywhere.
    "nsd": Core("nsd", nsd.discover, staged=True, eval_clips=nsd.eval_clips,
                discovery_keys=nsd.DISCOVERY_KEYS),
    "nsd_native": Core("nsd_native", nsd_native.discover, staged=True,
                       eval_clips=nsd_native.eval_clips,
                       discovery_keys=nsd_native.DISCOVERY_KEYS),
    # The REST cohorts are one run per subject, and that run rides in `aux` (AABC) or is streamed by
    # the core (aabc_native), so staging would buy nothing and schedule nothing.
    "aabc": Core("aabc_vendor", aabc.discover, discovery_keys=aabc.DISCOVERY_KEYS),
    "aabc_native": Core("aabc_native", aabc_native.discover, discovery_keys=aabc.DISCOVERY_KEYS),
    "ppmi": Core("fmriprep_dx", ppmi.discover),
    "ppmi_native": Core("openneuro", ppmi_native.discover),
    "abide": Core("fmriprep_dx", abide.discover),
    "abide_native": Core("openneuro", abide_native.discover),
    "adhd200": Core("fmriprep_dx", adhd200.discover),
    "adhd200_native": Core("openneuro", adhd200_native.discover),
}


def load_core(dataset: str) -> Core:
    """The `Core` for a config's ``dataset``."""
    try:
        return CORES[dataset]
    except KeyError:
        raise KeyError(f"unknown dataset {dataset!r}; known: {sorted(CORES)}") from None
