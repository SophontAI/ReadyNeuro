# Orientation for a coding agent

Read [`ARCHITECTURE.md`](ARCHITECTURE.md) first — it has the data-flow diagram, the four
seams, and a table mapping each method claim in the paper to the file that implements it.

## What is here

One package, `readyneuro`, builds the data and supplies the evaluation data loaders. The
Brainmarks benchmark suite (its own MIT license) the paper evaluates through is an `eval`
dependency pinned to a fork commit. The model code, which trains, validates and evaluates the
model, is in [its own repository](https://github.com/SophontAI/CortexMAE-ReadyNeuro).
[`BUILDING.md`](BUILDING.md) is the build walkthrough.

## What runs without data

The package installs and unit-tests offline, on CPU:

```bash
uv run --extra test pytest tests/
```

Nothing else is usefully executable without more: building data needs FreeSurfer, FSL and ANTs.
Evaluation needs the six benchmark stores, which are not released because the cohorts' terms
(gated access, data-use agreements or non-commercial share-alike licenses) do not allow us to
redistribute a derived store, so the benchmark numbers cannot be reproduced
without building them under your own data access. For the evaluation data loaders,
from the repository root:

```bash
pip install -e '.[eval]'
```

## Answering questions about the method

Prefer reading the code to reading prose about the code. The three files that carry the
most method per line:

- `preprocess/openneuro.py` — the preprocessing pipeline, end to end,
  in one standalone script. This is the paper's contribution.
- `configs/model.yaml` (model repository) — the entire architecture and recipe. One corpus
  from `configs/data/` is the only thing that varies between arms.
- `src/fmrifm/pretrain_data.py` (model repository) — how a shard becomes a training sample:
  the patch grid, the world-millimetre coordinates, the 96/32 draw.

When a number in the paper is in question, `ARCHITECTURE.md` says where it is implemented.
Several are derived rather than configured — the 7.1e-5 learning rate is
`base_lr × sqrt(batch/256)` — so grep for the rule, not the value.

## Invariants worth knowing before you suggest an edit

- `src/brainmarks/` (here and in the model repository) is a PEP 420 namespace
  overlay and must contain **no `__init__.py`**. Adding one makes plugin discovery find zero
  plugins, and the symptom is an empty registry rather than an error.
- Brainmarks is pinned at `d4640074168f` in `pyproject.toml`, and `readyneuro/io.py` reads its
  `datasets/` metadata from the same commit on GitHub (`$BRAINMARKS_METADATA_URL` overrides).
- `s3://openneuro/v1/` [^public] (the released corpus, on Cloudflare R2, read through the
  `openneuro-public` AWS profile) and `s3://openneuro.org` (the upstream archive) are real.
  `<RESTRICTED_ROOT>`, `<RAW_ROOT>` and `<EVAL_ROOT>` are deliberate placeholders for data
  that cannot be redistributed.

The store formats — `.npy` tar members for shard geometry, the baked-in clip windows — are
seams 3 and 4 in `ARCHITECTURE.md`.

## Building the data

**The method is one file.** `preprocess/openneuro.py` is the from-raw kernel the paper
describes: N4, SynthSeg, rigid T1→template and mean-EPI→T1 with SynthMorph, composed with
per-run motion correction, then a single resample onto a 2 mm grid. Every `native` build
runs it. Read it before anything else here.

**The cores are standalone by design.** Nothing under `preprocess/` may import `readyneuro`;
the driver runs them as a subprocess CLI, and a test pins it. Each lifts out of the repo
carrying at most its sibling `openneuro.py`, which `aabc_native.py` and `nsd_native.py` reuse.

**The core↔finalizer seam is a filename layout**, written by hand in the cores and parsed
by `readyneuro.io.BidsLayout`. Change one side and you must change the other.

**Gates live in the finalizers**, not the cores: BIDS completeness, the minimum grey-matter
voxel count, the 60% coverage rejection, and the per-voxel z-score plus 1 s resample.

**Offline check:** `pytest tests/` and
`python -m readyneuro.builder.driver --config dataset_configs/openneuro.yaml --plan`.
A real build needs FreeSurfer, FSL, ANTs, templateflow and object storage.

## Please do not

Do not download the corpus (15.88 TB) or attempt to fetch the benchmark stores. Do not "fix"
the placeholders into a guessed bucket name. Do not paste the corpus read key into any file;
it comes from the setup script.

[^public]: **Public release:** the bucket holds 817 of the paper's 838 accessions under
    CC0 1.0. The 21 studies since withdrawn from OpenNeuro (498 shards, 290 h) are removed,
    leaving 30,222 shards (29,256 training + 966 held out), 22,083 h of stored training BOLD,
    15.88 TB. The paper trained on all of them, and the manifests in
    `manifests/` still list them.
