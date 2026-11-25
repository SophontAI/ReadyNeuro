# How the pieces fit together

Three packages, one path from raw fMRI to a benchmark score. `readyneuro` is here; the model
code is in [CortexMAE-ReadyNeuro](https://github.com/SophontAI/CortexMAE-ReadyNeuro), and both
pin the same Brainmarks commit as an `eval` dependency. This describes how the two repositories join.

```
   public archives                ReadyNeuro                            object store
  (OpenNeuro, HCP-YA,   ──────▶   preprocess/<core>.py     ──────▶   WebDataset .tar   (pretraining)
   NSD, AABC, PPMI,               readyneuro.builder.driver            Arrow IPC store   (evaluation)
   ABIDE, ADHD-200)                                                        │
                                                                           ▼
                                  CortexMAE-ReadyNeuro          ┌──────────────────────┐
                                  fmrifm.train  ──────────────▶ │ logs/<run>/          │
                                  (200k updates, 1 GPU)         │   checkpoints/*.ckpt │
                                  fmrifm.validate  ───────────▶ │   validation/*.json  │
                                                                │   config.yaml        │
                                                                └──────────┬───────────┘
                                                                           ▼
                                  brainmarks (eval dependency)  ┌──────────────────────┐
                                  main_probe / main_logistic ──▶│ output/<run>/…/      │
                                  + the two plugin overlays     │   eval_table.csv ×14 │
                                                                └──────────────────────┘
```

## The four seams

Each seam is a file format, not a function call, so every stage can be run, inspected and
replaced on its own.

**1. Raw archive → preprocessed BIDS.** `preprocess/*.py` are standalone scripts: the
driver invokes them as a subprocess CLI and they import nothing from `readyneuro`, so each
lifts out of the repo carrying at most its sibling `openneuro.py`. The contract between them
and the rest of the pipeline is a filename layout, written by hand on one side and parsed by
`readyneuro.io.BidsLayout` on the other — `anat/sub-<label>_space-<space>_desc-preproc_T1w.nii.gz`,
`func/sub-<label>_run-<NN>_space-<space>_desc-preproc_bold.nii`, three masks per run, and
a `sub-<label>_desc-fmridata.json` sidecar mapping run indices back to real run keys.
Rename anything there and both sides must change together.

**2. Preprocessed BIDS → the two stores.** `readyneuro.builder.finalize_pretrain` and
`finalize_eval` read that tree and pack it. They hold every gate — BIDS completeness, the
minimum grey-matter voxel count, the 60% brain-coverage rejection — and the
normalisation: each voxel's time series is z-scored within its run and resampled to a 1 s
interval. Neither finalizer does geometry; all of that already happened in the core.

**3. The pretraining store → the model.** One `.tar` per subject, schema version 3,
enforced at `fmrifm.pretrain_data.SHARD_SCHEMA_VERSION`. The first WebDataset *sample* is
the subject bundle — its geometry `.npy` members plus a `.json` sidecar; every later sample
is one BOLD run, whose member keys begin `<subject>__`. The geometry arrives as `.npy`
**tar members** (`bold_idx`, `bold_gm`, `bold_cortical_gm`, `bold_affine`), not as JSON
keys — a reader that looks for them in the metadata finds nothing and rejects the corpus.
`values.npy` is `int8 [voxels, frames]` with a JSON `scale`; the loader transposes it. A
run's `values.npy` payload ÷ voxels ÷ 3600 is its hours, but whole-tar bytes are not a
duration proxy — the paper's hours are measured from run sidecars.

**4. The evaluation store → the frozen encoder.** Arrow IPC under
`<root>/func.<space>.arrow/<split>/` and `anat.<space>.arrow/<split>/`, documented in
`src/brainmarks/datasets/eval_store.py`. The clip conventions are baked
in at build time, not applied at read time: PPMI, ABIDE and ADHD-200 are fixed at a
300-frame window (longer runs clipped, shorter tile-padded), AABC at a 500-frame raw
window that becomes 400 after the 1 s resample.

**Outputs.** Each of the 14 evaluation array tasks writes its own
`output/<run>/<name_prefix>/<dataset>__<model>__patch__{attn,logistic}/eval_table.csv`, so a run
produces 14 CSVs and nothing here aggregates them;
`fmrifm.validate` writes `logs/<run>/validation/<ckpt>.json`. Both are plain files and the
end of the pipeline.

## How the three packages find each other

`brainmarks` uses [PEP 420 namespace-package plugin discovery](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/#using-namespace-packages).
The two packages here contribute into its namespace without modifying it:

| contributor | registers | discovered by |
|---|---|---|
| `CortexMAE-ReadyNeuro/src/brainmarks/models/fmrifm.py` | `fmrifm_native`, `fmrifm_mni` | `brainmarks.models.registry.import_model_plugins()` |
| `src/brainmarks/datasets/*_readyneuro.py` | the seven benchmark arms | `brainmarks.datasets.registry.import_dataset_plugins()` |

Neither overlay may contain an `__init__.py` at any level. Adding one shadows the
installed package and the scanner silently finds **zero** plugins — the failure is an
empty registry, not an `ImportError`.

Both packages pin `brainmarks` to the same git commit in their `eval` extra. To see the
wiring work, in one Python 3.14
environment, from the repository root:

```bash
pip install -e '.[eval]'
(cd /tmp && python -c "from brainmarks.datasets.registry import list_datasets; print(list_datasets())")
```

Seven names ending `_readyneuro` should appear. The same names are what
`CortexMAE-ReadyNeuro/scripts/eval_brainmarks.sbatch` asks for.

Brainmarks' `datasets/` folder doubles as the label source: the ABIDE, PPMI and ADHD-200
loaders fetch their target maps from the pinned commit on GitHub, so no bucket is involved.
AABC is the exception: Brainmarks keeps its per-subject maps out of git deliberately, so
those two arms need `targets_root` pointed at a directory holding
`aabc_target_map_{age_open,sex}.json`, which come with AABC access.

## Where each claim in the paper is implemented

| the paper says | look at |
|---|---|
| rigid alignment to a 2 mm `MNI152NLin2009cAsym` grid, under 16 min for the median subject on one core | `preprocess/openneuro.py` — the whole method is one file |
| 96 context and 32 target cells, 6×6×6 voxels, 16 frames in four temporal patches | `CortexMAE-ReadyNeuro/src/fmrifm/masking.py`, `pretrain_data.py` |
| ViT-L, no class or register token, cross-attending decoder | `CortexMAE-ReadyNeuro/configs/model.yaml`, `CortexMAE-ReadyNeuro/src/fmrifm/{encoder,decoder}.py` |
| peak LR 7.1e-5 | derived, not configured: `base_lr 1.0e-4 × sqrt(128/256)` in `CortexMAE-ReadyNeuro/src/fmrifm/config.py` |
| frozen encoder, attention pooling vs. mean-pool + logistic | `CortexMAE-ReadyNeuro/src/brainmarks/models/fmrifm.py`, and the two drivers in `brainmarks` |
| 838 accessions, 30,720 shards [^public] | `manifests/` — `corpus_manifest.tsv` (one row per accession) and `openneuro_rebuild_manifest.jsonl.gz` (one record per shard) |
| 22,373 BOLD hours and the six subset sizes [^public] | the header comment of each `CortexMAE-ReadyNeuro/configs/data/openneuro_*.yaml` |

## Placeholders

`s3://openneuro/v1/` [^public] is the released pretraining corpus; `s3://<RESTRICTED_ROOT>/`,
`<RAW_ROOT>` and `<EVAL_ROOT>` stand in for data that cannot be redistributed (see
[`README.md`](README.md)). That is the seam-3/seam-4 asymmetry: the pretraining store is
public, the evaluation stores are not.

## Reading order

Start at `preprocess/openneuro.py`; that single file is the paper's
contribution. Then `CortexMAE-ReadyNeuro/configs/model.yaml` for the recipe and
`CortexMAE-ReadyNeuro/src/fmrifm/pretrain_data.py` for how a shard becomes a training sample.

[^public]: **Public release:** the paper's counts describe the corpus it trained on, and the
    manifests and config comments still include the 21 studies since withdrawn from
    OpenNeuro (flagged in the `withdrawn` column of `corpus_manifest.tsv`). The public store,
    `s3://openneuro/v1/`, omits those studies (498 shards, 290 h) and serves 817 accessions:
    29,256 training shards over 816 studies plus 966 held out, 22,083 h of stored training
    BOLD. The six corpus configs shrink from 747, 1,520, 2,984, 5,919, 11,837 and 22,373 h to
    737, 1,502, 2,946, 5,836, 11,674 and 22,083 h.
