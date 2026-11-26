# Building ReadyNeuro

Builds the pretraining corpora and the six evaluation stores for *Scaling fMRI Foundation Models
with Native-Space Data from Heterogeneous Public Studies*, and houses the evaluation data loaders
that [CortexMAE-ReadyNeuro](https://github.com/SophontAI/CortexMAE-ReadyNeuro) reads.

**If you only want to train a model, you do not need to run any of this.** ReadyNeuro, the native-space
OpenNeuro corpus, is released at `s3://openneuro/v1/`, and the corpus configs in
[CortexMAE-ReadyNeuro](https://github.com/SophontAI/CortexMAE-ReadyNeuro) already name it. Everything else -- the HCP-YA
shards and the six evaluation stores -- you build yourself; see [Buckets](#buckets) for where
each one lives.

## Getting the released corpus

The corpus lives in a Cloudflare R2 bucket. R2's S3 API always signs requests, so there is no
anonymous mode and `--no-sign-request` does not work; read access comes from a public
read-only key that a setup script installs as an AWS profile. With AWS CLI v2.13 or later:

```sh
curl -fsS https://openneuro.sophont.med/clients/aws-setup.sh -o openneuro-aws-setup.sh
sh openneuro-aws-setup.sh
aws --profile openneuro-public s3 ls s3://openneuro/v1/     # 32 batch prefixes
aws --profile openneuro-public s3 cp s3://openneuro/v1/openneuro-pretrain-shards-v6-b1/train/native/ds004604__022.tar .   # one subject, 77 MB
```

Each batch is about 0.5 TB, and `b31` is the validation holdout; `s3 sync` a prefix to download
a whole batch. Each shard has a zero-byte `<shard>.tar.done` marker beside it; add `--exclude '*.done'` to skip
them.

The `openneuro-public` profile carries the key, the R2 endpoint and region `auto`. To train,
export `AWS_PROFILE=openneuro-public`: the loader streams each shard through `aws s3 cp` and
lists prefixes through `s3fs`, and both pick up the profile. The corpus configs already name
the paths, so training needs no further configuration.

The profile also sets the R2 endpoint, so while `AWS_PROFILE=openneuro-public` is exported
every S3 request goes to R2, including the anonymous reads this pipeline makes from public AWS
buckets (`s3://openneuro.org`, `s3://fcp-indi`, `s3://natural-scenes-dataset`,
`s3://hcp-openaccess`). Export it only for training or validation jobs, and unset it before
running a build or `scripts/smoke_openneuro.sh`.

The profile is not optional, because it is where the endpoint comes from. Without the
profile's R2 endpoint, `s3://openneuro` is looked up on AWS S3, which is not this bucket
(OpenNeuro's own archive is `s3://openneuro.org`), so the listing fails or comes back empty.

The bucket holds 817 of the paper's 838 accessions: 30,222 shards, 22,083 h of
stored training BOLD, 15.88 TB. The 21 studies since withdrawn from OpenNeuro are removed
(see [Caveats](#caveats)).

## What "native space" means here

Every subject's T1w image is rigidly registered (6 degrees of freedom: rotation and translation
only, no scaling and no shear) onto a shared 2 mm `MNI152NLin2009cAsym` grid. Orientation and
position are therefore fixed and comparable across subjects, while the anatomy stays the subject's
own -- no structure is stretched to match a template.

The `mni` half of each store is the contrasting condition -- each benchmark's own nonlinearly
normalised distribution, kept for the pipeline comparison.

## The pipeline

`preprocess/openneuro.py` is the from-raw kernel. It is a standalone script: the driver invokes it
as a subprocess CLI, and it imports nothing from `readyneuro`, so it lifts out of this repo unchanged.
Every `native` build in the paper runs it -- OpenNeuro, and the `*_native` arms of HCP-YA, PPMI,
ABIDE and ADHD-200. The vendor `mni` builds run the sibling scripts that read each distribution's
own preprocessed product: `hcp.py` (HCP-YA), `nsd.py` (NSD), `aabc_vendor.py` (AABC) and
`fmriprep_dx.py` (PPMI / ABIDE / ADHD-200); `aabc_native.py` and `nsd_native.py` are the two
from-raw arms that need more than the shared kernel. `builder/wrappers.py` is the full table.

1. **Bias-correct** the raw T1w with ANTs `N4BiasFieldCorrection` (shrink factor 4).
2. **Segment** it with FreeSurfer `mri_synthseg --robust` on a 1 mm grid. Fewer than 100,000 brain
   voxels means the segmentation failed and the subject is rejected.
3. **Conform and skull-strip**: `mri_convert -rt cubic` onto the segmentation's grid, then
   `mri_mask`.
4. **Register T1 to the template** with `mri_synthmorph register -m rigid` against the
   `MNI152NLin2009cAsym` brain, converted to FSL form by `lta_convert`. The brain, grey-matter,
   cortical-grey-matter masks and the full label volume are carried onto the 1 mm output grid and
   its exactly-nested 2 mm BOLD grid.
5. **Motion-correct** each BOLD run with FSL `mcflirt -meanvol -mats`, which also yields the mean
   BOLD reference and one matrix per frame.
6. **Register the mean BOLD to the T1**, again `mri_synthmorph` rigid.
7. **Resample once.** The per-frame motion transform, the BOLD-to-T1 transform and the
   T1-to-template transform compose into a single affine per frame, and the series is resampled
   straight onto the 2 mm grid, streamed frame by frame so peak memory stays bounded by one dense
   frame. The in-field-of-view masks fall out of the same pass.

**Cost.** On one pinned CPU core with inputs on local disk: about **340 s once per subject**,
plus about **50 s per BOLD run and 0.3 s per acquired frame** (a typical run costs ~130 s). The
four-run, 911-frame subject we timed took **14 min**; the median training record is about 16 min.
**Peak resident memory 14.5 GB**. The expense is registration, not interpolation -- the single
composed resample is 7--8 s of a run's ~130 s.

## The two products

* **Pretraining** -- one WebDataset tar per subject. BOLD values are int8 at in-mask voxels, sampled
  at 1 s, up to 20 runs per subject, alongside the 1 mm T1 layer and the tissue masks.
* **Evaluation** -- one Arrow IPC store per benchmark, one row per pre-sliced clip, float16. The
  schema is documented in `src/brainmarks/datasets/eval_store.py`.

**The evaluation windows are cut at BUILD time, in the store, not in the loader.** PPMI, ABIDE and
ADHD-200 fix every scan at a 300-frame five-minute window, clipping longer runs and tile-padding
shorter ones. AABC uses a 500-raw-frame resting window, which is 400 frames after the 0.8 -> 1.0 s
resample. HCP-YA Task21 and NSD store 16-frame clips cut at labelled onsets.

## Building

The test suite is the one thing here that needs no data, no credentials and no external tools --
a few dozen tests over the store codec, the finalizer's gates and the normalization, against a
synthetic BIDS subject, in a few seconds:

```
pip install -e '.[test]' && pytest
```

Install the package first -- the driver locates `preprocess/` by walking up from its own file, so
it has to run from a checkout:

```
pip install -e '.[build]'
```

Every build is then one command:

```
python -m readyneuro.builder.driver --config dataset_configs/<name>.yaml [--set key=value ...]
```

`--plan` prints the resolved core, roots, concurrency and discovery signature and exits without
touching the network -- safe on a login node. `--set` overrides any config key (value parsed as
YAML) and is repeatable.

### Pretraining corpora

The OpenNeuro corpus is **32 batches of 1,000 subjects**. Each batch needs its OWN `results_dir`
(the manifest cache and the per-subject `.done` sentinels live there) and its OWN `pretrain_root`,
but every batch must share the ONE frozen `openneuro_index.json` -- that file is the archive listing
the seeded shuffle is drawn from, and re-listing a live archive would redraw a different corpus.

Batch 0 lands in the UNSUFFIXED root (`...-v6`); batches 1-31 take a `-bN` suffix matching their
number. This is how the released corpus was built. The config default, `s3://openneuro/v1/`, is
the published read-only location, so to rebuild, point `pretrain_root` at your own bucket and set
`out_profile` to an AWS profile that can write to it.

```
for N in $(seq 0 31); do
  if [ "$N" = 0 ]; then SUF=; else SUF="-b$N"; fi
  python -m readyneuro.builder.driver --config dataset_configs/openneuro.yaml \
    --set openneuro_batch=$N \
    --set results_dir=/data/build/openneuro/b$N \
    --set out_profile=<YOUR_PROFILE> \
    --set pretrain_root=s3://<YOUR_BUCKET>/openneuro-pretrain-shards-v6$SUF
done

python -m readyneuro.builder.driver --config dataset_configs/hcp_ya-native.yaml
python -m readyneuro.builder.driver --config dataset_configs/hcp_ya-mni.yaml
```

The two HCP-YA configs write both the pretraining shards and the Task21 evaluation store.

### Evaluation stores

**No store here may be redistributed, so you have to build your own.** The cohorts' terms (gated
access, data-use agreements or non-commercial share-alike licenses) do not allow us to
redistribute derived stores. What each arm needs:

| benchmark | paper target | raw source | to get it |
|---|---|---|---|
| HCP-YA | Task21, 21-way (and the 880-subject pretraining corpus) | `s3://hcp-openaccess/HCP_1200` | register and accept the HCP open-access data use terms |
| NSD | viewed COCO category, 24-way | `s3://natural-scenes-dataset` | public bucket, under the NSD terms of use |
| AABC | age bracket, sex | session archives you stage to `s3://<RAW_ROOT>/` | apply to the study; access-controlled |
| PPMI | Parkinson's vs prodromal | BIDS and fMRIPrep derivatives you stage to `s3://<EVAL_ROOT>/` | apply to the study; access-controlled |
| ABIDE | autism vs control | native from `s3://fcp-indi/data/Projects/ABIDE/RawDataBIDS` (public, anonymous); MNI from fMRIPrep derivatives you stage | raw is open; the MNI arm needs derivatives you run yourself |
| ADHD-200 | ADHD vs control | native from `s3://fcp-indi/data/Projects/ADHD200/RawDataBIDS` (public, anonymous); MNI from fMRIPrep derivatives you stage | same |

The `native` arms rebuild from raw with the same kernel as the OpenNeuro corpus, so for ABIDE
and ADHD-200 they need nothing but public data. The `mni` arms are each benchmark's own
nonlinearly normalised product: HCP-YA ships one, NSD ships a transform, and the three
clinical cohorts need fMRIPrep run over the raw BIDS first.

One config per (benchmark, space), and the two together make one complete store: the from-raw
kernel cannot produce `mni` at all (raw BIDS ships no template warp), and the vendor cores build
only `mni`. The two HCP-YA configs ran above, so this is the ten remaining:

```
for c in dataset_configs/{nsd,aabc,ppmi,abide,adhd200}-{native,mni}.yaml; do
  python -m readyneuro.builder.driver --config $c
done
```

NSD is the exception to "one store, two spaces": its two arms live in separate roots, so the native
arm needs an explicit `eval_root` at read time.

**AABC also needs its evaluation split, which does not ship.** Both AABC builds take each
evaluation subject (Brainmarks batches 10-19) at one sampled visit and assign it to train,
validation or test. That assignment lists participants of an access-controlled cohort, so this
repository does not include it. Supply it as a local JSON file mapping each bare subject ID to
`[split, visit]`, for example `{"HCA…": ["train", "V1"], ...}`, and point the
`aabc_eval_split` key at it: replace the `<AABC_EVAL_SPLIT_JSON>` placeholder in both
`dataset_configs/aabc-*.yaml`, or pass `--set aabc_eval_split=/path/to/split.json`. Without it
both AABC builds stop before listing any data. The paper used Brainmarks' own AABC split:
[`datasets/AABC/scripts/make_aabc_eval.py`](https://github.com/Aceticia/Brainmarks/blob/d4640074168f1c6dcb0a94ec78801577792be3f4/datasets/AABC/scripts/make_aabc_eval.py) (seed 2912) draws one visit per subject
and splits 80/10/10, stratified by age, sex and fluid IQ. It needs the AABC `MNINonLinear` REST
download and builds the full AABC Arrow dataset on the way. Run
[`make_aabc_targets.py`](https://github.com/Aceticia/Brainmarks/blob/d4640074168f1c6dcb0a94ec78801577792be3f4/datasets/AABC/scripts/make_aabc_targets.py) first: without
`aabc_target_map_{age_open,sex,FluidIQ_Tr35_60y}.json` the stratification silently falls back to
one bin. Then read the `sub` and `visit` columns of each split it writes. The split depends on
which visits your download holds, so it can differ from the paper's.

## Reading the stores

```
pip install -e '.[eval]'
```

`brainmarks` is the benchmark suite the paper cites; the `eval` extra installs it from git,
pinned to commit `d4640074168f` of the [Aceticia/Brainmarks](https://github.com/Aceticia/Brainmarks)
fork. `brainmarks` on PyPI is a different, unpinned release without the attention-probe
padding-mask fix. Neither it nor torch is needed to BUILD a
store.

`src/brainmarks/` is a PEP 420 namespace overlay that registers seven datasets into Brainmarks'
plugin scanner.

| dataset id | target | frames per clip | root env var | needs your own `targets_root` |
|---|---|---|---|---|
| `hcpya_task21_readyneuro` | HCP-YA task condition, 21-way | 16 | `HCPYA_FMRI_EVAL_ROOT` | no |
| `nsd_cococlip_readyneuro` | NSD viewed COCO category, 24-way | 16 | `NSD_FMRI_EVAL_ROOT` | no |
| `aabc_age_readyneuro` | AABC age bracket | 400 | `AABC_FMRI_EVAL_ROOT` | yes |
| `aabc_sex_readyneuro` | AABC sex | 400 | `AABC_FMRI_EVAL_ROOT` | yes |
| `ppmi_dx_readyneuro` | Parkinson's vs prodromal | 300 | `PPMI_FMRI_EVAL_ROOT` | no |
| `abide_dx_readyneuro` | autism vs control | 300 | `ABIDE_FMRI_EVAL_ROOT` | no |
| `adhd200_dx_readyneuro` | ADHD vs control | 300 | `ADHD200_FMRI_EVAL_ROOT` | no |

Each takes `space` (`"native"` or `"mni"`) and optional `eval_root` / `storage_options`; the root
resolves as argument > env var > the module default.

ABIDE, PPMI and ADHD-200 join their labels from Brainmarks metadata at construction time, fetched
from the pinned commit's `datasets/` tree on GitHub with no configuration.

**AABC is the exception, and at read time it needs one thing from you.** Brainmarks keeps per-subject AABC
label maps out of git on purpose: the public repository carries only the aggregate
`aabc_target_info_*.json` summaries (class names and counts), while the maps themselves live
behind its private data root, because they are participant-level phenotype for an
access-controlled cohort. So the two files the age and sex arms need are not in this release
either. If you have AABC access you already have them; put

```
aabc_target_map_age_open.json
aabc_target_map_sex.json
```

in one directory and pass it as `dataset_kwargs.targets_root=<dir or s3:// prefix>`. Without
that, `aabc_age_readyneuro` and `aabc_sex_readyneuro` raise `FileNotFoundError` naming the file
they wanted; the other five arms are unaffected.

If you do not have the maps to hand, the recipe ships even though the data does not:
Brainmarks' [`datasets/AABC/scripts/make_aabc_targets.py`](https://github.com/Aceticia/Brainmarks/blob/d4640074168f1c6dcb0a94ec78801577792be3f4/datasets/AABC/scripts/make_aabc_targets.py) builds both
files from the AABC subject CSV of your own download, writing them into
`datasets/AABC/metadata/targets/` of a Brainmarks checkout; pass that directory as `targets_root`.

`targets_root` works for any arm, and `$BRAINMARKS_METADATA_URL` repoints the whole metadata
tree. Point either at a raw HTTP base and the label map becomes unpinned, so it can change
under you.

## Buckets

`s3://openneuro/v1/` is ReadyNeuro, the released OpenNeuro corpus, and the only output of this pipeline
that is public. `s3://<RESTRICTED_ROOT>/` is where it writes everything else -- the HCP-YA
shards and the six evaluation stores, none of which may be redistributed. Two input roots
appear in the configs and the discovery code, both placeholders for a bucket you control:

* `s3://<RAW_ROOT>/...` -- the access-controlled AABC session zips (raw REST + struct), staged.
  AABC only (`discover/aabc.py`, `discover/aabc_native.py`).
* `s3://<EVAL_ROOT>/...` -- the staged fMRIPrep derivatives for ABIDE, PPMI and ADHD-200
  (`.../{ABIDE,PPMI,ADHD200}/fmriprep`), PPMI's raw BIDS tree (`.../PPMI/bids`), and the AABC
  phenotype target mirror at `.../targets` that both AABC builds bake into their item meta
  (`discover/aabc.py`, overridable with `aabc_pheno_targets_root`). ABIDE and ADHD-200 native read
  their raw data from the public `s3://fcp-indi/` mirror and need no staging.

Curated splits, phenotype target maps and event onsets -- read by every build except
`openneuro.yaml` (`src/readyneuro/io.py`, `BENCHMARK_METADATA_ROOT`) -- come from the
pinned Brainmarks commit's `datasets/` tree on GitHub by default, so they need no bucket at all. `$BRAINMARKS_METADATA_URL`
repoints them at another local directory or a raw HTTP base (not `s3://`); `$FMRI_META_DIR` stages
a remote tree on local disk.

`out_profile: r2` and `input_profile: r2` name AWS profiles in `~/.aws/config` (endpoint included);
`input_storage_options: {anon: true}` marks a genuinely public bucket.

## External tools

Not python packages -- they must be on `PATH`:

* **FreeSurfer** -- `mri_synthseg`, `mri_synthmorph`, `mri_convert`, `mri_mask`, `lta_convert`,
  plus a FreeSurfer license (free registration; SynthSeg refuses to run without one)
* **FSL** -- `mcflirt` (FSL is free for non-commercial use; commercial use needs a license)
* **ANTs** -- `N4BiasFieldCorrection`
* **templateflow** -- fetches the `MNI152NLin2009cAsym` grid (`pip install '.[build]'`)
* **the `aws` CLI** (v2.13 or later) -- for staging inputs and fetching the released shards

```
source scripts/setup_preproc.sh    # verify the toolchain and prefetch the template
scripts/smoke_openneuro.sh         # download one real subject and run end to end
python scripts/native_smoke.py --config dataset_configs/<ds>-native.yaml --subject <key>
```

`native_smoke.py` runs discover -> download -> preflight -> core and then asserts the intermediate
BIDS tree, without evicting the scratch, so a failure can be attributed between the core and the
finalizer.

## Caveats

**OpenNeuro study versions were never pinned.** The corpus was drawn from a July 2026 listing of the
public mirror, and the archive is live. By the 17 August 2026 audit 21 accessions had been
withdrawn, so the archive can no longer reconstruct the store exactly.

**The released bucket omits those 21 studies.** They are flagged in the `withdrawn` column of
[`manifests/corpus_manifest.tsv`](manifests/corpus_manifest.tsv) and account for 498 shards
(484 training + 14 held-out), 290 h of stored training BOLD (1.3%) and 0.20 TB.
`s3://openneuro/v1/` serves the other 817 of 838 accessions (816 training; the `b31` held-out
set spans 439): 30,222 of 30,720 shards (29,256 training + 966 held-out), 22,083 h of stored
training BOLD, 15.88 TB, under all 32 batch prefixes. The paper trained on all 837 training
studies. The prefix-glob corpus configs in CortexMAE-ReadyNeuro need no change;
`openneuro_1520h.yaml`, which enumerates shards, has its 33 withdrawn rows commented out.

**Reproducing the exact subject draw needs the frozen `openneuro_index.json` plus seed
`20260713`.** The seed alone is not enough: it fixes one shuffle *of the eligible set*, and the
eligible set is derived from the archive listing, which changes. Re-listing and re-drawing with the
same seed yields a different corpus. That index file is **not part of this release**; a first build
regenerates it by listing the live archive, which produces a different eligible set and so a
different draw. What does ship is the draw itself:
[`manifests/openneuro_rebuild_manifest.jsonl.gz`](manifests/openneuro_rebuild_manifest.jsonl.gz)
records every shard's source URIs and when they were listed; [`manifests/`](manifests/README.md)
describes both files.

## License

Code is MIT ([`LICENSE`](LICENSE)). The corpus this pipeline releases is **CC0 1.0**, the terms
OpenNeuro requires every uploader to agree to, over 817 accessions: only the 21 studies since
withdrawn from OpenNeuro are removed, and nothing is withheld for its license.
[`DATA_LICENSE.md`](DATA_LICENSE.md) has the details. The six evaluation stores are not
released.
