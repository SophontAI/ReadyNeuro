# ReadyNeuro

Training-ready fMRI from OpenNeuro: about 30,000 subjects from 817 studies, 22,083 h of BOLD,
15.88 TB, released under CC0 1.0. Each subject is one WebDataset shard, rigidly aligned to a
shared 2 mm MNI152NLin2009cAsym grid. This repository builds the corpus and the evaluation stores
and provides the evaluation data loaders.

- Paper: <https://openreview.net/forum?id=rLMKAJCLc1>
- Blog post: <https://sophont.med/blog/native-fmri/>
- Corpus: `s3://openneuro/v1/` ([access](#get-the-data)); a 500 GB subset is at
  [medarc/ReadyNeuro](https://huggingface.co/datasets/medarc/ReadyNeuro)
- Model code: <https://github.com/SophontAI/CortexMAE-ReadyNeuro>
- Weights: <https://huggingface.co/medarc/CortexMAE-ReadyNeuro>
- Contact: paul@sophont.med

## Get the data

The corpus is on Cloudflare R2. This sets up a read-only `openneuro-public` profile (AWS CLI
v2.13 or later):

```sh
curl -fsS https://openneuro.sophont.med/clients/aws-setup.sh -o openneuro-aws-setup.sh
sh openneuro-aws-setup.sh
aws --profile openneuro-public s3 ls s3://openneuro/v1/
aws --profile openneuro-public s3 sync s3://openneuro/v1/openneuro-pretrain-shards-v6-b31/ ./openneuro-v1/openneuro-pretrain-shards-v6-b31/   # one batch
```

Batch `b0` is the prefix `openneuro-pretrain-shards-v6/`, and batches `b1`-`b31` are
`openneuro-pretrain-shards-v6-b<N>/`; shards sit under `<prefix>/train/native/`. Each shard has a
zero-byte `<shard>.tar.done` marker; add `--exclude '*.done'` to skip them.
Training reads the bucket through this profile (`export AWS_PROFILE=openneuro-public`). Unset it
before running builds, which read public AWS buckets.

## What is released

- **Corpus:** 30,222 shards, 29,256 for training (`b0`-`b30`) and 966 held out (`b31`). The
  paper trained on 837 studies and 22,373 h; 21 studies since withdrawn from OpenNeuro are
  removed here (498 shards, 290 h).
- **Manifests:** [`corpus_manifest.tsv`](manifests/corpus_manifest.tsv) lists
  each study's license and flags the withdrawn studies. Its `eval_overlap` column flags four
  studies that overlap HCP-YA evaluation (`ds003452`, `ds003465`, `ds000244`, `ds002685`). They
  are in the corpus; exclude them if you evaluate on HCP-YA.
- **Not released:** the six evaluation stores (HCP-YA, NSD, AABC, PPMI, ABIDE, ADHD-200), whose
  terms don't allow redistribution. Configs use the placeholders `s3://<RESTRICTED_ROOT>/`,
  `<RAW_ROOT>` and `<EVAL_ROOT>` for them.

## Repository

| directory | contents |
|---|---|
| [`src/readyneuro/`](src/readyneuro/) | builds the shards and evaluation stores |
| [`src/brainmarks/datasets/`](src/brainmarks/datasets/) | evaluation data loaders (Brainmarks dataset plugins) |
| [`preprocess/`](preprocess/), [`dataset_configs/`](dataset_configs/) | preprocessing scripts and one build config per dataset |
| [`manifests/`](manifests/) | corpus and shard manifests |

The `eval` extra installs [Brainmarks](https://github.com/MedARC-AI/Brainmarks) from the
[fork](https://github.com/Aceticia/Brainmarks/tree/d4640074168f1c6dcb0a94ec78801577792be3f4) pinned at
`d4640074168f`, which adds a padding mask to the attention probe.

```sh
git clone https://github.com/SophontAI/ReadyNeuro.git
cd ReadyNeuro
pip install -e '.[eval,test]'       # evaluation data loaders (Python 3.14)
pytest tests/                       # offline tests
# or, in a uv-managed venv:
uv run --extra test pytest tests/
```

With pip on Linux, add `--extra-index-url https://download.pytorch.org/whl/cu128` for the CUDA 12.8
torch the paper used (`uv` picks it up automatically). Building data is covered in
[`BUILDING.md`](BUILDING.md), and how the code maps to the paper
in [`ARCHITECTURE.md`](ARCHITECTURE.md). To evaluate a model, follow the
[model repository](https://github.com/SophontAI/CortexMAE-ReadyNeuro#install), which installs this
package as an evaluation dependency.

## NSD evaluation splits

The NSD store keeps the `shared1000` trials that Brainmarks holds out, so its splits have
35,184 / 5,857 / 5,794 clips instead of Brainmarks' 32,539 / 5,418 / 5,390, and 265 images
appear in both train and test. The paper's NSD scores use these stores; its published-model rows
use Brainmarks' splits. To match Brainmarks, also filter on `split` in `subject_run_clips`
([`nsd.py`](src/readyneuro/nsd.py)).

## License and citation

Code: MIT ([`LICENSE`](LICENSE)). Corpus: CC0 1.0
([`DATA_LICENSE.md`](DATA_LICENSE.md)). Brainmarks keeps its own MIT
license.

CC0 needs no attribution, but please cite the paper, OpenNeuro (Markiewicz et al. 2021, *eLife*
10:e71774), Brainmarks (Lane et al. 2026, ICML, arXiv:2510.13768) and the source studies in
`corpus_manifest.tsv`.

```bibtex
@inproceedings{liu2026scaling,
  title     = {Scaling {fMRI} Foundation Models with Native-Space Data from Heterogeneous Public Studies},
  author    = {Liu, Xujin Chris and Lane, Connor and Muhammad, Hassan and Abraham, Tanishq Mathew and Scotti, Paul Steven},
  booktitle = {NeurIPS 2026 Workshop on Foundation Models for the Brain and Body},
  year      = {2026},
  url       = {https://openreview.net/forum?id=rLMKAJCLc1}
}
```

Xujin Chris Liu (New York University) did this work during a summer 2026 internship at Sophont.
