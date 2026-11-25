# manifests

Two records of ReadyNeuro, the native-space OpenNeuro corpus, one per study and one per shard. The
paper's corpus is 838 accessions and 30,720 shards (16.08 TB). It trained on 837 studies,
29,740 shards and 22,373 h (15.56 TB), and held out 980 shards (448 studies, one of them
validation-only) for validation. Both files document that corpus, **including the 21 studies
since withdrawn from OpenNeuro**, which `s3://openneuro/v1/` no longer serves
(see [`../DATA_LICENSE.md`](../DATA_LICENSE.md)). Both files are released under CC0 1.0, like
the corpus.

## `corpus_manifest.tsv`

One tab-separated row per accession, 838 rows.

| column | meaning |
|---|---|
| `accession` | OpenNeuro accession (`ds…`) |
| `train_subjects` | unique subjects in the training batches `b0`–`b30` |
| `val_subjects` | unique subjects in the held-out batch `b31` |
| `license` | the study's own license field as recorded upstream, normalized: `CC0`, `PDDL`, `other: pd`, `unstated`, `CC-BY`, `CC-BY-SA`, `CC-BY-NC`, or `unknown` |
| `license_source` | where `license` came from: `api` (OpenNeuro API), `archived-bucket` (the archived `dataset_description.json` of a withdrawn study) or `unrecoverable` |
| `withdrawn` | `yes` if the study was withdrawn from OpenNeuro by the 2026-08-17 audit; these are not in the public bucket |
| `eval_overlap` | `subject-overlap(HCP-YA)` or `stimulus-overlap(HCP-task21)` where a study overlaps an evaluation benchmark |
| `overlap_note` | the adjudication behind `eval_overlap` |
| `duplicate_subjects` | `;`-separated subject labels shipped in two batches; the prefix-glob corpus configs sample them at double weight |

## `openneuro_rebuild_manifest.jsonl.gz`

Gzipped JSON Lines. The first line is a `_meta` record: the corpus, when the archive index
was frozen, the source URI template and how to infer upstream versions. Every other line is
one shipped shard of the paper's corpus (30,720 records, withdrawn studies included):

* `batch`, `shard` (`{accession}__{label}`), `accession`, `subject_dir`, `bytes`
* `age`, `sex` -- the subject's values as sampled, verbatim (`null` where the study gave none)
* `t1`, `runs` -- the exact `s3://openneuro.org/...` source URI of the T1 and of every BOLD
  run selected (at most 20 runs of at least 50 frames), each with its acquired TR and internal
  run label. The finalizer later dropped 1,361 of the 173,975 selected training runs (for
  example under 60% brain coverage), so a shard can hold fewer runs than listed; the stored
  ones are those whose `run` label appears in the shard's member names.
* `sampled`, `uploaded` -- when the URIs were listed and when the shard was stored

OpenNeuro study versions were never pinned. To rebuild an accession, use the latest snapshot
created before its `sampled` date; a snapshot created between `sampled` and `uploaded` makes
that accession ambiguous.

[`configs/data/make_openneuro_hours_urls.py`](https://github.com/SophontAI/CortexMAE-ReadyNeuro/blob/main/configs/data/make_openneuro_hours_urls.py)
reads this file to regenerate `openneuro_1520h.yaml`, the OpenNeuro subset matched to HCP-YA
by hours.
