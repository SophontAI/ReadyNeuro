# License of the released corpus

The code in this repository is MIT, copyright Sophont AI (see
[`LICENSE`](LICENSE)); Brainmarks, an evaluation dependency, keeps its own MIT license. The **data**
the pipeline produces is licensed separately, and the two should not be confused.

## ReadyNeuro, the native-space OpenNeuro corpus

Released under **CC0 1.0 Universal** at `s3://openneuro/v1/`.
[`BUILDING.md`](BUILDING.md#getting-the-released-corpus) has the access commands.

Every study in it comes from OpenNeuro, which requires uploaders to agree that their dataset
becomes public under CC0 (OpenNeuro FAQ and user guide, <https://docs.openneuro.org>;
Markiewicz et al. 2021, *eLife* 10:e71774). Datasets from the OpenfMRI era carry PDDL, also a
public-domain dedication. The shards derived from them are released under CC0 1.0.

The release holds 817 of the paper's 838 accessions under all 32 batch prefixes: 30,222 shards
(29,256 training over 816 studies in `b0`–`b30`, 966 held-out over 439 studies in `b31`),
22,083 h of stored training BOLD, 15.88 TB.

[`manifests/corpus_manifest.tsv`](manifests/corpus_manifest.tsv) is the per-study record. The
upstream `license` field it records for those 817 studies:

| CC0 | PDDL | other public domain | unstated | CC-BY | CC-BY-SA | CC-BY-NC |
|---:|---:|---:|---:|---:|---:|---:|
| 730 | 37 | 1 | 40 | 4 | 3 | 2 |

The release relies on OpenNeuro's platform-level CC0 terms for all of them, including the nine
whose own metadata names a CC-BY-family license: `ds001105`, `ds001218`, `ds001634`,
`ds001635` (CC-BY); `ds001439`, `ds001722`, `ds001748` (CC-BY-SA); `ds001338`, `ds001771`
(CC-BY-NC). A dataset owner who objects can open an issue
at <https://github.com/SophontAI/ReadyNeuro/issues>.

## What was removed

Only the 21 studies since withdrawn from OpenNeuro, flagged in the `withdrawn` column of
`corpus_manifest.tsv` (2026-08-17 audit): `ds001019` `ds001167` `ds001348` `ds001354`
`ds001482` `ds001583` `ds001619` `ds001966` `ds002351` `ds002355` `ds002749` `ds002809`
`ds002862` `ds002894` `ds002941` `ds003028` `ds004174` `ds004935` `ds005032` `ds006503`
`ds007082`. That is 498 shards (484 training + 14 held-out), 290 h of stored training BOLD
(1.3%), 0.20 TB. Nothing is withheld for license reasons.

The paper's corpus is 838 accessions and 30,720 shards (16.08 TB). It trained on 837 studies,
29,740 shards and 22,373 h (15.56 TB), and held out 980 shards (448 studies, one of them
validation-only) for validation. The two files in [`manifests/`](manifests/) document that
corpus, so they still list the withdrawn studies. The manifests are released under CC0 1.0,
like the corpus.

## Citing

CC0 does not require attribution. Please cite the paper, OpenNeuro
(Markiewicz et al. 2021) and the source datasets anyway; `corpus_manifest.tsv` lists every
accession.

## Not released

The six evaluation stores are **not released**. Each derives from a cohort — HCP-YA, NSD,
AABC, PPMI, ABIDE, ADHD-200 — whose terms (gated access, data-use agreements or non-commercial share-alike licenses) do not
allow us to redistribute a derived product. Build your
own with this pipeline under your own access; [`BUILDING.md`](BUILDING.md#evaluation-stores) lists
what each one requires.
