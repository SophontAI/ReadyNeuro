"""Shared discovery for the fMRIPrep-derivative diagnosis cohorts (ABIDE / ADHD200 / PPMI).

`discover_mni_only` + `dx_meta` take a per-dataset source module (``srcmod``, the cohort's own
discover module) supplying ``discover_work`` / ``anat_stem`` / ``anat_uri`` / ``bold_uri`` /
``sidecar_uri`` / ``read_repetition_time`` / ``dx_map``. PPMI has its OWN ``discover`` (its
derivatives are laid out differently), so it uses only the staged-input names and `dx_meta`."""

from __future__ import annotations

# Staged-input NAMES: the driver stages the shipped fMRIPrep derivatives under these, and
# ``preprocess/fmriprep_dx.py`` opens them under exactly these -- change one side, change both.
T1W = "t1w.nii.gz"
FWD_H5 = "from-T1w_to-MNI.h5"
BOLD_MNI = "bold_mni.nii.gz"


def dx_meta(srcmod, sub: str) -> dict:
    """Best-effort diagnosis label from a cohort's ``dx_map`` (PD/Prodromal, Autism/Control,
    ADHD/Control) -> the sidecar ``meta``. The eval also joins it consumer-side; ``{}`` if absent."""
    dxv = srcmod.dx_map().get(str(sub))
    return {"diagnosis": dxv, "label": dxv} if dxv else {}


def discover_mni_only(cfg: dict, srcmod) -> list[dict]:
    """`discover` for an MNI-BOLD-only fMRIPrep dx cohort (ABIDE / ADHD200, exact twins): one eval
    item per subject, carrying the shipped ``space-MNI152NLin6Asym`` run + the T1->MNI warp.

    The acquisition TR is read from the fMRIPrep sidecar INTO the manifest, so the core needs no
    remote read. ``sources.runs`` stays empty -- the one REST run's frames come from the staged
    ``aux`` BOLD, which is why these cohorts are not run-staged."""
    if list(cfg["spaces"]) != ["mni"]:
        raise ValueError(f"{__name__} builds only 'mni'; for native space use the *_native dataset (got {cfg['spaces']!r})")
    so = cfg.get("input_storage_options")
    works = srcmod.discover_work(only_subjects=cfg.get("subjects"), n_subjects=cfg.get("n_subjects"))

    items: list[dict] = []
    for w in works:
        stem = srcmod.anat_stem(w, so)
        items.append(
            {
                "key": w.sub,
                "deliverables": [{"kind": "eval_arrow", "space": "mni", "split": w.split}],
                "sources": {
                    "t1": {"name": T1W, "uri": srcmod.anat_uri(stem, "t1w")},
                    "aux": [{"name": FWD_H5, "uri": srcmod.anat_uri(stem, "fwd_h5")},
                            {"name": BOLD_MNI, "uri": srcmod.bold_uri(w)}],
                    "runs": [],
                    "tr_acq": srcmod.read_repetition_time(srcmod.sidecar_uri(w), so),
                    "meta": dx_meta(srcmod, w.sub),
                },
            }
        )
    return items
