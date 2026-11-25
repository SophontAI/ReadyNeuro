"""The two value operations the finalizers own: the per-voxel temporal z-score and the TR resample.

Everything geometric happened in ``preprocess/``; this module only touches values.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import scipy.interpolate
import scipy.signal

from readyneuro.io import ValidVolume


def _resample_timeseries(series: np.ndarray, tr: float, new_tr: float, *, antialias: bool) -> np.ndarray:
    """Resample a ``(n_samples, dim)`` series from ``tr`` to ``new_tr``: an order-8 Chebyshev-I
    anti-alias low-pass **on downsampling only** (scipy.signal.decimate's filter), then cubic
    interpolation onto the new sample times.

    The output length is ``int(tr*T/new_tr)`` MINUS a fencepost: samples past the last input time
    ``tr*(T-1)`` are trimmed, so upsampling a genuine window loses its final frame (200 frames @
    1.5 s -> 1.0 s spans 298.5 s -> 299, not 300). `finalize_eval` tile-pads that back up to
    ``window_frames`` rather than dropping the run.

    This is the benchmark suite's resampler, inlined so rebuilding a corpus does not require the
    benchmark package."""
    if tr == new_tr:
        return series
    if antialias and new_tr > tr:  # downsampling: band-limit to the new Nyquist first
        q = new_tr / tr
        sos = scipy.signal.cheby1(8, 0.05, 0.8 / q, output="sos")
        series = scipy.signal.sosfiltfilt(sos, series, axis=0, padtype="even")
    # Sample times as tr * arange(n) rather than arange(0, duration, tr) -- more reliable in
    # floating point.
    x = tr * np.arange(len(series))
    new_x = new_tr * np.arange(int(tr * len(series) / new_tr))
    new_x = new_x[new_x <= x[-1]]
    return scipy.interpolate.interp1d(x, series, kind="cubic", axis=0)(new_x)


def _resample_tr(z: np.ndarray, from_tr: float, to_tr: float) -> np.ndarray:
    """One TR-resample of a ``(M, T)`` z-block. `_resample_timeseries` works on ``(n_samples, dim)``,
    ours is ``(M, T)`` -> transpose around it. The antialias guard is filtfilt's (~>=30 samples).
    Caller gates ``abs(from_tr - to_tr) > 1e-3``."""
    return _resample_timeseries(
        z.T, float(from_tr), float(to_tr), antialias=z.shape[1] >= 30
    ).T.astype(np.float32)


def _lowpass_to_nyquist(z: np.ndarray, tr: float, cutoff_tr: float) -> np.ndarray:
    """Zero-phase Chebyshev-I low-pass of a ``(M, T)`` block sampled at ``tr``, capping the band at
    the Nyquist of ``cutoff_tr`` **without resampling** -- the exact anti-alias `_resample_tr`
    applies when decimating to ``cutoff_tr``, but keeping the block on its ``tr`` grid. No-op on
    short blocks (filtfilt needs ~>=30 samples)."""
    if z.shape[1] < 30:
        return z
    wn = 0.8 * float(tr) / float(cutoff_tr)  # normalized to this block's Nyquist (1/(2*tr))
    sos = scipy.signal.cheby1(8, 0.05, wn, output="sos")
    return scipy.signal.sosfiltfilt(sos, z, axis=1, padtype="even").astype(np.float32)


def zscore_resample(
    vol: ValidVolume,
    target_tr: float,
    *,
    common_tr: float | None = None,
    eps: float = 1e-3,
) -> ValidVolume:
    """Per-voxel temporal z-score then TR-resample, packed in -> packed out (no dense
    grid). Operates on `vol.values` ``(M, T)`` directly -- no scatter/gather.

    ``common_tr`` (**cross-TR bandwidth harmonization**, set by the multi-site diagnosis evals --
    PPMI / ABIDE / ADHD200 at 3.0 s): when coarser than the source TR, the series is first low-pass
    filtered IN PLACE at the native rate to the common Nyquist ``1/(2*common_tr)``, so runs of
    different native TRs share ONE effective bandwidth (the slowest run's), and only then resampled
    once to ``target_tr``. One filter + one resample: a run already at ``target_tr`` (PPMI's 1.0 s
    runs) is filtered but NOT resampled, keeping its genuine samples instead of a lossy
    downsample->upsample round trip. ``common_tr=None`` (the single-site corpora) is a plain resample.

    The output records two temporal facts: the stored rate ``tr`` and the limiting period
    ``nyquist_tr = max(src_tr, target_tr, common_tr)``. ``tr`` alone cannot distinguish a native
    1.0 s run from a 2.5 s run upsampled to 1.0 s, and the limiting period is what the model reads.
    """
    values = np.asarray(vol.values, dtype=np.float32)  # (M, T)
    src_tr = float(vol.tr)
    # Keep what the z-score divides out. `mu` IS the mean EPI on this grid -- the only anatomy left
    # in the run, and what a BOLD->T1 alignment check needs. `sigma` is the PRE-resample std, hence
    # the honest tSNR denominator (the returned series is TR-interpolated, which smooths variance).
    mu = values.mean(axis=1)

    # Flat voxels are floored to 0, NOT dropped: the func mask stays the full structural mask,
    # so a subject's every run shares one voxel order.
    sigma = values.std(axis=1, keepdims=True)
    z = (values - values.mean(axis=1, keepdims=True)) / np.maximum(sigma, eps)
    z[sigma[:, 0] < eps] = 0.0  # flat voxels -> 0 (not noise blown up by /eps)

    # Filter leg: band-limit finer runs to the common Nyquist in place (no resample).
    if common_tr is not None and float(common_tr) > src_tr + 1e-3:
        z = _lowpass_to_nyquist(z, src_tr, common_tr)
    # Resample leg: one resample to target_tr (a coarser run upsamples; a run at target skips).
    cur_tr = src_tr
    if abs(src_tr - float(target_tr)) > 1e-3:
        z = _resample_tr(z, src_tr, target_tr)
        cur_tr = float(target_tr)
    np.nan_to_num(z, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    nyq = max(src_tr, float(cur_tr), *((float(common_tr),) if common_tr is not None else ()))
    return ValidVolume(
        z,
        vol.mask,
        vol.affine,
        tr=float(cur_tr),
        nyquist_tr=float(nyq),
        mean=mu,
        std=sigma[:, 0].astype(np.float32),
    )


def global_zscore_mask(
    vol: nib.Nifti1Image,
    mask: nib.Nifti1Image,
    *,
    eps: float = 1e-3,
) -> ValidVolume:
    """Single global z-score over the masked voxels (not per-voxel) -> a `ValidVolume`
    (z at the in-mask voxels, mask carried explicitly; no dense NaN grid)."""
    data = np.asarray(vol.dataobj, dtype=np.float32)
    m = np.asanyarray(mask.dataobj) > 0
    vals = data[m]
    mu = float(vals.mean()) if vals.size else 0.0
    sigma = max(float(vals.std()), eps) if vals.size else 1.0
    return ValidVolume((vals - mu) / sigma, m, np.asarray(vol.affine))
