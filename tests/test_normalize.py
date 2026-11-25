"""Temporal z-score and TR resampling -- both silent when wrong.

A resampled run of the wrong length still trains; a voxel with no temporal variance divided by its
own zero standard deviation yields NaN that only surfaces much later; and a run harmonized to the
wrong band is indistinguishable from a native one by ``tr`` alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from readyneuro.io import ValidVolume
from readyneuro.tools.value_preproc import zscore_resample

RAD = np.array([[-2, 0, 0, 40], [0, 2, 0, -30], [0, 0, 2, -20], [0, 0, 0, 1]], float)


def _vol(t=10, tr=0.72, seed=0):
    mask = np.zeros((8, 8, 8), bool)
    mask[2:6, 2:6, 2:6] = True
    values = np.random.default_rng(seed).standard_normal((int(mask.sum()), t)).astype(np.float32)
    return ValidVolume(values, mask, RAD, tr=tr), mask


def test_zscore_is_unit_variance_and_drops_no_voxel():
    vol, mask = _vol(tr=1.0)
    out = zscore_resample(vol, target_tr=1.0)  # src_tr == target_tr -> no resample leg
    assert out.n_frames == 10
    assert np.array_equal(out.mask, mask)
    assert np.allclose(out.values.mean(1), 0, atol=1e-5)
    assert np.allclose(out.values.std(1), 1, atol=1e-3)


def test_resampled_length_is_int_src_tr_times_t_over_target():
    """``int(src_tr * T / target_tr)``, not ``floor((T-1) * r) + 1``. They differ for some run
    lengths -- at T=176, TR 0.72 s that is 126 vs 127, an extra tail frame the consumer never has.
    """
    vol, _ = _vol(t=176, tr=0.72)  # >= 30 frames, so the anti-alias path runs too
    out = zscore_resample(vol, target_tr=1.0)
    assert out.n_frames == int(0.72 * 176 / 1.0) == 126
    assert out.tr == pytest.approx(1.0)


def test_constant_voxel_is_kept_and_floored_to_zero():
    """A zero-variance voxel is KEPT and floored to exactly z=0, never dropped -- the func mask has
    to stay equal to the structural GM mask across every run of a subject."""
    mask = np.zeros((4, 4, 4), bool)
    mask[1, 1, 1] = mask[2, 2, 2] = True  # flat-index 21 precedes 42 in C order
    values = np.array([[3.0] * 5, [1.0, -1.0, 2.0, 0.0, -2.0]], np.float32)
    out = zscore_resample(ValidVolume(values, mask, RAD, tr=1.0), 1.0)
    assert np.array_equal(out.mask, mask)
    assert np.isfinite(out.values).all()
    assert np.allclose(out.values[0], 0.0)
    assert np.isclose(out.values[1].std(), 1.0, atol=1e-3)


def test_harmonized_runs_carry_the_shared_band_not_their_own():
    """``common_tr`` band-limits a fast run to the SLOWEST site's Nyquist before resampling, so the
    multi-site diagnosis stores share one effective bandwidth. Drop that leg and every stored row
    still claims ``nyquist_tr=3.0`` while carrying the run's own, wider band -- so assert the
    spectrum, not the label: a 4 s-period sinusoid is above the 3.0 s Nyquist and must be crushed."""
    mask = np.zeros((4, 4, 4), bool)
    mask[1, 1, 1] = mask[2, 2, 2] = True
    above_band = np.tile(np.sin(2 * np.pi * np.arange(100) / 4.0).astype(np.float32), (2, 1))
    fast = ValidVolume(above_band, mask, RAD, tr=1.0)

    kept = zscore_resample(fast, target_tr=1.0, common_tr=None)
    harmonized = zscore_resample(fast, target_tr=1.0, common_tr=3.0)
    assert kept.values.std() > 10 * harmonized.values.std()

    slow, _ = _vol(t=40, tr=2.5)
    out_slow = zscore_resample(slow, target_tr=1.0, common_tr=3.0)
    assert harmonized.tr == out_slow.tr == pytest.approx(1.0)
    assert harmonized.nyquist_tr == out_slow.nyquist_tr == pytest.approx(3.0)
