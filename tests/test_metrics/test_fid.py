"""Tests for FID math (feature extraction needs the `generative` extra and is
not exercised here — see extract_inception_features)."""

import numpy as np
import pytest

pytest.importorskip("scipy")

from vlm_evaluation_harness.metrics.generative.fid import compute_fid, frechet_distance


def test_identical_distributions_have_zero_fid():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(200, 16))
    fid = compute_fid(features, features)
    assert fid == pytest.approx(0.0, abs=1e-6)


def test_different_distributions_have_positive_fid():
    rng = np.random.default_rng(0)
    a = rng.normal(loc=0.0, size=(200, 16))
    b = rng.normal(loc=5.0, size=(200, 16))
    fid = compute_fid(a, b)
    assert fid > 0


def test_larger_mean_shift_increases_fid():
    rng = np.random.default_rng(0)
    a = rng.normal(loc=0.0, size=(200, 8))
    b_close = rng.normal(loc=1.0, size=(200, 8))
    b_far = rng.normal(loc=10.0, size=(200, 8))
    assert compute_fid(a, b_far) > compute_fid(a, b_close)


def test_frechet_distance_symmetric_for_identical_gaussians():
    mu = np.zeros(4)
    sigma = np.eye(4)
    assert frechet_distance(mu, sigma, mu, sigma) == pytest.approx(0.0, abs=1e-6)
