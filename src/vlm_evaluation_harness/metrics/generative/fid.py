"""FID (Frechet Inception Distance).

The distance math (`frechet_distance` / `compute_fid`) is pure numpy/scipy
and is unit-testable without any model weights or network access. Extracting
Inception features from real images (`extract_inception_features`) needs
torchvision and downloads pretrained weights on first use — that part is
only exercised when a `fid` metric is actually run against real images.
"""

from __future__ import annotations

import numpy as np

from vlm_evaluation_harness.metrics.base import MetricResult


def frechet_distance(
    mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6
) -> float:
    """Frechet distance between two multivariate Gaussians N(mu1, sigma1), N(mu2, sigma2)."""
    import scipy.linalg

    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def _statistics(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = features.mean(axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def compute_fid(features_a: np.ndarray, features_b: np.ndarray) -> float:
    """FID between two feature matrices of shape (n_samples, n_dims)."""
    mu1, sigma1 = _statistics(features_a)
    mu2, sigma2 = _statistics(features_b)
    return frechet_distance(mu1, sigma1, mu2, sigma2)


def extract_inception_features(images: list) -> np.ndarray:
    """Extract 2048-d InceptionV3 pool features for a batch of PIL images."""
    try:
        import torch
        from torchvision.models import Inception_V3_Weights, inception_v3
    except ImportError:
        raise ImportError("pip install vlm-evaluation-harness[generative]")

    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = inception_v3(weights=weights, aux_logits=True)
    model.fc = torch.nn.Identity()
    model.eval()

    preprocess = weights.transforms()
    batch = torch.stack([preprocess(img.convert("RGB")) for img in images])
    with torch.no_grad():
        features = model(batch)
    return features.numpy()


class FIDMetric:
    """Batch-level metric comparing all generated images in a run against a
    reference directory of real images.

    Needs a reasonably large sample to be meaningful (the original paper
    recommends >=2048 images per side); on small benchmarks treat the
    number as a rough signal, not a calibrated score.
    """

    def __init__(self, reference_dir: str):
        self._reference_dir = reference_dir

    def compute(self, images: list, metadata: list[dict] | None = None) -> MetricResult:
        from pathlib import Path

        from PIL import Image as PILImage

        ref_paths = sorted(p for p in Path(self._reference_dir).glob("*") if p.is_file())
        if not ref_paths:
            raise FileNotFoundError(f"No reference images found in {self._reference_dir}")
        ref_images = [PILImage.open(p) for p in ref_paths]

        gen_features = extract_inception_features(images)
        ref_features = extract_inception_features(ref_images)
        fid = compute_fid(gen_features, ref_features)

        return MetricResult(
            metric_name="fid",
            value=fid,
            n_samples=len(images),
            n_scored=len(images),
            metadata={"n_reference": len(ref_images)},
        )
