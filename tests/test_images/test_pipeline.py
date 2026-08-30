"""Tests for the image pipeline."""

import pytest
from PIL import Image

from vlm_harness.images.corruption import CORRUPTION_NAMES, apply_corruption
from vlm_harness.images.pipeline import ImagePipeline, ImagePipelineConfig


def make_image(w=512, h=512, color=(128, 64, 200)):
    return Image.new("RGB", (w, h), color=color)


class TestImagePipeline:
    def test_rgb_conversion(self):
        gray = Image.new("L", (100, 100))
        pipeline = ImagePipeline()
        result = pipeline.process(gray)
        assert result.image.mode == "RGB"

    def test_downscale_large_image(self):
        big = make_image(4096, 4096)
        cfg = ImagePipelineConfig(max_resolution=(2048, 2048))
        pipeline = ImagePipeline(cfg)
        result = pipeline.process(big)
        assert result.final_size[0] <= 2048
        assert result.final_size[1] <= 2048

    def test_small_image_not_upscaled_beyond_max(self):
        small = make_image(100, 100)
        pipeline = ImagePipeline()
        result = pipeline.process(small)
        # Should not exceed max_resolution
        assert result.final_size[0] <= 2048

    def test_hash_is_deterministic(self):
        img = make_image()
        pipeline = ImagePipeline()
        r1 = pipeline.process(img.copy())
        r2 = pipeline.process(img.copy())
        assert r1.hash == r2.hash

    def test_hash_differs_for_different_images(self):
        img1 = make_image(color=(255, 0, 0))
        img2 = make_image(color=(0, 255, 0))
        pipeline = ImagePipeline()
        r1 = pipeline.process(img1)
        r2 = pipeline.process(img2)
        assert r1.hash != r2.hash

    def test_batch_processing(self):
        imgs = [make_image() for _ in range(3)]
        pipeline = ImagePipeline()
        results = pipeline.process_batch(imgs)
        assert len(results) == 3


class TestCorruption:
    def test_all_corruptions_run(self):
        img = make_image()
        for corruption in CORRUPTION_NAMES:
            result = apply_corruption(img.copy(), corruption)
            assert isinstance(result, Image.Image)

    def test_jpeg_reduces_quality(self):
        img = make_image()
        compressed = apply_corruption(img, "jpeg_compression", severity=5)
        assert isinstance(compressed, Image.Image)

    def test_rotation_90_changes_dimensions_for_non_square(self):
        img = make_image(400, 200)
        rotated = apply_corruption(img, "rotation_90")
        # With expand=True, dimensions should swap
        assert rotated.size == (200, 400)

    def test_blur_produces_different_image(self):
        import numpy as np

        # Use a non-solid image so blur actually changes pixel values
        img = Image.new("RGB", (64, 64))
        pixels = img.load()
        for x in range(64):
            for y in range(64):
                pixels[x, y] = (x * 4 % 256, y * 4 % 256, (x + y) % 256)
        blurred = apply_corruption(img, "gaussian_blur", severity=3)
        arr_orig = np.array(img)
        arr_blur = np.array(blurred)
        assert not (arr_orig == arr_blur).all()

    def test_unknown_corruption_raises(self):
        with pytest.raises(ValueError):
            apply_corruption(make_image(), "unknown_corruption")
