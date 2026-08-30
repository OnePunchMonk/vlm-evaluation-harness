"""Image corruption functions for robustness probing."""

from __future__ import annotations

import io

from PIL import Image, ImageFilter

CORRUPTION_NAMES = [
    "gaussian_blur",
    "gaussian_noise",
    "jpeg_compression",
    "rotation_90",
    "rotation_180",
    "rotation_270",
    "center_crop_75",
    "center_crop_50",
    "resolution_2x",
    "resolution_4x",
]


def apply_corruption(image: Image.Image, corruption: str, severity: int = 2) -> Image.Image:
    """
    Apply a named corruption to an image.

    severity: 1 (mild) to 5 (severe)
    """
    if corruption == "gaussian_blur":
        sigma = [0.5, 1.0, 2.0, 4.0, 8.0][severity - 1]
        return image.filter(ImageFilter.GaussianBlur(radius=sigma))

    elif corruption == "gaussian_noise":
        import numpy as np

        sigma = [5, 10, 25, 40, 60][severity - 1]
        arr = np.array(image).astype(np.float32)
        noise = np.random.normal(0, sigma, arr.shape)
        noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)

    elif corruption == "jpeg_compression":
        quality = [75, 50, 30, 15, 5][severity - 1]
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).copy()

    elif corruption == "rotation_90":
        return image.rotate(90, expand=True)

    elif corruption == "rotation_180":
        return image.rotate(180, expand=True)

    elif corruption == "rotation_270":
        return image.rotate(270, expand=True)

    elif corruption == "center_crop_75":
        return _center_crop(image, 0.75)

    elif corruption == "center_crop_50":
        return _center_crop(image, 0.50)

    elif corruption == "resolution_2x":
        w, h = image.size
        small = image.resize((max(1, w // 2), max(1, h // 2)), Image.Resampling.LANCZOS)
        return small.resize((w, h), Image.Resampling.NEAREST)

    elif corruption == "resolution_4x":
        w, h = image.size
        small = image.resize((max(1, w // 4), max(1, h // 4)), Image.Resampling.LANCZOS)
        return small.resize((w, h), Image.Resampling.NEAREST)

    else:
        raise ValueError(f"Unknown corruption: {corruption}")


def _center_crop(image: Image.Image, fraction: float) -> Image.Image:
    w, h = image.size
    new_w, new_h = int(w * fraction), int(h * fraction)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    cropped = image.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((w, h), Image.Resampling.LANCZOS)
