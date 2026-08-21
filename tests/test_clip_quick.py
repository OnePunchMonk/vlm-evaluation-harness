"""
Quick CLIP smoke-test against the harness metrics.

CLIP is a contrastive model (image-text similarity), not generative,
so it maps to yes/no and multiple-choice tasks via zero-shot softmax —
rather than through the generate() adapter interface.

This script:
  1. Downloads openai/clip-vit-base-patch32 (~600 MB)
  2. Runs zero-shot classification on synthetic VQA-style samples
  3. Reports accuracy + F1 using the harness metric engine
  4. Cleans up the weights afterward
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# ── Synthetic test data ────────────────────────────────────────────────────────
# Each sample: an RGB image + a question + candidate labels + ground truth

def make_solid_image(color: tuple[int, int, int], size=(224, 224)) -> Image.Image:
    return Image.new("RGB", size, color=color)

def make_striped_image(color1, color2, size=(224, 224)) -> Image.Image:
    img = Image.new("RGB", size, color=color1)
    draw = ImageDraw.Draw(img)
    for y in range(0, size[1], 20):
        if (y // 20) % 2 == 0:
            draw.rectangle([0, y, size[0], y + 20], fill=color2)
    return img


SAMPLES = [
    # (image, candidate_texts, ground_truth_index)
    (
        make_solid_image((220, 50, 50)),     # red image
        ["a red solid color", "a blue solid color", "a green solid color"],
        0,  # red
    ),
    (
        make_solid_image((50, 50, 210)),     # blue image
        ["a red solid color", "a blue solid color", "a green solid color"],
        1,  # blue
    ),
    (
        make_solid_image((50, 180, 50)),     # green image
        ["a red solid color", "a blue solid color", "a green solid color"],
        2,  # green
    ),
    (
        make_striped_image((255, 255, 255), (0, 0, 0)),  # black & white stripes
        ["a colorful image", "a black and white striped pattern", "a solid blue image"],
        1,  # stripes
    ),
    (
        make_solid_image((255, 165, 0)),     # orange
        ["an orange colored image", "a purple image", "a white image"],
        0,  # orange
    ),
]


def run():
    print("=" * 60)
    print("CLIP Zero-Shot Classification — VLM-Harness Quick Test")
    print("=" * 60)

    # ── Load model ────────────────────────────────────────────────────────────
    print("\n[1/4] Loading openai/clip-vit-base-patch32 ...")
    import torch
    from transformers import CLIPModel, CLIPProcessor

    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    print(f"      Loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Inference ─────────────────────────────────────────────────────────────
    print("\n[2/4] Running zero-shot classification on 5 synthetic samples ...")
    predictions = []
    references  = []

    for i, (image, candidates, gt_idx) in enumerate(SAMPLES):
        inputs = processor(
            text=candidates,
            images=image,
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            outputs = model(**inputs)
        # image-text similarity scores
        logits = outputs.logits_per_image[0]  # shape: (num_candidates,)
        probs  = logits.softmax(dim=0).numpy()
        pred_idx = int(np.argmax(probs))

        predictions.append(str(pred_idx))
        references.append(str(gt_idx))

        status = "✓" if pred_idx == gt_idx else "✗"
        print(
            f"  Sample {i+1}: {status}  "
            f"pred='{candidates[pred_idx]}'  "
            f"(conf={probs[pred_idx]:.2%})"
        )

    # ── Harness metrics ───────────────────────────────────────────────────────
    print("\n[3/4] Computing harness metrics ...")
    from vlm_harness.metrics.accuracy import AccuracyMetric
    from vlm_harness.metrics.nlp import F1Metric

    acc_result = AccuracyMetric().compute(predictions, references, [{} for _ in predictions])
    f1_result  = F1Metric().compute(predictions, references, [{} for _ in predictions])

    print(f"\n  Accuracy : {acc_result.value:.2%}  ({acc_result.n_samples} samples)")
    print(f"  Token F1 : {f1_result.value:.4f}")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    print("\n[4/4] Deleting model weights from cache ...")
    import shutil

    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    deleted_bytes = 0
    for entry in hf_cache.iterdir() if hf_cache.exists() else []:
        if "clip-vit-base-patch32" in entry.name:
            size = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
            shutil.rmtree(entry)
            deleted_bytes += size
            print(f"  Deleted: {entry.name}  ({size / 1e6:.1f} MB)")

    if deleted_bytes == 0:
        print("  (No cache entries found — may have been stored elsewhere.)")
    else:
        print(f"  Total freed: {deleted_bytes / 1e6:.1f} MB")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  RESULT: {int(acc_result.value * len(predictions))}/{len(predictions)} correct")
    print(f"  Accuracy: {acc_result.value:.2%}")
    print("=" * 60)

    assert acc_result.value >= 0.6, (
        f"CLIP accuracy {acc_result.value:.2%} below 60% threshold — something is wrong."
    )
    print("\n  All assertions passed.")


if __name__ == "__main__":
    run()
