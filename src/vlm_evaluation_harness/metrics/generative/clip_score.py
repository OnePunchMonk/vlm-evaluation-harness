"""CLIPScore: prompt-image alignment via CLIP embedding cosine similarity.

Reference: Hessel et al., "CLIPScore: A Reference-free Evaluation Metric for
Image Captioning" (2021). Score = 2.5 * max(cos_sim(text_embed, image_embed), 0).
"""

from __future__ import annotations

from PIL import Image

from vlm_evaluation_harness.metrics.base import MetricResult

_DEFAULT_MODEL = "openai/clip-vit-base-patch32"


class CLIPScorer:
    """Wraps a HuggingFace CLIP model for prompt-image alignment scoring."""

    def __init__(self, model_id: str | None = None):
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError:
            raise ImportError("pip install vlm-evaluation-harness[generative]")

        self._torch = torch
        self._model_id = model_id or _DEFAULT_MODEL
        self._model = CLIPModel.from_pretrained(self._model_id)
        self._processor = CLIPProcessor.from_pretrained(self._model_id)
        self._model.eval()

    def score(self, prompt: str, image: Image.Image) -> float:
        inputs = self._processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
        with self._torch.no_grad():
            out = self._model(**inputs)
        img_emb = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        txt_emb = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        cos = float((img_emb * txt_emb).sum(dim=-1).item())
        return max(cos, 0.0) * 2.5

    def compute(
        self,
        prompts: list[str],
        images: list[Image.Image],
        metadata: list[dict] | None = None,
        sample_ids: list[str] | None = None,
    ) -> MetricResult:
        from vlm_evaluation_harness.metrics.base import NAN

        ids = sample_ids or [str(i) for i in range(len(prompts))]
        per_sample = {
            sample_id: self.score(p, im) for sample_id, p, im in zip(ids, prompts, images)
        }
        avg = sum(per_sample.values()) / len(per_sample) if per_sample else NAN
        return MetricResult(
            metric_name="clip_score",
            value=avg,
            n_samples=len(prompts),
            n_scored=len(per_sample),
            per_sample=per_sample,
            metadata={"clip_model": self._model_id},
        )
