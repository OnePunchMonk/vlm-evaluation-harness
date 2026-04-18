"""HuggingFace Transformers adapter for local VLMs."""

from __future__ import annotations

import time

from PIL import Image

from vlm_harness.adapters.base import ConversationTurn, VLMResponse


class HuggingFaceAdapter:
    """Adapter for local models via HuggingFace Transformers."""

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        torch_dtype: str = "auto",
        batch_size: int = 1,
        trust_remote_code: bool = True,
    ):
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForVision2Seq, pipeline
        except ImportError:
            raise ImportError("pip install vlm-harness[huggingface]")

        import torch as _torch
        from transformers import AutoProcessor, AutoModelForVision2Seq

        dtype_map = {
            "auto": "auto",
            "float16": _torch.float16,
            "bfloat16": _torch.bfloat16,
            "float32": _torch.float32,
        }
        dtype = dtype_map.get(torch_dtype, "auto")

        self._model_id = model_id
        self._batch_size = batch_size

        self._processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        self._model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def supports_multi_image(self) -> bool:
        return True

    @property
    def supports_video(self) -> bool:
        return False

    @property
    def max_resolution(self) -> tuple[int, int] | None:
        return None

    @property
    def cost_per_million_input_tokens(self) -> float | None:
        return None

    @property
    def cost_per_million_output_tokens(self) -> float | None:
        return None

    def generate(
        self,
        images: list[Image.Image | str],
        prompt: str,
        system: str | None = None,
        history: list[ConversationTurn] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> VLMResponse:
        import torch

        pil_images = [
            Image.open(img) if isinstance(img, str) else img for img in images
        ]

        full_prompt = (f"{system}\n\n" if system else "") + prompt

        inputs = self._processor(
            text=full_prompt,
            images=pil_images if pil_images else None,
            return_tensors="pt",
        ).to(self._model.device)

        gen_kwargs: dict = {"max_new_tokens": max_tokens}
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000

        input_len = inputs["input_ids"].shape[-1]
        new_ids = output_ids[:, input_len:]
        text = self._processor.batch_decode(new_ids, skip_special_tokens=True)[0]

        return VLMResponse(
            text=text.strip(),
            input_tokens=input_len,
            output_tokens=new_ids.shape[-1],
            latency_ms=latency_ms,
            model_id=self._model_id,
        )
