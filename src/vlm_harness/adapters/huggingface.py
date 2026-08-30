"""HuggingFace Transformers adapter for local VLMs."""

from __future__ import annotations

import time

from PIL import Image

from vlm_harness.adapters.base import ConversationTurn, VLMResponse

# Starting point for batch-size == "auto" probing. Deliberately optimistic --
# generate_auto_batch halves on the first OOM, so a too-high start just costs
# one failed attempt, while a too-low start would silently under-utilize the
# GPU for the whole run.
_AUTO_BATCH_START = 32


def _is_oom_error(exc: Exception) -> bool:
    """True for a CUDA/accelerator out-of-memory error, cheaply and without
    a hard torch import (torch's own OutOfMemoryError subclasses RuntimeError,
    but so does everything else CUDA raises -- so also pattern-match the
    message, matching torch's own convention of putting "out of memory" in
    it)."""
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except ImportError:
        pass
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


class HuggingFaceAdapter:
    """Adapter for local models via HuggingFace Transformers."""

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        torch_dtype: str = "auto",
        batch_size: int | str = 1,
        trust_remote_code: bool = True,
    ):
        try:
            import torch
        except ImportError:
            raise ImportError("pip install vlm-harness[huggingface]")

        try:
            # transformers >=4.46 renamed this; keep working on older pins too.
            from transformers import AutoModelForImageTextToText as _AutoModelForVLM
        except ImportError:
            from transformers import AutoModelForVision2Seq  # type: ignore[attr-defined]

            _AutoModelForVLM = AutoModelForVision2Seq  # type: ignore[misc,no-redef]
        from transformers import AutoProcessor

        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(torch_dtype, "auto")

        self._model_id = model_id
        # "auto" starts optimistic and halves on OOM (see generate_auto_batch);
        # an explicit int is used as-is and never adjusted.
        self._auto_batch_size = batch_size == "auto"
        self._batch_size = _AUTO_BATCH_START if self._auto_batch_size else int(batch_size)

        self._processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        self._model = _AutoModelForVLM.from_pretrained(
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

        pil_images = [Image.open(img) if isinstance(img, str) else img for img in images]

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
            output_ids = self._model.generate(**inputs, **gen_kwargs)  # type: ignore[misc]
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

    def generate_batch(
        self,
        requests: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> list[VLMResponse]:
        """Batched counterpart to `generate`.

        Each element of `requests` is a dict with the same keys as
        `generate`'s arguments (`images`, `prompt`, `system`). Left-padding
        the processor's tokenizer keeps every sequence's *new* tokens at a
        fixed offset from the right edge of `generate`'s output, so per-item
        input/output token counts can be recovered from the shared attention
        mask instead of guessing from the batch's padded input length.

        Must produce sample-for-sample identical text to calling `generate`
        once per request (temperature=0) -- see
        tests/test_adapters/test_huggingface_batching.py.
        """
        import torch

        if not requests:
            return []

        pad_side = self._processor.tokenizer.padding_side
        self._processor.tokenizer.padding_side = "left"
        try:
            pil_images_per_req = [
                [Image.open(img) if isinstance(img, str) else img for img in r.get("images", [])]
                for r in requests
            ]
            full_prompts = [
                (f"{r['system']}\n\n" if r.get("system") else "") + r["prompt"] for r in requests
            ]

            inputs = self._processor(
                text=full_prompts,
                images=pil_images_per_req if any(pil_images_per_req) else None,
                return_tensors="pt",
                padding=True,
            ).to(self._model.device)
        finally:
            self._processor.tokenizer.padding_side = pad_side

        gen_kwargs: dict = {"max_new_tokens": max_tokens}
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)  # type: ignore[misc]
        latency_ms = (time.perf_counter() - t0) * 1000

        input_len = inputs["input_ids"].shape[-1]
        new_ids = output_ids[:, input_len:]
        texts = self._processor.batch_decode(new_ids, skip_special_tokens=True)

        attention_mask = inputs["attention_mask"]
        responses = []
        for i, text in enumerate(texts):
            per_sample_input_len = int(attention_mask[i].sum().item())
            responses.append(
                VLMResponse(
                    text=text.strip(),
                    input_tokens=per_sample_input_len,
                    output_tokens=new_ids.shape[-1],
                    latency_ms=latency_ms / len(requests),
                    model_id=self._model_id,
                )
            )
        return responses

    def generate_auto_batch(
        self,
        requests: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> list[VLMResponse]:
        """Run `requests` through generate_batch(), sized to fit GPU memory.

        Starts at self._batch_size (32 when batch_size="auto", otherwise the
        configured size) and, on OOM, clears the CUDA cache and progressively
        halves the batch size until a chunk fits or the floor of 1 is hit (a
        floor-1 OOM propagates -- a single sample not fitting isn't a batch
        sizing problem). Once reduced, self._batch_size stays reduced for the
        rest of this adapter's life; it never grows back, since re-probing
        upward risks repeatedly re-triggering the OOM it just backed off
        from.
        """
        results: list[VLMResponse] = []
        i = 0
        while i < len(requests):
            chunk = requests[i : i + self._batch_size]
            try:
                results.extend(
                    self.generate_batch(chunk, max_tokens=max_tokens, temperature=temperature)
                )
                i += len(chunk)
            except Exception as exc:
                if not _is_oom_error(exc) or self._batch_size <= 1:
                    raise
                try:
                    import torch

                    torch.cuda.empty_cache()
                except ImportError:
                    pass
                self._batch_size = max(1, self._batch_size // 2)
        return results
