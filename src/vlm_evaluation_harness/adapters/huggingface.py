"""HuggingFace Transformers adapter for local VLMs."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from PIL import Image

from vlm_evaluation_harness.adapters.base import ChoiceScores, ConversationTurn, VLMResponse

logger = logging.getLogger(__name__)


@dataclass
class BatchGenerateRequest:
    """One sample's worth of `generate()` arguments, for `generate_batch`."""

    images: list
    prompt: str
    system: str | None = None
    history: list[ConversationTurn] | None = None
    max_tokens: int = 1024
    temperature: float = 0.0
    metadata: dict = field(default_factory=dict)


# Starting point for batch_size="auto". Deliberately optimistic --
# generate_batch's OOM backoff halves on the first failure, so a too-high
# start costs one failed attempt, while a too-low start would silently
# under-utilize the GPU for the whole run.
_AUTO_BATCH_START = 32


def _is_oom_error(exc: BaseException) -> bool:
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
        device_map: str | dict | None = None,
    ):
        try:
            import torch as _torch
            from transformers import AutoProcessor
        except ImportError:
            raise ImportError("pip install vlm-evaluation-harness[huggingface]")

        try:
            # transformers >=4.46 renamed this; keep working on older pins too.
            from transformers import AutoModelForImageTextToText as _AutoModelForVLM
        except ImportError:
            from transformers import AutoModelForVision2Seq  # type: ignore[attr-defined]

            _AutoModelForVLM = AutoModelForVision2Seq  # type: ignore[misc]

        dtype_map = {
            "auto": "auto",
            "float16": _torch.float16,
            "bfloat16": _torch.bfloat16,
            "float32": _torch.float32,
        }
        dtype = dtype_map.get(torch_dtype, "auto")

        self._model_id = model_id
        self._batch_size = _AUTO_BATCH_START if batch_size == "auto" else int(batch_size)

        # `device_map` (accelerate-backed sharding across multiple GPUs) takes
        # priority over `device` when explicitly given; `device` alone still
        # works exactly as before (it's passed straight through as
        # `device_map=device`, which is what makes "auto" already spread a
        # model across all visible GPUs when `accelerate` is installed).
        resolved_device_map = device_map if device_map is not None else device
        if resolved_device_map not in (None, "cpu") and not isinstance(
            resolved_device_map, dict
        ):
            try:
                import accelerate  # noqa: F401
            except ImportError:
                raise ImportError(
                    "device_map requires the 'accelerate' package: "
                    "pip install vlm-evaluation-harness[huggingface]"
                )

        self._processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        self._model = _AutoModelForVLM.from_pretrained(
            model_id,
            device_map=resolved_device_map,
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

    def _render_prompt(
        self,
        images: list,
        prompt: str,
        system: str | None,
        history: list[ConversationTurn] | None,
        prompt_parts: list | None = None,
    ) -> tuple[str, list]:
        """Render (system, history, prompt) into model input text.

        Uses the processor's/tokenizer's own `apply_chat_template` when the
        checkpoint ships one (true for most instruction-tuned VLMs) so the
        text matches what the model was actually fine-tuned on, instead of a
        hand-rolled format that silently produces worse-than-real numbers.
        Falls back to a plain concatenation when no chat template is
        available. Either way, `history` (e.g. multi-turn few-shot examples)
        is now actually included — previously it was accepted as a parameter
        here and silently dropped.

        `prompt_parts` (a list of PromptPart, see adapters/base.py), when
        given, is only consulted for the user message's content when a chat
        template is available -- it carries the exact interleaved order
        images should appear relative to the prompt text, in place of the
        images-then-text order `images`/`prompt` would otherwise produce.
        The plain-concatenation fallback below ignores it: without a chat
        template there's no per-image markup to interleave into, so images
        still all attach as a flat block regardless.
        """
        apply_chat_template = getattr(self._processor, "apply_chat_template", None)
        tokenizer = getattr(self._processor, "tokenizer", None)
        chat_template = getattr(self._processor, "chat_template", None) or getattr(
            tokenizer, "chat_template", None
        )
        if apply_chat_template is None and tokenizer is not None:
            apply_chat_template = getattr(tokenizer, "apply_chat_template", None)

        all_images: list = []
        if chat_template and apply_chat_template is not None:
            messages = []
            if system:
                messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
            for turn in history or []:
                content = [{"type": "image"} for _ in turn.images]
                content.append({"type": "text", "text": turn.text})
                messages.append({"role": turn.role, "content": content})
                all_images.extend(turn.images)
            if prompt_parts:
                content = []
                for p in prompt_parts:
                    if p.kind == "image":
                        content.append({"type": "image"})
                        all_images.append(p.image)
                    else:
                        content.append({"type": "text", "text": p.text})
            else:
                content = [{"type": "image"} for _ in images]
                content.append({"type": "text", "text": prompt})
                all_images.extend(images)
            messages.append({"role": "user", "content": content})
            text = apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return text, all_images

        parts = []
        if system:
            parts.append(system)
        for turn in history or []:
            parts.append(f"{turn.role}: {turn.text}")
            all_images.extend(turn.images)
        parts.append(prompt)
        all_images.extend(images)
        return "\n\n".join(parts), all_images

    def generate(
        self,
        images: list[Image.Image | str],
        prompt: str,
        system: str | None = None,
        history: list[ConversationTurn] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        parts: list | None = None,
    ) -> VLMResponse:
        import torch

        full_prompt, all_images = self._render_prompt(images, prompt, system, history, parts)
        pil_images = [
            Image.open(img) if isinstance(img, str) else img for img in all_images
        ]

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

    @property
    def supports_batch_inference(self) -> bool:
        """Whether `generate_batch` runs requests through one forward pass."""
        return True

    def generate_batch(self, requests: list[BatchGenerateRequest]) -> list[VLMResponse]:
        """Generate responses for multiple samples in as few forward passes
        as possible, backing off the batch size on out-of-memory errors.

        Requests may mix prompts with and without images; all requests in a
        given sub-batch are padded together by the processor. Order of the
        returned list matches the order of `requests`.
        """
        if not requests:
            return []
        return self._generate_batch_with_backoff(requests, self._batch_size)

    def _generate_batch_with_backoff(
        self, requests: list[BatchGenerateRequest], batch_size: int
    ) -> list[VLMResponse]:
        results: list[VLMResponse] = []
        for i in range(0, len(requests), max(1, batch_size)):
            chunk = requests[i : i + max(1, batch_size)]
            results.extend(self._run_chunk_with_backoff(chunk))
        return results

    def _run_chunk_with_backoff(
        self, chunk: list[BatchGenerateRequest]
    ) -> list[VLMResponse]:
        size = len(chunk)
        while True:
            try:
                return self._generate_forward(chunk)
            except Exception as exc:  # noqa: BLE001 - re-raised below if not OOM
                if not _is_oom_error(exc) or size <= 1:
                    raise
                try:
                    import torch

                    torch.cuda.empty_cache()
                except ImportError:
                    pass
                new_size = max(1, size // 2)
                logger.warning(
                    "HuggingFaceAdapter: OOM at batch size %d, retrying at %d",
                    size,
                    new_size,
                )
                size = new_size
                results: list[VLMResponse] = []
                for j in range(0, len(chunk), size):
                    results.extend(self._run_chunk_with_backoff(chunk[j : j + size]))
                return results

    def _generate_forward(self, chunk: list[BatchGenerateRequest]) -> list[VLMResponse]:
        import torch

        rendered = [
            self._render_prompt(req.images, req.prompt, req.system, req.history)
            for req in chunk
        ]
        full_prompts = [text for text, _ in rendered]
        pil_images_per_request = [
            [Image.open(img) if isinstance(img, str) else img for img in all_imgs]
            for _, all_imgs in rendered
        ]
        # transformers image-text processors accept a list of per-sample
        # image lists for batched multi-image inputs; a request with no
        # images contributes an empty list.
        has_any_images = any(pil_images_per_request)

        inputs = self._processor(
            text=full_prompts,
            images=pil_images_per_request if has_any_images else None,
            return_tensors="pt",
            padding=True,
        ).to(self._model.device)

        max_tokens = max(req.max_tokens for req in chunk)
        temperature = max(req.temperature for req in chunk)
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
        texts = self._processor.batch_decode(new_ids, skip_special_tokens=True)

        responses = []
        for req, text in zip(chunk, texts):
            responses.append(
                VLMResponse(
                    text=text.strip(),
                    input_tokens=input_len,
                    output_tokens=new_ids.shape[-1],
                    latency_ms=latency_ms / len(chunk),
                    model_id=self._model_id,
                    metadata=req.metadata,
                )
            )
        return responses

    @property
    def supports_choice_scoring(self) -> bool:
        return True

    def score_choices(
        self,
        images: list[Image.Image | str],
        prompt: str,
        choices: list[str],
        system: str | None = None,
    ) -> ChoiceScores:
        """Log-probability of each choice as a continuation of the prompt.

        This is the scoring path used by MC benchmarks with
        `scoring: loglikelihood`, and is what makes results comparable with
        published open-weight leaderboard numbers. Both the summed and the
        length-normalized log-probability are returned; the runner uses the
        normalized one by default so that longer options are not penalized.
        """
        import torch

        pil_images = [Image.open(img) if isinstance(img, str) else img for img in images]
        full_prompt = (f"{system}\n\n" if system else "") + prompt

        logprobs: list[float] = []
        per_token: list[float] = []

        t0 = time.perf_counter()
        for choice in choices:
            context = self._processor(
                text=full_prompt,
                images=pil_images if pil_images else None,
                return_tensors="pt",
            ).to(self._model.device)
            full = self._processor(
                text=full_prompt + choice,
                images=pil_images if pil_images else None,
                return_tensors="pt",
            ).to(self._model.device)

            context_len = context["input_ids"].shape[-1]
            input_ids = full["input_ids"]
            n_choice_tokens = input_ids.shape[-1] - context_len
            if n_choice_tokens <= 0:
                logprobs.append(float("-inf"))
                per_token.append(float("-inf"))
                continue

            with torch.no_grad():
                logits = self._model(**full).logits

            # Predict token t from position t-1.
            log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
            targets = input_ids[:, 1:]
            token_logprobs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
            choice_logprobs = token_logprobs[0, context_len - 1 :]

            total = float(choice_logprobs.sum())
            logprobs.append(total)
            per_token.append(total / n_choice_tokens)

        return ChoiceScores(
            logprobs=logprobs,
            logprobs_per_token=per_token,
            latency_ms=(time.perf_counter() - t0) * 1000,
            model_id=self._model_id,
        )
