"""HuggingFace Transformers adapter for local VLMs."""

from __future__ import annotations

import time

from PIL import Image

from vlm_evaluation_harness.adapters.base import ChoiceScores, ConversationTurn, VLMResponse


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
            import torch as _torch
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError:
            raise ImportError("pip install vlm-evaluation-harness[huggingface]")

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
