"""
Generic HuggingFace VLM wrapper.

Supports any model that can be loaded via:
  AutoModelForVision2Seq  (primary path — LLaVA, Qwen2-VL, PaliGemma, etc.)
  AutoModelForCausalLM    (fallback for LLaMA-3.2-Vision, Phi-3-Vision)

Two answer strategies:
  "generate"  — generate text, extract answer with regex (default, universal)
  "logprob"   — compute log-likelihood of each choice token, pick highest
                (more accurate for MC but not supported on all models)

Model-family-specific patches are applied automatically via _FAMILY_PATCHES.
"""

from __future__ import annotations

import re
import logging
from typing import Literal, Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)


# ── Model family detection ────────────────────────────────────────────────────

def _detect_family(model_id: str) -> str:
    mid = model_id.lower()
    if "llava" in mid:
        return "llava"
    if "qwen" in mid and ("vl" in mid or "vision" in mid):
        return "qwen_vl"
    if "internvl" in mid:
        return "internvl"
    if "paligemma" in mid:
        return "paligemma"
    if "llama" in mid and ("vision" in mid or "11b" in mid or "90b" in mid):
        return "llama_vision"
    if "phi" in mid and ("vision" in mid or "3.5" in mid or "4" in mid):
        return "phi_vision"
    if "idefics" in mid:
        return "idefics"
    if "molmo" in mid:
        return "molmo"
    return "generic"


# ── Per-family chat template formatters ──────────────────────────────────────

def _format_prompt_generic(prompt: str, processor) -> str:
    """Try chat template; fall back to plain prompt."""
    if hasattr(processor, "apply_chat_template"):
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        try:
            return processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        except Exception:
            pass
    return prompt


def _format_prompt_llava(prompt: str, processor) -> str:
    return f"USER: <image>\n{prompt}\nASSISTANT:"


def _format_prompt_qwen_vl(prompt: str, processor) -> str:
    # Qwen2-VL uses chat template with vision tokens handled by processor
    messages = [{"role": "user", "content": [
        {"type": "image", "image": "placeholder"},
        {"type": "text", "text": prompt},
    ]}]
    if hasattr(processor, "apply_chat_template"):
        try:
            return processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        except Exception:
            pass
    return f"<|im_start|>user\n<img/>\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def _format_prompt_paligemma(prompt: str, processor) -> str:
    return prompt  # PaliGemma processor handles formatting internally


def _format_prompt_llama_vision(prompt: str, processor) -> str:
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": prompt},
    ]}]
    if hasattr(processor, "apply_chat_template"):
        try:
            return processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        except Exception:
            pass
    return f"<|image|>\n{prompt}"


_FAMILY_FORMATTERS = {
    "llava": _format_prompt_llava,
    "qwen_vl": _format_prompt_qwen_vl,
    "paligemma": _format_prompt_paligemma,
    "llama_vision": _format_prompt_llama_vision,
    "generic": _format_prompt_generic,
}


# ── Main wrapper ──────────────────────────────────────────────────────────────

class VLMWrapper:
    """
    Wraps a HuggingFace VLM for benchmark evaluation.

    Usage:
        model = VLMWrapper("llava-hf/llava-1.5-7b-hf")
        answer = model.answer([pil_image], "What colour is the car?", ["Red","Blue"])
    """

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        strategy: Literal["generate", "logprob"] = "generate",
        max_new_tokens: int = 64,
        trust_remote_code: bool = True,
    ):
        self.model_id = model_id
        self.strategy = strategy
        self.max_new_tokens = max_new_tokens
        self.family = _detect_family(model_id)
        self._format_prompt = _FAMILY_FORMATTERS.get(self.family, _format_prompt_generic)

        logger.info(f"Loading {model_id!r} (family={self.family}, strategy={strategy})")

        quant_kwargs = {}
        if load_in_4bit or load_in_8bit:
            from transformers import BitsAndBytesConfig
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                bnb_4bit_compute_dtype=dtype,
            )

        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )

        # Try Vision2Seq first, fall back to CausalLM
        self.model = self._load_model(
            model_id, dtype, device, trust_remote_code, quant_kwargs
        )
        self.model.eval()

        # Determine device for input tensors
        if device == "auto":
            self.device = next(self.model.parameters()).device
        else:
            self.device = torch.device(device)

    def _load_model(self, model_id, dtype, device, trust_remote_code, quant_kwargs):
        from transformers import AutoModelForVision2Seq, AutoModelForCausalLM

        common = dict(
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=trust_remote_code,
            **quant_kwargs,
        )

        for cls in [AutoModelForVision2Seq, AutoModelForCausalLM]:
            try:
                return cls.from_pretrained(model_id, **common)
            except (ValueError, OSError, KeyError):
                continue

        raise RuntimeError(
            f"Could not load {model_id!r} via AutoModelForVision2Seq or AutoModelForCausalLM."
        )

    # ── Core interface ────────────────────────────────────────────────────────

    def answer(
        self,
        images: list[Image.Image],
        prompt: str,
        choices: Optional[list[str]] = None,
    ) -> str:
        """
        Generate an answer for the given image(s) + prompt.

        Args:
            images: list of PIL Images (most tasks have 1 image)
            prompt: the question / instruction text
            choices: if provided and strategy=="logprob", used for scoring

        Returns:
            Raw generated string (caller extracts letter/text as needed)
        """
        if self.strategy == "logprob" and choices:
            return self._logprob_answer(images, prompt, choices)
        return self._generate_answer(images, prompt)

    def _generate_answer(self, images: list[Image.Image], prompt: str) -> str:
        formatted = self._format_prompt(prompt, self.processor)

        try:
            inputs = self.processor(
                text=formatted,
                images=images if len(images) > 1 else images[0],
                return_tensors="pt",
                padding=True,
            ).to(self.device)
        except Exception:
            # Some processors want images as list always
            try:
                inputs = self.processor(
                    text=formatted,
                    images=images,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)
            except Exception as e:
                logger.warning(f"Processor failed: {e}")
                return ""

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        # Strip input tokens from output
        input_len = inputs["input_ids"].shape[1]
        new_ids = generated_ids[:, input_len:]
        return self.processor.decode(new_ids[0], skip_special_tokens=True).strip()

    def _logprob_answer(
        self, images: list[Image.Image], prompt: str, choices: list[str]
    ) -> str:
        """
        Score each choice by computing the log-probability of the choice
        token(s) given the prompt + image. Returns the highest-scoring choice letter.
        """
        letters = "ABCDEFGHIJ"
        formatted = self._format_prompt(prompt, self.processor)
        best_letter = "A"
        best_score = float("-inf")

        for i, choice in enumerate(choices):
            letter = letters[i]
            # Score just the letter token (most efficient for MC)
            full_text = formatted + " " + letter

            try:
                inputs = self.processor(
                    text=full_text,
                    images=images if len(images) > 1 else images[0],
                    return_tensors="pt",
                ).to(self.device)
            except Exception:
                try:
                    inputs = self.processor(
                        text=full_text,
                        images=images,
                        return_tensors="pt",
                    ).to(self.device)
                except Exception:
                    continue

            with torch.inference_mode():
                outputs = self.model(**inputs)
                logits = outputs.logits  # [1, seq_len, vocab]

            # Get log-prob of the letter token at the last position
            log_probs = torch.log_softmax(logits[0, -2, :], dim=-1)
            letter_token_ids = self.processor.tokenizer.encode(
                " " + letter, add_special_tokens=False
            )
            if not letter_token_ids:
                continue
            score = log_probs[letter_token_ids[-1]].item()

            if score > best_score:
                best_score = score
                best_letter = letter

        return best_letter

    # ── Utility ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"VLMWrapper(model_id={self.model_id!r}, family={self.family!r})"
