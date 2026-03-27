"""Local KV-press generation runner adapted from autokv/kvpress/pipeline.py."""

from __future__ import annotations

import contextlib
from typing import Any, Optional

import torch
from transformers import Cache, DynamicCache, PreTrainedModel, PreTrainedTokenizerBase

from benchmark.kvpress.base_press import BasePress

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


class KVPressTextGenerationRunner:
    """Run context prefill with optional press, then greedy decode answers."""

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(
        self,
        context: str,
        question: Optional[str] = None,
        questions: Optional[list[str]] = None,
        answer_prefix: Optional[str] = None,
        press: Optional[BasePress] = None,
        max_new_tokens: int = 50,
        max_context_length: Optional[int] = None,
        enable_thinking: bool = False,
        cache: Optional[Cache] = None,
    ) -> dict[str, str | list[str]]:
        single_question = questions is None
        assert question is None or questions is None, "Either question or questions should be provided, not both."
        questions = questions or ([question] if question else [""])

        if max_context_length is None:
            max_context_length = min(self.tokenizer.model_max_length, int(1e10))
        answer_prefix = answer_prefix or ""

        input_tensors = self.preprocess(
            context=context,
            questions=questions,
            answer_prefix=answer_prefix,
            max_context_length=max_context_length,
            enable_thinking=enable_thinking,
        )
        answers = self._forward(
            input_tensors=input_tensors,
            max_new_tokens=max_new_tokens,
            press=press,
            cache=cache,
        )

        if single_question:
            return {"answer": answers[0]}
        return {"answers": answers}

    def preprocess(
        self,
        context: str,
        questions: list[str],
        answer_prefix: str,
        max_context_length: int,
        enable_thinking: bool = False,
    ) -> dict[str, Any]:
        if self.tokenizer.chat_template is None:
            bos_token = getattr(self.tokenizer, "bos_token", "")
            context_with_prompt = bos_token + context
            question_suffix = "\n"
        else:
            separator = "#" * (len(context) + 10)
            messages = [{"role": "user", "content": context + separator}]
            try:
                templated = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=enable_thinking,
                )
            except TypeError:
                templated = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            if separator not in templated:
                raise RuntimeError("Failed to split context and question prompt using separator.")
            context_with_prompt, question_suffix = templated.split(separator, maxsplit=1)

        full_questions = [f"{question}{question_suffix}{answer_prefix}" for question in questions]

        context_ids = self.tokenizer.encode(context_with_prompt, return_tensors="pt", add_special_tokens=False)
        question_ids = [
            self.tokenizer.encode(full_question, return_tensors="pt", add_special_tokens=False)
            for full_question in full_questions
        ]

        if context_ids.shape[1] > max_context_length:
            context_ids = context_ids[:, :max_context_length]

        return {"context_ids": context_ids, "questions_ids": question_ids}

    def _forward(
        self,
        input_tensors: dict[str, Any],
        max_new_tokens: int = 50,
        press: Optional[BasePress] = None,
        cache: Optional[Cache] = None,
    ) -> list[str]:
        context_ids = input_tensors["context_ids"].to(self.model.device)
        context_length = context_ids.shape[1]

        if cache is None:
            cache = DynamicCache()

        with press(self.model) if press is not None else contextlib.nullcontext():
            backbone = self.model.model if hasattr(self.model, "model") else self.model
            backbone(
                input_ids=context_ids,
                past_key_values=cache,
            )

        answers: list[str] = []
        question_ids_list = input_tensors["questions_ids"]
        question_iterator = question_ids_list
        if tqdm is not None and len(question_ids_list) > 1:
            question_iterator = tqdm(
                question_ids_list,
                total=len(question_ids_list),
                desc="Generating Answers",
                leave=False,
            )

        for question_ids in question_iterator:
            cache_seq_lengths = [cache.get_seq_length(layer_idx) for layer_idx in range(len(cache))]
            answer = self.generate_answer(
                question_ids=question_ids.to(self.model.device),
                cache=cache,
                context_length=context_length,
                max_new_tokens=max_new_tokens,
            )
            self._remove_answer_from_cache(cache, cache_seq_lengths)
            answers.append(answer)

        return answers

    def _remove_answer_from_cache(self, cache: Cache, cache_seq_lengths: list[int]) -> None:
        for layer_idx, sequence_length in enumerate(cache_seq_lengths):
            cache_layer = cache.layers[layer_idx]
            if hasattr(cache_layer, "keys") and isinstance(cache_layer.keys, torch.Tensor):
                cache_layer.keys = cache_layer.keys[:, :, :sequence_length]
            if hasattr(cache_layer, "values") and isinstance(cache_layer.values, torch.Tensor):
                cache_layer.values = cache_layer.values[:, :, :sequence_length]
            if hasattr(cache_layer, "_quantized_keys") and isinstance(cache_layer._quantized_keys, torch.Tensor):
                cache_layer._quantized_keys = cache_layer._quantized_keys[:, :, :sequence_length]
            if hasattr(cache_layer, "_quantized_values") and isinstance(cache_layer._quantized_values, torch.Tensor):
                cache_layer._quantized_values = cache_layer._quantized_values[:, :, :sequence_length]

    def generate_answer(self, question_ids: torch.Tensor, cache: Cache, context_length: int, max_new_tokens: int) -> str:
        position_ids = torch.arange(
            context_length,
            context_length + question_ids.shape[1],
            device=self.model.device,
        ).unsqueeze(0)

        outputs = self.model(
            input_ids=question_ids,
            past_key_values=cache,
            position_ids=position_ids,
        )

        position_ids = position_ids[:, -1:] + 1
        generated_ids = [outputs.logits[0, -1].argmax()]

        eos_token_ids = self.model.generation_config.eos_token_id
        if eos_token_ids is None:
            eos_token_ids = []
        if not isinstance(eos_token_ids, list):
            eos_token_ids = [eos_token_ids]

        for step in range(max_new_tokens - 1):
            outputs = self.model(
                input_ids=generated_ids[-1].view(1, 1),
                past_key_values=cache,
                position_ids=position_ids + step,
            )
            new_id = outputs.logits[0, -1].argmax()
            generated_ids.append(new_id)
            if new_id.item() in eos_token_ids:
                break

        answer_ids = torch.stack(generated_ids).detach().cpu()
        return str(self.tokenizer.decode(answer_ids, skip_special_tokens=True))

