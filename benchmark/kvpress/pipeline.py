# Adapted from autokv/kvpress/pipeline.py
from __future__ import annotations

import contextlib
from typing import Any, Optional

import torch
from transformers import Cache, DynamicCache, PreTrainedModel, PreTrainedTokenizerBase

from benchmark.kvpress.base_press import BasePress

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class KVPressTextGenerationRunner:
    """Prefill context (optionally with a press), then greedy-decode answers.

    Pattern:  context tokens → model.model() w/ press → DynamicCache
              for each question: decode from cache, then trim cache back
    """

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
        self.model     = model
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
        cache: Optional[Cache] = None,
    ) -> dict[str, str | list[str]]:
        assert not (question and questions), "Provide question or questions, not both."
        single = questions is None
        questions = questions or ([question] if question else [""])

        max_ctx = max_context_length or min(self.tokenizer.model_max_length, int(1e10))
        tensors = self._tokenize(context, questions, answer_prefix or "", max_ctx)
        answers = self._run(tensors, press, max_new_tokens, cache)

        return {"answer": answers[0]} if single else {"answers": answers}

    # ── Tokenisation ──────────────────────────────────────────────────────────

    def _tokenize(
        self,
        context: str,
        questions: list[str],
        answer_prefix: str,
        max_context_length: int,
    ) -> dict[str, Any]:
        if self.tokenizer.chat_template is None:
            ctx_text      = getattr(self.tokenizer, "bos_token", "") + context
            question_sfx  = "\n"
        else:
            sep = "#" * (len(context) + 10)
            try:
                templated = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": context + sep}],
                    add_generation_prompt=True, tokenize=False,
                )
            except TypeError:
                templated = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": context + sep}],
                    add_generation_prompt=True, tokenize=False,
                    enable_thinking=False,
                )
            if sep not in templated:
                raise RuntimeError("Separator missing from chat template output.")
            ctx_text, question_sfx = templated.split(sep, maxsplit=1)

        ctx_ids = self.tokenizer.encode(ctx_text, return_tensors="pt", add_special_tokens=False)
        if ctx_ids.shape[1] > max_context_length:
            ctx_ids = ctx_ids[:, :max_context_length]

        q_ids = [
            self.tokenizer.encode(q + question_sfx + answer_prefix,
                                  return_tensors="pt", add_special_tokens=False)
            for q in questions
        ]
        return {"context_ids": ctx_ids, "questions_ids": q_ids}

    # ── Inference ─────────────────────────────────────────────────────────────

    def _run(
        self,
        tensors: dict[str, Any],
        press: Optional[BasePress],
        max_new_tokens: int,
        cache: Optional[Cache],
    ) -> list[str]:
        ctx_ids = tensors["context_ids"].to(self.model.device)
        ctx_len = ctx_ids.shape[1]
        if cache is None:
            cache = DynamicCache()

        # Prefill: run backbone only (no lm head needed)
        backbone = self.model.model if hasattr(self.model, "model") else self.model
        ctx_mgr  = press(self.model) if press is not None else contextlib.nullcontext()
        with ctx_mgr:
            backbone(input_ids=ctx_ids, past_key_values=cache)

        answers: list[str] = []
        q_list    = tensors["questions_ids"]
        iterator  = tqdm(q_list, desc="Decoding", leave=False) if (tqdm and len(q_list) > 1) else q_list

        for q_ids in iterator:
            snap = [cache.get_seq_length(i) for i in range(len(cache))]
            answers.append(self._decode(q_ids.to(self.model.device), cache, ctx_len, max_new_tokens))
            self._trim_cache(cache, snap)   # rewind to pre-question state for next question

        return answers

    def _decode(
        self,
        question_ids: torch.Tensor,
        cache: Cache,
        context_length: int,
        max_new_tokens: int,
    ) -> str:
        pos = torch.arange(context_length, context_length + question_ids.shape[1],
                           device=self.model.device).unsqueeze(0)
        out = self.model(input_ids=question_ids, past_key_values=cache, position_ids=pos)

        eos = self.model.generation_config.eos_token_id or []
        if not isinstance(eos, list):
            eos = [eos]

        pos    = pos[:, -1:] + 1
        tokens = [out.logits[0, -1].argmax()]
        for step in range(max_new_tokens - 1):
            out  = self.model(input_ids=tokens[-1].view(1, 1), past_key_values=cache,
                              position_ids=pos + step)
            tok  = out.logits[0, -1].argmax()
            tokens.append(tok)
            if tok.item() in eos:
                break

        return self.tokenizer.decode(torch.stack(tokens).cpu(), skip_special_tokens=True)

    def _trim_cache(self, cache: Cache, lengths: list[int]) -> None:
        """Rewind cache to lengths captured before the last question was decoded."""
        for i, seq_len in enumerate(lengths):
            layer = cache.layers[i]
            for attr in ("keys", "values"):
                t = getattr(layer, attr, None)
                if isinstance(t, torch.Tensor) and t.shape[2] > seq_len:
                    setattr(layer, attr, t[:, :, :seq_len])
            for attr in ("_quantized_keys", "_quantized_values"):
                t = getattr(layer, attr, None)
                if isinstance(t, torch.Tensor) and t.shape[2] > seq_len:
                    setattr(layer, attr, t[:, :, :seq_len])
