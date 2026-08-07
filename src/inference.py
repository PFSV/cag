"""Single and resumable batch inference using a pre-filled KV cache."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from time import perf_counter

import torch

from config import DEFAULT_MAX_NEW_TOKENS, DEFAULT_ROPE_THETA
from src.cache_builder import load_model_and_tokenizer, render_chat_prompt

LOGGER = logging.getLogger(__name__)


def clean_up(cache, origin_len: int) -> None:
    """Truncate query tokens so the pre-filled cache can be reused."""
    if hasattr(cache, "crop"):
        cache.crop(origin_len)
        return
    for index in range(len(cache.key_cache)):
        cache.key_cache[index] = cache.key_cache[index][:, :, :origin_len, :]
        cache.value_cache[index] = cache.value_cache[index][:, :, :origin_len, :]


def generate(model, input_ids, cache, max_new_tokens: int) -> torch.Tensor:
    """Greedily generate tokens after the cached prefix and question."""
    device = model.get_input_embeddings().weight.device
    input_ids = input_ids.to(device)
    prompt_len = input_ids.shape[-1]
    next_token = input_ids
    generated = input_ids.clone()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=next_token, past_key_values=cache, use_cache=True)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True).to(device)
            cache = outputs.past_key_values
            generated = torch.cat([generated, next_token], dim=1)
            eos_ids = model.config.eos_token_id
            eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
            if next_token.item() in eos_ids:
                break
    return generated[:, prompt_len:]


def answer_one(model, tokenizer, cache, origin_len: int, question: str, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS, prefix_messages: list[dict[str, str]] | None = None) -> str:
    """Answer one question using the cache and the tokenizer's chat format."""
    clean_up(cache, origin_len)
    if prefix_messages:
        prefix_prompt = render_chat_prompt(tokenizer, prefix_messages)
        full_prompt = render_chat_prompt(
            tokenizer,
            prefix_messages + [{"role": "user", "content": question}],
            add_generation_prompt=True,
        )
        prefix_ids = tokenizer(prefix_prompt, return_tensors="pt").input_ids
        full_ids = tokenizer(full_prompt, return_tensors="pt").input_ids
        if prefix_ids.shape[-1] != origin_len or not torch.equal(
            full_ids[:, :origin_len], prefix_ids
        ):
            raise ValueError(
                "The tokenizer did not preserve the cached prefix. "
                "Build and query the cache with the same model/tokenizer."
            )
        input_ids = full_ids[:, origin_len:]
    else:
        # Compatibility path for cache artifacts created by the old scripts.
        input_ids = tokenizer(f"[user]\n{question}\n[assistant]\n", return_tensors="pt").input_ids
    output = generate(model, input_ids, cache, max_new_tokens)
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


def load_cache(path: Path):
    """Load a cache artifact created by :func:`build_cache`."""
    saved = torch.load(path, weights_only=True)
    return saved["kv"], saved["origin_len"]


def run_query(cache_path: Path, question: str, model_name: str, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS, use_yarn: bool = False, rope_theta: float = DEFAULT_ROPE_THETA) -> str:
    """Load model/cache and answer one question."""
    model, tokenizer = load_model_and_tokenizer(model_name, use_yarn, rope_theta)
    saved = torch.load(cache_path, weights_only=True)
    cached_model = saved.get("model_name")
    if cached_model and cached_model != model_name:
        raise ValueError(f"Cache was built for {cached_model}, but query requested {model_name}")
    return answer_one(model, tokenizer, saved["kv"], saved["origin_len"], question, max_new_tokens, saved.get("messages"))


def run_batch(cache_path: Path, questions_path: Path, output_path: Path, model_name: str, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS, use_yarn: bool = False, rope_theta: float = DEFAULT_ROPE_THETA) -> tuple[int, int]:
    """Process one question per line and flush each JSONL result immediately."""
    questions = [line.strip() for line in questions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    model, tokenizer = load_model_and_tokenizer(model_name, use_yarn, rope_theta)
    saved = torch.load(cache_path, weights_only=True)
    cached_model = saved.get("model_name")
    if cached_model and cached_model != model_name:
        raise ValueError(f"Cache was built for {cached_model}, but query requested {model_name}")
    cache, origin_len = saved["kv"], saved["origin_len"]
    prefix_messages = saved.get("messages")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for index, question in enumerate(questions):
            started = perf_counter()
            result = {"index": index, "question": question, "answer": None, "elapsed_sec": None, "error": None}
            try:
                result["answer"] = answer_one(model, tokenizer, cache, origin_len, question, max_new_tokens, prefix_messages)
                success += 1
            except Exception as exc:  # preserve progress for long-running batches
                result["error"] = repr(exc)
                LOGGER.exception("Question %d failed", index)
                clean_up(cache, origin_len)
            result["elapsed_sec"] = round(perf_counter() - started, 3)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
    return success, len(questions) - success
