"""Pre-fill a Transformers causal language model and persist its KV cache."""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from config import DEFAULT_ROPE_THETA, ORIGINAL_CONTEXT_LENGTH, SYSTEM_PROMPT, YARN_FACTOR

LOGGER = logging.getLogger(__name__)
torch.serialization.add_safe_globals([DynamicCache, set])
try:
    from transformers.cache_utils import DynamicLayer
    torch.serialization.add_safe_globals([DynamicLayer])
except ImportError:
    pass


def build_prefix_messages(corpus: str) -> list[dict[str, str]]:
    """Return model-independent chat messages for the cached knowledge prefix."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            "Reference documents follow. Use them to answer the next question.\n"
            "------------------------------------------------\n"
            f"{corpus}\n"
            "------------------------------------------------"
        )},
    ]


def render_chat_prompt(tokenizer, messages: list[dict[str, str]], add_generation_prompt: bool = False) -> str:
    """Render messages with the selected tokenizer's template.

    Llama-family checkpoints generally provide a chat template, but the
    fallback keeps custom causal checkpoints usable when they do not.
    """
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    rendered = "".join(f"[{message['role']}]\n{message['content']}\n" for message in messages)
    return rendered + ("[assistant]\n" if add_generation_prompt else "")


def build_prefix_prompt(corpus: str, tokenizer=None) -> str:
    """Render the cache prefix, using a tokenizer template when supplied."""
    messages = build_prefix_messages(corpus)
    return render_chat_prompt(tokenizer, messages) if tokenizer is not None else "\n".join(
        f"[{message['role']}]\n{message['content']}" for message in messages
    )


def load_model_and_tokenizer(model_name: str, use_yarn: bool = False, rope_theta: float = DEFAULT_ROPE_THETA):
    """Load any compatible Transformers causal LM and its tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if use_yarn:
        config = AutoConfig.from_pretrained(model_name)
        config.rope_parameters = {
            "rope_type": "yarn", "factor": YARN_FACTOR,
            "original_max_position_embeddings": ORIGINAL_CONTEXT_LENGTH,
            "beta_fast": 64, "beta_slow": 2, "rope_theta": rope_theta,
        }
        config.max_position_embeddings = int(ORIGINAL_CONTEXT_LENGTH * YARN_FACTOR)
        kwargs["config"] = config
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model, tokenizer


def build_cache(corpus_path: Path, output_path: Path, model_name: str, use_yarn: bool = False, rope_theta: float = DEFAULT_ROPE_THETA) -> int:
    """Create a model-specific cache and return its pre-filled token length.

    A cache is not interchangeable across model checkpoints, even when both
    checkpoints use a Llama-family architecture.
    """
    corpus = corpus_path.read_text(encoding="utf-8")
    model, tokenizer = load_model_and_tokenizer(model_name, use_yarn, rope_theta)
    messages = build_prefix_messages(corpus)
    prefix_prompt = render_chat_prompt(tokenizer, messages)
    input_ids = tokenizer(prefix_prompt, return_tensors="pt").input_ids
    device = model.get_input_embeddings().weight.device
    started = perf_counter()
    with torch.no_grad():
        # Passing the config lets Transformers select the correct cache-layer
        # behavior for sliding-window/chunked attention variants.
        outputs = model(
            input_ids=input_ids.to(device),
            past_key_values=DynamicCache(config=model.config),
            use_cache=True,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "kv": outputs.past_key_values,
        "origin_len": input_ids.shape[-1],
        "model_name": model_name,
        "messages": messages,
        "prompt_format": "tokenizer_chat_template",
    }, output_path)
    LOGGER.info("Wrote %s (%d tokens, %.2fs)", output_path, input_ids.shape[-1], perf_counter() - started)
    return input_ids.shape[-1]
