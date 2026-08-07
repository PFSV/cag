"""Public Python API for reusable CAG with Llama-family causal LMs."""

__version__ = "0.1.0"

from src.cache_builder import build_cache, build_prefix_messages, build_prefix_prompt
from src.corpus_builder import build_corpus, build_documents, render_corpus
from src.inference import answer_one, run_batch, run_query

__all__ = [
    "answer_one", "build_cache", "build_corpus", "build_documents",
    "build_prefix_messages", "build_prefix_prompt", "render_corpus", "run_batch", "run_query",
]
