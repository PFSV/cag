# pycag — Cache-Augmented Generation for Llama-family Models

[![PyPI](https://img.shields.io/pypi/v/pycag)](https://pypi.org/project/pycag/)
[![CI](https://github.com/PFSV/cag/actions/workflows/ci.yml/badge.svg)](https://github.com/PFSV/cag/actions/workflows/ci.yml)

Reusable **Cache-Augmented Generation (CAG)** implementation for pre-filling any compatible Llama-family causal language model with a manageable knowledge base and serving queries without per-query vector retrieval. Kanana is one supported model instance, not the architectural limit.

This is a reusable implementation and application of the upstream [CAG repository](https://github.com/hhhuang/CAG) and paper, [*Don't Do RAG: When Cache-Augmented Generation is All You Need for Knowledge Tasks*](https://arxiv.org/abs/2412.15605). It is not a drop-in reproduction of the paper's benchmark code: the upstream implementation exposes SQuAD and HotpotQA experiments, while this package generalizes the cache-prefill mechanism to user-selected Llama-family checkpoints and data.

## Workflow

```text
documents -> corpus consolidation -> KV-cache pre-building (.pt) -> query execution
```

1. `01_build_corpus.py` consolidates TXT, Markdown, JSON, JSONL, CSV files and, optionally, the `key-answer` sheet of an FAQ workbook.
2. `02_build_cache.py` loads the selected causal LM, pre-fills it with the corpus, and saves the resulting cache.
3. `03_run_query.py` appends a question to that cached prefix. It supports one question or a line-oriented batch that writes JSONL incrementally.

## Install

```bash
pip install pycag
```

`pycag` is an alpha research utility. It downloads and runs user-selected language models;
hardware and memory requirements depend on the model and corpus size.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/01_build_corpus.py
python scripts/02_build_cache.py
python scripts/03_run_query.py --question "What should be replaced according to the guide?"
python scripts/03_run_query.py --questions questions.txt --output data/results/answers.jsonl
```

The package exposes the same workflow as command-line tools:

```bash
pycag-build-corpus
pycag-build-cache
pycag-query --question "What should be replaced according to the guide?"
```

## Bring your own data

The CAG engine does not depend on a particular dataset. Place text-like files under a data directory and run:

```bash
pycag-build-corpus --data-root ./my-data --output ./data/knowledge_corpus.txt
pycag-build-cache --corpus ./data/knowledge_corpus.txt --model meta-llama/Llama-3.1-8B-Instruct
pycag-query --cache ./data/cache/llama_kvcache.pt --question "Ask about my data"
```

For PDFs, database rows, web pages, or other binary/API sources, add an application-specific extraction step that converts records into UTF-8 text before corpus building. This keeps the cache engine independent of any one ingestion technology.

The default model is `meta-llama/Llama-3.1-8B-Instruct`, matching the original CAG implementation. Override it with `--model` or `CAG_MODEL`; Kanana works by passing its model ID instead. Cache creation and query must use the exact same model/tokenizer, and the artifact records that identity. Paths can be changed with `CAG_RAW_DATA_DIR`, `CAG_CORPUS_PATH`, and `CAG_CACHE_PATH`. If the corpus exceeds the native context, pass `--use-yarn` consistently to cache-building and query commands.

## Pros, cons, and lessons learned

**Pros:** no vector-search overhead per question; potentially lower time-to-first-token after cache loading; the full original context can be preserved instead of losing information through chunking and top-k selection.

**Cons:** a static cache consumes substantial VRAM/storage and is tied to the model, tokenizer, device setup, and RoPE configuration; raw-document updates require rebuilding and redistributing it; a full corpus can exceed context limits; CAG does not automatically provide source citations or improve factuality.

Treat YaRN as an experiment, not a quality guarantee at extended lengths. Benchmark CAG against retrieval-plus-generation and no-context generation using the same model and prompts. Record corpus size, cache build/load time, steady-state query latency, peak VRAM, and answer quality separately.

## Citation

If this implementation is used in research, cite the original CAG work:

```bibtex
@misc{chan2024dontragcacheaugmentedgeneration,
  title={Don't Do RAG: When Cache-Augmented Generation is All You Need for Knowledge Tasks},
  author={Brian J. Chan and Chao-Ting Chen and Jui-Hung Cheng and Hen-Hsen Huang},
  year={2024},
  eprint={2412.15605},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

## Data and GitHub hygiene

Put private source material in `data/raw/` locally. Git ignores it except for `data/raw/sample.txt`; generated corpora, caches, and results are ignored too. Do not commit internal FAQs, model weights, or `.pt` files. The legacy scripts/data remain in this checkout for comparison; the public-facing implementation lives in `src/` and `scripts/`.
