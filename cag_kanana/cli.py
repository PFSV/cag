"""Installed console entry points for corpus, cache, and query workflows."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config import CACHE_PATH, CORPUS_PATH, DEFAULT_MAX_NEW_TOKENS, DEFAULT_MODEL, DEFAULT_ROPE_THETA, RAW_DATA_DIR
from src.cache_builder import build_cache
from src.corpus_builder import build_corpus
from src.inference import run_batch, run_query


def _logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def build_corpus_cli() -> None:
    parser = argparse.ArgumentParser(prog="cag-build-corpus")
    parser.add_argument("--data-root", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--xlsx", type=Path)
    parser.add_argument("--output", type=Path, default=CORPUS_PATH)
    args = parser.parse_args()
    _logging()
    build_corpus(args.data_root, args.output, args.xlsx)


def build_cache_cli() -> None:
    parser = argparse.ArgumentParser(prog="cag-build-cache")
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--output", type=Path, default=CACHE_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--use-yarn", action="store_true")
    parser.add_argument("--rope-theta", type=float, default=DEFAULT_ROPE_THETA)
    args = parser.parse_args()
    _logging()
    build_cache(args.corpus, args.output, args.model, args.use_yarn, args.rope_theta)


def query_cli() -> None:
    parser = argparse.ArgumentParser(prog="cag-query")
    parser.add_argument("--cache", type=Path, default=CACHE_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--question")
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--use-yarn", action="store_true")
    parser.add_argument("--rope-theta", type=float, default=DEFAULT_ROPE_THETA)
    args = parser.parse_args()
    _logging()
    if bool(args.question) == bool(args.questions):
        parser.error("provide exactly one of --question or --questions")
    if args.question:
        print(run_query(args.cache, args.question, args.model, args.max_new_tokens, args.use_yarn, args.rope_theta))
        return
    if not args.output:
        parser.error("--output is required with --questions")
    success, failed = run_batch(args.cache, args.questions, args.output, args.model, args.max_new_tokens, args.use_yarn, args.rope_theta)
    logging.info("Finished batch: %d succeeded, %d failed; output=%s", success, failed, args.output)
