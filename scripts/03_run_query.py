#!/usr/bin/env python3
"""CLI for single-question or JSONL batch CAG inference."""
import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CACHE_PATH, DEFAULT_MAX_NEW_TOKENS, DEFAULT_MODEL, DEFAULT_ROPE_THETA
from src.inference import run_batch, run_query

parser = argparse.ArgumentParser()
parser.add_argument("--cache", type=Path, default=CACHE_PATH)
parser.add_argument("--model", default=DEFAULT_MODEL)
parser.add_argument("--question")
parser.add_argument("--questions", type=Path, help="UTF-8 text file, one question per line")
parser.add_argument("--output", type=Path, help="JSONL output for batch mode")
parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
parser.add_argument("--use-yarn", action="store_true")
parser.add_argument("--rope-theta", type=float, default=DEFAULT_ROPE_THETA)
args = parser.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
if bool(args.question) == bool(args.questions):
    parser.error("provide exactly one of --question or --questions")
if args.question:
    print(run_query(args.cache, args.question, args.model, args.max_new_tokens, args.use_yarn, args.rope_theta))
else:
    if not args.output:
        parser.error("--output is required with --questions")
    success, failed = run_batch(args.cache, args.questions, args.output, args.model, args.max_new_tokens, args.use_yarn, args.rope_theta)
    logging.info("Finished batch: %d succeeded, %d failed; output=%s", success, failed, args.output)
