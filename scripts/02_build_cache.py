#!/usr/bin/env python3
"""CLI for pre-filling a causal LM and saving a KV-cache artifact."""
import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CACHE_PATH, CORPUS_PATH, DEFAULT_MODEL, DEFAULT_ROPE_THETA
from src.cache_builder import build_cache

parser = argparse.ArgumentParser()
parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
parser.add_argument("--output", type=Path, default=CACHE_PATH)
parser.add_argument("--model", default=DEFAULT_MODEL)
parser.add_argument("--use-yarn", action="store_true")
parser.add_argument("--rope-theta", type=float, default=DEFAULT_ROPE_THETA)
args = parser.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
build_cache(args.corpus, args.output, args.model, args.use_yarn, args.rope_theta)
