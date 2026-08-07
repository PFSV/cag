#!/usr/bin/env python3
"""CLI for consolidating raw documents into one CAG corpus."""
import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CORPUS_PATH, RAW_DATA_DIR
from src.corpus_builder import build_corpus

parser = argparse.ArgumentParser()
parser.add_argument("--data-root", type=Path, default=RAW_DATA_DIR)
parser.add_argument("--xlsx", type=Path, default=None)
parser.add_argument("--output", type=Path, default=CORPUS_PATH)
parser.add_argument("--log-level", default="INFO")
args = parser.parse_args()
logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s %(message)s")
build_corpus(args.data_root, args.output, args.xlsx)
