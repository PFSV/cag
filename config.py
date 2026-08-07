"""Central configuration for the reusable CAG Llama-family implementation.

All paths are relative to this repository by default.  They can be overridden
with environment variables so the same commands work on a workstation or a
cluster without editing source files.
"""

from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = Path(os.getenv("CAG_RAW_DATA_DIR", PROJECT_ROOT / "data" / "raw"))
CORPUS_PATH = Path(os.getenv("CAG_CORPUS_PATH", PROJECT_ROOT / "data" / "knowledge_corpus.txt"))
CACHE_PATH = Path(os.getenv("CAG_CACHE_PATH", PROJECT_ROOT / "data" / "cache" / "llama_kvcache.pt"))
DEFAULT_MODEL = os.getenv("CAG_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

DEFAULT_MAX_NEW_TOKENS = int(os.getenv("CAG_MAX_NEW_TOKENS", "400"))
DEFAULT_DTYPE = os.getenv("CAG_DTYPE", "bfloat16")
ORIGINAL_CONTEXT_LENGTH = 32_768
YARN_FACTOR = 4.4
DEFAULT_ROPE_THETA = 8_000_000.0

SYSTEM_PROMPT = (
    "당신은 주어진 도메인 문서에 대해 답변하는 상담 어시스턴트입니다. "
    "문서 내용에만 근거해서 간결하고 정확하게 답변하세요. "
    "문서에 없는 내용은 모른다고 답하세요."
)
