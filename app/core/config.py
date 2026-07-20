from __future__ import annotations
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# --- Logging (shared across every module) ------------------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def _env_int(key: str, default: int) -> int:
    """Read an int-valued environment variable with a fallback default."""
    val = os.getenv(key)
    return int(val) if val not in (None, "") else default


def _env_float(key: str, default: float) -> float:
    """Read a float-valued environment variable with a fallback default."""
    val = os.getenv(key)
    return float(val) if val not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    # --- Filesystem ---
    base_dir: Path = Path(os.getenv("EDU_RAG_DATA_DIR", "./data")).resolve()
    raw_docs_dir: Path = field(init=False)
    page_text_dir: Path = field(init=False)
    raw_text_dump_dir: Path = field(init=False)
    chunk_store_dir: Path = field(init=False)
    chroma_dir: Path = field(init=False)
    bm25_dir: Path = field(init=False)
    evaluation_dir: Path = field(init=False)

    # --- OCR ---
    ocr_lang: str = os.getenv("OCR_LANG", "ben+eng")   # Tesseract language pack: Bangla + English
    ocr_dpi: int = _env_int("OCR_DPI", 300)

    # --- Chunking ---
    min_chunk_tokens: int = _env_int("MIN_CHUNK_TOKENS", 350)
    max_chunk_tokens: int = _env_int("MAX_CHUNK_TOKENS", 700)
    chunk_overlap_ratio: float = _env_float("CHUNK_OVERLAP_RATIO", 0.10)

    # --- Embeddings ---
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
    embedding_batch_size: int = _env_int("EMBEDDING_BATCH_SIZE", 16)
    embedding_dim: int = _env_int("EMBEDDING_DIM", 1024)

    # --- Reranker ---
    reranker_model_name: str = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
    # Adaptive context depth: simple questions (fact/definition/mcq/...) get a
    # small, cheap context; only questions flagged complex (see
    # HIGH_RISK_QUESTION_TYPES in query_understanding) get the full budget.
    # This is the single biggest lever on prompt token size, since each
    # reranked chunk (plus its neighbors) is echoed in full into the LLM prompt.
    rerank_top_k_simple: int = _env_int("RERANK_TOP_K_SIMPLE", 3)
    rerank_top_k_complex: int = _env_int("RERANK_TOP_K_COMPLEX", 6)

    # --- Retrieval ---
    dense_top_k: int = _env_int("DENSE_TOP_K", 25)
    sparse_top_k: int = _env_int("SPARSE_TOP_K", 25)
    rrf_k: int = _env_int("RRF_K", 60)                     # RRF damping constant
    fused_candidate_k: int = _env_int("FUSED_CANDIDATE_K", 25)  # candidate pool fed to the (local, free) reranker

    # --- Parent-context expansion (also complexity-adaptive) ---
    max_neighbors_simple: int = _env_int("MAX_NEIGHBORS_SIMPLE", 0)   # simple questions: core chunk only
    max_neighbors_complex: int = _env_int("MAX_NEIGHBORS_COMPLEX", 2)  # complex/multi-hop: prev + next

    # --- Prompt-size caps (hard ceiling regardless of chunk size settings) ---
    excerpt_max_words: int = _env_int("EXCERPT_MAX_WORDS", 220)     # per core excerpt, in the final prompt
    neighbor_max_words: int = _env_int("NEIGHBOR_MAX_WORDS", 100)   # per neighbor excerpt, in the final prompt

    # --- LLM (Groq) ---
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    groq_temperature: float = _env_float("GROQ_TEMPERATURE", 0.2)
    groq_max_tokens: int = _env_int("GROQ_MAX_TOKENS", 700)                    # final answer generation
    groq_max_tokens_analysis: int = _env_int("GROQ_MAX_TOKENS_ANALYSIS", 700)  # combined classify+expand call
    groq_max_tokens_hyde: int = _env_int("GROQ_MAX_TOKENS_HYDE", 150)          # HyDE hypothetical passage
    groq_max_tokens_validate: int = _env_int("GROQ_MAX_TOKENS_VALIDATE", 200)  # answer validation (JSON verdict)

    # --- Session ---
    session_max_turns: int = _env_int("SESSION_MAX_TURNS", 5)

    # --- Collection naming ---
    chroma_collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "edu_rag_chunks")

    # --- API / server ---
    api_cors_origins: str = os.getenv("API_CORS_ORIGINS", "*")  # comma-separated, "*" = allow all

    def __post_init__(self):
        object.__setattr__(self, "raw_docs_dir", self.base_dir / "raw_docs")
        object.__setattr__(self, "page_text_dir", self.base_dir / "page_text")
        object.__setattr__(self, "raw_text_dump_dir", self.base_dir / "raw_text")
        object.__setattr__(self, "chunk_store_dir", self.base_dir / "chunks")
        object.__setattr__(self, "chroma_dir", self.base_dir / "chroma")
        object.__setattr__(self, "bm25_dir", self.base_dir / "bm25")
        object.__setattr__(self, "evaluation_dir", self.base_dir / "evaluation")

    def ensure_dirs(self) -> None:        
        for d in (self.raw_docs_dir, self.page_text_dir, self.raw_text_dump_dir, self.chunk_store_dir,
                  self.chroma_dir, self.bm25_dir, self.evaluation_dir):
            d.mkdir(parents=True, exist_ok=True)


SETTINGS = Settings()
SETTINGS.ensure_dirs()

_log = get_logger("config")
_log.info("Settings initialized. base_dir=%s", SETTINGS.base_dir)


def load_groq_api_key() -> str:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file "
            "(see .env.example) or export it in the environment before starting the service."
        )
    return key
