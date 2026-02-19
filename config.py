"""Configuration from environment for case-clustering pipeline."""
import os
from pathlib import Path
from dotenv import load_dotenv


# Avoid Hugging Face tokenizers fork warning (set before any tokenizer use)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# # AWS credentials/region
# AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
# AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID")
# AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY")


try:
    
    load_dotenv()
except ImportError:
    pass

# Data and storage
CASES_PATH: str = os.getenv("CASES_PATH", "dummy_data_clean.csv")
CHECKPOINT_DIR: str = os.getenv("CHECKPOINT_DIR", "./checkpoints")
SAMPLE_SIZE: int = int(os.getenv("SAMPLE_SIZE", "0"))

# Embedding: "local" | "openai" | "bedrock"
EMBEDDING_BACKEND: str = os.getenv("EMBEDDING_BACKEND", "local").lower()
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2")
# EMBEDDING_BACKEND: str = os.getenv("EMBEDDING_BACKEND", "openai").lower()
# EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")



# Summarisation (CLIO): "openai" | "bedrock" – Kura uses Instructor provider string "openai/MODEL" or "bedrock/MODEL"
# For Bedrock set AWS credentials/region; MODEL e.g. "anthropic.claude-3-5-sonnet-20241022-v2:0"
# OpenAI examples: gpt-4o-mini, gpt-4o, gpt-4.1, o1 (use exact API model id)
SUMMARIZATION_BACKEND: str = os.getenv("SUMMARIZATION_BACKEND", "openai").lower()
SUMMARIZATION_MODEL: str = os.getenv("SUMMARIZATION_MODEL", "gpt-5-mini")
SUMMARIZATION_TEMPERATURE: float = float(os.getenv("SUMMARIZATION_TEMPERATURE", "0.2"))

# # Or AWS Bedrock (set AWS_REGION and credentials)
# SUMMARIZATION_BACKEND=bedrock
# SUMMARIZATION_MODEL=anthropic.claude-3-5-sonnet-20241022-v2:0


# Resolve paths relative to project root (directory containing config.py or cwd)
PROJECT_ROOT: Path = Path(__file__).resolve().parent


def resolve_cases_path() -> Path:
    """Return resolved path to cases file (CSV/JSON/Parquet)."""
    p = Path(CASES_PATH)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def get_checkpoint_dir() -> Path:
    """Return resolved checkpoint directory."""
    p = Path(CHECKPOINT_DIR)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p
