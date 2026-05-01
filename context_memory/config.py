"""
Centralized configuration — all swappable values in one place.
Reads from .env file. Every other module imports `settings` from here.
"""

from typing import Literal
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Every configurable value lives here. Nothing is hardcoded elsewhere."""

    # --- LLM Provider ---
    gemini_api_key: str = ""

    # --- Model Names (LiteLLM format: "gemini/<model-name>") ---
    pathfinder_model: str = "gemini/gemini-2.5-flash"
    summarizer_model: str = "gemini/gemini-2.5-flash"
    verifier_model: str = "gemini/gemini-2.5-flash"
    embedding_model: str = "gemini/gemini-embedding-001"

    # --- Database Connections ---
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    qdrant_url: str = "http://localhost:6333"
    sqlite_path: str = "./data/memory.db"

    # --- Tunable Parameters ---
    default_decay_mode: Literal["drift", "time", "hybrid"] = "drift"
    decay_lambda: float = 0.01
    max_backtrack: int = 5
    chunk_size_tokens: int = 500
    edge_weight_ceiling: float = 5.0
    broad_query_threshold: float = 0.6

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


# Singleton — import this everywhere
settings = Settings()
