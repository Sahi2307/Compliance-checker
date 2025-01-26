import logging
import os
from datetime import timedelta
from functools import lru_cache
from typing import Optional
from pydantic import BaseModel, Field

def setup_logging():
    """Configure basic logging for the application."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

class LLMSettings(BaseModel):
    """Base settings for Language Model configurations."""
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    max_retries: int = 3

class HuggingFaceSettings(LLMSettings):
    """Hugging Face-specific settings extending LLMSettings."""
    vector_store: str = "pgvector"
    model_name: str = "BAAI/bge-large-en-v1.5"
    api_key: str = ""
    cache_dir: str = "./hf_cache"

class DatabaseSettings(BaseModel):
    """Database connection settings."""
    service_url: str = "postgres://postgres:password@localhost:5432/postgres"

class VectorStoreSettings(BaseModel):
    """Settings for the VectorStore."""
    table_name: str = "embeddings"
    embedding_dimensions: int = 1024
    time_partition_interval: timedelta = timedelta(days=7)

class Settings(BaseModel):
    """Main settings class combining all sub-settings."""
    huggingface: HuggingFaceSettings = Field(default_factory=HuggingFaceSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)

@lru_cache()
def get_settings() -> Settings:
    """Create and return a cached instance of the Settings."""
    setup_logging()
    return Settings()
