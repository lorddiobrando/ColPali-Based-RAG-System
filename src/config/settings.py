from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppSettings:
    data_dir: Path
    hf_cache_dir: Path
    vidore_dataset: str
    vidore_output_dir: Path
    vidore_manifest_path: Path

    # Encoder
    colpali_model: str
    encoder_type: str  # "colpali" or "baseline"

    # Qdrant
    qdrant_path: Path
    qdrant_collection: str

    # OpenRouter
    openrouter_api_key: str | None
    openrouter_model: str


def get_settings() -> AppSettings:
    load_dotenv()
    data_dir = Path(os.getenv("DATA_DIR", "./data")).resolve()
    hf_cache_dir = Path(os.getenv("HF_CACHE_DIR", "./data/cache/huggingface")).resolve()
    vidore_output_dir = Path(os.getenv("VIDORE_OUTPUT_DIR", "./data/processed/vidore")).resolve()
    vidore_manifest_path = Path(
        os.getenv("VIDORE_MANIFEST_PATH", "./data/processed/vidore/vidore_manifest.jsonl")
    ).resolve()

    return AppSettings(
        data_dir=data_dir,
        hf_cache_dir=hf_cache_dir,
        vidore_dataset=os.getenv("VIDORE_DATASET", "vidore/vidore-benchmark"),
        vidore_output_dir=vidore_output_dir,
        vidore_manifest_path=vidore_manifest_path,
        # Encoder
        colpali_model=os.getenv("COLPALI_MODEL", "vidore/colqwen2-v1.0"),
        encoder_type=os.getenv("ENCODER_TYPE", "colpali"),
        # Qdrant
        qdrant_path=Path(os.getenv("QDRANT_PATH", "./data/indexes/qdrant")).resolve(),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "vidore_pages"),
        # OpenRouter
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        openrouter_model=os.getenv(
            "OPENROUTER_MODEL", "google/gemini-2.0-flash-001"
        ),
    )
