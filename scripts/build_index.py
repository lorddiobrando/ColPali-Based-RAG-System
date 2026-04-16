from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.settings import get_settings
from src.indexing.index_builder import IndexBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local vector index from ViDoRe manifest.")
    parser.add_argument("--collection", default=None, help="Qdrant collection name.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for encoding.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit per run.")
    parser.add_argument("--encoder", default=None, choices=["colpali", "baseline"],
                        help="Encoder type (default: from .env)")
    parser.add_argument("--reset", action="store_true", help="Delete progress file and re-index from scratch.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()

    manifest_path = settings.vidore_manifest_path
    qdrant_path = settings.qdrant_path
    qdrant_path.mkdir(parents=True, exist_ok=True)

    collection = args.collection or settings.qdrant_collection
    encoder_type = args.encoder or settings.encoder_type

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    builder = IndexBuilder(
        manifest_path=manifest_path,
        qdrant_path=qdrant_path,
        collection_name=collection,
        batch_size=args.batch_size,
        encoder_type=encoder_type,
        model_name=settings.colpali_model,
    )

    if args.reset:
        progress_file = manifest_path.parent / "index_progress.json"
        if progress_file.exists():
            progress_file.unlink()
            print("Progress file deleted — re-indexing from scratch.")

    result = builder.build(limit=args.limit)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
