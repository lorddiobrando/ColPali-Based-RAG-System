from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.settings import get_settings
from src.indexing.index_builder import IndexBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local vector index from ViDoRe manifest.")
    parser.add_argument("--collection", default="vidore_pages", help="Qdrant collection name.")
    parser.add_argument("--vector-dim", type=int, default=256, help="Embedding dimensionality.")
    parser.add_argument("--batch-size", type=int, default=64, help="Upsert batch size.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit per run.")
    parser.add_argument(
        "--qdrant-path",
        default="./data/indexes/qdrant",
        help="Local Qdrant storage path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    manifest_path = settings.vidore_manifest_path
    qdrant_path = Path(args.qdrant_path).resolve()
    qdrant_path.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    builder = IndexBuilder(
        manifest_path=manifest_path,
        qdrant_path=qdrant_path,
        collection_name=args.collection,
        vector_dim=args.vector_dim,
        batch_size=args.batch_size,
    )
    result = builder.build(limit=args.limit)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
