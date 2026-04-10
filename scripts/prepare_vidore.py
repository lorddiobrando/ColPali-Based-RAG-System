from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.settings import get_settings
from src.data.loaders.vidore_loader import (
    infer_doc_id,
    infer_language,
    infer_page_id,
    infer_page_num,
    infer_query_text,
    iter_split,
    load_vidore_dataset_by_id,
)
from src.data.schemas import VidoreRecord


def load_cfg(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def try_save_image(image_obj: Any, output_path: Path, image_format: str) -> str | None:
    if image_obj is None:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(image_obj, Image.Image):
        image_obj.save(output_path, format=image_format.upper())
        return str(output_path)

    # Datasets image feature often exposes .save()
    save_method = getattr(image_obj, "save", None)
    if callable(save_method):
        save_method(str(output_path))
        return str(output_path)
    return None


def sanitize_filename_part(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return sanitized[:180] if len(sanitized) > 180 else sanitized


def main() -> None:
    cfg = load_cfg(REPO_ROOT / "configs" / "vidore.yaml")
    settings = get_settings()

    settings.vidore_output_dir.mkdir(parents=True, exist_ok=True)
    settings.vidore_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_ids = cfg.get("dataset", {}).get("ids", [])
    preferred_splits = cfg.get("dataset", {}).get("preferred_splits", ["test", "validation", "train"])
    max_rows = cfg.get("processing", {}).get("max_rows_per_split")
    save_images = bool(cfg.get("processing", {}).get("save_images", True))
    image_format = cfg.get("processing", {}).get("image_format", "png")

    records: list[dict[str, Any]] = []

    for dataset_id in dataset_ids:
        ds = load_vidore_dataset_by_id(dataset_id, settings)
        split = next((s for s in preferred_splits if s in ds), None)
        if split is None:
            continue

        split_ds = ds[split]
        processed = 0
        for row in tqdm(iter_split(split_ds), desc=f"Processing {dataset_id}:{split}"):
            if max_rows is not None and processed >= int(max_rows):
                break

            query_id = str(row.get("query_id", row.get("id", f"{split}_{processed}")))
            query_text = infer_query_text(row)
            doc_id = infer_doc_id(row)
            page_id = infer_page_id(row, fallback=f"{doc_id}_{split}_{processed}")
            page_num = infer_page_num(row)
            language = infer_language(row)
            raw_payload = {
                k: v
                for k, v in row.items()
                if k not in {"image", "page_image"}
                and isinstance(v, (str, int, float, bool, type(None)))
            }

            image_path = None
            if save_images:
                image_obj = row.get("image") or row.get("page_image")
                safe_dataset_name = dataset_id.replace("/", "__")
                safe_page_id = sanitize_filename_part(page_id)
                target = (
                    settings.vidore_output_dir
                    / "images"
                    / safe_dataset_name
                    / split
                    / f"{safe_page_id}.{image_format}"
                )
                image_path = try_save_image(image_obj, target, image_format=image_format)

            record = VidoreRecord(
                split=f"{dataset_id}:{split}",
                query_id=query_id,
                query_text=query_text,
                doc_id=doc_id,
                page_id=page_id,
                page_num=page_num,
                language=language,
                image_path=image_path,
                raw=raw_payload,
            )
            records.append(record.to_dict())
            processed += 1

    with settings.vidore_manifest_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records to: {settings.vidore_manifest_path}")


if __name__ == "__main__":
    main()
