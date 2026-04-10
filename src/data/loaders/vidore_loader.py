from __future__ import annotations

from typing import Any, Iterable

from datasets import DatasetDict, load_dataset

from src.config.settings import AppSettings


def load_vidore_dataset(settings: AppSettings) -> DatasetDict:
    return load_dataset(settings.vidore_dataset, cache_dir=str(settings.hf_cache_dir))


def load_vidore_dataset_by_id(dataset_id: str, settings: AppSettings) -> DatasetDict:
    return load_dataset(dataset_id, cache_dir=str(settings.hf_cache_dir))


def infer_query_text(item: dict[str, Any]) -> str:
    for key in ("query", "question", "text"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def infer_doc_id(item: dict[str, Any]) -> str:
    for key in ("doc_id", "document_id", "docid"):
        value = item.get(key)
        if value is not None:
            return str(value)
    filename = item.get("image_filename")
    if isinstance(filename, str) and filename.strip():
        # Example: images/1810.10511_2.jpg -> 1810.10511
        stem = filename.strip().split("/")[-1].split("\\")[-1]
        if "_" in stem:
            return stem.split("_")[0]
        if "." in stem:
            return stem.split(".")[0]
    return "unknown_doc"


def infer_page_id(item: dict[str, Any], fallback: str) -> str:
    for key in ("page_id", "passage_id", "chunk_id", "id"):
        value = item.get(key)
        if value is not None:
            return str(value)
    return fallback


def infer_language(item: dict[str, Any]) -> str | None:
    for key in ("language", "lang"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return None


def infer_page_num(item: dict[str, Any]) -> int | None:
    for key in ("page_num", "page", "page_number"):
        value = item.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def iter_split(ds_split: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for row in ds_split:
        if not isinstance(row, dict):
            continue
        yield row
