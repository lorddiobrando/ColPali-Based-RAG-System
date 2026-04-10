from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class VidoreRecord:
    split: str
    query_id: str
    query_text: str
    doc_id: str
    page_id: str
    page_num: int | None
    language: str | None
    image_path: str | None
    raw: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
