from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path
from typing import Any

import requests

from src.config.settings import AppSettings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a precise document assistant. "
    "Answer the user's question using ONLY the information from the retrieved document pages provided below. "
    "If the answer is not found in the provided context, say so clearly. "
    "Always cite which document and page number your answer comes from."
)


def _build_context(results: list[dict[str, Any]]) -> str:
    """Build a text context block from retrieval results."""
    parts = []
    for i, r in enumerate(results, 1):
        doc_id = r.get("doc_id", "unknown")
        page_num = r.get("page_num", "?")
        score = r.get("score", 0)
        split = r.get("split", "")
        parts.append(
            f"[Source {i}] Document: {doc_id} | Page: {page_num} | "
            f"Split: {split} | Relevance: {score}"
        )
    return "\n".join(parts)


def generate_answer(
    settings: AppSettings,
    query: str,
    retrieval_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Generate an answer via OpenRouter API using retrieved context.

    Returns a dict with 'answer', 'citations', 'model', and 'timing'.
    If no API key is set, returns retrieval-only results.
    """
    citations = [
        {
            "doc_id": r.get("doc_id"),
            "page_num": r.get("page_num"),
            "score": r.get("score"),
            "image_path": r.get("image_path"),
        }
        for r in retrieval_results
    ]

    if not settings.openrouter_api_key:
        return {
            "mode": "retrieval_only",
            "answer": None,
            "citations": citations,
            "model": None,
            "timing": {"generation_seconds": 0},
        }

    # Build prompt
    context = _build_context(retrieval_results)
    user_message = (
        f"Retrieved document context metadata:\n{context}\n\n"
        f"User question: {query}\n\n"
        "Provide a concise, accurate answer using the provided page images and citations."
    )

    # Build the multimodal content array
    content_array = [{"type": "text", "text": user_message}]
    
    for r in retrieval_results:
        img_path = r.get("image_path")
        if img_path and Path(img_path).exists():
            try:
                from PIL import Image
                import io
                
                with Image.open(img_path) as img:
                    # Convert to RGB to ensure JPEG compatibility and discard alpha
                    img = img.convert("RGB")
                    # Resize to max 1024x1024 to keep payload size beneath EOF limits
                    img.thumbnail((1024, 1024))
                    
                    buffer = io.BytesIO()
                    img.save(buffer, format="JPEG", quality=85)
                    b64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    
                    content_array.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"}
                    })
            except Exception as e:
                logger.warning(f"Failed to read image {img_path}: {e}")

    t0 = time.perf_counter()
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.openrouter_model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": content_array},
                ],
                "max_tokens": 1024,
                "temperature": 0.2,
            },
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data["choices"][0]["message"]["content"]
        model_used = data.get("model", settings.openrouter_model)
    except Exception as e:
        logger.exception("OpenRouter generation failed")
        answer = f"Generation error: {e}"
        model_used = None

    t_gen = time.perf_counter() - t0

    return {
        "mode": "rag",
        "answer": answer,
        "citations": citations,
        "model": model_used,
        "timing": {"generation_seconds": round(t_gen, 3)},
    }
