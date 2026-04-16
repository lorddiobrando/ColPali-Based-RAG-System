from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flask import Blueprint, jsonify, render_template, request, send_file

from app.services.generation_service import generate_answer
from app.services.retrieval_service import run_retrieval
from src.config.settings import get_settings

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    settings = get_settings()
    # E.g. "vidore/colSmol-256M" -> "colSmol-256M"
    model_short = settings.colpali_model.split("/")[-1]
    return render_template("index.html", model_name=model_short)


@main_bp.route("/api/query", methods=["POST"])
def api_query():
    """
    Main query endpoint.
    Body: { "query": "...", "top_k": 5, "mode": "rag" | "retrieval_only" }
    """
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "query is required"}), 400

    top_k = min(int(data.get("top_k", 5)), 20)
    mode = data.get("mode", "rag")
    settings = get_settings()

    # Step 1: Retrieval
    retrieval = run_retrieval(settings, query, top_k=top_k)

    # Step 2: Generation (if mode=rag and API key present)
    if mode == "rag" and settings.openrouter_api_key:
        generation = generate_answer(settings, query, retrieval["results"])
    else:
        generation = {
            "mode": "retrieval_only",
            "answer": None,
            "citations": [
                {
                    "doc_id": r.get("doc_id"),
                    "page_num": r.get("page_num"),
                    "score": r.get("score"),
                    "image_path": r.get("image_path"),
                }
                for r in retrieval["results"]
            ],
            "model": None,
            "timing": {"generation_seconds": 0},
        }

    return jsonify({
        "query": query,
        "retrieval": retrieval,
        "generation": generation,
    })


@main_bp.route("/api/page-preview/<int:point_id>")
def page_preview(point_id: int):
    """
    Serve a page image thumbnail by its point ID.
    Looks up the image_path from the Qdrant payload.
    Reuses the retrieval service's shared Qdrant client to avoid lock conflicts.
    """
    from app.services.retrieval_service import _get_retriever

    settings = get_settings()
    retriever = _get_retriever(settings)
    points = retriever.store.client.retrieve(
        collection_name=settings.qdrant_collection,
        ids=[point_id],
        with_payload=True,
        with_vectors=False,
    )
    if not points:
        return jsonify({"error": "not found"}), 404

    image_path = (points[0].payload or {}).get("image_path")
    if not image_path or not Path(image_path).exists():
        return jsonify({"error": "image not found"}), 404

    return send_file(image_path, mimetype="image/png")
