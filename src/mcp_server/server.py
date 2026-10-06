from __future__ import annotations

import os
from pathlib import Path

from .annotations import AnnotationService
from storage.local import read_json
import hashlib
import re
from retrieval.spoiler_policy import SpoilerPolicy
from security.access import BookAccess


def main() -> None:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise SystemExit("Installez l’extra MCP avec : python -m pip install -e '.[mcp]'") from exc

    root = Path(os.getenv("NARRATIVELENS_DATA_DIR", "data"))
    user_id = os.getenv("NARRATIVELENS_USER_ID", "local")
    service = AnnotationService(root)
    server = FastMCP("NarrativeLens")

    def user_progress() -> dict:
        if user_id == "local":
            return read_json(root / "library/reading_progress.json", {})
        key = hashlib.sha256(user_id.encode()).hexdigest()[:24]
        return read_json(root / "library/users" / key / "reading_progress.json", {})

    def validate_evidence(work_id: str, chapter: int, evidence_ids: list[str]) -> None:
        BookAccess(root).require(user_id, work_id)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", work_id):
            raise ValueError("Invalid work ID.")
        chunks = read_json(root / "processed" / f"{work_id}.chunks.json", [])
        by_id = {row.get("chunk_id"): row for row in chunks if isinstance(row, dict)}
        progress = user_progress().get(work_id)
        limit = min(chapter, int(progress)) if progress else chapter
        if not evidence_ids or any(evidence_id not in by_id or not SpoilerPolicy.allows(by_id[evidence_id].get("chapter"), limit) for evidence_id in evidence_ids):
            raise ValueError("Evidence must exist in this book and stay within the user's reading progress.")

    @server.tool()
    def get_reading_progress(work_id: str) -> int | None:
        """Return this local user's reading progress for a book."""
        BookAccess(root).require(user_id, work_id)
        return user_progress().get(work_id)

    @server.tool()
    def get_user_annotations(work_id: str) -> list[dict]:
        """List annotations owned by the configured local MCP user."""
        BookAccess(root).require(user_id, work_id)
        return service.list(user_id, work_id)

    @server.tool()
    def get_book_metadata(work_id: str) -> dict:
        """Return metadata for a locally installed book."""
        BookAccess(root).require(user_id, work_id)
        from storage.local import read_json
        records = read_json(root / "library/installed_books.json", [])
        records += read_json(root / "annotations/work_manifest.json", [])
        return next((item for item in records if item.get("work_id") == work_id), {})

    @server.tool()
    def prepare_annotation(work_id: str, chapter: int, text: str, evidence_ids: list[str]) -> dict:
        """Prepare an annotation; this does not write it to disk."""
        validate_evidence(work_id, chapter, evidence_ids)
        return service.prepare(user_id, work_id, chapter, text, evidence_ids)

    @server.tool()
    def save_annotation(work_id: str, chapter: int, text: str, evidence_ids: list[str], approval_token: str) -> dict:
        """Save only the exact annotation that received a valid approval token."""
        validate_evidence(work_id, chapter, evidence_ids)
        return service.save(user_id, work_id, chapter, text, evidence_ids, approval_token)

    server.run()


if __name__ == "__main__":
    main()
