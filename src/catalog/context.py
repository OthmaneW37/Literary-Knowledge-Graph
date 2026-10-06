"""Supplementary catalog information, kept outside the primary text index."""
import json
from pathlib import Path

from storage.local import read_json


def metadata_for(index, work_ids):
    manifest = getattr(index, "manifest_path", None)
    if manifest is None:
        return []
    root = Path(manifest).parent.parent
    records = read_json(root / "library" / "installed_books.json", [])
    return [record for record in records if record.get("work_id") in work_ids]


def catalog_context(index, work_ids, max_chapter=None):
    # Whole-book synopses cannot be safely reduced to an arbitrary chapter.
    if max_chapter is not None:
        return ""
    contexts = []
    for record in metadata_for(index, work_ids):
        for source in record.get("metadata_sources", [])[:2]:
            if source.get("summary") or source.get("subjects"):
                contexts.append({"work_id": record["work_id"], "title": record["title"],
                                 "source": source.get("source_name", "Catalogue"), "url": source.get("source_url", ""),
                                 "summary": (source.get("summary") or "")[:1100], "subjects": (source.get("subjects") or [])[:8]})
    return json.dumps(contexts[:3], ensure_ascii=False) if contexts else ""


def candidate_people(index, work_id):
    return sorted({name.split("(")[0].strip() for record in metadata_for(index, [work_id])
                   for source in record.get("metadata_sources", []) for name in (source.get("people") or [])
                   if isinstance(name, str) and 2 <= len(name) <= 100})
