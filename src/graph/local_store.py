from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from rag.prompts import RELATIONS_SYSTEM
from storage.local import read_json, write_json


class LocalGraphStore:
    """Incremental, sourced graph extraction; Neo4j is an optional projection."""

    KINDS = {"character", "place", "organization", "event", "theme"}
    SCHEMA_VERSION = "2"
    FALSE_SINGLE_WORD_CHARACTERS = {
        "he", "she", "it", "they", "them", "him", "her", "his", "hers",
        "i", "we", "you", "who", "what", "someone", "somebody", "anyone",
        "everyone", "god", "hell", "yes", "no", "oh", "this", "that",
    }

    def __init__(self, index, directory: str | Path = "data/library/graphs") -> None:
        self.index = index
        self.directory = Path(directory)

    @staticmethod
    def fingerprint(passage) -> str:
        # Bump the version whenever extraction/validation rules change so
        # stale graph entries are rebuilt rather than silently reused.
        value = f"{LocalGraphStore.SCHEMA_VERSION}\0{passage.text}"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _path(self, work_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", work_id):
            raise ValueError("Identifiant de livre invalide.")
        return self.directory / f"{work_id}.json"

    def _read(self, work_id: str) -> dict:
        return read_json(self._path(work_id), {})

    @classmethod
    def validate(cls, payload: dict, passage) -> dict:
        normalize = lambda value: re.sub(r"\s+", " ", value).strip()
        source_text = normalize(passage.text)
        nodes = {}
        for item in payload.get("nodes", [])[:50]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            kind = item.get("kind", "character")
            is_false_character = (
                kind == "character"
                and len(name.split()) == 1
                and (name.casefold() in cls.FALSE_SINGLE_WORD_CHARACTERS or not name[:1].isupper())
            )
            if (
                name and len(name) < 100 and name.casefold() in source_text.casefold()
                and kind in cls.KINDS and not is_false_character
            ):
                nodes[name] = kind
        relations = []
        for item in payload.get("relations", [])[:30]:
            if not isinstance(item, dict):
                continue
            source, target = item.get("source"), item.get("target")
            quote = item.get("evidence", "")
            label = str(item.get("relation", "")).strip()[:80]
            if not isinstance(source, str) or not isinstance(target, str) or not isinstance(quote, str):
                continue
            if source not in nodes or target not in nodes or source == target or not label or len(quote) < 12:
                continue
            if normalize(quote) not in source_text:
                continue
            if not all(name.casefold() in quote.casefold() for name in (source, target)):
                continue
            relations.append({"source": source, "target": target, "relation": label, "evidence": quote,
                              "source_kind": nodes[source], "target_kind": nodes[target]})
        return {"fingerprint": cls.fingerprint(passage), "nodes": nodes, "relations": relations}

    def build(self, provider, model: str, work_ids: list[str], limit: int = 10,
              max_chapter: int | None = None, progress=None) -> dict:
        records = {work_id: self._read(work_id) for work_id in work_ids}
        eligible = [p for p in self.index.passages if p.work_id in work_ids
                    and (max_chapter is None or (str(p.chapter).isdigit() and int(p.chapter) <= max_chapter))]
        missing = [p for p in eligible if records[p.work_id].get(p.chunk_id, {}).get("fingerprint") != self.fingerprint(p)]
        batch = missing[:max(1, limit)]
        for position, passage in enumerate(batch):
            response = provider.chat(
                model=model, messages=[{"role": "system", "content": RELATIONS_SYSTEM},
                                       {"role": "user", "content": passage.text}],
                format="json", think=False, options={"temperature": 0, "num_predict": 1000, "num_ctx": 8192}, keep_alive="15m",
            )
            payload = json.loads(response["message"]["content"])
            if not isinstance(payload, dict):
                raise ValueError("Extraction invalide ; les passages déjà traités ont été conservés.")
            records[passage.work_id][passage.chunk_id] = self.validate(payload, passage)
            write_json(self._path(passage.work_id), records[passage.work_id])
            if progress:
                progress(position + 1, len(batch))
        return {"processed": len(batch), "remaining": len(missing) - len(batch), "total": len(eligible)}

    def explore(self, work_ids: list[str] | None = None, limit: int = 150):
        from retrieval.graph_retriever import GraphRelationship, GraphRetrievalResult

        relationships = []
        characters = set()
        selected = self.index.works if work_ids is None else work_ids
        for work_id in selected:
            for chunk_id, record in self._read(work_id).items():
                passage = self.index.get_passage(chunk_id)
                if passage is None or passage.work_id != work_id or record.get("fingerprint") != self.fingerprint(passage):
                    continue
                for item in record.get("relations", []):
                    relationships.append(GraphRelationship(
                        source=item["source"], target=item["target"], relation=item["relation"],
                        evidence_chunk_id=chunk_id, evidence=item["evidence"], work_id=work_id,
                        source_kind=item.get("source_kind", "character"), target_kind=item.get("target_kind", "character"),
                    ))
                characters.update(name for name, kind in record.get("nodes", {}).items() if kind == "character")
        return GraphRetrievalResult(
            characters=sorted(characters), relationships=relationships[:limit],
            evidence_chunk_ids=list(dict.fromkeys(r.evidence_chunk_id for r in relationships[:limit])),
            available=True,
        )

    def retrieve(self, entities: list[str], work_ids=None, limit=30):
        result = self.explore(work_ids, limit=10000)
        names = [entity.casefold() for entity in entities]
        result.relationships = [r for r in result.relationships if any(
            name in r.source.casefold() or name in r.target.casefold() for name in names
        )][:limit]
        result.evidence_chunk_ids = list(dict.fromkeys(r.evidence_chunk_id for r in result.relationships))
        return result

    def sync_neo4j(self, driver, work_ids: list[str]) -> int:
        """Idempotent projection of locally verified relations into Neo4j."""
        result = self.explore(work_ids, limit=100000)
        query = """
        UNWIND $rows AS row
        MERGE (a:Entity {work_id: row.work_id, canonical_name: row.source})
        SET a.kind = row.source_kind
        MERGE (b:Entity {work_id: row.work_id, canonical_name: row.target})
        SET b.kind = row.target_kind
        MERGE (a)-[r:LITERARY_RELATION {work_id: row.work_id, relation_type: row.relation,
            evidence_chunk_id: row.evidence_chunk_id}]->(b)
        SET r.evidence = row.evidence
        """
        with driver.session() as session:
            def replace_projection(tx):
                tx.run("MATCH ()-[r:LITERARY_RELATION]->() WHERE r.work_id IN $works DELETE r", works=work_ids).consume()
                tx.run(query, rows=[r.__dict__ for r in result.relationships]).consume()
            session.execute_write(replace_projection)
        return len(result.relationships)
