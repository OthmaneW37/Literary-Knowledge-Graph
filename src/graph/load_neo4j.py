from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

from extraction.entity_resolution import (
    EntityCandidate,
    Mention,
    absorb_entities,
    resolve_mention,
)

load_dotenv()

WORK_QUERY = """
MERGE (w:Work {work_id: $work_id})
SET w.title = $title,
    w.author = $author,
    w.source = $source,
    w.source_url = $source_url,
    w.language = $language
"""

CHUNK_QUERY = """
MERGE (c:Chunk {chunk_id: $chunk_id})
SET c.work_id = $work_id,
    c.chapter = $chapter,
    c.text = $text
"""

CHARACTER_QUERY = """
MERGE (p:Character {canonical_name: $canonical_name})
ON CREATE SET p.aliases = $aliases, p.mentions = $mentions
ON MATCH SET p.aliases = $aliases, p.mentions = $mentions
"""

THEME_QUERY = """
MERGE (t:Theme {name: $name})
ON CREATE SET t.mentions = 1
ON MATCH SET t.mentions = coalesce(t.mentions, 0) + 1
"""

EVENT_QUERY = """
MERGE (e:Event {summary: $summary})
ON CREATE SET e.mentions = 1
ON MATCH SET e.mentions = coalesce(e.mentions, 0) + 1
"""

MENTIONS_CHARACTER_QUERY = """
MATCH (c:Chunk {chunk_id: $chunk_id})
MATCH (p:Character {canonical_name: $canonical_name})
MERGE (c)-[:MENTIONS_CHARACTER]->(p)
"""

HAS_THEME_QUERY = """
MATCH (c:Chunk {chunk_id: $chunk_id})
MATCH (t:Theme {name: $name})
MERGE (c)-[:HAS_THEME]->(t)
"""

HAS_EVENT_QUERY = """
MATCH (c:Chunk {chunk_id: $chunk_id})
MATCH (e:Event {summary: $summary})
MERGE (c)-[:HAS_EVENT]->(e)
"""

HAS_CHUNK_QUERY = """
MATCH (w:Work {work_id: $work_id})
MATCH (c:Chunk {chunk_id: $chunk_id})
MERGE (w)-[:HAS_CHUNK]->(c)
"""


def get_driver():
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    if not uri or not user or not password:
        raise ValueError("Missing NEO4J_URI, NEO4J_USER, or NEO4J_PASSWORD")
    return GraphDatabase.driver(uri, auth=(user, password))


def load_graph(
    manifest_path: str | Path = "data/annotations/work_manifest.json",
    processed_dir: str | Path = "data/processed",
):
    manifest_path = Path(manifest_path)
    processed_dir = Path(processed_dir)
    works = json.loads(manifest_path.read_text(encoding="utf-8"))

    with get_driver() as driver:
        with driver.session() as session:
            for work in works:
                session.run(WORK_QUERY, **work)

                chunks_path = processed_dir / f'{work["work_id"]}.chunks.json'
                extractions_path = processed_dir / f'{work["work_id"]}.extractions.json'

                chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
                extractions = json.loads(extractions_path.read_text(encoding="utf-8"))
                extraction_by_chunk = {item["chunk_id"]: item for item in extractions}

                global_entities: list[EntityCandidate] = []

                for chunk in chunks:
                    session.run(CHUNK_QUERY, **chunk)
                    session.run(
                        HAS_CHUNK_QUERY,
                        work_id=chunk["work_id"],
                        chunk_id=chunk["chunk_id"],
                    )

                    extraction = extraction_by_chunk.get(chunk["chunk_id"], {})
                    mentions = extraction.get("mentions", [])

                    valid_mentions = []
                    for mention in mentions:
                        if not isinstance(mention, dict):
                            continue
                        text = mention.get("text", "")
                        if not text:
                            continue
                        low = text.lower().strip()
                        if len(low.split()) > 4:
                            continue
                        if low in {"he", "she", "it", "they", "his", "her", "their", "him", "them"}:
                            continue
                        if any(
                            x in low
                            for x in [
                                "clock",
                                "door",
                                "room",
                                "pain",
                                "silence",
                                "weather",
                                "train",
                                "picture",
                                "chair",
                                "muff",
                                "frame",
                                "table",
                                "lamp",
                                "budget",
                                "work",
                                "time",
                                "gas",
                                "gaslight",
                                "couch",
                                "newspaper",
                                "alarm",
                                "locksmith",
                            ]
                        ):
                            continue
                        valid_mentions.append(mention)

                    chunk_entities: list[EntityCandidate] = []

                    for mention in valid_mentions:
                        m = Mention(
                            text=mention.get("text", ""),
                            context=mention.get("context", ""),
                            chapter=mention.get("chapter", chunk.get("chapter", "")),
                        )

                        canonical_name, aliases, matched = resolve_mention(m, global_entities)
                        if not canonical_name:
                            continue

                        entity = next((e for e in global_entities if e.canonical_name == canonical_name), None)
                        if entity is None:
                            entity = EntityCandidate(canonical_name=canonical_name)
                            global_entities.append(entity)

                        entity.aliases.update(aliases)
                        if m.context:
                            entity.contexts.add(m.context)
                        if m.chapter:
                            entity.contexts.add(m.chapter)
                        entity.mentions += 1

                        chunk_entities.append(entity)

                    for entity in chunk_entities:
                        session.run(
                            CHARACTER_QUERY,
                            canonical_name=entity.canonical_name,
                            aliases=sorted(entity.aliases),
                            mentions=entity.mentions,
                        )
                        session.run(
                            MENTIONS_CHARACTER_QUERY,
                            chunk_id=chunk["chunk_id"],
                            canonical_name=entity.canonical_name,
                        )

                    for theme in extraction.get("themes", []):
                        if isinstance(theme, dict) and theme.get("name"):
                            session.run(THEME_QUERY, name=theme["name"])
                            session.run(
                                HAS_THEME_QUERY,
                                chunk_id=chunk["chunk_id"],
                                name=theme["name"],
                            )

                    for event in extraction.get("events", []):
                        if isinstance(event, dict) and event.get("summary"):
                            session.run(EVENT_QUERY, summary=event["summary"])
                            session.run(
                                HAS_EVENT_QUERY,
                                chunk_id=chunk["chunk_id"],
                                summary=event["summary"],
                            )

            global_entities = absorb_entities(global_entities)

    return True


if __name__ == "__main__":
    print(load_graph())
