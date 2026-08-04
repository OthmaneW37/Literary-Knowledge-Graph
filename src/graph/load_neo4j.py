from __future__ import annotations

import json
import os
from pathlib import Path

from neo4j import GraphDatabase
from dotenv import load_dotenv

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
MERGE (p:Character {name: $name})
"""

THEME_QUERY = """
MERGE (t:Theme {name: $name})
"""

EVENT_QUERY = """
MERGE (e:Event {summary: $summary})
"""

HAS_CHUNK_QUERY = """
MATCH (w:Work {work_id: $work_id})
MATCH (c:Chunk {chunk_id: $chunk_id})
MERGE (w)-[:HAS_CHUNK]->(c)
"""

MENTIONS_CHARACTER_QUERY = """
MATCH (c:Chunk {chunk_id: $chunk_id})
MATCH (p:Character {name: $name})
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

                for chunk in chunks:
                    session.run(CHUNK_QUERY, **chunk)
                    session.run(HAS_CHUNK_QUERY, work_id=chunk["work_id"], chunk_id=chunk["chunk_id"])

                    extraction = extraction_by_chunk.get(chunk["chunk_id"], {})
                    for name in extraction.get("characters", []):
                        session.run(CHARACTER_QUERY, name=name)
                        session.run(MENTIONS_CHARACTER_QUERY, chunk_id=chunk["chunk_id"], name=name)

                    for name in extraction.get("themes", []):
                        session.run(THEME_QUERY, name=name)
                        session.run(HAS_THEME_QUERY, chunk_id=chunk["chunk_id"], name=name)

                    for event in extraction.get("events", []):
                        summary = event.get("summary")
                        if summary:
                            session.run(EVENT_QUERY, summary=summary)
                            session.run(HAS_EVENT_QUERY, chunk_id=chunk["chunk_id"], summary=summary)

    return True


if __name__ == "__main__":
    print(load_graph())