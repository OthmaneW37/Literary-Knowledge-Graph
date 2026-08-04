from __future__ import annotations

WORK_NODE = """
MERGE (w:Work {work_id: $work_id})
SET w.title = $title,
    w.author = $author,
    w.source = $source,
    w.source_url = $source_url,
    w.language = $language
"""

CHUNK_NODE = """
MERGE (c:Chunk {chunk_id: $chunk_id})
SET c.work_id = $work_id,
    c.chapter = $chapter,
    c.text = $text
"""

CHARACTER_NODE = """
MERGE (p:Character {name: $name})
"""

THEME_NODE = """
MERGE (t:Theme {name: $name})
"""

EVENT_NODE = """
MERGE (e:Event {summary: $summary})
"""

HAS_CHUNK_REL = """
MATCH (w:Work {work_id: $work_id})
MATCH (c:Chunk {chunk_id: $chunk_id})
MERGE (w)-[:HAS_CHUNK]->(c)
"""

MENTIONS_CHARACTER_REL = """
MATCH (c:Chunk {chunk_id: $chunk_id})
MATCH (p:Character {name: $name})
MERGE (c)-[:MENTIONS_CHARACTER]->(p)
"""

HAS_THEME_REL = """
MATCH (c:Chunk {chunk_id: $chunk_id})
MATCH (t:Theme {name: $name})
MERGE (c)-[:HAS_THEME]->(t)
"""

HAS_EVENT_REL = """
MATCH (c:Chunk {chunk_id: $chunk_id})
MATCH (e:Event {summary: $summary})
MERGE (c)-[:HAS_EVENT]->(e)
"""