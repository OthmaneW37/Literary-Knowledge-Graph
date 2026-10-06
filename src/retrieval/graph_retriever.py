# src/retrieval/graph_retriever.py

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase


load_dotenv()


@dataclass
class GraphRelationship:
    source: str
    target: str
    relation: str
    evidence_chunk_id: str | None = None
    evidence: str = ""
    work_id: str = ""
    source_kind: str = "character"
    target_kind: str = "character"


@dataclass
class GraphRetrievalResult:
    characters: list[str] = field(default_factory=list)

    relationships: list[GraphRelationship] = field(
        default_factory=list
    )

    evidence_chunk_ids: list[str] = field(
        default_factory=list
    )

    available: bool = True
    # Mentions retain provenance even when no relation was extracted.
    nodes: list[dict] = field(default_factory=list)


class GraphRetriever:
    """
    Recherche des personnages et relations dans Neo4j.

    Neo4j reste optionnel :
    si la base n'est pas disponible, retrieve() renvoie un résultat vide.
    """

    def __init__(
        self,
        uri: str | None = None,
        username: str | None = None,
        password: str | None = None,
        enabled: bool | None = None,
        local_store=None,
    ) -> None:

        self.uri = uri if uri is not None else os.getenv("NEO4J_URI", "bolt://localhost:7687")

        self.username = username if username is not None else (
            os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME", "neo4j")
        )

        self.password = password if password is not None else os.getenv("NEO4J_PASSWORD", "")
        if enabled is None:
            enabled = os.getenv("NEO4J_ENABLED", "false").strip().casefold() in {
                "1", "true", "yes", "oui",
            }
        self.enabled = enabled

        self.driver = None
        self.local_store = local_store
        self.last_error = ""
        self._last_attempt = 0.0
        self._connection_attempted = False

    def _ensure_driver(self) -> bool:
        """Connect lazily so an offline optional Neo4j never delays startup."""
        if self.driver:
            return True
        if not self.enabled or not self.password:
            return False
        if self._connection_attempted and time.monotonic() - self._last_attempt < 30:
            return False
        self._connection_attempted = True
        self._last_attempt = time.monotonic()
        candidate = None
        try:
            candidate = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                connection_timeout=2.0,
                connection_acquisition_timeout=3.0,
                max_transaction_retry_time=0,
            )
            candidate.verify_connectivity()
            self.driver = candidate
        except Exception as exc:
            self.last_error = type(exc).__name__
            if candidate:
                candidate.close()
            self.driver = None
        return self.driver is not None

    @property
    def available(self) -> bool:
        return self._ensure_driver()

    def close(self) -> None:
        if self.driver:
            self.driver.close()
            self.driver = None

    def retrieve(
        self,
        entities: list[str],
        work_ids: list[str] | None = None,
        limit: int = 30,
    ) -> GraphRetrievalResult:

        if getattr(self, "local_store", None):
            local = self.local_store.retrieve(entities, work_ids, limit)
            if local.relationships or not self.enabled:
                return local
        if not self._ensure_driver():
            if getattr(self, "local_store", None):
                return local
            return GraphRetrievalResult(
                available=False
            )

        if not entities:
            return GraphRetrievalResult()

        normalized_entities = [
            entity.casefold().strip()
            for entity in entities
            if entity.strip()
        ]

        if not normalized_entities:
            return GraphRetrievalResult()

        try:
            records = self._query_relationships(
                normalized_entities,
                work_ids or [],
                limit,
            )
        except Exception:
            return GraphRetrievalResult(available=False)

        characters: list[str] = []
        relationships: list[GraphRelationship] = []
        evidence_ids: list[str] = []

        for record in records:

            source = record.get("source")
            target = record.get("target")
            relation = record.get("relation")
            evidence = record.get(
                "evidence_chunk_id"
            )

            if source and source not in characters:
                characters.append(source)

            if target and target not in characters:
                characters.append(target)

            if source and target and relation:

                relationship = GraphRelationship(
                    source=source,
                    target=target,
                    relation=relation,
                    evidence_chunk_id=evidence,
                    evidence=record.get("evidence") or "",
                    work_id=record.get("work_id") or "",
                    source_kind=record.get("source_kind") or "character",
                    target_kind=record.get("target_kind") or "character",
                )

                relationships.append(
                    relationship
                )

            if (
                evidence
                and evidence not in evidence_ids
            ):
                evidence_ids.append(evidence)

        return GraphRetrievalResult(
            characters=characters,
            relationships=relationships,
            evidence_chunk_ids=evidence_ids,
            available=True,
        )

    def explore(
        self,
        work_ids: list[str] | None = None,
        limit: int = 150,
    ) -> GraphRetrievalResult:
        """Return the sourced character relations for the selected local works."""
        if getattr(self, "local_store", None):
            local = self.local_store.explore(work_ids, limit)
            if local.relationships or not self.enabled:
                return local
        if not self._ensure_driver():
            if getattr(self, "local_store", None):
                return local
            return GraphRetrievalResult(available=False)
        cypher = """
        MATCH (source)-[r]->(target)
        WHERE type(r) IN ['CHARACTER_RELATION', 'LITERARY_RELATION']
          AND (size($work_ids) = 0 OR r.work_id IN $work_ids)
        RETURN DISTINCT
            source.canonical_name AS source,
            target.canonical_name AS target,
            coalesce(r.relation_type, type(r)) AS relation,
            r.evidence_chunk_id AS evidence_chunk_id,
            r.evidence AS evidence, r.work_id AS work_id,
            coalesce(source.kind, 'character') AS source_kind,
            coalesce(target.kind, 'character') AS target_kind
        LIMIT $limit
        """
        try:
            with self.driver.session() as session:
                records = session.run(
                    cypher,
                    work_ids=work_ids or [],
                    limit=max(1, min(limit, 500)),
                )
                relationships = [
                    GraphRelationship(
                        str(record["source"]),
                        str(record["target"]),
                        str(record["relation"]),
                        record.get("evidence_chunk_id"),
                        record.get("evidence") or "", record.get("work_id") or "",
                        record.get("source_kind") or "character", record.get("target_kind") or "character",
                    )
                    for record in records
                    if record.get("source") and record.get("target") and record.get("relation")
                ]
        except Exception:
            return GraphRetrievalResult(available=False)
        characters = sorted(
            {item for relation in relationships for item in (relation.source, relation.target)},
            key=str.casefold,
        )
        evidence_ids = list(dict.fromkeys(
            relation.evidence_chunk_id
            for relation in relationships
            if relation.evidence_chunk_id
        ))
        return GraphRetrievalResult(
            characters=characters,
            relationships=relationships,
            evidence_chunk_ids=evidence_ids,
            available=True,
        )

    @staticmethod
    def evidence_chapter(chunk_id: str | None) -> int | None:
        match = re.search(r"_ch(\d+)_", chunk_id or "", re.IGNORECASE)
        return int(match.group(1)) if match else None

    def _query_relationships(
        self,
        normalized_entities: list[str],
        work_ids: list[str],
        limit: int,
    ) -> list[dict[str, Any]]:

        cypher = """
        MATCH (source)-[r]->(target)

        WHERE
            any(
                entity IN $entities
                WHERE toLower(coalesce(source.canonical_name, '')) CONTAINS entity
                   OR toLower(coalesce(target.canonical_name, '')) CONTAINS entity
                   OR entity CONTAINS toLower(coalesce(source.canonical_name, ''))
                   OR entity CONTAINS toLower(coalesce(target.canonical_name, ''))
            )

        AND (
            size($work_ids) = 0
            OR r.work_id IN $work_ids
            OR source.work_id IN $work_ids
            OR target.work_id IN $work_ids
        )

        RETURN DISTINCT
            source.canonical_name AS source,
            target.canonical_name AS target,
            coalesce(r.relation_type, type(r)) AS relation,
            r.evidence_chunk_id AS evidence_chunk_id,
            r.evidence AS evidence, r.work_id AS work_id,
            coalesce(source.kind, 'character') AS source_kind,
            coalesce(target.kind, 'character') AS target_kind

        LIMIT $limit
        """

        with self.driver.session() as session:

            result = session.run(
                cypher,
                entities=normalized_entities,
                work_ids=work_ids,
                limit=limit,
            )

            return [
                dict(record)
                for record in result
            ]
