# src/retrieval/graph_retriever.py

from __future__ import annotations

import os
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
    ) -> None:

        self.uri = uri if uri is not None else os.getenv("NEO4J_URI", "bolt://localhost:7687")

        self.username = username if username is not None else (
            os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME", "neo4j")
        )

        self.password = password if password is not None else os.getenv("NEO4J_PASSWORD", "")

        self.driver = None

        if self.password:
            try:
                self.driver = GraphDatabase.driver(
                    self.uri,
                    auth=(
                        self.username,
                        self.password,
                    ),
                )

                self.driver.verify_connectivity()

            except Exception:
                if self.driver:
                    self.driver.close()
                self.driver = None

    @property
    def available(self) -> bool:
        return self.driver is not None

    def close(self) -> None:
        if self.driver:
            self.driver.close()

    def retrieve(
        self,
        entities: list[str],
        work_ids: list[str] | None = None,
        limit: int = 30,
    ) -> GraphRetrievalResult:

        if not self.driver:
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

    def _query_relationships(
        self,
        normalized_entities: list[str],
        work_ids: list[str],
        limit: int,
    ) -> list[dict[str, Any]]:

        cypher = """
        MATCH (source:Character)-[r:CHARACTER_RELATION]->(target:Character)

        WHERE
            any(
                entity IN $entities
                WHERE toLower(coalesce(source.canonical_name, source.name, '')) CONTAINS entity
                   OR toLower(coalesce(target.canonical_name, target.name, '')) CONTAINS entity
            )

        AND (
            size($work_ids) = 0
            OR r.work_id IN $work_ids
            OR source.work_id IN $work_ids
            OR target.work_id IN $work_ids
        )

        RETURN DISTINCT
            coalesce(source.canonical_name, source.name) AS source,
            coalesce(target.canonical_name, target.name) AS target,
            coalesce(r.relation_type, type(r)) AS relation,
            r.evidence_chunk_id AS evidence_chunk_id

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
