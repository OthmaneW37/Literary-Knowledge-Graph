from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import ollama
from dotenv import load_dotenv

from .local_index import LocalLiteraryIndex
from .models import Answer, GraphEdge, GraphNode, Passage, Visualization


load_dotenv()

DEFAULT_MODEL = "qwen2.5:7b-instruct"
RELATIONSHIP_HINTS = {
    "father", "mother", "parent", "son", "daughter", "brother", "sister",
    "family", "famille", "père", "pere", "mère", "mere", "frère", "frere",
    "sœur", "soeur", "fils", "fille", "relation", "relationship", "lié", "lie",
}


def _clean_json(value: str) -> dict[str, Any]:
    value = value.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?", "", value, flags=re.IGNORECASE).strip()
        value = re.sub(r"```$", "", value).strip()
    start, end = value.find("{"), value.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("The model did not return a JSON object")
    return json.loads(re.sub(r",\s*([}\]])", r"\1", value[start : end + 1]))


class LiteraryAssistant:
    def __init__(
        self,
        manifest_path: str | Path = "data/annotations/work_manifest.json",
        processed_dir: str | Path = "data/processed",
        model: str | None = None,
    ) -> None:
        self.index = LocalLiteraryIndex(manifest_path, processed_dir)
        self.model = model or os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)

    def retrieve(self, question: str, work_ids: list[str] | None = None, top_k: int = 6) -> list[Passage]:
        return self.index.search(question, work_ids=work_ids, top_k=top_k)

    def answer(
        self,
        question: str,
        work_ids: list[str] | None = None,
        top_k: int = 6,
        history: list[dict[str, str]] | None = None,
    ) -> Answer:
        expanded_query = self._expand_query(question, history)
        passages = self.retrieve(expanded_query, work_ids=work_ids, top_k=top_k)
        if not passages:
            return Answer(
                text="Je n’ai trouvé aucun passage suffisamment pertinent dans les œuvres sélectionnées.",
                citations=[],
                used_model=False,
            )

        try:
            payload = self._call_model(question, passages, history)
            return self._validate_answer(payload, passages)
        except Exception as exc:
            return self._extractive_fallback(question, passages, exc)

    def _expand_query(self, question: str, history: list[dict[str, str]] | None = None) -> str:
        """Add English literary search terms while preserving names and wording.

        The bundled novels are English while the UI is French. A short local
        translation/keyword pass gives lexical retrieval useful cross-language
        recall without any external API. Failure simply falls back to the
        original question.
        """
        recent_context = "\n".join(
            f"{item.get('role', '')}: {item.get('content', '')[:500]}"
            for item in (history or [])[-4:]
        )
        prompt = f"""
Transforme cette question en une requête de recherche pour retrouver des passages dans un roman anglais.
Conserve tous les noms propres. Ajoute des synonymes anglais utiles.
Résous les références comme « il », « elle » ou « ce personnage » à partir de l'historique.
Retourne uniquement une ligne courte de mots-clés, sans explication.

Historique récent :
{recent_context or '(aucun)'}

Question : {question}
""".strip()
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0, "num_predict": 80},
            )
            expansion = str(response["message"]["content"]).strip().replace("\n", " ")
            if expansion:
                return f"{question} {expansion[:500]}"
        except Exception:
            pass
        return question

    def _call_model(
        self,
        question: str,
        passages: list[Passage],
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        context = "\n\n".join(
            f"[SOURCE {passage.chunk_id}]\n"
            f"Œuvre: {passage.work_title}\nChapitre: {passage.chapter}\n"
            f"Texte: {passage.text}"
            for passage in passages
        )
        wants_graph = bool(set(re.findall(r"[^\W_]+", question.casefold())) & RELATIONSHIP_HINTS)
        recent_context = "\n".join(
            f"{item.get('role', '')}: {item.get('content', '')[:1000]}"
            for item in (history or [])[-6:]
        )
        allowed_ids = ", ".join(passage.chunk_id for passage in passages)
        prompt = f"""
Tu es un assistant d'analyse littéraire. Réponds dans la langue de la question.
Tu dois utiliser EXCLUSIVEMENT les sources fournies. N'invente aucun fait.
Si les sources sont insuffisantes, dis-le clairement.
Sépare les faits explicites de l'interprétation littéraire.

Retourne uniquement un objet JSON valide avec cette structure exacte :
{{
  "answer": "réponse claire avec des marqueurs de citation comme [chunk_id]",
  "citation_ids": ["chunk_id"],
  "visualization": {{
    "type": "relationship_graph ou timeline ou none",
    "title": "titre court",
    "nodes": [{{"id": "identifiant", "label": "nom", "kind": "character ou event"}}],
    "edges": [{{"source": "id", "target": "id", "label": "relation", "evidence_chunk_id": "chunk_id"}}]
  }}
}}

Règles :
- Chaque affirmation factuelle importante doit citer un identifiant fourni.
- Les SEULS identifiants autorisés sont : {allowed_ids}.
- Copie ces identifiants exactement dans citation_ids et evidence_chunk_id. N'écris jamais "chunk_id" ou "source_id".
- Une arête doit être explicitement soutenue par son evidence_chunk_id.
- Si aucune visualisation n'est utile, utilise type "none", nodes [] et edges [].
- Une visualisation de relation est souhaitée pour cette question : {str(wants_graph).lower()}.

Historique récent de la conversation :
{recent_context or '(aucun)'}

Question actuelle : {question}

Sources :
{context}
""".strip()
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0, "num_ctx": 16384, "num_predict": 1200},
        )
        return _clean_json(response["message"]["content"])

    def _validate_answer(self, payload: dict[str, Any], passages: list[Passage]) -> Answer:
        allowed = {passage.chunk_id: passage for passage in passages}
        answer_text = str(payload.get("answer", "")).strip()
        if not answer_text:
            raise ValueError("The model returned an empty answer")

        citation_ids = payload.get("citation_ids", [])
        if not isinstance(citation_ids, list):
            citation_ids = []
        # Also accept valid inline citations if the model omitted citation_ids.
        inline_ids = re.findall(r"\[([^\]]+)\]", answer_text)
        valid_ids = []
        for chunk_id in [*citation_ids, *inline_ids]:
            if chunk_id in allowed and chunk_id not in valid_ids:
                valid_ids.append(chunk_id)
        if not valid_ids:
            valid_ids = [passages[0].chunk_id]

        visualization = self._validate_visualization(payload.get("visualization"), allowed, valid_ids)
        return Answer(
            text=answer_text,
            citations=[allowed[chunk_id] for chunk_id in valid_ids],
            visualization=visualization,
            used_model=True,
            raw=payload,
        )

    @staticmethod
    def _validate_visualization(
        raw: Any,
        allowed: dict[str, Passage],
        preferred_evidence_ids: list[str] | None = None,
    ) -> Visualization:
        if not isinstance(raw, dict):
            return Visualization()
        graph_type = str(raw.get("type", "none")).strip().lower()
        if graph_type not in {"relationship_graph", "timeline"}:
            return Visualization()

        nodes: list[GraphNode] = []
        node_ids: set[str] = set()
        for item in raw.get("nodes", []):
            if not isinstance(item, dict):
                continue
            node_id = str(item.get("id", "")).strip()
            label = str(item.get("label", "")).strip()
            if not node_id or not label or node_id in node_ids:
                continue
            node_ids.add(node_id)
            nodes.append(GraphNode(node_id, label, str(item.get("kind", "character"))))

        edges: list[GraphEdge] = []
        for item in raw.get("edges", []):
            if not isinstance(item, dict):
                continue
            source = str(item.get("source", "")).strip()
            target = str(item.get("target", "")).strip()
            evidence = str(item.get("evidence_chunk_id", "")).strip()
            label = str(item.get("label", "relation")).strip()
            # Some local models correctly extract the relation but copy the
            # schema placeholder instead of the source id. Repair only when
            # the answer itself has exactly one validated supporting passage.
            placeholder = bool(re.fullmatch(r"(?:chunk|source)(?:_id)?(?:_\d+)?", evidence, flags=re.IGNORECASE))
            if (
                evidence not in allowed
                and placeholder
                and preferred_evidence_ids
                and len(preferred_evidence_ids) == 1
            ):
                evidence = preferred_evidence_ids[0]
            if source in node_ids and target in node_ids and evidence in allowed:
                edges.append(GraphEdge(source, target, label, evidence))

        if not nodes or (graph_type == "relationship_graph" and not edges):
            return Visualization()
        return Visualization(graph_type, str(raw.get("title", "")).strip(), nodes, edges)

    @staticmethod
    def _extractive_fallback(question: str, passages: list[Passage], error: Exception) -> Answer:
        excerpts = []
        for passage in passages[:3]:
            text = passage.text.strip().replace("\n", " ")
            excerpts.append(f"- **{passage.citation_label}** : {text[:500]}…")
        return Answer(
            text=(
                "Le modèle local n’a pas pu générer une synthèse. Voici les passages les plus pertinents "
                "retrouvés par le moteur local :\n\n" + "\n\n".join(excerpts)
            ),
            citations=passages[:3],
            used_model=False,
            raw={"error": str(error), "question": question},
        )


def visualization_to_dot(visualization: Visualization) -> str:
    """Convert a validated visualization to Graphviz DOT for Streamlit."""
    lines = ["digraph LiteraryGraph {", "rankdir=TB;", 'node [shape=box, style="rounded,filled", fillcolor="#F5E9D8"];']
    for node in visualization.nodes:
        safe_id = json.dumps(node.id)
        safe_label = json.dumps(node.label, ensure_ascii=False)
        shape = "ellipse" if node.kind == "character" else "box"
        lines.append(f"{safe_id} [label={safe_label}, shape={shape}];")
    for edge in visualization.edges:
        safe_source = json.dumps(edge.source)
        safe_target = json.dumps(edge.target)
        safe_label = json.dumps(edge.label, ensure_ascii=False)
        lines.append(f"{safe_source} -> {safe_target} [label={safe_label}];")
    lines.append("}")
    return "\n".join(lines)
