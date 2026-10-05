from __future__ import annotations

import argparse
import json
from pathlib import Path

from catalog import GutendexClient, LibraryImporter
from rag.engine import LiteraryAssistant


def main() -> None:
    parser = argparse.ArgumentParser(description="Local AI Reading Assistant")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    sub = parser.add_subparsers(dest="command", required=True)
    upload = sub.add_parser("import", help="import a local book")
    upload.add_argument("path", type=Path)
    upload.add_argument("--title", default="")
    upload.add_argument("--author", default="Auteur inconnu")
    download = sub.add_parser("download", help="download a public-domain Gutenberg book")
    download.add_argument("id", type=int)
    sub.add_parser("books", help="list the local library")
    sub.add_parser("demo", help="download and index the two Kafka demo editions")
    for name in ("ask", "search"):
        command = sub.add_parser(name)
        command.add_argument("question")
        command.add_argument("--work", action="append")
        command.add_argument("--through-chapter", type=int)
        command.add_argument("--top-k", type=int, default=4)
        if name == "ask":
            command.add_argument("--mode", default="Ask")
    index = sub.add_parser("index", help="prepare cached embeddings or reindex one book")
    index.add_argument("--work")
    for name in ("archive", "restore"):
        command = sub.add_parser(name)
        command.add_argument("work_id")
    graph = sub.add_parser("graph", help="extract an incremental batch of sourced relations")
    graph.add_argument("--work", action="append", required=True)
    graph.add_argument("--limit", type=int, default=5)
    graph.add_argument("--through-chapter", type=int)
    graph.add_argument("--sync", action="store_true")
    sub.add_parser("doctor", help="check local services and models")
    args = parser.parse_args()
    library = LibraryImporter(args.data_dir)
    try:
        if args.command == "import":
            result = library.install_upload(args.path.read_bytes(), args.path.name, args.title, args.author)
            print(f"{result.title}: {result.work_id}")
            return
        if args.command in {"download", "demo"}:
            client = GutendexClient()
            try:
                if args.command == "download":
                    book = client.get_book(args.id)
                    print(library.install(book, client.download(book)))
                else:
                    for work_id, provider_id in [("metamorphosis", 5200), ("the_trial", 7849)]:
                        book = client.get_book(provider_id)
                        downloaded = client.download(book)
                        record = {"work_id": work_id, "title": book.title, "author": book.author_display,
                                  "language": "en", "source": "Project Gutenberg", "source_url": downloaded.source_url,
                                  "raw_filename": f"{work_id}{downloaded.suffix}"}
                        print(library._install_content(work_id, book.title, downloaded.content, downloaded.suffix, record))
            finally:
                client.close()
            return
        if args.command == "books":
            for record in library.records():
                print(f"{record['work_id']}: {record['title']}" + (" [archivé]" if record["work_id"] in library.archived_ids() else ""))
            return
        if args.command in {"archive", "restore"}:
            library.set_archived(args.work_id, args.command == "archive")
            print("Bibliothèque mise à jour. Les fichiers originaux sont conservés.")
            return
        if args.command == "index" and args.work:
            library.reindex(args.work)
        assistant = LiteraryAssistant(args.data_dir / "annotations/work_manifest.json", args.data_dir / "processed")
        if args.command == "index":
            semantic = assistant.hybrid_retriever.semantic
            if not semantic:
                raise ValueError("OLLAMA_EMBED_MODEL est désactivé.")
            print(f"{semantic.prepare()} passages prêts")
        elif args.command == "search":
            for passage in assistant.retrieve(args.question, args.work, args.top_k, max_chapter=args.through_chapter):
                print(f"\n{passage.citation_label}\n{passage.text}\n")
        elif args.command == "ask":
            answer = assistant.answer(args.question, args.work, args.top_k, max_chapter=args.through_chapter, mode=args.mode)
            print(answer.text)
            for passage in answer.citations:
                print(f"\nSource: {passage.citation_label}\n{passage.text}")
            print(f"\nRecherche {answer.retrieval_ms} ms · génération {answer.generation_ms} ms")
        elif args.command == "graph":
            service = assistant.hybrid_retriever.graph
            result = service.local_store.build(assistant.llm, assistant.model, args.work, args.limit, args.through_chapter,
                                               progress=lambda done, total: print(f"{done}/{total}", flush=True))
            print(json.dumps(result))
            if args.sync:
                if not service.available:
                    raise ValueError("Neo4j indisponible ; l’extraction locale a été conservée.")
                print(f"{service.local_store.sync_neo4j(service.driver, args.work)} relations synchronisées")
        elif args.command == "doctor":
            models = assistant.llm.client.list()
            names = {model.model for model in models.models}
            print("Ollama : disponible")
            for model in [assistant.model, assistant.config.embedding_model]:
                print(f"{model}: {'installé' if model in names else 'à installer avec ollama pull'}")
            print(f"Neo4j : {'disponible' if assistant.hybrid_retriever.graph.available else 'optionnel, indisponible'}")
        assistant.hybrid_retriever.graph.close()
    except Exception as error:
        parser.exit(2, f"Erreur : {error}\n")


if __name__ == "__main__":
    main()
