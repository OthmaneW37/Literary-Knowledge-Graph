from __future__ import annotations

import argparse
from pathlib import Path

from rag.config import RAGConfig
from rag.local_index import LocalLiteraryIndex
from retrieval import SemanticRetriever

from .gutendex import GutendexClient
from .library import LibraryImporter


def main() -> None:
    parser = argparse.ArgumentParser(description="Search and install public-domain novels.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    search_parser = subparsers.add_parser("search", help="search Gutendex")
    search_parser.add_argument("query")
    search_parser.add_argument("--language", choices=["en", "fr", "es", "de", "it", "pt"])
    import_parser = subparsers.add_parser("import", help="download and index one Gutenberg book")
    import_parser.add_argument("book_id", type=int)
    import_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    upload_parser = subparsers.add_parser(
        "upload", help="index a local EPUB, TXT, Markdown, PDF or DOCX"
    )
    upload_parser.add_argument("path", type=Path)
    upload_parser.add_argument("--title", required=True)
    upload_parser.add_argument("--author", default="Auteur inconnu")
    upload_parser.add_argument("--language", choices=["en", "fr", "es", "de", "it", "pt"], default="")
    upload_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    embeddings_parser = subparsers.add_parser(
        "embeddings", help="build or refresh the local semantic index"
    )
    embeddings_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    embeddings_parser.add_argument("--model", default=None)
    args = parser.parse_args()

    client = None
    try:
        if args.command == "search":
            client = GutendexClient()
            page = client.search(args.query, language=args.language)
            print(f"{page.count} résultat(s)")
            for book in page.books:
                print(f"{book.provider_id}: {book.title} — {book.author_display} [{book.language_display}]")
        elif args.command == "import":
            client = GutendexClient()
            book = client.get_book(args.book_id)
            result = LibraryImporter(args.data_dir).install(book, client.download(book))
            status = "déjà installé" if result.already_installed else "installé et indexé"
            print(f"{result.title}: {status}")
        elif args.command == "upload":
            result = LibraryImporter(args.data_dir).install_upload(
                content=args.path.read_bytes(),
                filename=args.path.name,
                title=args.title,
                author=args.author,
                language=args.language,
            )
            status = "déjà installé" if result.already_installed else "installé et indexé"
            print(f"{result.title}: {status}")
        else:
            config = RAGConfig.from_env()
            model = args.model or config.embedding_model
            if not model:
                raise ValueError("OLLAMA_EMBED_MODEL is empty; choose a model with --model")
            index = LocalLiteraryIndex(
                args.data_dir / "annotations" / "work_manifest.json",
                args.data_dir / "processed",
            )
            retriever = SemanticRetriever(
                index,
                model=model,
                cache_dir=args.data_dir / "library" / "embeddings",
                batch_size=config.embedding_batch_size,
                keep_alive=config.keep_alive,
            )
            count = retriever.prepare()
            print(f"Index sémantique prêt : {count} passages avec {model}")
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
