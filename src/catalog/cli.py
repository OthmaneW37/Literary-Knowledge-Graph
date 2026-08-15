from __future__ import annotations

import argparse
from pathlib import Path

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
    args = parser.parse_args()

    client = GutendexClient()
    try:
        if args.command == "search":
            page = client.search(args.query, language=args.language)
            print(f"{page.count} résultat(s)")
            for book in page.books:
                print(f"{book.provider_id}: {book.title} — {book.author_display} [{book.language_display}]")
        else:
            book = client.get_book(args.book_id)
            result = LibraryImporter(args.data_dir).install(book, client.download(book))
            status = "déjà installé" if result.already_installed else "installé et indexé"
            print(f"{result.title}: {status}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
