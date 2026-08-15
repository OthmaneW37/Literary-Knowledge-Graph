from __future__ import annotations

import json

import httpx

from catalog import CatalogBook, DownloadedBook, GutendexClient, LibraryImporter
from rag.local_index import LocalLiteraryIndex


def catalog_payload() -> dict:
    return {
        "count": 1,
        "next": None,
        "previous": None,
        "results": [
            {
                "id": 84,
                "title": "Frankenstein",
                "authors": [{"name": "Shelley, Mary Wollstonecraft"}],
                "languages": ["en"],
                "subjects": ["Gothic fiction"],
                "summaries": ["A scientist creates a living being."],
                "formats": {
                    "text/plain; charset=utf-8": "https://www.gutenberg.org/files/84/84-0.txt"
                },
                "download_count": 123,
                "copyright": False,
            }
        ],
    }


def test_gutendex_search_and_download() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "gutendex.test":
            assert request.url.params["search"] == "Frankenstein"
            return httpx.Response(200, json=catalog_payload())
        return httpx.Response(200, content=b"word " * 200)

    http_client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    client = GutendexClient("https://gutendex.test", client=http_client)

    page = client.search("Frankenstein", language="en")
    downloaded = client.download(page.books[0])

    assert page.count == 1
    assert page.books[0].work_id == "gutenberg_84"
    assert downloaded.suffix == ".txt"
    assert downloaded.content.startswith(b"word")


def test_install_book_and_load_it_into_local_index(tmp_path) -> None:
    data_dir = tmp_path / "data"
    annotations = data_dir / "annotations"
    annotations.mkdir(parents=True)
    manifest_path = annotations / "work_manifest.json"
    manifest_path.write_text("[]", encoding="utf-8")

    book = CatalogBook(
        provider_id=84,
        title="Frankenstein",
        authors=("Shelley, Mary Wollstonecraft",),
        languages=("en",),
    )
    text = (
        "*** START OF THE PROJECT GUTENBERG EBOOK FRANKENSTEIN ***\n"
        + "Victor Frankenstein creates a living creature. " * 160
        + "\n*** END OF THE PROJECT GUTENBERG EBOOK FRANKENSTEIN ***"
    )
    downloaded = DownloadedBook(
        content=text.encode("utf-8"),
        mime_type="text/plain; charset=utf-8",
        suffix=".txt",
        source_url="https://www.gutenberg.org/files/84/84-0.txt",
    )

    importer = LibraryImporter(data_dir)
    installed = importer.install(book, downloaded)
    installed_again = importer.install(book, downloaded)
    index = LocalLiteraryIndex(manifest_path, data_dir / "processed")

    assert not installed.already_installed
    assert installed_again.already_installed
    assert "gutenberg_84" in index.works
    assert index.search("Victor creature", work_ids=["gutenberg_84"])
    records = json.loads((data_dir / "library" / "installed_books.json").read_text(encoding="utf-8"))
    assert records[0]["provider_id"] == 84
