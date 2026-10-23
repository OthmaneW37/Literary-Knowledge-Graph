from __future__ import annotations

import csv
import gzip
import io
import os
import time
from types import SimpleNamespace

import httpx
import pytest

from catalog.gutendex import CatalogError
from catalog.gutenberg_catalog import GutenbergCatalog
from catalog.models import DownloadedBook
from catalog.opener import BookOpener

ROWS = [
    {"Text#": "28054", "Type": "Text", "Title": "The Brothers Karamazov", "Language": "en",
     "Authors": "Dostoyevsky, Fyodor, 1821-1881; Garnett, Constance, 1861-1946 [Translator]",
     "Bookshelves": "Best Books Ever Listings; Category: Novels"},
    {"Text#": "2554", "Type": "Text", "Title": "Crime and Punishment", "Language": "en",
     "Authors": "Dostoyevsky, Fyodor, 1821-1881", "Bookshelves": ""},
    {"Text#": "17489", "Type": "Text", "Title": "Les misérables Tome I: Fantine", "Language": "fr",
     "Authors": "Hugo, Victor, 1802-1885", "Bookshelves": ""},
    {"Text#": "99999", "Type": "Sound", "Title": "Karamazov audio", "Language": "en", "Authors": "", "Bookshelves": ""},
]


def catalog_bytes(rows=ROWS) -> bytes:
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=["Text#", "Type", "Issued", "Title", "Language", "Authors", "Subjects", "LoCC", "Bookshelves"])
    writer.writeheader()
    for row in rows:
        writer.writerow({"Issued": "2009-01-01", "Subjects": "Fiction", "LoCC": "PG", **row})
    return gzip.compress(text.getvalue().encode("utf-8"))


def make_catalog(tmp_path, handler=None):
    calls = []

    def default(request):
        calls.append(str(request.url))
        return httpx.Response(200, content=catalog_bytes())
    client = httpx.Client(transport=httpx.MockTransport(handler or default))
    return GutenbergCatalog(tmp_path, client=client, url="https://www.gutenberg.org/catalog.csv.gz"), calls


def test_catalog_is_downloaded_once_and_searched_across_languages(tmp_path):
    catalog, calls = make_catalog(tmp_path)
    found = catalog.search("les frères Karamazov")
    assert found.books[0].provider_id == 28054
    assert found.books[0].author_display == "Fyodor Dostoyevsky"
    assert catalog.search("Dostoïevski").count == 2  # transliteration tolerance
    assert catalog.search("misérables", language="fr").books[0].languages == ("fr",)
    assert catalog.search("Karamazov audio").count == 1  # audio entries are not books
    assert len(calls) == 1 and (tmp_path / "pg_catalog.csv.gz").exists()


def test_books_point_to_gutenberg_epub_and_text(tmp_path):
    catalog, _ = make_catalog(tmp_path)
    book = catalog.get_book(28054)
    assert book.work_id == "gutenberg_28054"
    assert book.formats["application/epub+zip"] == "https://www.gutenberg.org/ebooks/28054.epub3.images"
    assert "txt.utf-8" in book.formats["text/plain; charset=utf-8"]


def test_cached_catalog_survives_a_network_failure(tmp_path):
    (tmp_path / "pg_catalog.csv.gz").write_bytes(catalog_bytes())
    old = time.time() - 30 * 24 * 3600
    os.utime(tmp_path / "pg_catalog.csv.gz", (old, old))
    catalog, _ = make_catalog(tmp_path, lambda request: httpx.Response(503))
    assert catalog.search("Karamazov").books[0].provider_id == 28054


def test_missing_catalog_and_network_gives_a_clear_error(tmp_path):
    catalog, _ = make_catalog(tmp_path, lambda request: httpx.Response(503))
    with pytest.raises(CatalogError, match="Gutenberg"):
        catalog.search("Karamazov")


class FakeLibrary:
    def __init__(self):
        self.installed = set()

    def installed_ids(self):
        return set(self.installed)

    def install(self, book, downloaded):
        self.installed.add(book.work_id)
        return SimpleNamespace(work_id=book.work_id, title=book.title, already_installed=False)


class FakeSemantic:
    ready = False

    def prepare(self, progress=None):
        for done in (8, 16):
            progress(done, 16)
        self.ready = True


def wait_for(job, states=("done", "failed")):
    deadline = time.time() + 5
    while job.state not in states and time.time() < deadline:
        time.sleep(0.01)
    return job


def test_opening_a_book_indexes_it_then_embeds_in_the_background(tmp_path):
    catalog, _ = make_catalog(tmp_path)
    catalog.downloader = SimpleNamespace(download=lambda book: DownloadedBook(b"PK...", "application/epub+zip", ".epub", "https://www.gutenberg.org/x"))
    library = FakeLibrary()
    semantic = FakeSemantic()
    reloads = []
    opener = BookOpener(catalog, library, lambda: reloads.append(1) or SimpleNamespace(hybrid_retriever=SimpleNamespace(semantic=semantic)))
    job = wait_for(opener.open(28054))
    assert job.state == "done" and job.work_id == "gutenberg_28054" and job.title == "The Brothers Karamazov"
    assert (job.embedded, job.to_embed) == (16, 16) and reloads == [1]
    again = wait_for(opener.open(28054))
    assert again.already_installed is True


def test_failed_download_is_reported_on_the_job(tmp_path):
    catalog, _ = make_catalog(tmp_path)

    def fail(book):
        raise CatalogError("Téléchargement impossible.")
    catalog.downloader = SimpleNamespace(download=fail)
    job = wait_for(BookOpener(catalog, FakeLibrary(), lambda: None).open(2554))
    assert job.state == "failed" and "Téléchargement" in job.error


def test_translated_titles_find_the_english_edition(tmp_path):
    from catalog.discovery import BookDiscovery
    catalog, _ = make_catalog(tmp_path)

    class Provider:
        def chat(self, **kwargs):
            assert kwargs["messages"][1]["content"] == "crime et châtiment"
            return {"message": {"content": '{"title": "Crime and Punishment", "author": "Fyodor Dostoyevsky"}'}}
    result = BookDiscovery(catalog).search("crime et châtiment", source="gutendex", provider=Provider(), model="m")
    assert result["books"][0]["title"] == "Crime and Punishment" and result["books"][0]["can_download"] is True
    # Without a model the reader's own words are still searched.
    assert BookDiscovery(catalog).search("Karamazov", source="gutendex")["books"][0]["provider_id"] == "28054"


def wikisource_client():
    from urllib.parse import parse_qs
    long_text = " ".join(["Alexis regardait son père Fédor Pavlovitch."] * 400)
    pages = {
        "Les Frères": '<div class="headertemplate"><span class="headertemplate-author">Fédor Dostoïevski</span></div>'
                      '<p><a title="Les Frères/Livre II" href="/wiki/x">Livre II</a> <a title="Les Frères/Livre I" href="/wiki/y">Livre I</a></p>',
        "Les Frères/Livre I": f'<span class="pagenum">12</span><h2>I</h2><p>{long_text}<sup class="reference">[1]</sup></p>',
        "Les Frères/Livre II": f'<h2>II</h2><p>{long_text}</p>',
        "Éditions": "<p>Liste des éditions.</p>",
    }

    def handler(request):
        query = {k: v[0] for k, v in parse_qs(request.url.query.decode()).items()}
        assert "github.com" in request.headers["user-agent"]  # Wikimedia robot policy
        if query.get("list") == "search":
            return httpx.Response(200, json={"query": {"search": [{"title": "Les Frères/Livre I"}, {"title": "Les Frères"}, {"title": "Éditions"}]}})
        return httpx.Response(200, json={"parse": {"title": query["page"], "text": pages[query["page"]]}})
    from catalog.wikisource import USER_AGENT
    return httpx.Client(transport=httpx.MockTransport(handler), headers={"User-Agent": USER_AGENT})


def test_wikisource_assembles_a_french_work_in_reading_order():
    from catalog.wikisource import WikisourceCatalog
    from ingestion.chunk_text import split_into_chapters
    source = WikisourceCatalog(client=wikisource_client(), pause_seconds=0)
    found = source.search("Les Frères")
    assert [b["title"] for b in found] == ["Les Frères", "Éditions"] and found[0]["languages"] == ["fr"]
    book = source.fetch("fr:Les Frères")
    assert book["author"] == "Fédor Dostoïevski"
    assert book["text"].index("Livre II") < book["text"].index("Livre I\n")  # page order, not alphabetical
    assert "[1]" not in book["text"] and "12" not in book["text"].split("\n")[2]
    assert len(split_into_chapters(book["text"])) >= 2
    with pytest.raises(CatalogError, match="édition"):
        source.fetch("fr:Éditions")


def test_opening_a_wikisource_book_installs_the_french_text(tmp_path):
    from catalog.wikisource import WikisourceCatalog, work_id_for
    library = FakeLibrary()
    installed = {}
    library.install_wikisource = lambda work_id, book: installed.setdefault(work_id, book) and SimpleNamespace(work_id=work_id, already_installed=False)
    opener = BookOpener(None, library, lambda: SimpleNamespace(hybrid_retriever=SimpleNamespace(semantic=None)),
                        wikisource=WikisourceCatalog(client=wikisource_client(), pause_seconds=0))
    job = wait_for(opener.open("fr:Les Frères", "wikisource"))
    assert job.state == "done" and job.work_id == work_id_for("Les Frères") and installed[job.work_id]["language"] == "fr"
