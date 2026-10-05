from __future__ import annotations

import json
from pathlib import Path

import pytest
from ebooklib import epub

from catalog.library import LibraryImporter
from graph.local_store import LocalGraphStore
from ingestion.chunk_text import build_chunks_for_work, chunk_text, split_into_chapters
from ingestion.loaders import DocumentLoader, detect_language
from rag.engine import LiteraryAssistant
from rag.local_index import LocalLiteraryIndex
from rag.models import Passage
from retrieval.reranker import diversify
from storage.local import ConversationStore


def test_epub_uses_spine_order_and_real_metadata(tmp_path):
    book = epub.EpubBook()
    book.set_identifier("new-novel")
    book.set_title("The Unknown Novel")
    book.set_language("en")
    book.add_author("Jane Reader")
    first = epub.EpubHtml(title="Chapter 1", file_name="first.xhtml")
    first.content = '<h1>Chapter 1</h1><p>' + 'Alice met Bernard at the station. ' * 30 + '</p>'
    second = epub.EpubHtml(title="Chapter 2", file_name="second.xhtml")
    second.content = '<h1>Chapter 2</h1><p>' + 'Later Bernard returned to Paris. ' * 30 + '</p>'
    book.add_item(second)
    book.add_item(first)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = [first, second]
    source = tmp_path / "book.epub"
    epub.write_epub(str(source), book)
    loaded = DocumentLoader().load(source)
    assert loaded.title == "The Unknown Novel"
    assert loaded.author == "Jane Reader"
    assert loaded.language == "en"
    assert loaded.text.index("Alice met") < loaded.text.index("Later Bernard")
    imported = LibraryImporter(tmp_path / "data").install_upload(source.read_bytes(), source.name)
    index = LocalLiteraryIndex(tmp_path / "data/annotations/work_manifest.json", tmp_path / "data/processed")
    assert index.works[imported.work_id].title == "The Unknown Novel"
    assert index.search("Bernard", [imported.work_id], max_chapter=1)


def test_epub_skips_project_gutenberg_boilerplate_spine_items():
    book = epub.EpubBook()
    book.set_identifier("gutenberg-like")
    book.set_title("A Sample Novel")
    book.set_language("en")
    header = epub.EpubHtml(title="Header", file_name="pg-header.xhtml")
    header.content = "<h1>Chapter I. Contents</h1><p>Chapter I Chapter II</p>"
    chapter = epub.EpubHtml(title="Chapter", file_name="chapter.xhtml")
    chapter.content = "<h1>Chapter I</h1><p>" + "Alice began her journey. " * 40 + "</p>"
    footer = epub.EpubHtml(title="Footer", file_name="pg-footer.xhtml")
    footer.content = "<p>*** END OF THE PROJECT GUTENBERG EBOOK ***</p>"
    for item in (header, chapter, footer):
        book.add_item(item)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = [header, chapter, footer]
    import io

    data = io.BytesIO()
    epub.write_epub(data, book)
    loaded = DocumentLoader().load(Path("sample.epub"), content=data.getvalue())
    assert "Alice began her journey" in loaded.text
    assert "END OF THE PROJECT GUTENBERG" not in loaded.text
    assert loaded.text.count("Chapter I") == 1


def test_chunking_preserves_short_ending_and_bounded_overlap():
    text = " ".join(f"word{i}" for i in range(601)) + "\n\nFinal sentence."
    chunks = chunk_text(text, 100, 20)
    assert "Final sentence." in chunks[-1]
    assert all(len(chunk.split()) <= 100 for chunk in chunks)
    assert set(text.split()) <= set(" ".join(chunks).split())
    chapters = split_into_chapters("Chapter 1\n" + "scene " * 60 + "\nChapter 2\nThe end.")
    assert len(chapters) == 2 and "The end." in chapters[-1][1]


def test_pdf_page_inherited_when_chapter_begins_midpage(tmp_path):
    text = "[[PAGE:7]]\nChapter 1\n" + "first scene " * 60 + "\nChapter 2\n" + "last scene " * 60
    source, target = tmp_path / "book.txt", tmp_path / "chunks.json"
    source.write_text(text, encoding="utf-8")
    build_chunks_for_work("book", source, target)
    records = json.loads(target.read_text(encoding="utf-8"))
    plain = text.replace("[[PAGE:7]]\n", "")
    assert all(r["page"] == 7 for r in records)
    assert all(plain[r["start_char"]:r["end_char"]] == r["text"] for r in records)


def test_archive_restore_and_reindex_keep_source(tmp_path):
    data = tmp_path / "data"
    library = LibraryImporter(data)
    book = library.install_upload(("Alice rencontre Bernard dans la ville. " * 60).encode(), "novel.txt", "Roman")
    paths = (data / "annotations/work_manifest.json", data / "processed")
    assert LocalLiteraryIndex(*paths).search("Alice")
    library.set_archived(book.work_id)
    assert not LocalLiteraryIndex(*paths).works
    library.set_archived(book.work_id, False)
    library.reindex(book.work_id)
    assert LocalLiteraryIndex(*paths).search("Alice")
    assert (data / "raw" / f"{book.work_id}.txt").exists()


def test_discussions_are_separated_by_scope_and_survive_restart(tmp_path):
    store = ConversationStore(tmp_path)
    messages = [{"role": "user", "content": "Who is Alice?"}]
    identifier = store.save("book1:chapter1", messages)
    restarted = ConversationStore(tmp_path)
    assert restarted.load(identifier, "book1:chapter1") == messages
    assert not restarted.list("book2:chapter1")
    with pytest.raises(ValueError):
        restarted.load(identifier, "book1:chapter2")
    with pytest.raises(ValueError):
        restarted.archive("../../other")
    restarted.archive(identifier)
    assert not restarted.list("book1:chapter1")


def test_graph_rejects_invented_evidence_and_keeps_provenance():
    passage = Passage("book", "Book", 1, "book_ch01_p001", "Alice is the mother of Bernard. They live in Paris.")
    payload = {
        "nodes": [{"name": "Alice", "kind": "character"}, {"name": "Bernard", "kind": "character"}],
        "relations": [{"source": "Alice", "target": "Bernard", "relation": "MOTHER_OF", "evidence": "Alice is the mother of Bernard."}],
    }
    assert len(LocalGraphStore.validate(payload, passage)["relations"]) == 1
    payload["relations"][0]["evidence"] = "Alice is married to Bernard."
    assert not LocalGraphStore.validate(payload, passage)["relations"]


def test_graph_does_not_treat_pronouns_or_common_exclamations_as_characters():
    passage = Passage("book", "Book", 1, "a", 'He said, "Go to Hell!"')
    payload = {"nodes": [{"name": "he", "kind": "character"}, {"name": "Hell", "kind": "character"}], "relations": []}
    assert LocalGraphStore.validate(payload, passage)["nodes"] == {}


def test_answer_rejects_citation_not_sent_to_model():
    assistant = LiteraryAssistant.__new__(LiteraryAssistant)
    passages = [Passage("book", "Book", 1, "a", "Visible source."), Passage("book", "Book", 2, "b", "Unseen source.")]
    with pytest.raises(ValueError, match="unseen"):
        assistant._validate_answer({"answer": "Claim [b]", "citation_ids": ["b"], "_context_ids": ["a"]}, passages)


def test_quotes_must_be_verbatim_from_their_own_source():
    assistant = LiteraryAssistant.__new__(LiteraryAssistant)
    passage = Passage("book", "Book", 1, "a", "Alice met Bernard at the station.")
    payload = {"answer": "Alice meets Bernard [a]", "citation_ids": ["a"], "_require_quotes": True,
               "evidence_quotes": {"a": "Alice met Bernard at the station."}}
    assert assistant._validate_answer(payload, [passage]).used_model
    payload["evidence_quotes"]["a"] = "Alice murdered Bernard."
    with pytest.raises(ValueError, match="quote"):
        assistant._validate_answer(payload, [passage])


def test_answer_reconciles_an_exact_quote_misattributed_to_a_context_source():
    assistant = LiteraryAssistant.__new__(LiteraryAssistant)
    first = Passage("book", "Book", 1, "a", "Alice followed the White Rabbit into the garden.")
    second = Passage("book", "Book", 2, "b", "She took up the key and hurried off to the garden door.")
    payload = {
        "answer": "Alice follows the White Rabbit and hurries into the garden.",
        "citation_ids": ["a", "b"],
        "evidence_quotes": {"a": "Alice followed the White Rabbit", "b": "She took up the key and hurried off"},
        "_context_ids": ["a", "b"], "_require_quotes": True,
    }
    answer = assistant._validate_answer(payload, [first, second])
    assert [passage.chunk_id for passage in answer.citations] == ["a", "b"]


def test_diversity_reranking_preserves_relevance_and_avoids_duplicate_context():
    passages = [Passage("book", "Book", 1, "a", "Alice met Bernard at the station."),
                Passage("book", "Book", 1, "b", "Alice met Bernard at the station."),
                Passage("book", "Book", 2, "c", "Bernard went home after buying a ticket.")]
    assert [p.chunk_id for p in diversify(passages, 2)] == ["a", "c"]


@pytest.mark.parametrize("text, expected", [
    ("Le jeune homme était dans la maison avec sa famille.", "fr"),
    ("The young man was in the house with his family.", "en"),
])
def test_book_language(text, expected):
    assert detect_language(text) == expected
