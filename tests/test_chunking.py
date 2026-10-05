from ingestion.chunk_text import build_chunks_for_work, chunk_text, split_into_chapters


def test_split_into_chapters_ignores_table_of_contents() -> None:
    body = " ".join(["substantive literary text"] * 60)
    text = f"""BOOK
Chapter One
Chapter Two

Chapter One
{body}
Chapter Two
{body}
"""

    chapters = split_into_chapters(text)

    assert len(chapters) == 2
    assert chapters[0][0] == 1
    assert "substantive literary text" in chapters[0][1]


def test_split_standalone_roman_numerals() -> None:
    body = " ".join(["Gregor woke from troubled dreams"] * 60)
    chapters = split_into_chapters(f"I\n{body}\nII\n{body}\nIII\n{body}")
    assert [number for number, _ in chapters] == [1, 2, 3]


def test_chunking_prefers_paragraph_boundaries_and_stores_pdf_page(tmp_path) -> None:
    paragraph_a = "A useful literary paragraph has complete sentences. " * 12
    paragraph_b = "Another paragraph supplies different evidence. " * 12
    source = tmp_path / "book.clean.txt"
    target = tmp_path / "book.chunks.json"
    source.write_text(f"[[PAGE:4]]\n{paragraph_a}\n\n[[PAGE:5]]\n{paragraph_b}", encoding="utf-8")

    import json

    build_chunks_for_work("book", source, target)
    chunks = json.loads(target.read_text(encoding="utf-8"))

    assert len(chunks) == 1
    assert chunks[0]["page"] == 4
    assert chunks[0]["pages"] == [4, 5]
    assert "\n\n" in chunks[0]["text"]
    assert chunk_text(paragraph_a + "\n\n" + paragraph_b)[0].startswith("A useful")
