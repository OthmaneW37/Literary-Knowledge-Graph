from ingestion.chunk_text import split_into_chapters


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
