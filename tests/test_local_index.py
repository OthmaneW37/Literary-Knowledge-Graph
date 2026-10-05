import json

from rag.local_index import LocalLiteraryIndex, tokenize


def make_index(tmp_path):
    annotations = tmp_path / "annotations"
    processed = tmp_path / "processed"
    annotations.mkdir()
    processed.mkdir()
    manifest = [
        {"work_id": "trial", "title": "The Trial", "author": "Kafka", "language": "en"},
        {"work_id": "meta", "title": "Metamorphosis", "author": "Kafka", "language": "en"},
    ]
    (annotations / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (processed / "trial.chunks.json").write_text(
        json.dumps([{"work_id": "trial", "chapter": 1, "chunk_id": "trial_1", "text": "Josef K. was arrested one morning by two warders."}]),
        encoding="utf-8",
    )
    (processed / "meta.chunks.json").write_text(
        json.dumps([{"work_id": "meta", "chapter": 1, "chunk_id": "meta_1", "text": "Gregor Samsa woke transformed into a vermin."}]),
        encoding="utf-8",
    )
    return LocalLiteraryIndex(annotations / "manifest.json", processed)


def test_tokenize_normalizes_accents() -> None:
    assert "pere" in tokenize("Le père de Grégoire")
    assert "gregoire" in tokenize("Le père de Grégoire")
    assert tokenize("arrests arrested") == ["arrest", "arrest"]


def test_search_finds_and_filters_work(tmp_path) -> None:
    index = make_index(tmp_path)
    assert index.search("Josef arrested")[0].chunk_id == "trial_1"
    assert index.search("Gregor", work_ids=["trial"]) == []
    assert index.search("Who arrests Josef?")[0].chunk_id == "trial_1"
    assert index.get_neighbors("trial_1") == []
