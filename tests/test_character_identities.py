import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from app.api import graph_dict
from graph.identities import CharacterIdentities
from graph.local_store import LocalGraphStore
from rag.engine import LiteraryAssistant
from rag.models import Passage
from storage.local import write_json


@pytest.fixture
def story(tmp_path, monkeypatch):
    passages = [
        Passage("book", "Book", 1, "one", 'Gregor Samsa waited. "Mr. Samsa", the clerk called to Gregor. His mother, Gregor\'s mother, answered.'),
        Passage("book", "Book", 2, "two", '"Mr. Samsa!", the lodger called to Gregor\'s father. His father rose. The boss himself was absent.'),
    ]
    assistant = LiteraryAssistant.__new__(LiteraryAssistant)
    assistant.index = SimpleNamespace(passages=passages, works={"book": object()},
                                     get_passage=lambda key: next((p for p in passages if p.chunk_id == key), None))
    store = LocalGraphStore(assistant.index, tmp_path / "graphs")
    assistant.hybrid_retriever = SimpleNamespace(graph=SimpleNamespace(local_store=store))
    names = [["Gregor Samsa", "Gregor", "Mr. Samsa", "his mother", "Gregor's mother"],
             ["Mr. Samsa", "Gregor's father", "his father", "the boss himself"]]
    records = {p.chunk_id: store.validate({"nodes": [{"name": n, "kind": "character"} for n in ns], "relations": []}, p)
               for p, ns in zip(passages, names)}
    write_json(store._path("book"), records)
    monkeypatch.setattr(CharacterIdentities, "ANNOTATION_DIRECTORY", tmp_path / "review")
    def entity(name, label, p):
        return {"name": name, "label": label, "kind": "character", "anchor": p.chunk_id,
                "anchor_fingerprint": store.fingerprint(p), "quote": name}
    review = {"entities": {"g": entity("Gregor Samsa", "Gregor Samsa", passages[0]),
                           "m": entity("Gregor's mother", "Mère de Gregor", passages[0]),
                           "f": entity("Gregor's father", "Père de Gregor", passages[1]),
                           "b": entity("The boss himself", "Patron de Gregor", passages[1])},
              "passages": {"one": {"fingerprint": store.fingerprint(passages[0]),
                         "assignments": dict(zip(names[0], ["g", "g", "g", "m", "m"]))},
                          "two": {"fingerprint": store.fingerprint(passages[1]),
                         "assignments": dict(zip(names[1], ["f", "f", "f", "b"]))}}, "relations": []}
    write_json(CharacterIdentities.ANNOTATION_DIRECTORY / "book.json", review)
    return assistant, store, review


def test_mother_aliases_become_one_identity_with_all_original_mentions(story):
    assistant, store, _ = story
    result = graph_dict(store.explore(), assistant, ["book"], None)
    mother = next(n for n in result["nodes"] if n["label"] == "Mère de Gregor")
    assert {"his mother", "Gregor's mother"} <= set(mother["aliases"])
    assert sum(n["label"] == "Mère de Gregor" for n in result["nodes"]) == 1
    assert mother["mentions"][0]["chunk_id"] == "one"
    assert result["identities"]["remaining"] == 0


def test_same_title_maps_to_different_people_in_different_passages(story):
    _, store, _ = story
    mapping = CharacterIdentities(store, "book").mapping()
    assert mapping[("one", "Mr. Samsa")]["label"] == "Gregor Samsa"
    assert mapping[("two", "Mr. Samsa")]["label"] == "Père de Gregor"
    assert mapping[("two", "the boss himself")]["label"] == "Patron de Gregor"


def test_reviewed_identities_do_not_reveal_future_chapters(story):
    assistant, store, _ = story
    result = graph_dict(store.explore(), assistant, ["book"], 1)
    assert {n["label"] for n in result["nodes"]} == {"Gregor Samsa", "Mère de Gregor"}
    assert all(m["chapter"] == 1 for n in result["nodes"] for m in n["mentions"])


def test_changed_anchor_invalidates_the_review(story):
    assistant, store, _ = story
    p = assistant.index.passages[0]
    assistant.index.passages[0] = replace(p, text=p.text + " Changed text.")
    record = store._read("book")
    record["one"]["fingerprint"] = store.fingerprint(assistant.index.passages[0])
    write_json(store._path("book"), record)
    resolver = CharacterIdentities(store, "book")
    assert resolver.mapping() is None
    assert resolver.remaining > 0


def test_review_cannot_be_applied_to_another_book(story):
    _, store, _ = story
    assert CharacterIdentities(store, "another").reviewed is None


def test_uncertain_occurrences_are_not_extra_people(story):
    assistant, store, review = story
    review["passages"]["two"]["assignments"]["the boss himself"] = None
    write_json(CharacterIdentities.ANNOTATION_DIRECTORY / "book.json", review)
    result = graph_dict(store.explore(), assistant, ["book"], None)
    assert result["identities"]["unresolved_mentions"] == 1
    assert not any(n["label"] == "Patron de Gregor" for n in result["nodes"])


def test_invalid_quote_never_publishes_a_reviewed_edge(story):
    assistant, store, review = story
    review["relations"] = [{"source_entity": "f", "target_entity": "b", "label": "identique à", "category": "family",
                             "evidence_chunk_id": "two", "fingerprint": store.fingerprint(assistant.index.passages[1]),
                             "evidence": "An invented quotation."}]
    write_json(CharacterIdentities.ANNOTATION_DIRECTORY / "book.json", review)
    assert graph_dict(store.explore(), assistant, ["book"], None)["edges"] == []


def test_incomplete_model_output_is_retryable_and_not_saved(story):
    _, store, _ = story
    resolver = CharacterIdentities(store, "book")
    resolver.reviewed = None
    resolver.data["entities"] = [{"name": "Gregor Samsa", "label": "Gregor Samsa", "kind": "character", "aliases": []}]
    provider = SimpleNamespace(chat=lambda **kw: {"message": {"content": '{"assignments": [], "relation_labels": []}'}})
    with pytest.raises(ValueError, match="incomplète"):
        resolver.step(provider, "test")
    assert resolver.data["assignments"] == {}
    assert resolver.data["done"] == []
    assert not resolver.path.exists()


def test_second_identity_pass_can_reject_proposed_assignments(story):
    _, store, _ = story
    resolver = CharacterIdentities(store, "book")
    resolver.reviewed = None
    resolver.data["entities"] = [{"name": "Gregor Samsa", "label": "Gregor Samsa", "kind": "character", "aliases": []}]
    responses = iter([{"assignments": [{"id": i, "entity": 0} for i in range(len(resolver.mentions))], "relation_labels": []},
                      {"accepted": [i for i, (_, name) in enumerate(resolver.mentions) if name in {"Gregor", "Gregor Samsa"}]}])
    provider = SimpleNamespace(chat=lambda **kw: {"message": {"content": json.dumps(next(responses))}})
    resolver.step(provider, "test")
    assert resolver.mapping()[("two", "the boss himself")] is None
    assert resolver.mapping()[("one", "Gregor")]["name"] == "Gregor Samsa"


def test_corpus_review_does_not_merge_father_boss_or_group_members():
    path = CharacterIdentities.ANNOTATION_DIRECTORY / "metamorphosis.json"
    review = json.loads(path.read_text(encoding="utf-8"))
    records = review["passages"]
    assert records["metamorphosis_ch01_p022"]["assignments"]["Mr. Samsa"] == "gregor"
    assert records["metamorphosis_ch03_p022"]["assignments"]["Mr. Samsa"] == "father"
    assert records["metamorphosis_ch01_p033"]["assignments"]["you, sir"] == "clerk"
    assert records["metamorphosis_ch01_p033"]["assignments"]["the boss himself"] == "boss"
    assert review["entities"]["lodgers"]["kind"] == "group"
    assert review["entities"]["middle"]["kind"] == "character"
