from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation import CATEGORIES, EvaluationRunner, classify, evaluate_checks, load_scenarios, validate_against_corpus
from evaluation.scenarios import Scenario
from rag.engine import LiteraryAssistant
from rag.models import Answer, Passage
from reranking.llm import LLMReranker
from retrieval import HybridRetriever, QueryAnalysis
from retrieval.query_expander import Expansion
from workflows.narrative_graph import NarrativeWorkflow

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "data/annotations/scenarios.json"


def passage(chunk_id="m_ch01_001", chapter=1, text="Samsa was a travelling salesman and a son.", work_id="meta"):
    return Passage(work_id=work_id, work_title="Meta", chapter=chapter, chunk_id=chunk_id, text=text)


def scenario(**values):
    base = dict(id="s1", category="facts", title="t", question="Quel métier exerce Gregor ?", work_ids=["meta"],
                expect="answer", expected="voyageur")
    return Scenario.from_dict({**base, **values})


# Dataset -----------------------------------------------------------------

def test_annotated_dataset_is_well_formed_and_covers_every_category():
    scenarios = load_scenarios(SCENARIOS)
    assert len(scenarios) >= 70
    assert {s.category for s in scenarios} == set(CATEGORIES)
    assert sum(s.showcase for s in scenarios) >= len(CATEGORIES)
    for s in scenarios:
        assert s.expected and s.question
        if s.expect == "answer":
            assert s.gold or s.must_include_any, s.id


def test_annotated_quotes_exist_in_the_local_corpus_when_installed():
    processed = ROOT / "data/processed"
    if not (processed / "metamorphosis.chunks.json").exists() or not (processed / "the_trial.chunks.json").exists():
        pytest.skip("Corpus Gutenberg non installé sur cette machine")
    passages = [passage(c["chunk_id"], c["chapter"], c["text"], c["work_id"])
                for work in ("metamorphosis", "the_trial")
                for c in json.loads((processed / f"{work}.chunks.json").read_text(encoding="utf-8"))]
    problems = validate_against_corpus(load_scenarios(SCENARIOS), passages, ROOT / "data/annotations/scenario_images")
    assert problems == []


def test_validation_rejects_invented_quotes_and_gold_beyond_the_reading_limit():
    corpus = [passage(), passage("m_ch03_001", 3, "Gregor was dead and the family took the tram.")]
    problems = validate_against_corpus([
        scenario(id="missing", gold=[{"work_id": "meta", "contains": "an invented sentence"}]),
        scenario(id="late", max_chapter=1, gold=[{"work_id": "meta", "contains": "took the tram"}]),
        scenario(id="fine", max_chapter=1, gold=[{"work_id": "meta", "contains": "travelling  SALESMAN"}]),
    ], corpus)
    assert any(p.startswith("missing") for p in problems)
    assert any(p.startswith("late") for p in problems)
    assert not any(p.startswith("fine") for p in problems)


def test_duplicate_scenario_ids_are_refused(tmp_path):
    path = tmp_path / "s.json"
    item = scenario().to_dict()
    path.write_text(json.dumps({"scenarios": [item, item]}), encoding="utf-8")
    with pytest.raises(ValueError, match="double"):
        load_scenarios(path)


# Verdicts ----------------------------------------------------------------

def test_verified_answer_with_gold_passes_every_check():
    s = scenario(gold=[{"work_id": "meta", "contains": "travelling salesman"}], must_include_any=["voyageur"])
    answer = Answer(text="Gregor est voyageur de commerce. [m_ch01_001]", citations=[passage()])
    checks = evaluate_checks(s, answer, classify(answer))
    assert all(c.passed for c in checks), [c.detail for c in checks if not c.passed]


def test_citation_beyond_reading_limit_fails_the_spoiler_check():
    s = scenario(category="spoiler", expect="safe", max_chapter=1, must_not_include=["mort"])
    answer = Answer(text="Something happens.", citations=[passage("m_ch03_001", 3)])
    failed = [c for c in evaluate_checks(s, answer, classify(answer)) if not c.passed]
    assert [c.name for c in failed] == ["Limite de chapitre respectée"]


def test_forbidden_terms_match_whole_words_only():
    s = scenario(expect="safe", must_not_include=["dies"])
    clean = Answer(text="He studies the text.", citations=[], used_model=False)
    leaked = Answer(text="Gregor dies at the end.", citations=[], used_model=False)
    assert all(c.passed for c in evaluate_checks(s, clean, classify(clean)))
    assert not all(c.passed for c in evaluate_checks(s, leaked, classify(leaked)))


def test_outcomes_distinguish_fallback_premise_and_spoiler_guard():
    assert classify(Answer("x", [passage()], used_model=False)) == "fallback"
    assert classify(Answer("x", [], raw={"premise_rejected": True})) == "premise_rejected"
    assert classify(Answer("x", [], used_model=False, raw={"spoiler_guard": True})) == "spoiler_guarded"
    assert classify(Answer("x", [])) == "abstained"
    premise = scenario(category="false_premise", expect="reject_premise")
    assert all(c.passed for c in evaluate_checks(premise, Answer("Le texte ne le dit pas.", [], raw={"premise_rejected": True}), "premise_rejected"))
    assert not all(c.passed for c in evaluate_checks(premise, Answer("Parce que.", [passage()]), "answered"))


# Runner -------------------------------------------------------------------

class FakeAssistant:
    model = "fake-model"

    def __init__(self, passages):
        self.index = SimpleNamespace(passages=passages)
        self.calls = []

    def answer(self, question, **kwargs):
        self.calls.append(question)
        return Answer(text="Gregor est voyageur de commerce. [m_ch01_001]", citations=[self.index.passages[0]])

    @staticmethod
    def _answer_language(question):
        return LiteraryAssistant._answer_language(question)


def write_scenarios(path, *items):
    path.write_text(json.dumps({"scenarios": [s.to_dict() for s in items]}), encoding="utf-8")
    return path


def test_runner_uses_the_chat_workflow_and_records_runs(tmp_path):
    assistant = FakeAssistant([passage(), passage("m_ch03_001", 3, "the end")])
    path = write_scenarios(tmp_path / "s.json",
                           scenario(id="fact", gold=[{"work_id": "meta", "contains": "travelling salesman"}]),
                           scenario(id="spoil", category="spoiler", expect="abstain", max_chapter=1,
                                    question="Que devient Gregor à la fin du livre ?"))
    runner = EvaluationRunner(lambda: assistant, path, tmp_path / "runs", tmp_path)
    run = runner.run_all()
    rows = {row["scenario_id"]: row for row in run["results"]}
    assert rows["fact"]["passed"] and rows["fact"]["outcome"] == "answered"
    # The spoiler guard answers before any retrieval or model call.
    assert rows["spoil"]["passed"] and rows["spoil"]["outcome"] == "spoiler_guarded"
    assert assistant.calls == ["Quel métier exerce Gregor ?"]
    assert run["summary"]["passed"] == 2 and runner.runs()[0]["run_id"] == run["run_id"]

    single = runner.run_scenario(runner.scenario("fact"))
    runner.record_single(single)
    assert runner.latest_results()["fact"]["ran_at"] == single["ran_at"]


def test_runner_reports_engine_errors_as_failed_scenarios(tmp_path):
    class Broken(FakeAssistant):
        def answer(self, question, **kwargs):
            raise RuntimeError("Ollama indisponible")
    path = write_scenarios(tmp_path / "s.json", scenario())
    runner = EvaluationRunner(lambda: Broken([passage()]), path, tmp_path / "runs", tmp_path)
    row = runner.run_scenario(runner.scenario("s1"))
    assert row["passed"] is False and "Ollama" in row["error"]


# Engine improvements measured by the scenarios -----------------------------

def test_spoiler_guard_refuses_ending_questions_only_before_the_last_chapter():
    assistant = FakeAssistant([passage(), passage("m_ch03_001", 3, "the end")])
    workflow = NarrativeWorkflow(assistant)
    base = {"work_ids": ["meta"], "top_k": 4, "history": [], "mode": "Ask"}
    refused = workflow.invoke({**base, "question": "Que devient Gregor à la fin ?", "max_chapter": 1})
    assert refused["status"] == "refused" and "chapitre 1" in refused["answer"].text and not refused["answer"].citations
    assert workflow.invoke({**base, "question": "Que se passe-t-il à la fin du chapitre 1 ?", "max_chapter": 1})["status"] == "answered"
    assert workflow.invoke({**base, "question": "Que devient Gregor à la fin ?", "max_chapter": 3})["status"] == "answered"
    assert workflow.invoke({**base, "question": "Que devient Gregor à la fin ?", "max_chapter": None})["status"] == "answered"


def test_false_premise_correction_keeps_only_quote_verified_citations():
    allowed = {"m_ch01_001": passage()}
    answer = LiteraryAssistant._premise_correction(
        {"citation_ids": ["m_ch01_001", "m_ch01_009"], "evidence_quotes": {"m_ch01_001": "Samsa was a travelling salesman"},
         "_language": "French"},
        "Le texte ne dit pas cela : Gregor est voyageur de commerce [m_ch01_001] [m_ch01_009].", allowed)
    assert answer.raw["premise_rejected"] is True
    assert [p.chunk_id for p in answer.citations] == ["m_ch01_001"]
    assert "m_ch01_009" not in answer.text and "[m_ch01_001]" in answer.text


def test_premise_answers_pass_the_workflow_evidence_check_without_citations():
    answer = Answer("Le texte ne le dit pas.", [], raw={"premise_rejected": True})
    assert NarrativeWorkflow._check_evidence({"answer": answer, "max_chapter": None})["status"] == "answered"


class Analyzer:
    def __init__(self, entities=()):
        self.entities = list(entities)

    def analyze(self, question, history=None):
        return QueryAnalysis(rewritten_query=question, entities=self.entities)


class RecordingLexical:
    def __init__(self, results):
        self.results = results
        self.queries = []
        self.index = SimpleNamespace(works={"meta": SimpleNamespace(title="Meta", language="en")},
                                     passages=list(results))

    def retrieve(self, query, work_ids=None, top_k=6, max_chapter=None):
        self.queries.append(query)
        return list(self.results)


def test_query_expansion_adds_an_english_lexical_probe_and_hypothetical_passage():
    class Expander:
        def expand(self, question, language):
            assert language == "en"
            return Expansion(keywords="street Charlottenstrasse lived", passage="He lived in Charlottenstrasse.")

    class Semantic:
        def retrieve_many(self, queries, work_ids=None, top_k=6, max_chapter=None, documents=()):
            assert documents == ["He lived in Charlottenstrasse."]
            return [[passage("a")], [passage("b")]]

    lexical = RecordingLexical([passage("c", text="he lived in Charlottenstrasse street")])
    result = HybridRetriever(lexical, Analyzer(), semantic=Semantic(), expander=Expander()).retrieve(
        "Dans quelle rue habite la famille ?", work_ids=["meta"], top_k=3)
    assert lexical.queries == ["Dans quelle rue habite la famille ?", "street Charlottenstrasse lived"]
    assert {p.chunk_id for p in result.passages} == {"a", "b", "c"}


def test_bare_initial_entity_does_not_count_as_an_absent_character():
    lexical = RecordingLexical([passage(text="The guards Franz and Willem arrest K. one morning.")])
    retriever = HybridRetriever(lexical, Analyzer(entities=["K"]))
    assert retriever._has_sufficient_evidence("guards arrest", lexical.results, [], entities=["K"], work_ids=["meta"])


def test_llm_reranker_moves_named_excerpts_first_and_ignores_bad_labels():
    class Provider:
        def chat(self, **kwargs):
            assert "P3" in kwargs["messages"][1]["content"]
            return {"message": {"content": json.dumps({"relevant": ["P3", "P9", "x", "P3"]})}}
    pool = [passage(f"p{i}") for i in range(1, 4)]
    ranked = LLMReranker("m", Provider()).rerank("question", pool, top_k=2)
    assert [p.chunk_id for p in ranked] == ["p3", "p1"]


# API ----------------------------------------------------------------------

def test_evaluation_routes_list_and_run_scenarios(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    sys.path.insert(0, str(ROOT))
    from app import api as api_module

    assistant = FakeAssistant([passage()])
    assistant.index.works = {"meta": SimpleNamespace(title="Meta")}
    path = write_scenarios(tmp_path / "s.json", scenario(gold=[{"work_id": "meta", "contains": "travelling salesman"}]))
    monkeypatch.setattr(api_module, "get_assistant", lambda: assistant)
    monkeypatch.setattr(api_module, "evaluation", EvaluationRunner(lambda: assistant, path, tmp_path / "runs", tmp_path))
    client = TestClient(api_module.api)

    listing = client.get("/api/evaluation/scenarios").json()
    assert listing["scenarios"][0]["available"] is True and listing["scenarios"][0]["last_result"] is None
    run = client.post("/api/evaluation/scenarios/s1/run").json()
    assert run["passed"] is True and run["checks"]
    assert client.get("/api/evaluation/scenarios").json()["scenarios"][0]["last_result"]["passed"] is True
    assert client.post("/api/evaluation/scenarios/unknown/run").status_code == 404
    assert client.get("/api/evaluation/images/../../secret.png").status_code == 404


def test_rejected_rewording_falls_back_to_verified_quotes_not_raw_passages():
    assistant = LiteraryAssistant.__new__(LiteraryAssistant)
    source = passage("t_ch06_015", 6, "\"We're going now to see Dr. Huld, the lawyer,\" he said.", work_id="trial")
    answer = assistant._quoted_answer("Who is Dr. Huld?", {
        "citation_ids": ["t_ch06_015"], "evidence_quotes": {"t_ch06_015": "We’re going now to see Dr. Huld, the lawyer"}}, [source])
    assert answer.raw["quoted"] is True and [p.chunk_id for p in answer.citations] == ["t_ch06_015"]
    assert "> « We’re going now to see Dr. Huld, the lawyer » [t_ch06_015]" in answer.text
    assert classify(answer) == "quoted"
    s = scenario(question="Who is Dr. Huld?", gold=[{"work_id": "trial", "contains": "Dr. Huld, the lawyer"}],
                 must_include_any=["lawyer"], work_ids=["trial"])
    assert all(c.passed for c in evaluate_checks(s, answer, "quoted"))


def test_a_wrong_second_quote_drops_only_that_citation():
    assistant = LiteraryAssistant.__new__(LiteraryAssistant)
    first, second = passage("a", text="Grete played the violin beautifully."), passage("b", text="The gentlemen left the flat.")
    payload = {"answer": "Grete joue du violon [a] [b].", "citation_ids": ["a", "b"], "_context_ids": ["a", "b"],
               "evidence_quotes": {"a": "Grete played the violin", "b": "she cleaned the room"}}
    assistant._drop_unquoted_citations(payload, [first, second])
    assert payload["citation_ids"] == ["a"] and "[b]" not in payload["answer"] and payload["_dropped_citations"] == ["b"]
    payload["evidence_quotes"] = {"a": "invented words"}
    with pytest.raises(ValueError, match="own exact"):
        assistant._drop_unquoted_citations(payload, [first, second])


def test_translated_character_names_pass_through_the_edition_language_rewrite():
    lexical = RecordingLexical([passage(text="Alyosha was the third son of Fyodor Pavlovitch Karamazov.")])
    retriever = HybridRetriever(lexical, Analyzer(entities=["Aliocha", "Fiodor Pavlovitch"]))
    rewrite = ("Alyosha Fyodor Pavlovitch son", lexical.results)
    assert retriever._has_sufficient_evidence("Qui est Aliocha ?", [], [], entities=["Aliocha"], work_ids=["meta"], expanded=rewrite)
    absent = ("Raskolnikov murder motive", [passage(text="Gregor woke up as a vermin.")])
    assert not retriever._has_sufficient_evidence("Who is Raskolnikov?", [], [], entities=["Raskolnikov"], work_ids=["meta"], expanded=absent)


def test_name_spellings_are_matched_to_the_book_vocabulary():
    from rag.local_index import LocalLiteraryIndex
    index = LocalLiteraryIndex.__new__(LocalLiteraryIndex)
    index.works = {"meta": SimpleNamespace(title="Meta", language="en")}
    index.passages = [passage(text="Alyosha was the third son of Fyodor Pavlovitch.")]
    index._postings = {"alyosha": [0], "fyodor": [0], "third": [0], "son": [0]}
    lexical = SimpleNamespace(index=index, retrieve=lambda *a, **k: [])
    retriever = HybridRetriever(lexical, Analyzer())
    assert retriever._respell_names(["Aliocha", "Fiodor Pavlovitch"], ["meta"]) == {"aliocha": "alyosha", "fiodor": "fyodor"}
    assert retriever._respell_names(["Raskolnikov"], ["meta"]) == {}
