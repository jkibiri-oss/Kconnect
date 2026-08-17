from types import SimpleNamespace

from app.schemas.knowledge import KnowledgeSearchResult
from app.schemas.response import Intent
from app.services.gemini_rag import (
    GeneratedSuggestion,
    GeminiRagService,
)
from app.services.rag_orchestration import orchestrate_rag


class FakeGeminiService:
    def __init__(self, intent, suggestion="Use the passenger helmet."):
        self.intent = intent
        self.suggestion = suggestion
        self.generation_records = None

    def detect_intent(self, _transcript):
        return self.intent

    def generate_suggestion(self, transcript, intent, records):
        self.generation_records = records
        return self.suggestion


class FakeGeminiModels:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def generate_content(self, **request):
        self.requests.append(request)
        return self.responses.pop(0)


def helmet_intent():
    return Intent(
        category="transport",
        sub_category="moto",
        situation="helmet_use",
        search_query="moto passenger helmet requirement in Rwanda",
    )


def helmet_record(score=0.92):
    return KnowledgeSearchResult(
        id="transport_002",
        category="transport",
        sub_category="moto",
        situation="helmet_use",
        rwanda_context=(
            "In Rwanda, moto riders and passengers wear helmets."
        ),
        suggested_tip="Fasten the helmet before the moto moves.",
        source="Rwanda National Police",
        score=score,
    )


def test_orchestration_retrieves_and_generates_grounded_tip(monkeypatch):
    service = FakeGeminiService(helmet_intent())
    captured = {}

    def fake_search(**kwargs):
        captured.update(kwargs)
        return [helmet_record()]

    monkeypatch.setattr(
        "app.services.rag_orchestration.search_knowledge",
        fake_search,
    )

    result = orchestrate_rag(
        database=object(),
        transcript="Do I really need to wear this helmet?",
        gemini_service=service,
    )

    assert captured["category"] == "transport"
    assert captured["sub_category"] == "moto"
    assert captured["situation"] == "helmet_use"
    assert captured["query"] == (
        "moto passenger helmet requirement in Rwanda"
    )
    assert result.cultural_tip == "Use the passenger helmet."
    assert result.source == "Rwanda National Police"
    assert service.generation_records == [helmet_record()]


def test_orchestration_prints_each_successful_trace_stage(
    monkeypatch,
    capsys,
):
    service = FakeGeminiService(helmet_intent())
    monkeypatch.setattr(
        "app.services.rag_orchestration.search_knowledge",
        lambda **_kwargs: [helmet_record()],
    )

    orchestrate_rag(
        database=object(),
        transcript="Do I really need to wear this helmet?",
        gemini_service=service,
    )

    output = capsys.readouterr().out

    for stage in (
        "pipeline.started",
        "intent.started",
        "intent.detected",
        "retrieval.started",
        "retrieval.completed",
        "grounding.completed",
        "suggestion.started",
        "suggestion.generated",
        "pipeline.completed",
    ):
        assert f"[rag trace] {stage} |" in output

    assert '"query": "moto passenger helmet requirement in Rwanda"' in output
    assert '"suggestion": "Use the passenger helmet."' in output


def test_orchestration_does_not_generate_without_useful_result(monkeypatch):
    service = FakeGeminiService(helmet_intent())
    monkeypatch.setattr(
        "app.services.rag_orchestration.search_knowledge",
        lambda **_kwargs: [helmet_record(score=0.4)],
    )

    result = orchestrate_rag(
        database=object(),
        transcript="Do I need this helmet?",
        gemini_service=service,
        minimum_score=0.75,
    )

    assert result.intent == helmet_intent()
    assert result.cultural_tip is None
    assert result.source is None
    assert service.generation_records is None


def test_orchestration_stops_when_intent_is_not_relevant(monkeypatch):
    service = FakeGeminiService(None)

    def unexpected_search(**_kwargs):
        raise AssertionError("Retrieval should not run")

    monkeypatch.setattr(
        "app.services.rag_orchestration.search_knowledge",
        unexpected_search,
    )

    result = orchestrate_rag(
        database=object(),
        transcript="Hello there",
        gemini_service=service,
    )

    assert result.intent is None
    assert result.cultural_tip is None


def test_gemini_service_validates_intent_and_generates_tip():
    models = FakeGeminiModels(
        [
            SimpleNamespace(
                parsed=Intent(
                    category="Transport",
                    sub_category="Moto",
                    situation="helmet_use",
                    search_query="moto passenger helmet Rwanda",
                ),
                text=None,
            ),
            SimpleNamespace(
                parsed=GeneratedSuggestion(
                    suggestion=" Fasten the passenger helmet. "
                ),
                text=None,
            ),
        ]
    )
    service = GeminiRagService(
        client=SimpleNamespace(models=models),
        model="test-generation-model",
    )

    intent = service.detect_intent("Do I need this helmet?")
    suggestion = service.generate_suggestion(
        transcript="Do I need this helmet?",
        intent=intent,
        records=[helmet_record()],
    )

    assert intent == helmet_intent().model_copy(
        update={"search_query": "moto passenger helmet Rwanda"}
    )
    assert suggestion == "Fasten the passenger helmet."
    assert len(models.requests) == 2
    suggestion_instruction = models.requests[1]["config"].system_instruction
    assert "Write the suggestion in English only" in suggestion_instruction


def test_gemini_service_requests_english_tip_for_kinyarwanda_transcript():
    models = FakeGeminiModels(
        [
            SimpleNamespace(
                parsed=GeneratedSuggestion(
                    suggestion="Fasten the passenger helmet."
                ),
                text=None,
            )
        ]
    )
    service = GeminiRagService(
        client=SimpleNamespace(models=models),
        model="test-generation-model",
    )

    suggestion = service.generate_suggestion(
        transcript="Ese nkeneye kwambara ingofero?",
        intent=helmet_intent(),
        records=[helmet_record()],
    )

    assert suggestion == "Fasten the passenger helmet."
    instruction = models.requests[0]["config"].system_instruction
    assert "Write the suggestion in English only" in instruction
    assert "same language as the original transcript" not in instruction


def test_gemini_service_rejects_unknown_taxonomy_labels():
    models = FakeGeminiModels(
        [
            SimpleNamespace(
                parsed=Intent(
                    category="unknown",
                    sub_category="unknown",
                    situation="unknown",
                    search_query="unknown",
                ),
                text=None,
            )
        ]
    )
    service = GeminiRagService(
        client=SimpleNamespace(models=models),
        model="test-generation-model",
    )

    assert service.detect_intent("Something unrelated") is None


def test_gemini_service_retries_empty_intent_response():
    expected_intent = helmet_intent()
    models = FakeGeminiModels(
        [
            SimpleNamespace(parsed=None, text=None),
            SimpleNamespace(parsed=expected_intent, text=None),
        ]
    )
    service = GeminiRagService(
        client=SimpleNamespace(models=models),
        model="test-generation-model",
    )

    assert service.detect_intent("Do I need this helmet?") == expected_intent
    assert len(models.requests) == 2


def test_gemini_service_retries_truncated_suggestion_response():
    models = FakeGeminiModels(
        [
            SimpleNamespace(parsed=None, text="Here is"),
            SimpleNamespace(
                parsed=GeneratedSuggestion(
                    suggestion="Fasten the passenger helmet."
                ),
                text=None,
            ),
        ]
    )
    service = GeminiRagService(
        client=SimpleNamespace(models=models),
        model="test-generation-model",
    )

    suggestion = service.generate_suggestion(
        transcript="Do I need this helmet?",
        intent=helmet_intent(),
        records=[helmet_record()],
    )

    assert suggestion == "Fasten the passenger helmet."
    assert len(models.requests) == 2
