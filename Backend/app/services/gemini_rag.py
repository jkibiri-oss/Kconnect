import json
from functools import lru_cache
from typing import Any, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError

from app.core.config import GEMINI_API_KEY, GEMINI_MODEL
from app.schemas.knowledge import KnowledgeSearchResult
from app.schemas.response import Intent
from app.services.knowledge_dataset import load_knowledge_dataset
from app.services.rag_trace import trace_rag


class GeminiRagError(RuntimeError):
    pass


STRUCTURED_RESPONSE_ATTEMPTS = 2
STRUCTURED_RESPONSE_MAX_TOKENS = 1024
StructuredModel = TypeVar("StructuredModel", bound=BaseModel)


class GeneratedSuggestion(BaseModel):
    suggestion: str | None = Field(
        default=None,
        description=(
            "A short English suggestion grounded only in the retrieved records, "
            "or null when the records do not support one."
        ),
    )


@lru_cache(maxsize=1)
def knowledge_taxonomy() -> dict[str, dict[str, list[str]]]:
    taxonomy: dict[str, dict[str, set[str]]] = {}

    for item in load_knowledge_dataset():
        taxonomy.setdefault(item.category, {}).setdefault(
            item.sub_category,
            set(),
        ).add(item.situation)

    return {
        category: {
            sub_category: sorted(situations)
            for sub_category, situations in sorted(sub_categories.items())
        }
        for category, sub_categories in sorted(taxonomy.items())
    }


class GeminiRagService:
    def __init__(
        self,
        client: Any | None = None,
        model: str = GEMINI_MODEL,
    ) -> None:
        if client is None:
            if not GEMINI_API_KEY:
                raise GeminiRagError("GEMINI_API_KEY is not configured.")

            client = genai.Client(api_key=GEMINI_API_KEY)

        if not model:
            raise GeminiRagError("GEMINI_MODEL is not configured.")

        self.client = client
        self.model = model

    def detect_intent(self, transcript: str) -> Intent | None:
        taxonomy = knowledge_taxonomy()
        system_instruction = (
            "Classify the speech transcript for KConnect's Rwanda cultural "
            "knowledge retrieval. Use only exact category, sub_category, and "
            "situation labels from the supplied taxonomy. Create a concise "
            "English semantic-search query. The transcript may be English or "
            "Kinyarwanda. When it does not confidently match the taxonomy, "
            "return null for every field.\n\n"
            f"Allowed taxonomy:\n{json.dumps(taxonomy, sort_keys=True)}"
        )

        intent = self._generate_structured(
            contents=transcript,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=Intent,
                temperature=0.0,
                max_output_tokens=STRUCTURED_RESPONSE_MAX_TOKENS,
                thinking_config=types.ThinkingConfig(
                    thinking_level=types.ThinkingLevel.LOW,
                ),
            ),
            schema=Intent,
        )
        return self._validate_intent(intent, transcript, taxonomy)

    def generate_suggestion(
        self,
        transcript: str,
        intent: Intent,
        records: list[KnowledgeSearchResult],
    ) -> str | None:
        payload = {
            "original_transcript": transcript,
            "detected_intent": intent.model_dump(),
            "retrieved_records": [
                record.model_dump() for record in records
            ],
        }
        system_instruction = (
            "You generate one short cultural or situational suggestion for "
            "KConnect. Use only facts supported by the retrieved records. "
            "Do not add laws, prices, customs, or safety advice that is not "
            "present in those records. Write the suggestion in English only, "
            "regardless of the language of the original transcript. Keep the "
            "suggestion to one or two short sentences. Return null when the "
            "records do not support a useful suggestion."
        )

        generated = self._generate_structured(
            contents=json.dumps(payload, ensure_ascii=True),
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=GeneratedSuggestion,
                temperature=0.0,
                max_output_tokens=STRUCTURED_RESPONSE_MAX_TOKENS,
                thinking_config=types.ThinkingConfig(
                    thinking_level=types.ThinkingLevel.LOW,
                ),
            ),
            schema=GeneratedSuggestion,
        )

        if not generated.suggestion:
            return None

        suggestion = generated.suggestion.strip()
        return suggestion or None

    def _generate(
        self,
        contents: str,
        config: types.GenerateContentConfig,
    ) -> types.GenerateContentResponse:
        try:
            return self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
        except Exception as error:
            raise GeminiRagError("Gemini generation request failed.") from error

    def _generate_structured(
        self,
        contents: str,
        config: types.GenerateContentConfig,
        schema: type[StructuredModel],
    ) -> StructuredModel:
        last_error: GeminiRagError | None = None

        schema_name = schema.__name__

        for attempt in range(1, STRUCTURED_RESPONSE_ATTEMPTS + 1):
            trace_rag(
                "gemini.request.started",
                model=self.model,
                schema=schema_name,
                attempt=attempt,
            )
            response = self._generate(contents=contents, config=config)

            try:
                parsed = self._parse_response(response, schema)
                trace_rag(
                    "gemini.response.valid",
                    schema=schema_name,
                    attempt=attempt,
                )
                return parsed
            except GeminiRagError as error:
                last_error = error
                trace_rag(
                    "gemini.response.invalid",
                    schema=schema_name,
                    attempt=attempt,
                    reason=str(error),
                )

        raise GeminiRagError(
            "Gemini did not return a valid structured response after retrying."
        ) from last_error

    @staticmethod
    def _parse_response(
        response: types.GenerateContentResponse,
        schema: type[StructuredModel],
    ) -> StructuredModel:
        try:
            parsed = getattr(response, "parsed", None)

            if isinstance(parsed, schema):
                return parsed

            if parsed is not None:
                return schema.model_validate(parsed)

            text = getattr(response, "text", None)

            if not isinstance(text, (str, bytes, bytearray)) or not text:
                raise ValueError("Gemini response did not contain output text.")

            return schema.model_validate_json(text)
        except (ValidationError, ValueError, TypeError) as error:
            raise GeminiRagError(
                "Gemini returned an invalid structured response."
            ) from error

    @staticmethod
    def _validate_intent(
        intent: Intent,
        transcript: str,
        taxonomy: dict[str, dict[str, list[str]]],
    ) -> Intent | None:
        category = (intent.category or "").strip().lower()
        sub_category = (intent.sub_category or "").strip().lower()

        if category not in taxonomy:
            return None

        if sub_category not in taxonomy[category]:
            return None

        situation = (intent.situation or "").strip().lower() or None

        if (
            situation
            and situation not in taxonomy[category][sub_category]
        ):
            situation = None

        search_query = (intent.search_query or "").strip()

        return Intent(
            category=category,
            sub_category=sub_category,
            situation=situation,
            search_query=search_query or transcript.strip(),
        )
