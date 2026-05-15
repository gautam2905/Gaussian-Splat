"""OpenAI orchestrator (production mode).

Per Amendment K: uses gpt-4o-mini by default. JSON-mode with response_format.
Retry-with-feedback on schema validation failure.

Falls through to stub.stub_orchestrate(...) when settings.orchestrator_mode == 'stub'.

Acceptable orchestrator_model values:
  - gpt-4o-mini  (default, ~$0.001/exhibit at v1 token counts)
  - gpt-4o       (higher quality, ~$0.01/exhibit)
  - claude-haiku-4-5 (future swap-back to Anthropic, requires SDK change)
"""
import json

from pydantic import ValidationError

from ..config import get_settings
from ..schemas import Manifest
from . import stub
from .prompts import SYSTEM_PROMPT_V1, build_user_message, load_few_shot

_client = None


def _client_lazy():
    global _client
    if _client is None:
        from openai import OpenAI

        settings = get_settings()
        if not settings.openai_api_key:
            raise OrchestratorFailed(
                "OPENAI_API_KEY not set; either set it in backend/.env "
                "or use ORCHESTRATOR_MODE=stub for offline dev."
            )
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def _build_messages(user_prompt: str, validation_error: str | None) -> list[dict]:
    examples = load_few_shot()
    fewshot_text = "\n\n".join(
        f"Example prompt: {ex['user_prompt']!r}\n"
        f"Example manifest:\n{json.dumps(ex, indent=2)}"
        for ex in examples
    )
    user = build_user_message(user_prompt)
    if validation_error is not None:
        user += (
            f"\n\nYour previous output failed validation:\n{validation_error}"
            "\n\nFix and emit valid JSON."
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT_V1 + "\n\n---\n\n" + fewshot_text},
        {"role": "user", "content": user},
    ]


class OrchestratorRefused(Exception):
    def __init__(self, reason: str, category: str = "other"):
        super().__init__(reason)
        self.reason = reason
        self.category = category


class OrchestratorFailed(Exception):
    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


def orchestrate(user_prompt: str, exhibit_id: str) -> Manifest:
    """Plan an exhibit. Dispatches to stub or OpenAI based on settings."""
    settings = get_settings()
    if settings.orchestrator_mode == "stub":
        return stub.stub_orchestrate(user_prompt, exhibit_id)
    return _orchestrate_openai(user_prompt, exhibit_id)


def _orchestrate_openai(user_prompt: str, exhibit_id: str) -> Manifest:
    client = _client_lazy()
    settings = get_settings()

    last_error: str | None = None
    for _ in range(2):
        messages = _build_messages(user_prompt, last_error)
        response = client.chat.completions.create(
            model=settings.orchestrator_model,
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=4096,
        )
        raw = response.choices[0].message.content
        if raw is None:
            last_error = "Empty response"
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            last_error = f"Invalid JSON: {e}"
            continue
        if obj.get("refused"):
            raise OrchestratorRefused(
                reason=obj.get("reason", "Refused without reason."),
                category=obj.get("category", "other"),
            )
        obj["exhibit_id"] = exhibit_id
        obj["user_prompt"] = user_prompt
        # Rewrite scene_url to use the real exhibit_id
        if "scene" in obj:
            obj["scene"]["scene_url"] = f"/exhibits/{exhibit_id}/scene/scene.glb"
        try:
            return Manifest.model_validate(obj)
        except ValidationError as e:
            last_error = str(e)
            continue
    raise OrchestratorFailed(last_error or "Two retries exhausted.")
