"""Orchestrator tests. Stub mode tests run with NO API key.
Production (OpenAI) tests are skipped unless OPENAI_API_KEY is set AND RUN_OPENAI_TESTS=1.
"""
import json
import os
from pathlib import Path

import pytest

from app.config import get_settings
from app.orchestrator.openai_client import orchestrate
from app.schemas import Manifest

FEW_SHOT = Path(__file__).parent / "fixtures" / "orchestrator" / "few_shot"
STUBS = Path(__file__).parent / "fixtures" / "orchestrator" / "stub_responses"


@pytest.mark.parametrize("name", ["anatomy", "horology", "botany"])
def test_few_shot_validates(name: str) -> None:
    payload = json.loads((FEW_SHOT / f"{name}.json").read_text())
    Manifest.model_validate(payload)


@pytest.mark.parametrize("name", ["anatomy", "horology", "botany"])
def test_stub_response_validates(name: str) -> None:
    payload = json.loads((STUBS / f"{name}.json").read_text())
    Manifest.model_validate(payload)


def test_stub_orchestrate_anatomy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORCHESTRATOR_MODE", "stub")
    get_settings.cache_clear()
    m = orchestrate("An anatomy exhibit", exhibit_id="TEST_001")
    assert m.exhibit_id == "TEST_001"
    assert m.user_prompt == "An anatomy exhibit"
    ids = [o.id.lower() for o in m.objects]
    assert any("heart" in i or "brain" in i or "skeleton" in i for i in ids)


def test_stub_orchestrate_horology(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORCHESTRATOR_MODE", "stub")
    get_settings.cache_clear()
    m = orchestrate("A horology exhibit with three pocket watches", exhibit_id="TEST_002")
    assert m.exhibit_id == "TEST_002"
    ids = [o.id.lower() for o in m.objects]
    assert any("watch" in i or "clock" in i for i in ids)


def test_stub_orchestrate_botany(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORCHESTRATOR_MODE", "stub")
    get_settings.cache_clear()
    m = orchestrate("A botany exhibit with fruits and seeds", exhibit_id="TEST_003")
    assert m.exhibit_id == "TEST_003"
    ids = [o.id.lower() for o in m.objects]
    assert any("pomegranate" in i or "apple" in i or "avocado" in i for i in ids)


def test_stub_orchestrate_scene_url_rewritten(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORCHESTRATOR_MODE", "stub")
    get_settings.cache_clear()
    m = orchestrate("An anatomy exhibit", exhibit_id="EX42")
    assert "EX42" in m.scene.scene_url


def test_stub_orchestrate_default_falls_back_to_anatomy(monkeypatch: pytest.MonkeyPatch) -> None:
    """When prompt doesn't match any topic, default to anatomy."""
    monkeypatch.setenv("ORCHESTRATOR_MODE", "stub")
    get_settings.cache_clear()
    m = orchestrate("a totally unrelated thing", exhibit_id="TEST_004")
    assert m.exhibit_id == "TEST_004"


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="needs OpenAI key for live test",
)
@pytest.mark.skipif(
    os.environ.get("RUN_OPENAI_TESTS") != "1",
    reason="set RUN_OPENAI_TESTS=1 to enable",
)
def test_orchestrate_openai_real_call() -> None:
    """Live OpenAI call. Only runs when OPENAI_API_KEY is set AND RUN_OPENAI_TESTS=1 (opt-in)."""
    os.environ["ORCHESTRATOR_MODE"] = "openai"
    get_settings.cache_clear()
    m = orchestrate("A Victorian study with a pocket watch", exhibit_id="LIVE_001")
    assert 3 <= len(m.objects) <= 5
