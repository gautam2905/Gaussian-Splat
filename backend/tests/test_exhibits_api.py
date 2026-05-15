"""Exhibits API tests after Phase 2C routing wiring.

Pipeline runs in stub mode by default — no OpenAI, no Modal.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import get_settings

client = TestClient(app)


def test_create_returns_202_with_events_url() -> None:
    r = client.post("/api/exhibits", json={"prompt": "an anatomy exhibit"})
    assert r.status_code == 202
    body = r.json()
    assert "events_url" in body
    assert body["events_url"].startswith("/api/exhibits/events?prompt=")


def test_rejects_empty_prompt() -> None:
    r = client.post("/api/exhibits", json={"prompt": ""})
    assert r.status_code == 422


def test_rejects_missing_prompt() -> None:
    r = client.post("/api/exhibits", json={})
    assert r.status_code == 422


def test_unknown_exhibit_404() -> None:
    r = client.get("/api/exhibits/01HZ8X9A2B0000000000000001")
    assert r.status_code == 404
