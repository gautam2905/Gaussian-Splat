"""Exhibits API tests — M0 expects 501 from every ML stage."""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_create_returns_501_in_m0() -> None:
    r = client.post("/api/exhibits", json={"prompt": "a Victorian study"})
    assert r.status_code == 501
    assert r.json()["detail"]["stage"] == "orchestrator"


def test_rejects_empty_prompt() -> None:
    r = client.post("/api/exhibits", json={"prompt": ""})
    assert r.status_code == 422


def test_rejects_missing_prompt() -> None:
    r = client.post("/api/exhibits", json={})
    assert r.status_code == 422


def test_unknown_exhibit_404() -> None:
    r = client.get("/api/exhibits/01HZ8X9A2B0000000000000001")
    assert r.status_code == 404
