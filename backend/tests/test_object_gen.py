"""Object generation tests. Stub mode runs with NO API key, no Modal."""
import json
import shutil
from pathlib import Path

import pytest

from app.object_gen import generate, hints_for_object
from app.object_gen.stub import _tokenize, best_match


def test_library_manifest_exists() -> None:
    p = Path(__file__).resolve().parent.parent / "stub_objects" / "library.json"
    assert p.exists(), "stub library manifest missing"
    data = json.loads(p.read_text())
    assert data["schema_version"] == "v1"
    assert len(data["objects"]) >= 8, f"need >= 8 stub objects; got {len(data['objects'])}"


def test_library_glbs_exist_on_disk() -> None:
    root = Path(__file__).resolve().parent.parent / "stub_objects"
    data = json.loads((root / "library.json").read_text())
    for entry in data["objects"]:
        p = root / entry["glb_path"]
        assert p.exists(), f"stub GLB missing: {p}"
        assert p.stat().st_size > 100, f"stub GLB suspiciously small: {p}"


def test_tokenize_basic() -> None:
    assert _tokenize("A pocket watch, c. 1880") == {"a", "pocket", "watch", "c", "1880"}


def test_best_match_avocado() -> None:
    entry = best_match("an avocado halved with pit visible")
    assert entry["id"] == "avocado"


def test_best_match_helmet() -> None:
    entry = best_match("an armored knight's helmet")
    assert entry["id"] == "helmet"


def test_best_match_unknown_falls_back() -> None:
    """No keywords match → return some library entry (don't crash)."""
    entry = best_match("xyzzy-foo-blah-quux-nothing-matches")
    assert "id" in entry  # successfully returned something


def test_stub_object_gen_copies_glb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """stub_object_gen copies a GLB into the exhibit objects dir."""
    monkeypatch.setenv("OBJECT_GEN_MODE", "stub")
    from app.config import get_settings
    get_settings.cache_clear()
    result = generate(
        image_prompt="avocado halved, pit visible",
        exhibit_id="TEST_EX",
        object_id="obj_1",
        data_dir=tmp_path,
    )
    assert "glb_url" in result
    assert result["glb_url"].endswith("/objects/obj_1.glb")
    expected_path = tmp_path / "exhibits" / "TEST_EX" / "objects" / "obj_1.glb"
    assert expected_path.exists()
    assert expected_path.stat().st_size > 100


def test_stub_object_gen_response_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OBJECT_GEN_MODE", "stub")
    from app.config import get_settings
    get_settings.cache_clear()
    result = generate(
        image_prompt="brass pocket watch back open",
        exhibit_id="TEST_EX2",
        object_id="watch_1",
        data_dir=tmp_path,
    )
    assert set(result.keys()) >= {"image_url", "glb_url", "elapsed_ms", "stub_id"}
    assert result["image_url"] is None  # stub doesn't generate hero images
    assert isinstance(result["elapsed_ms"], int)
    assert result["elapsed_ms"] >= 0


def test_hints_for_dissectable_object() -> None:
    h = hints_for_object("avocado halved")
    assert h["dissectable"] is True
    assert h["preferred_clip_axis"] in ("x", "y", "z")
    assert 0.0 <= h["preferred_clip_t"] <= 1.0


def test_modal_mode_raises_not_implemented(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Production Modal path is intentionally NotImplemented in v1 (budget pivot)."""
    monkeypatch.setenv("OBJECT_GEN_MODE", "modal")
    from app.config import get_settings
    get_settings.cache_clear()
    with pytest.raises(NotImplementedError):
        generate(
            image_prompt="anything",
            exhibit_id="X",
            object_id="o",
            data_dir=tmp_path,
        )
