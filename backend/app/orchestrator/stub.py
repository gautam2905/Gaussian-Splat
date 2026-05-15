"""Stub orchestrator — returns canned manifests by prompt-substring match.

Used when settings.orchestrator_mode == 'stub' so the demo runs for $0.
"""
import json
from pathlib import Path

from ..schemas import Manifest

_FIXTURES_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "tests"
    / "fixtures"
    / "orchestrator"
    / "stub_responses"
)


def _lazy_load_fixtures() -> dict[str, dict]:
    """Map of {topic: manifest_dict} keyed by primary topic word."""
    out: dict[str, dict] = {}
    if not _FIXTURES_DIR.exists():
        return out
    for path in _FIXTURES_DIR.glob("*.json"):
        topic = path.stem.lower()  # e.g. "anatomy"
        out[topic] = json.loads(path.read_text())
    return out


_FIXTURES: dict[str, dict] | None = None


def stub_orchestrate(user_prompt: str, exhibit_id: str) -> Manifest:
    """Pick the best-matching canned manifest by keyword.

    Match logic: lowercase + substring. If "anatom" / "heart" / "brain" / "skeleton"
    in prompt, return anatomy. If "watch" / "clock" / "horology" / "mechan" in
    prompt, return horology. If "botan" / "plant" / "fruit" / "seed" / "garden" in
    prompt, return botany. Otherwise default to anatomy.
    """
    global _FIXTURES
    if _FIXTURES is None:
        _FIXTURES = _lazy_load_fixtures()
    if not _FIXTURES:
        raise RuntimeError(
            f"No stub fixtures found at {_FIXTURES_DIR}. "
            "Run the test suite or check the fixtures directory."
        )

    p = user_prompt.lower()

    def matches(topic_keywords: list[str]) -> bool:
        return any(k in p for k in topic_keywords)

    if matches(["anatom", "heart", "brain", "skeleton", "body", "human"]):
        key = "anatomy"
    elif matches(["watch", "clock", "horology", "mechan", "gear", "time"]):
        key = "horology"
    elif matches(["botan", "plant", "fruit", "seed", "garden", "pomegranate", "leaf"]):
        key = "botany"
    else:
        key = "anatomy"  # default

    raw = _FIXTURES.get(key)
    if raw is None:
        # Fallback to first available fixture
        raw = next(iter(_FIXTURES.values()))

    # Inject the real exhibit_id and prompt
    data = json.loads(json.dumps(raw))  # deep copy
    data["exhibit_id"] = exhibit_id
    data["user_prompt"] = user_prompt
    # Rewrite URLs to use the real exhibit_id
    data["scene"]["scene_url"] = f"/exhibits/{exhibit_id}/scene/scene.glb"

    return Manifest.model_validate(data)
