"""Fast tests: no model calls, no Blender. Run with `uv run pytest`."""

import json
from pathlib import Path

import pytest

from dancebuddy import agent, llm
from dancebuddy.baseline import pose_copy
from dancebuddy.checks import metrics, run_checks
from dancebuddy.perception import build_dance_score, load_keypoints
from dancebuddy.primitives import compile_plan
from dancebuddy.synth import make

ROOT = Path(__file__).resolve().parent.parent
CARD = {"name": "capsule", "root_bone": "root",
        "bones": [{"name": n} for n in ("root", "lower", "upper", "top")],
        "limits": {"lean": 35, "twist": 45, "sway": 35, "lift": 0.35, "squash": 0.35}}
PLAN = json.loads((ROOT / "examples" / "handwritten_plan_synth.json").read_text())


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = make(tmp_path_factory.mktemp("synth"))
    kp = load_keypoints(d / "keypoints.npz")
    return kp, build_dance_score(kp)


def test_accents_match_the_synthetic_beats(synth):
    _, score = synth
    truth = [round((0.5 + i * 60 / 110) * 30) for i in range(22)]
    hits = sum(any(abs(a - t) <= 3 for a in score["accents"]) for t in truth)
    assert hits / len(truth) >= 0.75
    assert len(score["units"]) == 6


def test_handwritten_plan_passes_checks(synth):
    _, score = synth
    res = run_checks(compile_plan(PLAN, score, CARD), score, CARD)
    assert res["passed"], res["checks"]


def test_pose_copy_loses_arm_sections(synth):
    kp, score = synth
    m = metrics(pose_copy(kp, CARD), score, CARD)
    assert m["idle_units"] >= 2 and m["accent_recall"] < 0.5


def test_limit_violation_is_reported(synth):
    _, score = synth
    bad = json.loads(json.dumps(PLAN))
    bad["units"][0]["primitives"][0]["amplitude"] = 80
    res = run_checks(compile_plan(bad, score, CARD), score, CARD)
    assert not res["checks"]["limits"]["passed"]


def test_agent_revises_after_failed_check(synth, tmp_path, monkeypatch):
    _, score = synth
    bad = json.loads(json.dumps(PLAN))
    bad["units"][0]["primitives"][0]["amplitude"] = 80
    script = iter([
        {"text": "", "tool_calls": [{"id": "a", "name": "get_dance_score", "arguments": {}}]},
        {"text": "", "tool_calls": [{"id": "b", "name": "submit_plan", "arguments": bad}]},
        {"text": "", "tool_calls": [{"id": "c", "name": "submit_plan", "arguments": PLAN}]},
        {"text": "done", "tool_calls": []},
    ])
    monkeypatch.setattr(llm, "call", lambda *a, **k: next(script))
    summary = agent.run_agent(score, CARD, tmp_path)
    assert summary["attempts"] == 2 and summary["passed"] and summary["best_attempt"] == 2


def test_cut_out_finds_the_character_on_a_plain_background(tmp_path):
    from dancebuddy.puppet import cut_out
    info = cut_out(ROOT / "examples" / "sample_character.png", tmp_path / "char")
    filled = sum(map(sum, info["grid"])) / (info["rows"] * info["cols"])
    assert 0.5 < filled < 0.95 and info["height_px"] > info["width_px"]
