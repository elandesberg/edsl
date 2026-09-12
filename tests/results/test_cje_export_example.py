"""The export contract runs without CJE; inference checks run when installed."""

import importlib.util
import json
from pathlib import Path

import pytest

from edsl import Agent, Model, Results, ScenarioList

_path = Path(__file__).resolve().parents[2] / "docs/examples/cje_export.py"
_spec = importlib.util.spec_from_file_location("cje_export_example", _path)
example = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(example)


@pytest.fixture
def saved():
    return example.synthetic_example()


def export(results, labels=None):
    return example.to_cje_draws(
        results, judge_question="judge_score", oracle_labels=labels
    )


def test_native_json_round_trip_and_reordering_preserve_sparse_labels(saved):
    results, labels = saved
    before = export(results, labels)
    restored = Results.from_dict(json.loads(json.dumps(results.to_dict())))
    restored.data.reverse()
    after = export(restored, labels)
    keyed = lambda rows: {
        row["observation_id"]: row for policy in rows.values() for row in policy
    }
    assert keyed(before) == keyed(after)
    assert set(before) == {"baseline", "candidate"}  # Both use the same test judge.
    assert sum("oracle_label" in r for r in keyed(before).values()) == 48
    assert all(r["draw_idx"] == 0 for r in keyed(before).values())


def test_native_survey_judging_lifecycle_is_offline(saved, monkeypatch, tmp_path):
    import httpx
    import platformdirs
    import requests

    def unexpected_network(*args, **kwargs):
        raise AssertionError("The synthetic judge must not make HTTP requests")

    monkeypatch.setattr(requests.sessions.Session, "request", unexpected_network)
    monkeypatch.setattr(httpx.Client, "send", unexpected_network)
    monkeypatch.setattr(httpx.AsyncClient, "send", unexpected_network)
    monkeypatch.setattr(platformdirs, "user_data_dir", lambda *a, **kw: str(tmp_path))
    seed, _ = saved
    native = (
        seed.survey.by(ScenarioList([r["scenario"] for r in seed[:2]]))
        .by(Model("test", canned_response="50"))
        .run(
            progress_bar=False,
            cache=False,
            disable_remote_cache=True,
            disable_remote_inference=True,
        )
    )
    restored = Results.from_dict(json.loads(json.dumps(native.to_dict())))
    draws = export(restored)
    assert set(draws) == {"baseline", "candidate"}
    assert [row["judge_score"] for rows in draws.values() for row in rows] == [50, 50]


def test_identical_response_text_on_distinct_draws_and_policies_is_distinct(saved):
    results, _ = saved
    first, other_policy, next_draw = results[:3]
    next_draw["scenario"].update(first["scenario"])
    next_draw["scenario"]["draw_idx"] = 7
    other_policy["scenario"]["response"] = first["scenario"]["response"]
    rows = [
        r for group in export([first, other_policy, next_draw]).values() for r in group
    ]
    assert len({r["observation_id"] for r in rows}) == 3
    assert {r["draw_idx"] for r in rows} == {0, 7}


def test_zero_scores_and_labels_are_preserved_without_scale_guessing(saved):
    results, _ = saved
    results[0]["answer"]["judge_score"] = 0
    obs = export(results)["baseline"][0]["observation_id"]
    row = export(results, {obs: 0})["baseline"][0]
    assert row["judge_score"] == row["oracle_label"] == 0
    results[0]["answer"]["judge_score"] = 75
    assert export(results)["baseline"][0]["judge_score"] == 75


@pytest.mark.parametrize("field", ["response", "rubric_id", "prompt", "draw_idx"])
def test_changed_response_or_rubric_cannot_reuse_labels(saved, field):
    results, _ = saved
    one = [results[0]]
    obs = export(one)["baseline"][0]["observation_id"]
    one[0]["scenario"][field] = 1 if field == "draw_idx" else "changed"
    with pytest.raises(ValueError, match="unknown or stale"):
        export(one, {obs: 3})


def test_changed_scenario_context_cannot_reuse_labels(saved):
    results, _ = saved
    one = [results[0]]
    one[0]["scenario"]["context"] = "Original reference passage"
    obs = export(one)["baseline"][0]["observation_id"]
    one[0]["scenario"]["context"] = "A different reference passage"
    with pytest.raises(ValueError, match="unknown or stale"):
        export(one, {obs: 3})


@pytest.mark.parametrize("bad", [None, True, "75", float("nan"), float("inf")])
def test_invalid_judge_scores_fail_with_row_and_question_context(saved, bad):
    results, _ = saved
    results[0]["answer"]["judge_score"] = bad
    with pytest.raises(ValueError, match=r"row 0: answer.judge_score"):
        export(results)


@pytest.mark.parametrize("bad", [None, True, "3", float("nan"), float("inf")])
def test_invalid_oracle_labels_fail(saved, bad):
    results, _ = saved
    obs = export(results)["baseline"][0]["observation_id"]
    with pytest.raises(ValueError, match="oracle_label"):
        export(results, {obs: bad})


@pytest.mark.parametrize("bad", [None, True, -1, 0.5, "0"])
def test_draw_indices_are_explicit_generation_identifiers(saved, bad):
    results, _ = saved
    results[0]["scenario"]["draw_idx"] = bad
    with pytest.raises(ValueError, match="scenario.draw_idx"):
        export(results)


def test_ambiguous_identity_and_missing_answers_are_rejected(saved):
    results, _ = saved
    with pytest.raises(ValueError, match="duplicate response draw"):
        export([results[0], results[0]])
    results[1]["scenario"]["prompt"] = "Different task with reused prompt ID"
    with pytest.raises(ValueError, match="conflicting text"):
        export(results)
    with pytest.raises(ValueError, match="missing definition"):
        example.to_cje_draws([results[0]], judge_question="other_score")
    results[0]["answer"].clear()
    with pytest.raises(ValueError, match="answer.judge_score"):
        export([results[0]])


@pytest.mark.parametrize(
    "field", ["policy_id", "prompt_id", "prompt", "response", "rubric_id"]
)
def test_required_identity_is_never_inferred(saved, field):
    results, _ = saved
    del results[0]["scenario"][field]
    with pytest.raises(ValueError, match=f"scenario.{field}"):
        export(results)


@pytest.mark.parametrize("field", ["model", "agent", "question", "rubric"])
def test_mixed_judging_configurations_are_rejected(saved, field):
    results, _ = saved
    if field == "model":
        results[1]["model"] = Model("test", temperature=0.9)
    elif field == "agent":
        results[1]["agent"] = Agent(traits={"style": "strict"})
    elif field == "question":
        results[1]["question_to_attributes"] = {
            "judge_score": {"question_text": "A different rubric"}
        }
    else:
        results[1]["scenario"]["rubric_id"] = "other-rubric"
    with pytest.raises(ValueError, match="mixed judge"):
        export(results)


def test_empty_results_and_positional_labels_are_rejected(saved):
    results, _ = saved
    with pytest.raises(ValueError, match="at least one"):
        export([])
    with pytest.raises(ValueError, match="must map"):
        export(results, [1, 2, 3])


def test_released_cje_uses_full_oracle_means_in_explicit_units(saved):
    cje = pytest.importorskip("cje")
    results, _ = saved
    draws = export(results)
    labels = {
        row["observation_id"]: 2 if policy == "baseline" else 4
        for policy, rows in draws.items()
        for row in rows
    }
    estimate = cje.analyze_dataset(
        fresh_draws_data=export(results, labels),
        fresh_judge_scale=(0, 100),
        fresh_oracle_scale=(1, 5),
    )
    assert dict(zip(estimate.target_policies, estimate.estimates)) == pytest.approx(
        {"baseline": 2, "candidate": 4}
    )
    assert set(estimate.metadata["claim_tier_by_policy"].values()) == {
        "DIRECT_ORACLE_MEAN"
    }


def test_released_cje_calibrated_and_raw_claims_remain_visible(saved, caplog):
    cje = pytest.importorskip("cje")
    results, labels = saved
    common = dict(fresh_judge_scale=(0, 100), fresh_oracle_scale=(1, 5))
    calibrated = cje.analyze_dataset(fresh_draws_data=export(results, labels), **common)
    assert set(calibrated.metadata["claim_tier_by_policy"].values()) == {
        "CALIBRATED_ORACLE_MEAN"
    }
    assert calibrated.metadata["transport_status"] == "NOT_CHECKED"
    assert "NOT_CHECKED" in calibrated.summary()
    comparison = calibrated.compare_policies(0, 1)
    assert comparison["ci_lower"] < comparison["ci_upper"]
    assert comparison["method"]
    assert comparison["gate_flagged"]
    assert "REFUSE-LEVEL" in caplog.text
    raw = cje.analyze_dataset(fresh_draws_data=export(results), **common)
    assert set(raw.metadata["claim_tier_by_policy"].values()) == {"RAW_JUDGE_MEAN"}
    assert "UNCALIBRATED raw judge-score mean" in raw.summary()
    assert "No oracle labels found" in caplog.text
