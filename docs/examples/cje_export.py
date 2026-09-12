"""Export already-judged EDSL responses to CJE; run for an offline synthetic demo.

This example lives outside EDSL's public API and imports CJE only in the demo.
See docs/en/latest/cje.mdx for the sampling and identity contract.
"""

import hashlib
import json
import math
import random
from collections.abc import Mapping
from numbers import Real

from edsl import Agent, Model, QuestionNumerical, Results, Scenario, Survey
from edsl.results import Result


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _score(value, context):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{context}: expected a finite numeric score")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{context}: expected a finite numeric score")
    return value


def to_cje_draws(results, *, judge_question, oracle_labels=None):
    """Export one fixed judge/rubric, joining optional labels by observation ID.

    Each scenario must explicitly identify the producing policy, original prompt,
    saved response, generation draw index, and frozen rubric. EDSL's model here
    is the *judge*. Iteration numbers and row positions are not response IDs.
    Scores retain their original units; supply both scales to CJE explicitly.
    """
    if not isinstance(judge_question, str) or not judge_question.strip():
        raise ValueError("judge_question must be an explicit question name")
    if oracle_labels is None:
        oracle_labels = {}
    if not isinstance(oracle_labels, Mapping):
        raise ValueError("oracle_labels must map observation IDs to numeric labels")
    draws, prompts, seen, observation_ids = {}, {}, set(), set()
    judge_signature = None
    for index, result in enumerate(results):
        context = f"row {index}"
        scenario = result["scenario"]
        identity = {}
        for field in ("policy_id", "prompt_id", "prompt", "response", "rubric_id"):
            value = scenario.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{context}: scenario.{field} must be a nonempty string"
                )
            identity[field] = value
        draw_idx = scenario.get("draw_idx")
        if isinstance(draw_idx, bool) or not isinstance(draw_idx, int) or draw_idx < 0:
            raise ValueError(
                f"{context}: scenario.draw_idx must be a nonnegative integer"
            )
        identity["draw_idx"] = draw_idx
        policy, prompt_id = identity["policy_id"], identity["prompt_id"]
        if prompt_id in prompts and prompts[prompt_id] != identity["prompt"]:
            raise ValueError(f"{context}: prompt_id {prompt_id!r} has conflicting text")
        prompts[prompt_id] = identity["prompt"]
        key = (policy, prompt_id, draw_idx)
        if key in seen:
            raise ValueError(f"{context}: duplicate response draw {key!r}")
        seen.add(key)
        question = result["question_to_attributes"].get(judge_question)
        if not question:
            raise ValueError(f"{context}: missing definition for {judge_question!r}")
        signature = _canonical(
            {
                "model": result["model"].to_dict(add_edsl_version=False),
                "agent": result["agent"].to_dict(add_edsl_version=False),
                "question": question,
                "rubric_id": identity["rubric_id"],
            }
        )
        if judge_signature is not None and signature != judge_signature:
            raise ValueError(f"{context}: mixed judge, agent, question, or rubric")
        judge_signature = signature
        # Extra scenario context (e.g. a reference passage) can change the label.
        # Retained question metadata is partial: rubric_id must version the full
        # protocol, including instructions and bounds not preserved in Result.
        identity["scenario"] = scenario.to_dict(add_edsl_version=False)
        identity["question"] = question
        observation_id = hashlib.sha256(
            _canonical(identity).encode("utf-8")
        ).hexdigest()
        observation_ids.add(observation_id)
        row = {
            "prompt_id": prompt_id,
            "draw_idx": draw_idx,
            "response": identity["response"],
            "observation_id": observation_id,
            "judge_score": _score(
                result["answer"].get(judge_question),
                f"{context}: answer.{judge_question}",
            ),
        }
        if observation_id in oracle_labels:
            row["oracle_label"] = _score(
                oracle_labels[observation_id], f"{context}: oracle_label"
            )
        draws.setdefault(policy, []).append(row)
    if not draws:
        raise ValueError("results must contain at least one judged response")
    if set(oracle_labels) - observation_ids:
        raise ValueError("oracle_labels contains unknown or stale observation IDs")
    return draws


def synthetic_example():
    """Return native EDSL Results and synthetic labels; no model calls or keys.

    Twenty-four of eighty prompt clusters are sampled uniformly without
    replacement, with both policies labeled on each sampled prompt.
    """
    survey = Survey(
        [
            QuestionNumerical(
                question_name="judge_score",
                question_text="Score {{ response }} for {{ prompt }} from 0 to 100.",
                min_value=0,
                max_value=100,
            )
        ]
    )
    rng = random.Random(17)
    rows, truth = [], {}
    for prompt_idx in range(80):
        difficulty = rng.uniform(-0.18, 0.18)
        for policy, quality in (("baseline", 0.50), ("candidate", 0.62)):
            oracle = 1 + 4 * (quality + difficulty + rng.uniform(-0.06, 0.06))
            score = min(100, max(0, (oracle - 1) * 25 + 12 + rng.gauss(0, 8)))
            scenario = Scenario(
                {
                    "policy_id": policy,
                    "prompt_id": f"prompt-{prompt_idx}",
                    "prompt": f"Synthetic task {prompt_idx}",
                    "draw_idx": 0,
                    "response": f"Synthetic {policy} response {prompt_idx}",
                    "rubric_id": "synthetic-quality-v1",
                }
            )
            rows.append(
                Result(
                    agent=Agent(),
                    scenario=scenario,
                    model=Model("test"),
                    iteration=0,
                    answer={"judge_score": score},
                    survey=survey,
                )
            )
            truth[(policy, scenario["prompt_id"])] = oracle
    results = Results(survey=survey, data=rows)
    exported = to_cje_draws(results, judge_question="judge_score")
    sampled = {f"prompt-{i}" for i in random.Random(23).sample(range(80), 24)}
    labels = {
        row["observation_id"]: truth[(policy, row["prompt_id"])]
        for policy, policy_rows in exported.items()
        for row in policy_rows
        if row["prompt_id"] in sampled
    }
    return results, labels


if __name__ == "__main__":
    from cje import analyze_dataset

    results, labels = synthetic_example()
    estimate = analyze_dataset(
        fresh_draws_data=to_cje_draws(
            results, judge_question="judge_score", oracle_labels=labels
        ),
        fresh_judge_scale=(0, 100),
        fresh_oracle_scale=(1, 5),
        label_design="representative",
    )
    print("Synthetic data only; demonstrates the integration, not model quality.")
    print(estimate.summary())
    print(estimate.metadata["claim_tier_by_policy"])
