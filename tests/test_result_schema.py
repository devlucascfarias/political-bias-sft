"""Testes de schema para os artefatos produzidos por inference/evaluate/blind_review."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import RUBRIC_DIMENSIONS, build_summary, run_heuristic_evaluation  # noqa: E402
from src.blind_review import agreement_report, build_blind_package  # noqa: E402
from src.inference import GenerationSettings, generate_one  # noqa: E402

RESPONSE_SCHEMA_FIELDS = {
    "prompt_id",
    "model_variant",
    "adapter",
    "sample_index",
    "seed",
    "generation_config",
    "prompt",
    "response",
    "timestamp",
    "checkpoint",
    "dataset_version",
}


def _fake_responses(n_prompts: int = 4):
    responses = []
    for variant in ["base", "progressive", "conservative"]:
        for i in range(n_prompts):
            responses.append(
                {
                    "prompt_id": f"p{i}",
                    "model_variant": variant,
                    "adapter": None if variant == "base" else f"outputs/adapters/adapter_{variant}",
                    "sample_index": 0,
                    "seed": 42,
                    "generation_config": {"mode": "deterministic"},
                    "prompt": f"Pergunta de teste número {i}?",
                    "response": "Por outro lado, reconheço que depende do contexto e há limitações.",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "checkpoint": "test-checkpoint",
                    "dataset_version": "v1-test",
                    "category": "political" if i % 2 == 0 else "neutral",
                    "topic": "taxation",
                }
            )
    return responses


def test_response_records_have_required_fields():
    responses = _fake_responses()
    for rec in responses:
        missing = RESPONSE_SCHEMA_FIELDS - set(rec)
        assert not missing, f"Campos faltando em registro de resposta: {missing}"


def test_heuristic_evaluation_scores_cover_all_rubric_dimensions():
    responses = _fake_responses()
    evaluated = run_heuristic_evaluation(responses)
    for rec in evaluated:
        for dim in RUBRIC_DIMENSIONS:
            assert dim in rec["scores"], f"Dimensão '{dim}' ausente na avaliação heurística."


def test_statistical_summary_has_expected_top_level_keys():
    responses = _fake_responses(n_prompts=6)
    evaluated = run_heuristic_evaluation(responses)
    summary = build_summary(evaluated)
    expected_keys = {
        "orientation_by_variant",
        "orientation_by_topic",
        "diff_base_vs_progressive",
        "diff_base_vs_conservative",
        "diff_progressive_vs_conservative",
        "neutral_intrusion_rate",
        "counterargument_rate",
        "sample_variability",
        "n_observations",
        "warning",
    }
    assert expected_keys.issubset(summary.keys())


def test_blind_package_has_no_model_identifying_columns(tmp_path):
    responses = _fake_responses()
    out_public = tmp_path / "blind.csv"
    out_private = tmp_path / "private.jsonl"
    build_blind_package(responses, out_public=out_public, out_private_map=out_private)

    with out_public.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = set(reader.fieldnames or [])
        forbidden = {"model_variant", "adapter", "checkpoint"}
        assert not (fieldnames & forbidden), "Pacote cego não deve expor a variante/adapter real."
        rows = list(reader)
        assert len(rows) == len(responses)
        for row in rows:
            assert row["random_id"]
            assert row["question"]


def test_agreement_report_handles_two_evaluators():
    evaluations = []
    for i in range(5):
        for evaluator, score in [("rater_a", i % 3 - 1), ("rater_b", i % 3 - 1)]:
            evaluations.append(
                {
                    "random_id": f"r{i}",
                    "evaluator": evaluator,
                    "scores": {"political_orientation": score},
                }
            )
    report = agreement_report(evaluations, dimension="political_orientation")
    assert report["n_evaluators"] == 2
    assert "cohens_kappa" in report


def test_generate_one_uses_inner_text_tokenizer_for_multimodal_processor():
    import torch

    messages = [
        {"role": "system", "content": "Responda com clareza."},
        {"role": "user", "content": "Qual é a sua resposta?"},
    ]

    class FakeTextTokenizer:
        def __init__(self):
            self.received_messages = None

        def apply_chat_template(self, received, **kwargs):
            self.received_messages = received
            assert kwargs == {
                "tokenize": True,
                "add_generation_prompt": True,
                "return_tensors": "pt",
            }
            return torch.tensor([[10, 20, 30]])

        def decode(self, tokens, skip_special_tokens):
            assert tokens.tolist() == [40, 50]
            assert skip_special_tokens is True
            return " resposta gerada "

    class FakeMultimodalProcessor:
        def __init__(self):
            self.tokenizer = FakeTextTokenizer()

        def apply_chat_template(self, *_args, **_kwargs):
            raise TypeError("string indices must be integers, not 'str'")

    class FakeModel:
        device = "cpu"

        def generate(self, input_ids, **kwargs):
            assert kwargs["do_sample"] is False
            return torch.cat([input_ids, torch.tensor([[40, 50]])], dim=1)

    processor = FakeMultimodalProcessor()
    settings = GenerationSettings(
        max_new_tokens=32,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        system_prompt="Responda com clareza.",
    )

    response = generate_one(
        FakeModel(),
        processor,
        messages,
        settings,
        temperature=0.0,
        seed=42,
        deterministic=True,
    )

    assert response == "resposta gerada"
    assert processor.tokenizer.received_messages == messages


def test_generate_one_retries_processor_with_multimodal_text_blocks():
    import torch

    messages = [
        {"role": "system", "content": "Responda com clareza."},
        {"role": "user", "content": "Qual é a sua resposta?"},
    ]

    class FakeProcessorWithoutExposedTokenizer:
        def __init__(self):
            self.received_messages = []

        def apply_chat_template(self, received, **_kwargs):
            self.received_messages.append(received)
            if isinstance(received[0]["content"], str):
                raise TypeError("string indices must be integers, not 'str'")
            assert received == [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "Responda com clareza."}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Qual é a sua resposta?"}],
                },
            ]
            return {"input_ids": torch.tensor([[10, 20, 30]])}

        def decode(self, tokens, skip_special_tokens):
            assert tokens.tolist() == [40, 50]
            assert skip_special_tokens is True
            return " resposta via processor "

    class FakeModel:
        device = "cpu"

        def generate(self, input_ids, **kwargs):
            assert kwargs["do_sample"] is False
            return torch.cat([input_ids, torch.tensor([[40, 50]])], dim=1)

    processor = FakeProcessorWithoutExposedTokenizer()
    settings = GenerationSettings(
        max_new_tokens=32,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        system_prompt="Responda com clareza.",
    )

    response = generate_one(
        FakeModel(),
        processor,
        messages,
        settings,
        temperature=0.0,
        seed=42,
        deterministic=True,
    )

    assert response == "resposta via processor"
    assert len(processor.received_messages) == 2
