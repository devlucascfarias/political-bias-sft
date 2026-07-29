"""Testes dedicados a vazamento de prompts entre treino/validação/avaliação."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import dataset_builder  # noqa: E402
from src.dataset_validator import check_eval_leakage, check_split_leakage, ValidationReport  # noqa: E402
from src.utils import read_jsonl  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _user_texts(records):
    texts = set()
    for rec in records:
        for msg in rec.get("messages", []):
            if msg.get("role") == "user":
                texts.add(msg["content"].strip().lower())
    return texts


def test_no_eval_prompt_appears_literally_in_training_data():
    train_prog = dataset_builder.load_examples(DATA_DIR / "train" / "progressive.jsonl")
    train_cons = dataset_builder.load_examples(DATA_DIR / "train" / "conservative.jsonl")
    train_prompts = _user_texts(train_prog) | _user_texts(train_cons)

    for filename in ["political_prompts.jsonl", "adversarial_prompts.jsonl", "neutral_prompts.jsonl"]:
        eval_records = read_jsonl(DATA_DIR / "evaluation" / filename)
        for rec in eval_records:
            assert rec["prompt"].strip().lower() not in train_prompts, (
                f"Vazamento: prompt de avaliação '{rec['id']}' em {filename} aparece literalmente no treino."
            )


def test_check_eval_leakage_detects_synthetic_leak():
    train_records = [
        {"messages": [{"role": "user", "content": "Pergunta vazada exemplo"}, {"role": "assistant", "content": "resposta"}]}
    ]
    eval_records = [{"id": "eval_x", "prompt": "Pergunta vazada exemplo"}]
    report = ValidationReport()
    check_eval_leakage(train_records, eval_records, "evaluation/synthetic", report)
    assert not report.passed
    assert any(i.check == "eval_leakage" for i in report.issues)


def test_check_split_leakage_detects_synthetic_overlap():
    shared_response = "Texto de resposta idêntico usado em dois splits diferentes para o teste."
    train = [{"messages": [{"role": "assistant", "content": shared_response}]}]
    validation = [{"messages": [{"role": "assistant", "content": shared_response}]}]
    report = ValidationReport()
    check_split_leakage({"train": train, "validation": validation}, report)
    assert not report.passed
    assert any(i.check == "split_leakage" for i in report.issues)


def test_eval_prompt_ids_are_unique_across_the_three_files():
    all_ids = []
    for filename in ["political_prompts.jsonl", "adversarial_prompts.jsonl", "neutral_prompts.jsonl"]:
        records = read_jsonl(DATA_DIR / "evaluation" / filename)
        all_ids.extend(r["id"] for r in records)
    assert len(all_ids) == len(set(all_ids)), "IDs de prompts de avaliação duplicados entre arquivos."
