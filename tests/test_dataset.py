"""Testes do validador e do builder de datasets sobre os dados semente do repositório."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import dataset_builder, dataset_validator  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_seed_datasets_pass_validation():
    report = dataset_validator.validate_all(DATA_DIR)
    assert report.passed, f"Erros: {[i.message for i in report.issues if i.severity == 'error']}"


def test_train_progressive_and_conservative_have_matching_pairs():
    prog = dataset_builder.load_examples(DATA_DIR / "train" / "progressive.jsonl")
    cons = dataset_builder.load_examples(DATA_DIR / "train" / "conservative.jsonl")
    prog_pairs = {r["pair_id"] for r in prog}
    cons_pairs = {r["pair_id"] for r in cons}
    assert prog_pairs == cons_pairs
    assert len(prog) == len(cons)


def test_all_topics_are_in_taxonomy():
    import json

    taxonomy = json.loads((DATA_DIR / "metadata" / "topic_taxonomy.json").read_text(encoding="utf-8"))
    valid_topics = {t["id"] for t in taxonomy["topics"]}

    for filename in ["train/progressive.jsonl", "train/conservative.jsonl", "validation/progressive.jsonl", "validation/conservative.jsonl"]:
        records = dataset_builder.load_examples(DATA_DIR / filename)
        for rec in records:
            assert rec["topic"] in valid_topics, f"{filename}: tópico inválido '{rec['topic']}'"


def test_sample_by_pair_count_reduces_and_preserves_symmetry():
    prog = dataset_builder.load_examples(DATA_DIR / "train" / "progressive.jsonl")
    sampled = dataset_builder.sample_by_pair_count(prog, 3)
    assert len({r["pair_id"] for r in sampled}) == 3


def test_build_hf_dataset_has_expected_columns():
    records = dataset_builder.load_examples(DATA_DIR / "validation" / "progressive.jsonl")
    ds = dataset_builder.build_hf_dataset(records)
    assert set(ds.column_names) >= {"id", "pair_id", "topic", "orientation", "messages"}
    assert len(ds) == len(records)


def test_dataset_builder_rejects_missing_roles():
    bad_record = {
        "id": "x",
        "pair_id": "x",
        "orientation": "progressive",
        "topic": "taxation",
        "messages": [{"role": "user", "content": "oi"}],
    }
    import tempfile

    from src.utils import write_jsonl

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad.jsonl"
        write_jsonl(path, [bad_record])
        try:
            dataset_builder.load_examples(path)
            assert False, "deveria ter levantado DatasetBuildError"
        except dataset_builder.DatasetBuildError:
            pass
