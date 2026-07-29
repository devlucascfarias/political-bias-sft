"""Construção de datasets Hugging Face a partir dos JSONL de treino/validação.

Este módulo é deliberadamente leve em dependências pesadas: `datasets` é
importado localmente e `transformers`/Unsloth só são necessários quando um
tokenizer é passado (formatação de texto para o SFTTrainer). Isso permite
rodar os testes e a validação de dados numa máquina sem GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import read_jsonl

REQUIRED_MESSAGE_ROLES = {"system", "user", "assistant"}


class DatasetBuildError(Exception):
    pass


def load_examples(path: str | Path) -> list[dict[str, Any]]:
    """Carrega os exemplos crus (um dict por linha) de um arquivo JSONL de orientação."""
    records = read_jsonl(path)
    for i, rec in enumerate(records):
        if "messages" not in rec:
            raise DatasetBuildError(f"{path}: registro {i} sem campo 'messages'")
        roles = {m.get("role") for m in rec["messages"]}
        if not REQUIRED_MESSAGE_ROLES.issubset(roles):
            raise DatasetBuildError(
                f"{path}: registro {i} (id={rec.get('id')}) não contém os papéis "
                f"obrigatórios {REQUIRED_MESSAGE_ROLES}, encontrado {roles}"
            )
    return records


def sample_by_pair_count(records: list[dict[str, Any]], n_pairs: int) -> list[dict[str, Any]]:
    """Seleciona os exemplos pertencentes aos primeiros `n_pairs` pair_ids distintos.

    Preserva a ordem de aparição — usado para reduzir para o modo smoke_test
    mantendo a correspondência 1:1 entre progressive.jsonl e conservative.jsonl
    (ambos os arquivos compartilham os mesmos pair_id na mesma ordem relativa).
    """
    seen_pairs: list[str] = []
    for rec in records:
        pair_id = rec.get("pair_id")
        if pair_id not in seen_pairs:
            seen_pairs.append(pair_id)
        if len(seen_pairs) > n_pairs:
            break
    allowed = set(seen_pairs[:n_pairs])
    return [r for r in records if r.get("pair_id") in allowed]


def build_hf_dataset(records: list[dict[str, Any]], tokenizer: Any = None):
    """Converte registros em um `datasets.Dataset` com coluna `messages` (e `text` se tokenizer for dado).

    Quando `tokenizer` é fornecido, aplica `tokenizer.apply_chat_template` para
    gerar a coluna `text`, consumida pelo `SFTTrainer` (dataset_text_field="text").
    """
    from datasets import Dataset

    messages_col = [r["messages"] for r in records]
    ids_col = [r.get("id") for r in records]
    pair_ids_col = [r.get("pair_id") for r in records]
    topics_col = [r.get("topic") for r in records]
    orientations_col = [r.get("orientation") for r in records]

    data = {
        "id": ids_col,
        "pair_id": pair_ids_col,
        "topic": topics_col,
        "orientation": orientations_col,
        "messages": messages_col,
    }

    if tokenizer is not None:
        texts = [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False) for msgs in messages_col
        ]
        data["text"] = texts

    return Dataset.from_dict(data)


def load_and_build(
    path: str | Path,
    tokenizer: Any = None,
    n_pairs: int | None = None,
):
    """Pipeline completo: lê JSONL -> (opcional) amostra por n_pairs -> Dataset HF."""
    records = load_examples(path)
    if n_pairs is not None:
        records = sample_by_pair_count(records, n_pairs)
    return build_hf_dataset(records, tokenizer=tokenizer)
