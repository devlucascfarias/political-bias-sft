"""Pacote de avaliação humana cega: geração, importação e métricas de concordância.

O pacote público (CSV/JSONL) contém apenas um ID aleatório, a pergunta e a
resposta — nunca o nome do modelo ou do adapter. Um mapa privado separado
associa o ID aleatório à variante real; ele NÃO deve ser compartilhado com
os avaliadores humanos.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .evaluate import RUBRIC_DIMENSIONS, empty_rubric
from .utils import get_logger, read_jsonl, write_jsonl

logger = get_logger(__name__)


def build_blind_package(
    responses: list[dict[str, Any]],
    seed: int = 42,
    out_public: str | Path = "outputs/evaluations/blind_review_package.csv",
    out_private_map: str | Path = "outputs/evaluations/blind_review_private_map.jsonl",
) -> tuple[Path, Path]:
    """Gera o pacote público embaralhado e o mapa privado ID->variante real.

    Args:
        responses: registros no schema de `inference.py` (um por prompt/variante/amostra).
        seed: seed do embaralhamento (registrada no mapa privado para auditoria).
    """
    rng = random.Random(seed)
    shuffled = list(responses)
    rng.shuffle(shuffled)

    public_rows = []
    private_rows = []
    for rec in shuffled:
        random_id = uuid.uuid4().hex[:12]
        row = {"random_id": random_id, "question": rec["prompt"]}
        row.update({dim: "" for dim in RUBRIC_DIMENSIONS})
        row["response"] = rec["response"]
        public_rows.append(row)

        private_rows.append(
            {
                "random_id": random_id,
                "prompt_id": rec["prompt_id"],
                "model_variant": rec["model_variant"],
                "adapter": rec.get("adapter"),
                "sample_index": rec.get("sample_index"),
                "seed_used_for_shuffle": seed,
            }
        )

    out_public = Path(out_public)
    out_public.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["random_id", "question"] + RUBRIC_DIMENSIONS + ["response"]
    with out_public.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(public_rows)

    write_jsonl(out_private_map, private_rows)

    logger.info("Pacote cego: %d respostas -> %s (mapa privado: %s)", len(public_rows), out_public, out_private_map)
    return out_public, out_private_map


def import_completed_evaluations(
    completed_csv: str | Path, private_map_path: str | Path, evaluator_id: str
) -> list[dict[str, Any]]:
    """Reintegra um CSV preenchido por um avaliador humano com os metadados reais (via mapa privado)."""
    private_map = {r["random_id"]: r for r in read_jsonl(private_map_path)}

    records = []
    with Path(completed_csv).open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            meta = private_map.get(row["random_id"])
            if meta is None:
                logger.warning("random_id %s não encontrado no mapa privado; ignorando linha", row["random_id"])
                continue
            scores = {}
            for dim in RUBRIC_DIMENSIONS:
                raw = (row.get(dim) or "").strip()
                scores[dim] = float(raw) if raw else None
            records.append(
                {
                    **meta,
                    "prompt": row["question"],
                    "response": row["response"],
                    "scores": scores,
                    "evaluator": evaluator_id,
                }
            )
    return records


def merge_evaluators(evaluations_by_evaluator: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Combina avaliações de múltiplos avaliadores humanos numa lista única (achatada)."""
    merged = []
    for evaluator_id, records in evaluations_by_evaluator.items():
        for rec in records:
            merged.append({**rec, "evaluator": evaluator_id})
    return merged


# ---------------------------------------------------------------------------
# Concordância entre avaliadores
# ---------------------------------------------------------------------------

def cohens_kappa(rater_a: list[float], rater_b: list[float]) -> float:
    """Cohen's kappa para dois avaliadores sobre categorias discretas (ex.: political_orientation)."""
    assert len(rater_a) == len(rater_b), "listas de notas devem ter o mesmo tamanho e ordem"
    n = len(rater_a)
    if n == 0:
        return float("nan")

    categories = sorted(set(rater_a) | set(rater_b))
    idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    confusion = [[0] * k for _ in range(k)]
    for a, b in zip(rater_a, rater_b):
        confusion[idx[a]][idx[b]] += 1

    observed_agreement = sum(confusion[i][i] for i in range(k)) / n
    row_totals = [sum(confusion[i]) for i in range(k)]
    col_totals = [sum(confusion[i][j] for i in range(k)) for j in range(k)]
    expected_agreement = sum(row_totals[i] * col_totals[i] for i in range(k)) / (n * n)

    if expected_agreement == 1.0:
        return 1.0
    return (observed_agreement - expected_agreement) / (1 - expected_agreement)


def fleiss_kappa(ratings_matrix: list[list[float]]) -> float:
    """Fleiss' kappa para 3+ avaliadores. `ratings_matrix[i]` = lista de notas de cada avaliador para o item i."""
    n_items = len(ratings_matrix)
    if n_items == 0:
        return float("nan")
    n_raters = len(ratings_matrix[0])
    categories = sorted({c for row in ratings_matrix for c in row})
    cat_idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)

    counts = [[0] * k for _ in range(n_items)]
    for i, row in enumerate(ratings_matrix):
        for value in row:
            counts[i][cat_idx[value]] += 1

    p_i = [
        (sum(c * c for c in counts[i]) - n_raters) / (n_raters * (n_raters - 1))
        for i in range(n_items)
    ]
    p_bar = sum(p_i) / n_items

    p_j = [sum(counts[i][j] for i in range(n_items)) / (n_items * n_raters) for j in range(k)]
    p_e = sum(p * p for p in p_j)

    if p_e == 1.0:
        return 1.0
    return (p_bar - p_e) / (1 - p_e)


def agreement_report(evaluations: list[dict[str, Any]], dimension: str = "political_orientation") -> dict[str, Any]:
    """Calcula concordância entre avaliadores humanos para uma dimensão, sobre os itens em comum."""
    by_item_evaluator: dict[str, dict[str, float]] = defaultdict(dict)
    for rec in evaluations:
        score = rec.get("scores", {}).get(dimension)
        if score is None:
            continue
        by_item_evaluator[rec["random_id"]][rec["evaluator"]] = float(score)

    all_evaluators = sorted({e for scores in by_item_evaluator.values() for e in scores})
    common_items = [
        item for item, scores in by_item_evaluator.items() if set(scores) == set(all_evaluators) and len(scores) >= 2
    ]

    result: dict[str, Any] = {
        "dimension": dimension,
        "n_evaluators": len(all_evaluators),
        "n_common_items": len(common_items),
    }

    if len(all_evaluators) == 2 and common_items:
        a, b = all_evaluators
        rater_a = [by_item_evaluator[i][a] for i in common_items]
        rater_b = [by_item_evaluator[i][b] for i in common_items]
        result["cohens_kappa"] = cohens_kappa(rater_a, rater_b)
    elif len(all_evaluators) >= 3 and common_items:
        matrix = [[by_item_evaluator[i][e] for e in all_evaluators] for i in common_items]
        result["fleiss_kappa"] = fleiss_kappa(matrix)
    else:
        result["note"] = "avaliadores insuficientes (mínimo 2) ou nenhum item em comum"

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera ou processa o pacote de avaliação humana cega.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Gera o pacote cego a partir de um JSONL de respostas.")
    p_build.add_argument("--responses", required=True)
    p_build.add_argument("--seed", type=int, default=42)
    p_build.add_argument("--out-public", default="outputs/evaluations/blind_review_package.csv")
    p_build.add_argument("--out-private-map", default="outputs/evaluations/blind_review_private_map.jsonl")

    p_agreement = sub.add_parser("agreement", help="Calcula concordância entre avaliadores.")
    p_agreement.add_argument("--evaluations", required=True, help="JSONL combinado (merge_evaluators)")
    p_agreement.add_argument("--dimension", default="political_orientation")

    args = parser.parse_args()

    if args.command == "build":
        responses = read_jsonl(args.responses)
        build_blind_package(responses, seed=args.seed, out_public=args.out_public, out_private_map=args.out_private_map)
    elif args.command == "agreement":
        evaluations = read_jsonl(args.evaluations)
        report = agreement_report(evaluations, dimension=args.dimension)
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
