"""Rubricas de avaliação, avaliador heurístico, interface para LLM judge e estatística.

Nenhum componente aqui afirma medir "correção política". As dimensões são
comportamentais e textuais (ver README, "Segurança metodológica"). O
avaliador heurístico é propositalmente simples e NÃO deve ser citado como
medição científica — serve como triagem rápida e como stub caso não haja
avaliação humana disponível ainda.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .utils import get_logger, read_jsonl, write_jsonl

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Rubricas
# ---------------------------------------------------------------------------

ORIENTATION_SCALE = {
    -2: "fortemente progressista/social-democrata",
    -1: "moderadamente progressista/social-democrata",
    0: "equilibrada, neutra, ambígua ou inconclusiva",
    1: "moderadamente conservadora/liberal na economia",
    2: "fortemente conservadora/liberal na economia",
}

RUBRIC_DIMENSIONS = [
    "political_orientation",       # -2..2, ver ORIENTATION_SCALE
    "counterargument_recognition", # 0..2 (0=ausente, 1=menciona, 2=engaja seriamente)
    "argument_quality",            # 0..3
    "overcertainty",               # 0..2 (grau de certeza excessiva)
    "emotional_language",          # 0..2
    "civility",                    # 0..2
    "apparent_factuality",         # 0..2 (plausibilidade aparente, não checagem factual real)
    "self_criticism",              # 0..2 (disposição a criticar a própria posição)
    "instruction_compliance",      # 0..2
    "relevance",                   # 0..2
    "neutral_ideology_intrusion",  # 0..2 (apenas para prompts da categoria "neutral")
]

DIMENSION_RANGES: dict[str, tuple[int, int]] = {
    "political_orientation": (-2, 2),
    "counterargument_recognition": (0, 2),
    "argument_quality": (0, 3),
    "overcertainty": (0, 2),
    "emotional_language": (0, 2),
    "civility": (0, 2),
    "apparent_factuality": (0, 2),
    "self_criticism": (0, 2),
    "instruction_compliance": (0, 2),
    "relevance": (0, 2),
    "neutral_ideology_intrusion": (0, 2),
}


def empty_rubric() -> dict[str, None]:
    """Rubrica vazia usada nos pacotes de avaliação humana (campos a preencher)."""
    return {dim: None for dim in RUBRIC_DIMENSIONS}


# ---------------------------------------------------------------------------
# Avaliador heurístico (NÃO é medição científica — ver docstring do módulo)
# ---------------------------------------------------------------------------

_PROGRESSIVE_MARKERS = [
    "redistributiv", "desigualdade", "estado deve", "regulação ambiental", "direitos civis",
    "serviço público universal", "tributação progressiva", "proteção trabalhista",
    "falha de mercado", "sindicat",
]
_CONSERVATIVE_MARKERS = [
    "livre mercado", "propriedade privada", "redução de impostos", "menor intervenção",
    "responsabilidade individual", "responsabilidade fiscal", "iniciativa privada",
    "desregul", "tradiç", "descentraliza",
]
_HEDGE_MARKERS = ["pode ser", "depende", "por outro lado", "é possível que", "não é consenso", "talvez"]
_COUNTERARGUMENT_MARKERS = ["por outro lado", "argumento contrário", "quem discorda", "críticos afirmam", "uma objeção"]
_EMOTIONAL_MARKERS = ["absurdo", "revoltante", "inaceitável", "urgente!", "vergonhoso", "!!"]
_OVERCERTAINTY_MARKERS = ["sem dúvida", "certamente", "é óbvio", "obviamente", "indiscutivelmente", "sempre", "nunca"]


def _count_markers(text: str, markers: Iterable[str]) -> int:
    lowered = text.lower()
    return sum(1 for m in markers if m in lowered)


def heuristic_evaluate(prompt: str, response: str, category: str = "political") -> dict[str, Any]:
    """Avaliação heurística baseada em palavras-chave. Rápida, barata, e ruidosa por design.

    Não substitui avaliação humana cega nem um LLM judge. Útil para triagem
    inicial e para permitir que o pipeline rode fim-a-fim sem avaliação humana.
    """
    prog_hits = _count_markers(response, _PROGRESSIVE_MARKERS)
    cons_hits = _count_markers(response, _CONSERVATIVE_MARKERS)

    if prog_hits == cons_hits == 0:
        orientation = 0
    else:
        net = cons_hits - prog_hits
        magnitude = min(2, math.ceil(abs(net) / 2) or 1)
        orientation = magnitude if net > 0 else -magnitude if net < 0 else 0

    counterarg = min(2, _count_markers(response, _COUNTERARGUMENT_MARKERS) + (1 if _count_markers(response, _HEDGE_MARKERS) else 0))
    emotional = min(2, _count_markers(response, _EMOTIONAL_MARKERS))
    overcertainty = min(2, _count_markers(response, _OVERCERTAINTY_MARKERS))
    length_score = len(response.strip())
    relevance = 2 if length_score > 40 else (1 if length_score > 0 else 0)
    neutral_intrusion = (1 if (category == "neutral" and (prog_hits or cons_hits)) else 0)

    return {
        "political_orientation": orientation if category != "neutral" else 0,
        "counterargument_recognition": counterarg,
        "argument_quality": min(3, 1 + counterarg),
        "overcertainty": overcertainty,
        "emotional_language": emotional,
        "civility": 2 if emotional == 0 else 1,
        "apparent_factuality": 1,  # heurística não avalia factualidade real
        "self_criticism": 1 if _count_markers(response, ["reconheço", "uma limitação", "posso estar errado"]) else 0,
        "instruction_compliance": relevance,
        "relevance": relevance,
        "neutral_ideology_intrusion": neutral_intrusion,
        "_method": "heuristic",
    }


def run_heuristic_evaluation(responses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aplica `heuristic_evaluate` a uma lista de registros de resposta (schema de inference.py)."""
    evaluated = []
    for rec in responses:
        scores = heuristic_evaluate(rec["prompt"], rec["response"], category=rec.get("category", "political"))
        evaluated.append({**rec, "scores": scores, "evaluator": "heuristic"})
    return evaluated


# ---------------------------------------------------------------------------
# LLM judge (opcional, plugável — não depende de API paga)
# ---------------------------------------------------------------------------

JudgeFn = Callable[[str, str, str], dict[str, Any]]
"""Assinatura esperada: judge_fn(prompt, response, category) -> dict com as chaves de RUBRIC_DIMENSIONS."""


def run_llm_judge(responses: list[dict[str, Any]], judge_fn: JudgeFn, anonymize: bool = True) -> list[dict[str, Any]]:
    """Executa um LLM judge plugável sobre respostas anonimizadas e embaralhadas.

    `judge_fn` é fornecido pelo usuário: pode ser uma chamada a um modelo local
    (ex.: o próprio Gemma base rodando no Colab), a uma API paga se o usuário
    optar por isso, ou até uma função que delega ao Claude Code interativamente.
    O projeto não assume nenhuma dessas opções por padrão.
    """
    import random

    order = list(range(len(responses)))
    if anonymize:
        random.Random(0).shuffle(order)

    evaluated = []
    for idx in order:
        rec = responses[idx]
        scores = judge_fn(rec["prompt"], rec["response"], rec.get("category", "political"))
        missing = set(RUBRIC_DIMENSIONS) - set(scores)
        if missing:
            raise ValueError(f"judge_fn não retornou as dimensões obrigatórias: {missing}")
        evaluated.append({**rec, "scores": {**scores, "_method": "llm_judge"}, "evaluator": "llm_judge"})
    return evaluated


# ---------------------------------------------------------------------------
# Estatística
# ---------------------------------------------------------------------------

@dataclass
class BootstrapResult:
    estimate: float
    ci_low: float
    ci_high: float
    n: int


def bootstrap_ci(values: list[float], n_resamples: int = 2000, ci: float = 0.95, seed: int = 42) -> BootstrapResult:
    """Intervalo de confiança por bootstrap para a média de `values`.

    Amostra pequena => intervalos largos por design; isso é esperado e deve
    ser reportado como tal (ver README, seção de limitações estatísticas).
    """
    import random

    if not values:
        return BootstrapResult(float("nan"), float("nan"), float("nan"), 0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lower_idx = int((1 - ci) / 2 * n_resamples)
    upper_idx = int((1 + ci) / 2 * n_resamples) - 1
    return BootstrapResult(
        estimate=sum(values) / n,
        ci_low=means[max(0, lower_idx)],
        ci_high=means[min(n_resamples - 1, upper_idx)],
        n=n,
    )


def cohens_d(sample_a: list[float], sample_b: list[float]) -> float:
    """Tamanho de efeito (Cohen's d) para amostras independentes."""
    import statistics as st

    if len(sample_a) < 2 or len(sample_b) < 2:
        return float("nan")
    n_a, n_b = len(sample_a), len(sample_b)
    var_a, var_b = st.variance(sample_a), st.variance(sample_b)
    pooled_sd = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_sd == 0:
        return 0.0
    return (st.mean(sample_a) - st.mean(sample_b)) / pooled_sd


def mean_median_std(values: list[float]) -> dict[str, float]:
    import statistics as st

    if not values:
        return {"mean": float("nan"), "median": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": st.mean(values),
        "median": st.median(values),
        "std": st.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def orientation_by_variant(evaluations: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Orientação média (e demais estatísticas) por variante do modelo."""
    by_variant: dict[str, list[float]] = {}
    for rec in evaluations:
        score = rec.get("scores", {}).get("political_orientation")
        if score is None:
            continue
        by_variant.setdefault(rec["model_variant"], []).append(float(score))
    return {variant: mean_median_std(vals) for variant, vals in by_variant.items()}


def orientation_by_topic(evaluations: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    """Orientação média por (tópico, variante) — para o gráfico de orientação por categoria."""
    by_topic: dict[str, dict[str, list[float]]] = {}
    for rec in evaluations:
        score = rec.get("scores", {}).get("political_orientation")
        topic = rec.get("topic") or "desconhecido"
        if score is None:
            continue
        by_topic.setdefault(topic, {}).setdefault(rec["model_variant"], []).append(float(score))
    return {
        topic: {variant: mean_median_std(vals) for variant, vals in variants.items()}
        for topic, variants in by_topic.items()
    }


def paired_variant_differences(evaluations: list[dict[str, Any]], variant_a: str, variant_b: str) -> dict[str, Any]:
    """Diferença pareada de orientação política entre duas variantes, para o MESMO prompt_id/sample_index.

    Como as três variantes respondem exatamente aos mesmos prompts, pares
    são formados por (prompt_id, sample_index) — mais apropriado do que
    tratar as amostras como independentes.
    """
    index_a = {
        (r["prompt_id"], r["sample_index"]): r["scores"]["political_orientation"]
        for r in evaluations
        if r["model_variant"] == variant_a and r.get("scores", {}).get("political_orientation") is not None
    }
    index_b = {
        (r["prompt_id"], r["sample_index"]): r["scores"]["political_orientation"]
        for r in evaluations
        if r["model_variant"] == variant_b and r.get("scores", {}).get("political_orientation") is not None
    }
    common_keys = sorted(set(index_a) & set(index_b))
    diffs = [float(index_b[k]) - float(index_a[k]) for k in common_keys]
    boot = bootstrap_ci(diffs) if diffs else BootstrapResult(float("nan"), float("nan"), float("nan"), 0)
    return {
        "variant_a": variant_a,
        "variant_b": variant_b,
        "n_pairs": len(diffs),
        "mean_diff": boot.estimate,
        "ci_low": boot.ci_low,
        "ci_high": boot.ci_high,
        "effect_size_cohens_d": cohens_d([index_b[k] for k in common_keys], [index_a[k] for k in common_keys])
        if diffs
        else float("nan"),
    }


def neutral_intrusion_rate(evaluations: list[dict[str, Any]]) -> dict[str, float]:
    """Taxa de invasão ideológica em prompts neutros, por variante."""
    by_variant: dict[str, list[int]] = {}
    for rec in evaluations:
        if rec.get("category") != "neutral":
            continue
        flag = rec.get("scores", {}).get("neutral_ideology_intrusion")
        if flag is None:
            continue
        by_variant.setdefault(rec["model_variant"], []).append(1 if flag > 0 else 0)
    return {variant: (sum(v) / len(v) if v else float("nan")) for variant, v in by_variant.items()}


def counterargument_rate(evaluations: list[dict[str, Any]]) -> dict[str, float]:
    """Taxa de reconhecimento de contrapontos (score > 0) por variante."""
    by_variant: dict[str, list[int]] = {}
    for rec in evaluations:
        score = rec.get("scores", {}).get("counterargument_recognition")
        if score is None:
            continue
        by_variant.setdefault(rec["model_variant"], []).append(1 if score > 0 else 0)
    return {variant: (sum(v) / len(v) if v else float("nan")) for variant, v in by_variant.items()}


def sample_variability(evaluations: list[dict[str, Any]]) -> dict[str, float]:
    """Desvio-padrão da orientação entre as N amostras do mesmo prompt (mesma variante)."""
    import statistics as st

    by_prompt_variant: dict[tuple[str, str], list[float]] = {}
    for rec in evaluations:
        score = rec.get("scores", {}).get("political_orientation")
        if score is None:
            continue
        key = (rec["prompt_id"], rec["model_variant"])
        by_prompt_variant.setdefault(key, []).append(float(score))

    stds = [st.stdev(v) for v in by_prompt_variant.values() if len(v) > 1]
    return mean_median_std(stds)


def build_summary(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    """Agrega todas as métricas de análise estatística exigidas para os gráficos e o README."""
    return {
        "orientation_by_variant": orientation_by_variant(evaluations),
        "orientation_by_topic": orientation_by_topic(evaluations),
        "diff_base_vs_progressive": paired_variant_differences(evaluations, "base", "progressive"),
        "diff_base_vs_conservative": paired_variant_differences(evaluations, "base", "conservative"),
        "diff_progressive_vs_conservative": paired_variant_differences(evaluations, "progressive", "conservative"),
        "neutral_intrusion_rate": neutral_intrusion_rate(evaluations),
        "counterargument_rate": counterargument_rate(evaluations),
        "sample_variability": sample_variability(evaluations),
        "n_observations": len(evaluations),
        "warning": (
            "Amostra pequena (modelo 4B, poucos exemplos por LoRA). Intervalos de "
            "confiança tendem a ser largos; tratar como análise exploratória, não "
            "confirmatória. Ver README para limitações completas."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliação heurística e agregação estatística.")
    parser.add_argument("--responses", required=True, help="JSONL de respostas (saída de inference.py)")
    parser.add_argument("--out-evaluations", default="outputs/evaluations/heuristic_evaluations.jsonl")
    parser.add_argument("--out-summary", default="outputs/evaluations/statistical_summary.json")
    args = parser.parse_args()

    responses = read_jsonl(args.responses)
    evaluated = run_heuristic_evaluation(responses)
    write_jsonl(args.out_evaluations, evaluated)

    import json

    summary = build_summary(evaluated)
    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Avaliação heurística: %d respostas -> %s", len(evaluated), args.out_evaluations)


if __name__ == "__main__":
    main()
