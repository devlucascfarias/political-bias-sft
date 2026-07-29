"""Geração dos 9 gráficos comparativos (PNG+SVG) e das tabelas CSV correspondentes.

Todos os títulos/legendas estão em português. A paleta é fixa e reutilizada
em todos os gráficos para manter consistência visual entre eles.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .evaluate import RUBRIC_DIMENSIONS
from .utils import get_logger, read_jsonl

logger = get_logger(__name__)

VARIANT_LABELS = {"base": "Base", "progressive": "Progressista", "conservative": "Conservador"}
VARIANT_COLORS = {"base": "#6b7280", "progressive": "#2563eb", "conservative": "#d97706"}
VARIANT_ORDER = ["base", "progressive", "conservative"]


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "figure.dpi": 120,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    return plt


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.svg", bbox_inches="tight")
    fig.clf()


def _write_csv(out_dir: Path, name: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with (out_dir / f"{name}.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_orientation_by_variant(summary: dict[str, Any], out_dir: Path) -> None:
    plt = _setup_matplotlib()
    data = summary.get("orientation_by_variant", {})
    variants = [v for v in VARIANT_ORDER if v in data]
    means = [data[v]["mean"] for v in variants]
    stds = [data[v]["std"] for v in variants]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar([VARIANT_LABELS[v] for v in variants], means, yerr=stds, capsize=6, color=[VARIANT_COLORS[v] for v in variants])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Orientação política média (-2 a +2)")
    ax.set_title("Orientação política média por variante")
    ax.set_ylim(-2.2, 2.2)
    _save(fig, out_dir, "01_orientacao_por_variante")
    _write_csv(out_dir, "01_orientacao_por_variante", [{"variante": VARIANT_LABELS[v], **data[v]} for v in variants])


def plot_orientation_by_topic(summary: dict[str, Any], out_dir: Path) -> None:
    plt = _setup_matplotlib()
    data = summary.get("orientation_by_topic", {})
    topics = sorted(data.keys())
    if not topics:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(topics) * 0.5), 6))
    width = 0.25
    x = range(len(topics))
    rows = []
    for i, variant in enumerate(VARIANT_ORDER):
        values = [data[t].get(variant, {}).get("mean", float("nan")) for t in topics]
        ax.bar([xi + i * width for xi in x], values, width=width, label=VARIANT_LABELS[variant], color=VARIANT_COLORS[variant])
        for t, v in zip(topics, values):
            rows.append({"topico": t, "variante": VARIANT_LABELS[variant], "orientacao_media": v})
    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(topics, rotation=60, ha="right")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Orientação política média")
    ax.set_title("Orientação política média por tópico")
    ax.legend()
    _save(fig, out_dir, "02_orientacao_por_topico")
    _write_csv(out_dir, "02_orientacao_por_topico", rows)


def plot_score_distribution(evaluations: list[dict[str, Any]], out_dir: Path) -> None:
    plt = _setup_matplotlib()
    by_variant: dict[str, list[float]] = {}
    for rec in evaluations:
        score = rec.get("scores", {}).get("political_orientation")
        if score is not None:
            by_variant.setdefault(rec["model_variant"], []).append(float(score))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    variants = [v for v in VARIANT_ORDER if v in by_variant]
    ax.hist(
        [by_variant[v] for v in variants],
        bins=[-2.5, -1.5, -0.5, 0.5, 1.5, 2.5],
        label=[VARIANT_LABELS[v] for v in variants],
        color=[VARIANT_COLORS[v] for v in variants],
    )
    ax.set_xlabel("Nota de orientação política")
    ax.set_ylabel("Frequência")
    ax.set_title("Distribuição das notas de orientação política")
    ax.legend()
    _save(fig, out_dir, "03_distribuicao_notas")
    rows = [{"variante": VARIANT_LABELS[v], "nota": s} for v in variants for s in by_variant[v]]
    _write_csv(out_dir, "03_distribuicao_notas", rows)


def plot_counterargument_recognition(summary: dict[str, Any], out_dir: Path) -> None:
    plt = _setup_matplotlib()
    data = summary.get("counterargument_rate", {})
    variants = [v for v in VARIANT_ORDER if v in data]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar([VARIANT_LABELS[v] for v in variants], [data[v] * 100 for v in variants], color=[VARIANT_COLORS[v] for v in variants])
    ax.set_ylabel("Taxa de reconhecimento de contrapontos (%)")
    ax.set_title("Reconhecimento de contrapontos por variante")
    ax.set_ylim(0, 100)
    _save(fig, out_dir, "04_reconhecimento_contrapontos")
    _write_csv(out_dir, "04_reconhecimento_contrapontos", [{"variante": VARIANT_LABELS[v], "taxa_pct": data[v] * 100} for v in variants])


def plot_neutral_intrusion(summary: dict[str, Any], out_dir: Path) -> None:
    plt = _setup_matplotlib()
    data = summary.get("neutral_intrusion_rate", {})
    variants = [v for v in VARIANT_ORDER if v in data]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar([VARIANT_LABELS[v] for v in variants], [data[v] * 100 for v in variants], color=[VARIANT_COLORS[v] for v in variants])
    ax.set_ylabel("Taxa de invasão ideológica (%)")
    ax.set_title("Invasão ideológica em tarefas neutras")
    ax.set_ylim(0, 100)
    _save(fig, out_dir, "05_invasao_ideologica_neutra")
    _write_csv(out_dir, "05_invasao_ideologica_neutra", [{"variante": VARIANT_LABELS[v], "taxa_pct": data[v] * 100} for v in variants])


def plot_sample_variability(evaluations: list[dict[str, Any]], out_dir: Path) -> None:
    plt = _setup_matplotlib()
    import statistics as st

    by_prompt_variant: dict[tuple[str, str], list[float]] = {}
    for rec in evaluations:
        score = rec.get("scores", {}).get("political_orientation")
        if score is None:
            continue
        key = (rec["prompt_id"], rec["model_variant"])
        by_prompt_variant.setdefault(key, []).append(float(score))

    by_variant_std: dict[str, list[float]] = {}
    for (prompt_id, variant), values in by_prompt_variant.items():
        if len(values) > 1:
            by_variant_std.setdefault(variant, []).append(st.stdev(values))

    variants = [v for v in VARIANT_ORDER if v in by_variant_std]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    means = [sum(by_variant_std[v]) / len(by_variant_std[v]) for v in variants]
    ax.bar([VARIANT_LABELS[v] for v in variants], means, color=[VARIANT_COLORS[v] for v in variants])
    ax.set_ylabel("Desvio-padrão médio entre amostras do mesmo prompt")
    ax.set_title("Variabilidade entre gerações amostrais")
    _save(fig, out_dir, "06_variabilidade_amostras")
    _write_csv(out_dir, "06_variabilidade_amostras", [{"variante": VARIANT_LABELS[v], "std_medio": m} for v, m in zip(variants, means)])


def plot_difference_matrix(summary: dict[str, Any], out_dir: Path) -> None:
    plt = _setup_matplotlib()
    import numpy as np

    pairs = {
        ("base", "progressive"): summary.get("diff_base_vs_progressive", {}).get("mean_diff"),
        ("base", "conservative"): summary.get("diff_base_vs_conservative", {}).get("mean_diff"),
        ("progressive", "conservative"): summary.get("diff_progressive_vs_conservative", {}).get("mean_diff"),
    }
    matrix = np.zeros((3, 3))
    for i, a in enumerate(VARIANT_ORDER):
        for j, b in enumerate(VARIANT_ORDER):
            if a == b:
                matrix[i, j] = 0
            elif (a, b) in pairs:
                matrix[i, j] = pairs[(a, b)] or 0
            elif (b, a) in pairs:
                matrix[i, j] = -(pairs[(b, a)] or 0)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels([VARIANT_LABELS[v] for v in VARIANT_ORDER])
    ax.set_yticklabels([VARIANT_LABELS[v] for v in VARIANT_ORDER])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="black")
    ax.set_title("Matriz de diferenças de orientação entre variantes")
    fig.colorbar(im, ax=ax, label="Diferença média (linha - coluna)")
    _save(fig, out_dir, "07_matriz_diferencas")
    rows = [{"linha": VARIANT_LABELS[VARIANT_ORDER[i]], "coluna": VARIANT_LABELS[VARIANT_ORDER[j]], "diferenca": matrix[i, j]} for i in range(3) for j in range(3)]
    _write_csv(out_dir, "07_matriz_diferencas", rows)


def plot_training_loss(log_history_progressive: list[dict[str, Any]], log_history_conservative: list[dict[str, Any]], out_dir: Path) -> None:
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rows = []
    for label, history, color in [
        ("Progressista (treino)", log_history_progressive, VARIANT_COLORS["progressive"]),
        ("Conservador (treino)", log_history_conservative, VARIANT_COLORS["conservative"]),
    ]:
        steps = [h["step"] for h in history if "loss" in h]
        losses = [h["loss"] for h in history if "loss" in h]
        ax.plot(steps, losses, label=label, color=color)
        rows += [{"serie": label, "step": s, "loss": l} for s, l in zip(steps, losses)]
    for label, history, color in [
        ("Progressista (validação)", log_history_progressive, VARIANT_COLORS["progressive"]),
        ("Conservador (validação)", log_history_conservative, VARIANT_COLORS["conservative"]),
    ]:
        steps = [h["step"] for h in history if "eval_loss" in h]
        losses = [h["eval_loss"] for h in history if "eval_loss" in h]
        ax.plot(steps, losses, label=label, color=color, linestyle="--")
        rows += [{"serie": label, "step": s, "loss": l} for s, l in zip(steps, losses)]
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Loss de treinamento e validação por adapter")
    ax.legend(fontsize=8)
    _save(fig, out_dir, "08_loss_treinamento")
    _write_csv(out_dir, "08_loss_treinamento", rows)


def plot_effect_sizes(summary: dict[str, Any], out_dir: Path) -> None:
    plt = _setup_matplotlib()
    comparisons = [
        ("Base vs. Progressista", summary.get("diff_base_vs_progressive", {})),
        ("Base vs. Conservador", summary.get("diff_base_vs_conservative", {})),
        ("Progressista vs. Conservador", summary.get("diff_progressive_vs_conservative", {})),
    ]
    labels = [c[0] for c in comparisons]
    means = [c[1].get("mean_diff", float("nan")) for c in comparisons]
    los = [c[1].get("ci_low", float("nan")) for c in comparisons]
    his = [c[1].get("ci_high", float("nan")) for c in comparisons]
    errs = [[m - l for m, l in zip(means, los)], [h - m for h, m in zip(his, means)]]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    y = range(len(labels))
    ax.errorbar(means, y, xerr=errs, fmt="o", capsize=5, color="#2563eb")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Diferença média de orientação (IC 95% via bootstrap)")
    ax.set_title("Tamanho do efeito com intervalos de confiança")
    _save(fig, out_dir, "09_tamanho_efeito_ic")
    rows = [{"comparacao": c[0], "diferenca_media": c[1].get("mean_diff"), "ci_low": c[1].get("ci_low"), "ci_high": c[1].get("ci_high"), "cohens_d": c[1].get("effect_size_cohens_d")} for c in comparisons]
    _write_csv(out_dir, "09_tamanho_efeito_ic", rows)


def generate_all_plots(
    summary_path: str | Path,
    evaluations_path: str | Path,
    out_dir: str | Path,
    log_history_progressive_path: str | Path | None = None,
    log_history_conservative_path: str | Path | None = None,
) -> None:
    """Gera os 9 gráficos + tabelas CSV a partir dos artefatos de avaliação já produzidos."""
    out_dir = Path(out_dir)
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    evaluations = read_jsonl(evaluations_path)

    plot_orientation_by_variant(summary, out_dir)
    plot_orientation_by_topic(summary, out_dir)
    plot_score_distribution(evaluations, out_dir)
    plot_counterargument_recognition(summary, out_dir)
    plot_neutral_intrusion(summary, out_dir)
    plot_sample_variability(evaluations, out_dir)
    plot_difference_matrix(summary, out_dir)

    def _load_log_history(path: str | Path | None) -> list[dict[str, Any]]:
        if not path or not Path(path).exists():
            return []
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        return state.get("log_history", [])  # trainer_state.json do HF Trainer

    log_prog = _load_log_history(log_history_progressive_path)
    log_cons = _load_log_history(log_history_conservative_path)
    plot_training_loss(log_prog, log_cons, out_dir)

    plot_effect_sizes(summary, out_dir)
    logger.info("9 gráficos (+CSVs) gerados em %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera os gráficos comparativos do experimento.")
    parser.add_argument("--summary", default="outputs/evaluations/statistical_summary.json")
    parser.add_argument("--evaluations", default="outputs/evaluations/heuristic_evaluations.jsonl")
    parser.add_argument("--out-dir", default="outputs/figures")
    parser.add_argument("--log-history-progressive", default="outputs/adapters/adapter_progressive/trainer_state.json")
    parser.add_argument("--log-history-conservative", default="outputs/adapters/adapter_conservative/trainer_state.json")
    args = parser.parse_args()
    generate_all_plots(args.summary, args.evaluations, args.out_dir, args.log_history_progressive, args.log_history_conservative)


if __name__ == "__main__":
    main()
