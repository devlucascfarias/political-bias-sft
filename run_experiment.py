#!/usr/bin/env python
"""CLI orquestradora do experimento political-bias-sft (equivalente não-notebook).

Cada etapa também pode ser executada isoladamente via `python -m src.<modulo>`.
Este script apenas encadeia as etapas com uma interface única.

Exemplos:
    python run_experiment.py validate
    python run_experiment.py train --orientation progressive --smoke-test
    python run_experiment.py train --orientation conservative --smoke-test
    python run_experiment.py infer --mode deterministic
    python run_experiment.py evaluate --responses outputs/responses/responses_deterministic.jsonl
    python run_experiment.py blind-build --responses outputs/responses/responses_deterministic.jsonl
    python run_experiment.py plot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import get_logger  # noqa: E402

logger = get_logger("run_experiment")


def cmd_validate(args: argparse.Namespace) -> None:
    from src.dataset_validator import validate_all

    report = validate_all(args.data_dir)
    out_json = Path(args.out_dir) / "dataset_validation_report.json"
    out_md = Path(args.out_dir) / "dataset_validation_report.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(report.to_markdown(), encoding="utf-8")
    logger.info("Validação: %d erro(s), %d aviso(s)", report.n_errors, report.n_warnings)
    if not report.passed:
        sys.exit(1)


def cmd_train(args: argparse.Namespace) -> None:
    from src.train import train_adapter

    config_path = f"configs/{args.orientation}.yaml"
    result = train_adapter(config_path, smoke_test=args.smoke_test, resume_from_checkpoint=args.resume_from_checkpoint)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_infer(args: argparse.Namespace) -> None:
    from src.config import load_config
    from src.inference import load_eval_prompts, run_comparative_inference
    from src.utils import write_jsonl

    cfg = load_config(args.config)
    prompts = load_eval_prompts(args.data_dir)
    adapter_dirs = {"progressive": args.adapter_progressive, "conservative": args.adapter_conservative}
    results = run_comparative_inference(cfg, prompts, adapter_dirs, mode=args.mode, dataset_version=args.dataset_version)
    out_path = Path(args.out or f"outputs/responses/responses_{args.mode}.jsonl")
    write_jsonl(out_path, results)
    logger.info("Inferência: %d registros -> %s", len(results), out_path)


def cmd_evaluate(args: argparse.Namespace) -> None:
    from src.evaluate import build_summary, run_heuristic_evaluation
    from src.utils import read_jsonl, write_jsonl

    responses = read_jsonl(args.responses)
    evaluated = run_heuristic_evaluation(responses)
    write_jsonl(args.out_evaluations, evaluated)
    summary = build_summary(evaluated)
    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Avaliação heurística: %d respostas -> %s", len(evaluated), args.out_evaluations)


def cmd_blind_build(args: argparse.Namespace) -> None:
    from src.blind_review import build_blind_package
    from src.utils import read_jsonl

    responses = read_jsonl(args.responses)
    build_blind_package(responses, seed=args.seed, out_public=args.out_public, out_private_map=args.out_private_map)


def cmd_plot(args: argparse.Namespace) -> None:
    from src.plotting import generate_all_plots

    generate_all_plots(
        args.summary,
        args.evaluations,
        args.out_dir,
        args.log_history_progressive,
        args.log_history_conservative,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Orquestrador do experimento political-bias-sft.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate", help="Valida os datasets.")
    p_validate.add_argument("--data-dir", default="data")
    p_validate.add_argument("--out-dir", default="outputs/evaluations")
    p_validate.set_defaults(func=cmd_validate)

    p_train = sub.add_parser("train", help="Treina um adapter LoRA (requer GPU).")
    p_train.add_argument("--orientation", choices=["progressive", "conservative"], required=True)
    p_train.add_argument("--smoke-test", action="store_true")
    p_train.add_argument("--resume-from-checkpoint", default=None)
    p_train.set_defaults(func=cmd_train)

    p_infer = sub.add_parser("infer", help="Inferência comparativa base/progressive/conservative (requer GPU).")
    p_infer.add_argument("--config", default="configs/base.yaml")
    p_infer.add_argument("--data-dir", default="data")
    p_infer.add_argument("--adapter-progressive", default="outputs/adapters/adapter_progressive")
    p_infer.add_argument("--adapter-conservative", default="outputs/adapters/adapter_conservative")
    p_infer.add_argument("--mode", choices=["deterministic", "sampling"], default="deterministic")
    p_infer.add_argument("--dataset-version", default="v1")
    p_infer.add_argument("--out", default=None)
    p_infer.set_defaults(func=cmd_infer)

    p_eval = sub.add_parser("evaluate", help="Avaliação heurística + resumo estatístico.")
    p_eval.add_argument("--responses", required=True)
    p_eval.add_argument("--out-evaluations", default="outputs/evaluations/heuristic_evaluations.jsonl")
    p_eval.add_argument("--out-summary", default="outputs/evaluations/statistical_summary.json")
    p_eval.set_defaults(func=cmd_evaluate)

    p_blind = sub.add_parser("blind-build", help="Gera o pacote de avaliação humana cega.")
    p_blind.add_argument("--responses", required=True)
    p_blind.add_argument("--seed", type=int, default=42)
    p_blind.add_argument("--out-public", default="outputs/evaluations/blind_review_package.csv")
    p_blind.add_argument("--out-private-map", default="outputs/evaluations/blind_review_private_map.jsonl")
    p_blind.set_defaults(func=cmd_blind_build)

    p_plot = sub.add_parser("plot", help="Gera os 9 gráficos comparativos + CSVs.")
    p_plot.add_argument("--summary", default="outputs/evaluations/statistical_summary.json")
    p_plot.add_argument("--evaluations", default="outputs/evaluations/heuristic_evaluations.jsonl")
    p_plot.add_argument("--out-dir", default="outputs/figures")
    p_plot.add_argument("--log-history-progressive", default="outputs/adapters/adapter_progressive/trainer_state.json")
    p_plot.add_argument("--log-history-conservative", default="outputs/adapters/adapter_conservative/trainer_state.json")
    p_plot.set_defaults(func=cmd_plot)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
