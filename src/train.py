"""Treinamento QLoRA (4-bit) de um adapter LoRA sobre o Gemma 3 4B-it via Unsloth + TRL.

Este script é usado duas vezes — uma para `configs/progressive.yaml` e outra
para `configs/conservative.yaml` — com hiperparâmetros IDÊNTICOS (garantido
por `config.assert_matching_hyperparameters`). A única diferença entre as
duas execuções é o arquivo de dados.

Todas as importações pesadas (unsloth, torch, trl) são feitas dentro das
funções para permitir que o restante do projeto (config, validação, testes)
rode em máquinas sem GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import ExperimentConfig, load_config
from .utils import (
    clear_gpu_memory,
    detect_bf16_support,
    environment_summary,
    get_logger,
    gpu_info,
    set_global_seed,
    sha256_file,
    timer,
)

logger = get_logger(__name__)


def load_base_model(cfg: ExperimentConfig):
    """Carrega o modelo base quantizado em 4-bit e o tokenizer via Unsloth FastModel."""
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template

    model, tokenizer = FastModel.from_pretrained(
        model_name=cfg.model.resolved_model_id(),
        max_seq_length=cfg.model.max_seq_length,
        load_in_4bit=cfg.model.load_in_4bit,
        dtype=None,
    )
    tokenizer = get_chat_template(tokenizer, chat_template=cfg.model.chat_template)
    return model, tokenizer


def attach_lora(model, cfg: ExperimentConfig):
    """Anexa adapters LoRA treináveis ao modelo base congelado (QLoRA)."""
    from unsloth import FastModel

    model = FastModel.get_peft_model(
        model,
        r=cfg.lora.r,
        target_modules=cfg.lora.target_modules,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        bias=cfg.lora.bias,
        use_gradient_checkpointing=cfg.lora.use_gradient_checkpointing,
        random_state=cfg.seed,
        use_rslora=cfg.lora.use_rslora,
    )
    return model


def build_training_args(cfg: ExperimentConfig, output_dir: Path):
    """Constrói `SFTConfig` a partir da config resolvida, com detecção automática de bf16/fp16.

    Os kwargs são filtrados pela assinatura real de `SFTConfig` instalada, pois
    o TRL renomeia/remove parâmetros entre versões (ex.: `group_by_length` foi
    removido em versões recentes) — assim o treino não quebra por causa de um
    parâmetro não essencial que mudou de nome.
    """
    import inspect

    from trl import SFTConfig

    bf16 = detect_bf16_support() if cfg.training.bf16 == "auto" else bool(cfg.training.bf16)
    fp16 = (not bf16) if cfg.training.fp16 == "auto" else bool(cfg.training.fp16)

    kwargs: dict[str, Any] = dict(
        output_dir=str(output_dir),
        num_train_epochs=cfg.training.epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        warmup_ratio=cfg.training.warmup_ratio,
        weight_decay=cfg.training.weight_decay,
        learning_rate=cfg.training.learning_rate,
        logging_steps=cfg.training.logging_steps,
        eval_strategy=cfg.training.eval_strategy,
        eval_steps=cfg.training.eval_steps,
        save_strategy="steps",
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        seed=cfg.training.seed,
        bf16=bf16,
        fp16=fp16,
        optim=cfg.training.optim,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        max_grad_norm=cfg.training.max_grad_norm,
        group_by_length=cfg.training.group_by_length,
        report_to=cfg.training.report_to,
        dataset_text_field="text",
        max_seq_length=cfg.model.max_seq_length,
        packing=cfg.training.packing,
    )
    if cfg.training.max_steps:
        kwargs["max_steps"] = cfg.training.max_steps
        kwargs.pop("num_train_epochs", None)

    accepted_params = set(inspect.signature(SFTConfig.__init__).parameters)
    dropped = {k: v for k, v in kwargs.items() if k not in accepted_params}
    if dropped:
        logger.warning(
            "SFTConfig instalado não aceita os parâmetros %s (versão de TRL diferente da esperada); ignorando.",
            sorted(dropped),
        )
    kwargs = {k: v for k, v in kwargs.items() if k in accepted_params}

    return SFTConfig(**kwargs)


def build_trainer(model, tokenizer, train_dataset, eval_dataset, cfg: ExperimentConfig, output_dir: Path):
    """Monta o SFTTrainer e, se configurado, restringe a loss aos tokens de resposta do assistente."""
    from trl import SFTTrainer

    args = build_training_args(cfg, output_dir)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=args,
    )

    if cfg.model.train_on_responses_only:
        from unsloth.chat_templates import train_on_responses_only

        trainer = train_on_responses_only(
            trainer,
            instruction_part="<start_of_turn>user\n",
            response_part="<start_of_turn>model\n",
        )
    return trainer


def train_adapter(config_path: str | Path, smoke_test: bool = False, resume_from_checkpoint: str | None = None) -> dict[str, Any]:
    """Executa o pipeline completo de treinamento de um adapter e retorna metadados para o manifesto."""
    cfg = load_config(config_path, smoke_test=smoke_test)
    set_global_seed(cfg.seed)

    from .dataset_builder import load_and_build

    output_dir = cfg.adapter_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Carregando modelo base: %s", cfg.model.resolved_model_id())
    model, tokenizer = load_base_model(cfg)
    model = attach_lora(model, cfg)

    n_pairs = cfg.data_sizes.train_pairs if not smoke_test else None  # smoke_test já reduz via config
    n_val_pairs = cfg.data_sizes.validation_pairs if not smoke_test else None

    train_dataset = load_and_build(cfg.train_file(), tokenizer=tokenizer, n_pairs=n_pairs)
    eval_dataset = load_and_build(cfg.validation_file(), tokenizer=tokenizer, n_pairs=n_val_pairs)

    logger.info("Exemplos de treino: %d | validação: %d", len(train_dataset), len(eval_dataset))

    trainer = build_trainer(model, tokenizer, train_dataset, eval_dataset, cfg, output_dir)

    gpu_before = gpu_info()
    with timer() as t:
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    gpu_after = gpu_info()

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics_path = output_dir / "train_metrics.json"
    metrics_path.write_text(json.dumps(train_result.metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_entry = {
        "orientation": cfg.orientation,
        "config_path": str(config_path),
        "adapter_dir": str(output_dir),
        "model_id": cfg.model.resolved_model_id(),
        "seed": cfg.seed,
        "smoke_test": smoke_test,
        "n_train_examples": len(train_dataset),
        "n_validation_examples": len(eval_dataset),
        "duration_seconds": t["seconds"],
        "train_metrics": train_result.metrics,
        "gpu_before": gpu_before,
        "gpu_after_peak_free_gb": gpu_after.get("free_memory_gb"),
        "environment": environment_summary(),
    }
    if Path(cfg.train_file()).exists():
        manifest_entry["train_file_sha256"] = sha256_file(cfg.train_file())

    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest_entry, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Libera o modelo da GPU antes de treinar o próximo adapter.
    del trainer, model, tokenizer
    clear_gpu_memory()

    logger.info("Treinamento concluído em %.1fs. Adapter salvo em %s", t["seconds"], output_dir)
    return manifest_entry


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina um adapter LoRA (progressive ou conservative).")
    parser.add_argument("config", help="Caminho para configs/progressive.yaml ou configs/conservative.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    result = train_adapter(args.config, smoke_test=args.smoke_test, resume_from_checkpoint=args.resume_from_checkpoint)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
