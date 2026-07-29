"""Carregamento e validação de configuração YAML para o experimento political-bias-sft.

`progressive.yaml` e `conservative.yaml` usam `extends: base.yaml` para herdar
os blocos `model`, `lora` e `training`. Isso garante, por construção, que as
duas execuções de treinamento usem exatamente os mesmos hiperparâmetros — a
única diferença permitida é o conteúdo apontado por `data.*`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


class ConfigError(Exception):
    """Erro de configuração (arquivo ausente, campo obrigatório faltando etc.)."""


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Mescla `override` sobre `base` recursivamente, sem mutar os originais."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Arquivo de configuração não encontrado: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Configuração inválida (esperado mapeamento) em {path}")
    return data


def load_raw_config(config_path: str | Path) -> dict[str, Any]:
    """Carrega um YAML de config resolvendo `extends` (uma única cadeia, sem ciclos)."""
    path = Path(config_path)
    if not path.is_absolute():
        candidate = CONFIG_DIR / path.name
        path = candidate if candidate.exists() else path
    data = _load_yaml(path)
    parent_name = data.get("extends")
    if parent_name:
        parent_path = path.parent / parent_name
        parent_data = _load_yaml(parent_path)
        data = _deep_merge(parent_data, data)
    return data


@dataclass
class ModelConfig:
    base_model_id: str
    base_model_id_prequantized: str
    use_prequantized: bool
    max_seq_length: int
    load_in_4bit: bool
    dtype: str | None
    chat_template: str
    train_on_responses_only: bool

    def resolved_model_id(self) -> str:
        return self.base_model_id_prequantized if self.use_prequantized else self.base_model_id


@dataclass
class LoraConfig:
    r: int
    alpha: int
    dropout: float
    bias: str
    target_modules: list[str]
    use_gradient_checkpointing: str | bool
    use_rslora: bool


@dataclass
class TrainingConfig:
    epochs: int
    learning_rate: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    warmup_ratio: float
    weight_decay: float
    logging_steps: int
    eval_strategy: str
    eval_steps: int
    save_steps: int
    save_total_limit: int
    seed: int
    bf16: str | bool
    fp16: str | bool
    optim: str
    lr_scheduler_type: str
    max_grad_norm: float
    packing: bool
    group_by_length: bool
    report_to: str
    max_steps: int | None = None  # sobrescrito em smoke_test


@dataclass
class DataSizes:
    train_pairs: int
    validation_pairs: int
    eval_political_prompts: int
    eval_adversarial_prompts: int
    eval_neutral_prompts: int


@dataclass
class ExperimentConfig:
    """Configuração totalmente resolvida para uma orientação (progressive/conservative)."""

    name: str
    orientation: str
    seed: int
    drive_root: str
    model: ModelConfig
    lora: LoraConfig
    training: TrainingConfig
    data_sizes: DataSizes
    validation: dict[str, Any]
    generation: dict[str, Any]
    paths: dict[str, str]
    data: dict[str, str]
    output: dict[str, str]
    smoke_test_enabled: bool = False
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def train_file(self) -> Path:
        return Path(self.data["train_file"])

    def validation_file(self) -> Path:
        return Path(self.data["validation_file"])

    def adapter_dir(self) -> Path:
        return Path(self.output["adapter_dir"])


def load_config(config_path: str | Path, smoke_test: bool = False) -> ExperimentConfig:
    """Carrega uma config de orientação (progressive.yaml/conservative.yaml) já resolvida.

    Args:
        config_path: caminho ou nome de arquivo dentro de `configs/`.
        smoke_test: se True, aplica os overrides de `smoke_test` (menos dados/steps).
    """
    raw = load_raw_config(config_path)

    required_top = ["experiment", "model", "lora", "training", "data_sizes", "orientation", "data", "output"]
    missing = [k for k in required_top if k not in raw]
    if missing:
        raise ConfigError(f"Campos obrigatórios ausentes em {config_path}: {missing}")

    data_sizes_raw = dict(raw["data_sizes"])
    training_raw = dict(raw["training"])
    max_steps = None

    if smoke_test:
        st = raw.get("smoke_test", {})
        data_sizes_raw.update(
            {
                "train_pairs": st.get("train_pairs", data_sizes_raw["train_pairs"]),
                "validation_pairs": st.get("validation_pairs", data_sizes_raw["validation_pairs"]),
                "eval_political_prompts": st.get("eval_political_prompts", data_sizes_raw["eval_political_prompts"]),
                "eval_adversarial_prompts": st.get(
                    "eval_adversarial_prompts", data_sizes_raw["eval_adversarial_prompts"]
                ),
                "eval_neutral_prompts": st.get("eval_neutral_prompts", data_sizes_raw["eval_neutral_prompts"]),
            }
        )
        training_raw["eval_steps"] = st.get("eval_steps", training_raw["eval_steps"])
        training_raw["save_steps"] = st.get("save_steps", training_raw["save_steps"])
        training_raw["logging_steps"] = st.get("logging_steps", training_raw["logging_steps"])
        max_steps = st.get("max_steps")

    model_cfg = ModelConfig(**raw["model"])
    lora_cfg = LoraConfig(**raw["lora"])
    training_cfg = TrainingConfig(**training_raw, max_steps=max_steps)
    data_sizes_cfg = DataSizes(**data_sizes_raw)

    return ExperimentConfig(
        name=raw["experiment"]["name"],
        orientation=raw["orientation"],
        seed=raw["experiment"]["seed"],
        drive_root=raw["experiment"]["drive_root"],
        model=model_cfg,
        lora=lora_cfg,
        training=training_cfg,
        data_sizes=data_sizes_cfg,
        validation=raw.get("validation", {}),
        generation=raw.get("generation", {}),
        paths=raw.get("paths", {}),
        data=raw["data"],
        output=raw["output"],
        smoke_test_enabled=smoke_test,
        raw=raw,
    )


def assert_matching_hyperparameters(cfg_a: ExperimentConfig, cfg_b: ExperimentConfig) -> None:
    """Garante que duas configs de orientação diferem apenas em `data`/`output`/`orientation`.

    Usado como salvaguarda antes de iniciar os dois treinamentos, para impedir
    que uma edição futura quebre o controle experimental.
    """
    if cfg_a.model != cfg_b.model:
        raise ConfigError("Configurações de modelo divergem entre as orientações.")
    if cfg_a.lora != cfg_b.lora:
        raise ConfigError("Configurações de LoRA divergem entre as orientações.")
    a_training = {k: v for k, v in vars(cfg_a.training).items()}
    b_training = {k: v for k, v in vars(cfg_b.training).items()}
    if a_training != b_training:
        raise ConfigError("Hiperparâmetros de treinamento divergem entre as orientações.")
    if cfg_a.seed != cfg_b.seed:
        raise ConfigError("Seeds divergem entre as orientações.")
