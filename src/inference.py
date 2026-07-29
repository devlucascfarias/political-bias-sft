"""Inferência comparativa: modelo base vs. adapter progressive vs. adapter conservative.

Garante que as três variantes recebem exatamente o mesmo system prompt, chat
template, generation_config e (no modo amostral) o mesmo conjunto de seeds —
a única variável entre execuções é qual adapter (se algum) está carregado.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

from .config import ExperimentConfig, load_config
from .utils import clear_gpu_memory, get_logger, read_jsonl, write_jsonl

logger = get_logger(__name__)

ModelVariant = Literal["base", "progressive", "conservative"]


@dataclass
class GenerationSettings:
    max_new_tokens: int
    top_p: float
    top_k: int
    repetition_penalty: float
    system_prompt: str

    @classmethod
    def from_config(cls, cfg: ExperimentConfig) -> "GenerationSettings":
        g = cfg.generation
        return cls(
            max_new_tokens=g.get("max_new_tokens", 400),
            top_p=g.get("top_p", 0.9),
            top_k=g.get("top_k", 50),
            repetition_penalty=g.get("repetition_penalty", 1.1),
            system_prompt=g.get("system_prompt", "Responda de forma clara, respeitosa e argumentativa."),
        )


def load_variant_model(cfg: ExperimentConfig, variant: ModelVariant, adapter_dirs: dict[str, str]):
    """Carrega o modelo base uma única vez e, se `variant` != base, aplica o adapter correspondente."""
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template

    model, tokenizer = FastModel.from_pretrained(
        model_name=cfg.model.resolved_model_id(),
        max_seq_length=cfg.model.max_seq_length,
        load_in_4bit=cfg.model.load_in_4bit,
        dtype=None,
    )
    tokenizer = get_chat_template(tokenizer, chat_template=cfg.model.chat_template)

    if variant != "base":
        adapter_path = adapter_dirs.get(variant)
        if not adapter_path:
            raise ValueError(f"Caminho do adapter para '{variant}' não informado.")
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)

    from unsloth import FastModel as _FM

    _FM.for_inference(model)
    return model, tokenizer


def _text_chat_tokenizer(tokenizer):
    """Retorna o tokenizer textual quando Unsloth fornece um processor multimodal.

    Modelos Gemma 3 podem ser carregados como ``ProcessorMixin`` mesmo em uma
    execução somente de texto. Nesse caso, ``processor.apply_chat_template``
    tenta inspecionar cada ``content`` como uma lista de blocos multimodais e
    falha com ``TypeError: string indices must be integers`` para mensagens
    textuais comuns. O tokenizer interno aceita diretamente o schema padrão
    ``[{"role": ..., "content": "..."}]`` usado neste experimento.
    """
    inner_tokenizer = getattr(tokenizer, "tokenizer", None)
    if inner_tokenizer is not None and hasattr(inner_tokenizer, "apply_chat_template"):
        return inner_tokenizer
    return tokenizer


def generate_one(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    settings: GenerationSettings,
    temperature: float,
    seed: int,
    deterministic: bool,
) -> str:
    """Gera uma única resposta a partir de uma lista de mensagens (system+user)."""
    import torch

    torch.manual_seed(seed)

    chat_tokenizer = _text_chat_tokenizer(tokenizer)
    inputs = chat_tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    gen_kwargs: dict[str, Any] = dict(
        max_new_tokens=settings.max_new_tokens,
        repetition_penalty=settings.repetition_penalty,
        do_sample=not deterministic,
    )
    if deterministic:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs.update(temperature=temperature, top_p=settings.top_p, top_k=settings.top_k)

    output = model.generate(input_ids=inputs, **gen_kwargs)
    new_tokens = output[0][inputs.shape[-1] :]
    return chat_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_comparative_inference(
    cfg: ExperimentConfig,
    prompts: list[dict[str, Any]],
    adapter_dirs: dict[str, str],
    mode: Literal["deterministic", "sampling"] = "deterministic",
    checkpoint_label: str = "unknown",
    dataset_version: str = "unknown",
) -> list[dict[str, Any]]:
    """Executa inferência para todas as variantes x prompts, retornando registros no schema padronizado."""
    settings = GenerationSettings.from_config(cfg)
    num_samples = cfg.generation.get("num_samples", 5) if mode == "sampling" else 1
    base_seed = cfg.generation.get("base_seed", 42)
    temperature = 0.0 if mode == "deterministic" else cfg.generation.get("temperature_sampling", 0.7)

    results: list[dict[str, Any]] = []
    for variant in ("base", "progressive", "conservative"):
        logger.info("Carregando variante: %s", variant)
        model, tokenizer = load_variant_model(cfg, variant, adapter_dirs)

        for prompt_rec in prompts:
            messages = [
                {"role": "system", "content": settings.system_prompt},
                {"role": "user", "content": prompt_rec["prompt"]},
            ]
            for sample_index in range(num_samples):
                seed = base_seed + sample_index
                response = generate_one(
                    model,
                    tokenizer,
                    messages,
                    settings,
                    temperature=temperature,
                    seed=seed,
                    deterministic=(mode == "deterministic"),
                )
                results.append(
                    {
                        "prompt_id": prompt_rec["id"],
                        "model_variant": variant,
                        "adapter": None if variant == "base" else adapter_dirs.get(variant),
                        "sample_index": sample_index,
                        "seed": seed,
                        "generation_config": {
                            "mode": mode,
                            "temperature": temperature,
                            "top_p": settings.top_p,
                            "top_k": settings.top_k,
                            "repetition_penalty": settings.repetition_penalty,
                            "max_new_tokens": settings.max_new_tokens,
                        },
                        "prompt": prompt_rec["prompt"],
                        "response": response,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "checkpoint": checkpoint_label,
                        "dataset_version": dataset_version,
                        "category": prompt_rec.get("category"),
                        "topic": prompt_rec.get("topic"),
                    }
                )

        del model, tokenizer
        clear_gpu_memory()

    return results


def load_eval_prompts(data_dir: str | Path) -> list[dict[str, Any]]:
    """Carrega e unifica os três grupos de prompts de avaliação, com `category` anotada."""
    data_dir = Path(data_dir)
    prompts: list[dict[str, Any]] = []
    for filename, category in [
        ("political_prompts.jsonl", "political"),
        ("adversarial_prompts.jsonl", "adversarial"),
        ("neutral_prompts.jsonl", "neutral"),
    ]:
        path = data_dir / "evaluation" / filename
        for rec in read_jsonl(path):
            rec.setdefault("category", category)
            prompts.append(rec)
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Inferência comparativa base/progressive/conservative.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--adapter-progressive", default="outputs/adapters/adapter_progressive")
    parser.add_argument("--adapter-conservative", default="outputs/adapters/adapter_conservative")
    parser.add_argument("--mode", choices=["deterministic", "sampling"], default="deterministic")
    parser.add_argument("--out", default=None)
    parser.add_argument("--dataset-version", default="v1")
    args = parser.parse_args()

    cfg = load_config(args.config)
    prompts = load_eval_prompts(args.data_dir)
    adapter_dirs = {"progressive": args.adapter_progressive, "conservative": args.adapter_conservative}

    results = run_comparative_inference(
        cfg, prompts, adapter_dirs, mode=args.mode, dataset_version=args.dataset_version
    )

    out_path = Path(args.out or f"outputs/responses/responses_{args.mode}.jsonl")
    write_jsonl(out_path, results)
    logger.info("Salvos %d registros em %s", len(results), out_path)


if __name__ == "__main__":
    main()
