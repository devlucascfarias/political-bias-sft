"""Utilidades compartilhadas: seeds, logging, I/O de JSONL, hashing, info de GPU/memória."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Retorna um logger configurado de forma idempotente (evita handlers duplicados)."""
    if name in _LOGGERS:
        return _LOGGERS[name]
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
    logger.propagate = False
    _LOGGERS[name] = logger
    return logger


def set_global_seed(seed: int) -> None:
    """Fixa seeds de Python, NumPy e (se disponível) PyTorch para reprodutibilidade."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Lê um arquivo JSONL retornando a lista de objetos. Levanta erro com número da linha."""
    path = Path(path)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: JSON inválido ({exc})") from exc
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    """Escreve uma lista de objetos como JSONL, criando diretórios pais se necessário."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def sha256_file(path: str | Path) -> str:
    """Calcula o hash SHA-256 de um arquivo (usado no manifesto de reprodutibilidade)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    """Context manager que mede duração em segundos: `with timer() as t: ...; t["seconds"]`."""
    state: dict[str, float] = {"start": time.time(), "seconds": 0.0}
    try:
        yield state
    finally:
        state["seconds"] = time.time() - state["start"]


def gpu_info() -> dict[str, Any]:
    """Coleta informações da GPU disponível (nome, memória total/livre, driver, CUDA)."""
    info: dict[str, Any] = {"available": False}
    try:
        import torch

        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
            info.update(
                {
                    "available": True,
                    "name": props.name,
                    "total_memory_gb": round(total_bytes / (1024**3), 2),
                    "free_memory_gb": round(free_bytes / (1024**3), 2),
                    "cuda_version": torch.version.cuda,
                    "torch_version": torch.__version__,
                    "bf16_supported": torch.cuda.is_bf16_supported(),
                }
            )
    except ImportError:
        info["error"] = "torch não instalado"
    return info


def clear_gpu_memory() -> None:
    """Libera cache de memória da GPU entre treinamentos/inferências consecutivas."""
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass


def detect_bf16_support() -> bool:
    """Detecta suporte a bfloat16 na GPU atual; retorna False se não houver GPU/torch."""
    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    except ImportError:
        return False


def environment_summary() -> dict[str, Any]:
    """Resumo do ambiente para o experiment_manifest.json (versões, GPU, plataforma)."""
    import platform

    summary: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "gpu": gpu_info(),
    }
    for pkg in ["torch", "transformers", "trl", "peft", "datasets", "bitsandbytes", "unsloth"]:
        try:
            module = __import__(pkg)
            summary[f"{pkg}_version"] = getattr(module, "__version__", "unknown")
        except ImportError:
            summary[f"{pkg}_version"] = None
    return summary
