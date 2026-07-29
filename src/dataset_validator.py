"""Validador dos datasets de treino/validação/avaliação.

Verifica estrutura, simetria entre pares ideológicos, qualidade textual básica
e ausência de vazamento entre splits. NÃO julga se uma resposta está
"politicamente correta" — isso é fora de escopo por design (ver README,
seção "Segurança metodológica").

Uso:
    python -m src.dataset_validator --data-dir data --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .utils import get_logger, read_jsonl, write_jsonl  # noqa: F401  (write_jsonl usado por chamadores externos)

logger = get_logger(__name__)

REQUIRED_FIELDS = {"id", "pair_id", "orientation", "topic", "messages"}
REQUIRED_ROLES_ORDER = ["system", "user", "assistant"]

# Lista mínima e deliberadamente conservadora de termos ofensivos em pt-BR.
# NÃO substitui revisão humana; serve apenas como rede de segurança grosseira.
DEFAULT_OFFENSIVE_TERMS = [
    "vagabundo",
    "vagabunda",
    "imbecil",
    "idiota",
    "retardado",
    "subumano",
    "subumanos",
    "extermin",
    "genocídio",
    "nazista",
    "terrorista",
]


@dataclass
class Issue:
    severity: str  # "error" | "warning"
    check: str
    message: str
    ids: list[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def error(self, check: str, message: str, ids: list[str] | None = None) -> None:
        self.issues.append(Issue("error", check, message, ids or []))

    def warning(self, check: str, message: str, ids: list[str] | None = None) -> None:
        self.issues.append(Issue("warning", check, message, ids or []))

    @property
    def n_errors(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def n_warnings(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    @property
    def passed(self) -> bool:
        return self.n_errors == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "n_errors": self.n_errors,
            "n_warnings": self.n_warnings,
            "stats": self.stats,
            "issues": [asdict(i) for i in self.issues],
        }

    def to_markdown(self) -> str:
        lines = ["# Relatório de validação dos datasets", ""]
        status = "✅ APROVADO" if self.passed else "❌ REPROVADO"
        lines.append(f"**Status:** {status}  ")
        lines.append(f"**Erros:** {self.n_errors}  **Avisos:** {self.n_warnings}")
        lines.append("")
        lines.append("## Estatísticas")
        for key, value in self.stats.items():
            lines.append(f"- **{key}**: {value}")
        lines.append("")
        if self.issues:
            lines.append("## Ocorrências")
            for issue in self.issues:
                icon = "🛑" if issue.severity == "error" else "⚠️"
                ids_str = f" (`{', '.join(issue.ids[:10])}`{'...' if len(issue.ids) > 10 else ''})" if issue.ids else ""
                lines.append(f"- {icon} **[{issue.check}]** {issue.message}{ids_str}")
        else:
            lines.append("Nenhuma ocorrência.")
        return "\n".join(lines)


def _assistant_text(record: dict[str, Any]) -> str:
    for msg in record.get("messages", []):
        if msg.get("role") == "assistant":
            return msg.get("content", "") or ""
    return ""


def _user_text(record: dict[str, Any]) -> str:
    for msg in record.get("messages", []):
        if msg.get("role") == "user":
            return msg.get("content", "") or ""
    return ""


def check_required_fields(records: list[dict[str, Any]], source: str, report: ValidationReport) -> None:
    for i, rec in enumerate(records):
        missing = REQUIRED_FIELDS - set(rec.keys())
        if missing:
            report.error("required_fields", f"{source}[{i}] faltam campos {sorted(missing)}", [rec.get("id", f"idx{i}")])


def check_roles(records: list[dict[str, Any]], source: str, report: ValidationReport) -> None:
    for rec in records:
        roles = [m.get("role") for m in rec.get("messages", [])]
        if set(REQUIRED_ROLES_ORDER) - set(roles):
            report.error(
                "roles",
                f"{source}: exemplo não contém todos os papéis obrigatórios {REQUIRED_ROLES_ORDER}",
                [rec.get("id", "?")],
            )
        # system deve vir antes de user, que deve vir antes de assistant
        role_order = [r for r in roles if r in REQUIRED_ROLES_ORDER]
        if role_order != sorted(role_order, key=REQUIRED_ROLES_ORDER.index):
            report.warning("roles_order", "ordem de papéis fora do padrão system->user->assistant", [rec.get("id", "?")])


def check_unique_ids(records: list[dict[str, Any]], source: str, report: ValidationReport) -> None:
    ids = [r.get("id") for r in records]
    counts = Counter(ids)
    dupes = [i for i, c in counts.items() if c > 1]
    if dupes:
        report.error("unique_ids", f"{source}: IDs duplicados", dupes)


def check_pair_symmetry(
    progressive: list[dict[str, Any]],
    conservative: list[dict[str, Any]],
    source: str,
    report: ValidationReport,
    max_length_diff_ratio: float,
) -> None:
    prog_by_pair = {r["pair_id"]: r for r in progressive if "pair_id" in r}
    cons_by_pair = {r["pair_id"]: r for r in conservative if "pair_id" in r}

    only_prog = set(prog_by_pair) - set(cons_by_pair)
    only_cons = set(cons_by_pair) - set(prog_by_pair)
    if only_prog:
        report.error("pair_symmetry", f"{source}: pair_id sem contraparte conservadora", sorted(only_prog))
    if only_cons:
        report.error("pair_symmetry", f"{source}: pair_id sem contraparte progressista", sorted(only_cons))

    mismatched_topics: list[str] = []
    length_outliers: list[str] = []
    for pair_id in set(prog_by_pair) & set(cons_by_pair):
        p, c = prog_by_pair[pair_id], cons_by_pair[pair_id]
        if p.get("topic") != c.get("topic"):
            mismatched_topics.append(pair_id)
        len_p, len_c = len(_assistant_text(p)), len(_assistant_text(c))
        if max(len_p, len_c) > 0:
            diff_ratio = abs(len_p - len_c) / max(len_p, len_c)
            if diff_ratio > max_length_diff_ratio:
                length_outliers.append(pair_id)

    if mismatched_topics:
        report.error("pair_topic_match", f"{source}: pares com topic divergente", mismatched_topics)
    if length_outliers:
        report.warning(
            "pair_length_balance",
            f"{source}: pares com diferença de comprimento acima de {max_length_diff_ratio:.0%}",
            length_outliers,
        )


def check_orientation_balance(progressive: list[dict[str, Any]], conservative: list[dict[str, Any]], source: str, report: ValidationReport) -> None:
    n_p, n_c = len(progressive), len(conservative)
    if n_p == 0 or n_c == 0:
        report.error("orientation_balance", f"{source}: uma das orientações está vazia (progressive={n_p}, conservative={n_c})")
        return
    ratio = abs(n_p - n_c) / max(n_p, n_c)
    if ratio > 0.05:
        report.warning("orientation_balance", f"{source}: desbalanceamento entre orientações (progressive={n_p}, conservative={n_c})")


def check_empty_and_length(
    records: list[dict[str, Any]], source: str, report: ValidationReport, min_chars: int, max_chars: int
) -> None:
    empty_ids, short_ids, long_ids = [], [], []
    for rec in records:
        text = _assistant_text(rec).strip()
        if not text:
            empty_ids.append(rec.get("id", "?"))
        elif len(text) < min_chars:
            short_ids.append(rec.get("id", "?"))
        elif len(text) > max_chars:
            long_ids.append(rec.get("id", "?"))
    if empty_ids:
        report.error("empty_response", f"{source}: respostas vazias", empty_ids)
    if short_ids:
        report.warning("short_response", f"{source}: respostas abaixo de {min_chars} caracteres", short_ids)
    if long_ids:
        report.warning("long_response", f"{source}: respostas acima de {max_chars} caracteres", long_ids)


def check_duplicates(records: list[dict[str, Any]], source: str, report: ValidationReport) -> None:
    seen: dict[str, str] = {}
    dupes = []
    for rec in records:
        text = _assistant_text(rec).strip().lower()
        if not text:
            continue
        if text in seen:
            dupes.append(rec.get("id", "?"))
        else:
            seen[text] = rec.get("id", "?")
    if dupes:
        report.warning("duplicates", f"{source}: respostas duplicadas (texto idêntico)", dupes)


def check_offensive_terms(records: list[dict[str, Any]], source: str, report: ValidationReport, extra_terms: list[str] | None = None) -> None:
    terms = list(DEFAULT_OFFENSIVE_TERMS) + list(extra_terms or [])
    flagged = []
    for rec in records:
        text = _assistant_text(rec).lower()
        if any(term in text for term in terms):
            flagged.append(rec.get("id", "?"))
    if flagged:
        report.error("offensive_terms", f"{source}: termos potencialmente ofensivos encontrados", flagged)


def check_topic_distribution(records: list[dict[str, Any]], source: str, report: ValidationReport) -> dict[str, int]:
    counts = Counter(r.get("topic", "unknown") for r in records)
    report.stats[f"{source}_topic_distribution"] = dict(sorted(counts.items()))
    if len(counts) > 1:
        values = list(counts.values())
        if max(values) > 3 * min(values):
            report.warning("topic_distribution", f"{source}: distribuição de tópicos muito desbalanceada {dict(counts)}")
    return dict(counts)


def _eval_prompt_text(record: dict[str, Any]) -> str:
    """Extrai o texto do prompt de um registro de avaliação (campo `prompt`, não `messages`)."""
    return record.get("prompt", "") or ""


def check_eval_leakage(train_records: list[dict[str, Any]], eval_records: list[dict[str, Any]], eval_source: str, report: ValidationReport) -> None:
    train_prompts = {_user_text(r).strip().lower() for r in train_records}
    leaked = [
        r.get("id", "?")
        for r in eval_records
        if _eval_prompt_text(r).strip() and _eval_prompt_text(r).strip().lower() in train_prompts
    ]
    if leaked:
        report.error("eval_leakage", f"{eval_source}: prompt(s) de avaliação também presentes no treino", leaked)


def check_split_leakage(splits: dict[str, list[dict[str, Any]]], report: ValidationReport) -> None:
    """Verifica sobreposição literal de texto de resposta entre train/validation/eval."""
    texts_by_split: dict[str, set[str]] = {
        name: {_assistant_text(r).strip().lower() for r in recs if _assistant_text(r).strip()} for name, recs in splits.items()
    }
    names = list(texts_by_split)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = texts_by_split[names[i]] & texts_by_split[names[j]]
            if overlap:
                report.error("split_leakage", f"vazamento literal de resposta entre '{names[i]}' e '{names[j]}' ({len(overlap)} ocorrência(s))")


def validate_all(data_dir: str | Path, max_length_diff_ratio: float = 0.35, min_chars: int = 60, max_chars: int = 4000, extra_offensive_terms: list[str] | None = None) -> ValidationReport:
    """Executa todas as checagens sobre a estrutura padrão de `data/`."""
    data_dir = Path(data_dir)
    report = ValidationReport()

    def _safe_read(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            report.error("missing_file", f"arquivo esperado não encontrado: {path}")
            return []
        try:
            return read_jsonl(path)
        except ValueError as exc:
            report.error("invalid_json", str(exc))
            return []

    train_prog = _safe_read(data_dir / "train" / "progressive.jsonl")
    train_cons = _safe_read(data_dir / "train" / "conservative.jsonl")
    val_prog = _safe_read(data_dir / "validation" / "progressive.jsonl")
    val_cons = _safe_read(data_dir / "validation" / "conservative.jsonl")
    eval_political = _safe_read(data_dir / "evaluation" / "political_prompts.jsonl")
    eval_adversarial = _safe_read(data_dir / "evaluation" / "adversarial_prompts.jsonl")
    eval_neutral = _safe_read(data_dir / "evaluation" / "neutral_prompts.jsonl")

    all_train = train_prog + train_cons
    all_val = val_prog + val_cons
    all_eval = eval_political + eval_adversarial + eval_neutral

    report.stats["n_train_progressive"] = len(train_prog)
    report.stats["n_train_conservative"] = len(train_cons)
    report.stats["n_validation_progressive"] = len(val_prog)
    report.stats["n_validation_conservative"] = len(val_cons)
    report.stats["n_eval_political"] = len(eval_political)
    report.stats["n_eval_adversarial"] = len(eval_adversarial)
    report.stats["n_eval_neutral"] = len(eval_neutral)

    for source, records in [("train", all_train), ("validation", all_val)]:
        check_required_fields(records, source, report)
        check_roles(records, source, report)
        check_unique_ids(records, source, report)
        check_empty_and_length(records, source, report, min_chars, max_chars)
        check_duplicates(records, source, report)
        check_offensive_terms(records, source, report, extra_offensive_terms)

    check_unique_ids(all_train + all_val + all_eval, "global", report)

    check_pair_symmetry(train_prog, train_cons, "train", report, max_length_diff_ratio)
    check_pair_symmetry(val_prog, val_cons, "validation", report, max_length_diff_ratio)
    check_orientation_balance(train_prog, train_cons, "train", report)
    check_orientation_balance(val_prog, val_cons, "validation", report)

    check_topic_distribution(train_prog, "train_progressive", report)
    check_topic_distribution(train_cons, "train_conservative", report)

    for eval_name, eval_recs in [
        ("evaluation/political_prompts", eval_political),
        ("evaluation/adversarial_prompts", eval_adversarial),
        ("evaluation/neutral_prompts", eval_neutral),
    ]:
        check_eval_leakage(all_train, eval_recs, eval_name, report)

    check_split_leakage({"train": all_train, "validation": all_val, "evaluation": all_eval}, report)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Valida os datasets do experimento political-bias-sft.")
    parser.add_argument("--data-dir", default="data", help="Diretório raiz de dados (default: data)")
    parser.add_argument("--max-length-diff-ratio", type=float, default=0.35)
    parser.add_argument("--min-chars", type=int, default=60)
    parser.add_argument("--max-chars", type=int, default=4000)
    parser.add_argument("--out-json", default="outputs/evaluations/dataset_validation_report.json")
    parser.add_argument("--out-md", default="outputs/evaluations/dataset_validation_report.md")
    args = parser.parse_args()

    report = validate_all(args.data_dir, args.max_length_diff_ratio, args.min_chars, args.max_chars)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    out_md = Path(args.out_md)
    out_md.write_text(report.to_markdown(), encoding="utf-8")

    logger.info("Validação concluída: %d erro(s), %d aviso(s)", report.n_errors, report.n_warnings)
    logger.info("Relatórios salvos em %s e %s", out_json, out_md)

    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
