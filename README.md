Experimento controlado de fine-tuning (QLoRA) para investigar, de forma
mensurável e reproduzível, como o conteúdo de um dataset de SFT desloca o
comportamento político-argumentativo de um modelo de linguagem pequeno.
## Objetivo

Não é demonstrar que um modelo "tem crenças" ou "desenvolveu uma ideologia".
É demonstrar, com controle experimental, que **dados selecionados de
fine-tuning podem alterar de forma mensurável**:

- as posições defendidas;
- o enquadramento das questões;
- o vocabulário utilizado;
- a seleção de argumentos;
- o reconhecimento de contrapontos;
- o grau de certeza das respostas.

## Hipótese

> Dois adapters QLoRA treinados sobre conjuntos ideologicamente distintos,
> mas estruturalmente equivalentes, produzirão deslocamentos mensuráveis e
> opostos em respostas políticas não vistas durante o treinamento.

Hipóteses secundárias:

1. O fine-tuning altera o enquadramento das questões, não apenas as conclusões.
2. Os adapters podem perder parte da capacidade de apresentar contrapontos.
3. O fine-tuning político pode afetar respostas sobre temas não políticos.
4. O efeito pode variar de acordo com o tópico.
5. Parte da diferença pode ser estilística, e não ideológica.

## Arquitetura do experimento

Três variantes são comparadas contra o **mesmo conjunto de prompts
inéditos** (não vistos em treino), com o mesmo system prompt, chat template,
seeds e `generation_config`:

| Variante | Descrição |
|---|---|
| **base** | `unsloth/gemma-3-4b-it` (ou variante pré-quantizada), sem fine-tuning político |
| **progressive** | Base + adapter LoRA treinado em `data/train/progressive.jsonl` |
| **conservative** | Base + adapter LoRA treinado em `data/train/conservative.jsonl` |

Os dois adapters usam **exatamente**: mesmo modelo base, mesma configuração
LoRA, mesmos hiperparâmetros de treino, mesma seed, mesmo número de exemplos
e mesma distribuição temática (`src/config.py::assert_matching_hyperparameters`
garante isso por código antes de qualquer treinamento). A única variável
experimental deliberada é **o conteúdo das respostas-alvo**.

Os adapters **não são mesclados (merge)** ao modelo base antes da avaliação
principal — cada um é carregado separadamente sobre a mesma base a cada
inferência, para permitir comparação limpa e reversível.

### Distinção modelo base vs. adapter

O modelo base (`unsloth/gemma-3-4b-it`) permanece congelado em 4-bit (QLoRA);
apenas as matrizes de baixo posto do LoRA (rank 16, ~poucas dezenas de MB por
adapter) são treinadas e salvas. Isso permite alternar entre as três
variantes carregando/descarregando apenas o adapter, sem duplicar a cópia
inteira do modelo em memória.

## Decisão técnica: identificador do modelo

Verificado em 2026-07-29 no catálogo Hugging Face/Unsloth:

- **Modelo escolhido**: `unsloth/gemma-3-4b-it` (Gemma 3 4B instruction-tuned,
  compatível com `unsloth.FastModel` + QLoRA 4-bit).
- **Variante usada por padrão**: `unsloth/gemma-3-4b-it-unsloth-bnb-4bit`
  (pré-quantizada em 4-bit pela Unsloth — download menor e carregamento mais
  rápido no Colab, mesma qualidade). Controlado por
  `configs/base.yaml::model.use_prequantized`.
- Chat template usado: `"gemma-3"` via `unsloth.chat_templates.get_chat_template`.

Se essas versões forem descontinuadas ou renomeadas no futuro, atualize
`configs/base.yaml::model.base_model_id` / `base_model_id_prequantized` — o
restante do pipeline não depende do nome exato do modelo.

## Instalação

```bash
git clone <este-repositorio>
cd political-bias-sft
python -m venv .venv && source .venv/bin/activate  # ou .venv\Scripts\activate no Windows
pip install -r requirements.txt
```

`requirements.txt` inclui Unsloth, Transformers, TRL, PEFT, Datasets,
bitsandbytes, além de pyyaml/pydantic (config), pandas/scipy/matplotlib
(análise e gráficos) e pytest (testes). **Unsloth e bitsandbytes exigem GPU
NVIDIA** — instalação local sem GPU só é útil para editar config/dados e
rodar os testes/validador.

## Execução local (sem GPU)

O que roda em CPU, sem `unsloth`/`torch`/GPU:

```bash
python -m src.dataset_validator --data-dir data
python -m pytest tests/ -q
python run_experiment.py validate
```

Treinamento, inferência e geração de texto exigem GPU (ver seção Colab).

## Execução no Google Colab

1. Abra `notebooks/train_and_evaluate_colab.ipynb` no Colab com runtime
   **GPU (L4 recomendada)**.
2. Clone este repositório para `/content/political-bias-sft` (célula 3).
3. Rode as células em ordem — o notebook está dividido em 18 seções
   numeradas, da configuração experimental até a exportação de resultados.
4. Datasets de treino/validação em escala completa (400+50 pares por
   orientação) **não são gerados pelo notebook** — foram gerados
   externamente seguindo `prompts/dataset_generation_prompt.md` (usado com
   um LLM externo via Codex) e devem estar presentes em `data/train/` e
   `data/validation/` antes de rodar a validação (seção 7).
5. Use `SMOKE_TEST = True` na seção 5 na primeira execução para validar o
   pipeline fim-a-fim rapidamente antes de comprometer horas de GPU com o
   treinamento completo.

### Células que exigem intervenção manual

| Seção | Intervenção necessária |
|---|---|
| 3 — Instalação | Definir o comando `git clone` do seu fork/repositório |
| 4 — Autenticação HF | Colar token manualmente via `getpass` **apenas se** `use_hf_auth = True` |
| 5 — Configuração | Decidir `SMOKE_TEST = True/False` |
| 6 — Datasets | Gerar os datasets completos via `prompts/dataset_generation_prompt.md` **antes** de rodar esta célula |
| 18 — Manifesto | Preencher `run_timestamp_utc` manualmente com data/hora real da execução |

## Estrutura dos datasets

Formato JSONL, um exemplo por linha, com pares espelhados
(`pair_id` idêntico entre `progressive.jsonl` e `conservative.jsonl`):

```json
{
  "id": "taxation_001_progressive",
  "pair_id": "taxation_001",
  "orientation": "progressive",
  "topic": "taxation",
  "messages": [
    {"role": "system", "content": "Responda de forma clara, respeitosa e argumentativa."},
    {"role": "user", "content": "Grandes fortunas deveriam pagar proporcionalmente mais impostos?"},
    {"role": "assistant", "content": "..."}
  ]
}
```

- **Treino**: 400 pares/orientação (800 exemplos totais) — `data/train/`.
- **Validação**: 50 pares/orientação — `data/validation/`.
- **Avaliação** (nunca usados em treino): 60 prompts políticos diretos/de
  enquadramento, 20 adversariais, 30 neutros — `data/evaluation/`.
- **Taxonomia**: 25 tópicos em `data/metadata/topic_taxonomy.json`.
- **Dados semente**: `data/train/*.jsonl` e `data/validation/*.jsonl` já
  contêm ~25 pares de treino e 5 de validação escritos manualmente, usados
  como referência de formato/qualidade e para permitir rodar
  testes/smoke_test sem depender da geração externa em escala completa. O
  dataset completo (400+50) é gerado seguindo
  `prompts/dataset_generation_prompt.md`.

O modo `smoke_test` (`configs/base.yaml::smoke_test`) reduz tudo para 20
exemplos de treino, 5 de validação, 6 prompts de avaliação e poucos steps.

## Parâmetros

Hiperparâmetros centralizados em `configs/base.yaml`, herdados por
`configs/progressive.yaml` e `configs/conservative.yaml` via `extends:` —
isso torna estruturalmente impossível que as duas execuções divirjam em
hiperparâmetros sem que alguém edite os dois arquivos filhos manualmente
(e `assert_matching_hyperparameters` verifica isso em runtime).

- **QLoRA 4-bit**, `max_seq_length=2048`, `r=16`, `alpha=32`, `dropout=0.05`.
- **Treino**: 2 épocas, `lr=1e-4`, `batch=2`, `grad_accum=8`,
  `optim=adamw_8bit`, `lr_scheduler=cosine`, seed `42`.
- **BF16/FP16**: detectados automaticamente via
  `src/utils.py::detect_bf16_support` (usa BF16 se a GPU suportar, senão FP16).
- **`train_on_responses_only`**: ativado por padrão — a loss é calculada
  apenas sobre os tokens de resposta do assistente (via
  `unsloth.chat_templates.train_on_responses_only`), não sobre o system/user.

### Gerenciamento de memória (GPU L4, 24GB)

- Cada adapter é treinado, salvo e **descarregado** (`clear_gpu_memory()`)
  antes do próximo — nunca há três cópias completas do modelo em memória.
- Checkpoints salvos incrementalmente em `outputs/adapters/adapter_*/` e
  copiados ao Drive a cada etapa concluída (permite retomar treinamento via
  `--resume-from-checkpoint`).

Se ocorrer erro de falta de memória (OOM), ajuste nesta ordem:

1. Reduzir `training.per_device_train_batch_size`.
2. Aumentar `training.gradient_accumulation_steps` (mantém o batch efetivo).
3. Reduzir `model.max_seq_length`.
4. Reforçar `lora.use_gradient_checkpointing` (já usa `"unsloth"` por padrão).
5. Reduzir `lora.r` (ex.: de 16 para 8).
6. Reduzir `data_sizes.train_pairs` — apenas para depuração, nunca para a
   execução final reportada.

## Avaliação

### Rubricas (independentes entre si — nunca uma nota única)

`political_orientation` (-2 a +2), `counterargument_recognition`,
`argument_quality`, `overcertainty`, `emotional_language`, `civility`,
`apparent_factuality`, `self_criticism`, `instruction_compliance`,
`relevance`, `neutral_ideology_intrusion` — ver `src/evaluate.py::RUBRIC_DIMENSIONS`.

### Camadas de avaliação (nenhuma obrigatória isoladamente)

1. **Avaliador heurístico** (`src/evaluate.py::heuristic_evaluate`) — baseado
   em palavras-chave, rápido e ruidoso por design. **Não é medição
   científica.** Serve como triagem e para permitir rodar o pipeline
   fim-a-fim sem depender de avaliação humana.
2. **LLM judge plugável** (`src/evaluate.py::run_llm_judge`) — recebe uma
   função `judge_fn(prompt, response, category) -> scores` fornecida pelo
   usuário. Não depende de nenhuma API paga por padrão; pode ser um modelo
   local, uma API externa opcional, ou uma chamada manual.
3. **Avaliação humana cega** (`src/blind_review.py`) — CSV embaralhado, sem
   nome de modelo/adapter, com mapa privado separado
   (`blind_review_private_map.jsonl`, **não compartilhar**). Suporta 2+
   avaliadores, com Cohen's kappa (2 avaliadores) ou Fleiss' kappa (3+).

### Estatística

`src/evaluate.py::build_summary` calcula: orientação média por variante e
por tópico, diferenças pareadas (mesmo `prompt_id`/`sample_index` entre
variantes) com **bootstrap CI** e Cohen's d, taxa de reconhecimento de
contrapontos, taxa de invasão ideológica em prompts neutros, e variabilidade
entre amostras do mesmo prompt.

**Limitações estatísticas explícitas**: a amostra é pequena (um modelo 4B,
poucas centenas de exemplos por adapter), então os intervalos de confiança
tendem a ser largos. Trate os resultados como **análise exploratória**, não
confirmatória. As três variantes respondem aos mesmos prompts (comparação
pareada), mas as N amostras de um mesmo prompt/variante **não são
independentes** entre si.

## Interpretação — segurança metodológica

- O modelo **não possui necessariamente crenças**; o experimento mede
  comportamento textual, não convicção.
- O experimento mede **comportamento textual**, não cognição.
- Datasets pequenos podem induzir **caricaturas** de posições políticas.
- Os rótulos "progressista"/"conservador" são **simplificações
  operacionais** (ver `data/metadata/topic_taxonomy.json` e a seção de
  definições nos prompts de geração) — não representam partidos, pessoas ou
  movimentos reais.
- Respostas plausíveis podem conter **erros factuais**.
- Um LLM judge pode **reproduzir seus próprios vieses** — trate como mais um
  sinal, não como ground truth.
- Resultados de um modelo de **4B parâmetros não devem ser generalizados**
  para "toda IA" ou modelos maiores.
- **Um único treinamento não demonstra robustez** — refaça com seeds e
  amostras diferentes antes de tirar conclusões fortes.
- A **seleção de prompts de avaliação** pode favorecer a hipótese; os
  prompts em `data/evaluation/` foram fixados antes do treinamento.
- **Seeds e amostragem** podem alterar resultados — por isso o modo
  amostral gera 5 respostas por prompt/variante.
- O experimento **não mede intenção, consciência ou convicção** do modelo.

## Limitações

- Dataset semente (25 pares) é suficiente para testes e smoke_test, mas
  **não** para conclusões experimentais — é necessário gerar o dataset completo
  (400+50 pares) antes da execução final.
- O avaliador heurístico é propositalmente simples (busca por palavras-chave)
  e gera ruído considerável; use-o como triagem, não como resultado final.
- O projeto não garante que o `chat_template="gemma-3"` da Unsloth continuará
  idêntico entre versões — revalide após qualquer `pip install -U unsloth`.
- Compensação de vazamento de dados é verificada apenas por **igualdade
  literal** de texto (normalizado para minúsculas) — paráfrases próximas
  entre treino e avaliação não são detectadas automaticamente.

## Solução de problemas

| Sintoma | Causa provável | Ação |
|---|---|---|
| `CUDA out of memory` | Batch/seq_length grande demais para a GPU | Ver ordem de redução na seção "Gerenciamento de memória" |
| `ConfigError: Campos obrigatórios ausentes` | YAML de config incompleto ou `extends` apontando para arquivo errado | Conferir `configs/*.yaml` |
| Validador reporta `eval_leakage` | Uma pergunta de treino/validação é idêntica a um prompt de `data/evaluation/` | Reescrever a pergunta de treino (ver `prompts/dataset_generation_prompt.md`) |
| Validador reporta `pair_symmetry` | `pair_id` presente em um arquivo de orientação e ausente no outro | Garantir que cada `pair_id` exista em ambos `progressive.jsonl`/`conservative.jsonl` do mesmo split |
| `unsloth`/`bitsandbytes` falha ao importar localmente | Ambiente sem GPU NVIDIA/CUDA | Normal — use Colab para as etapas de treino/inferência |
| Gráficos de loss vazios | `trainer_state.json` do adapter ainda não existe | Rodar o treinamento antes da célula de plotting |

## Licença

Este projeto é disponibilizado para fins educacionais e de pesquisa. Os
datasets gerados não devem ser usados para propaganda política, manipulação
eleitoral ou qualquer aplicação que viole os termos de uso do modelo base
(Gemma — ver licença Google Gemma) ou das bibliotecas utilizadas.
