# Prompt para geração dos datasets de fine-tuning político (usar no GPT-5.6 / Codex)

Cole este arquivo inteiro como instrução para o agente. Ele descreve exatamente
o que gerar, em que formato, com quais regras de qualidade e como validar o
resultado antes de considerar a tarefa concluída.

---

## Contexto

Este repositório (`political-bias-sft/`) é um experimento controlado para um
vídeo sobre viés induzido por fine-tuning em LLMs. Ele já contém toda a
infraestrutura de treinamento, validação e avaliação — **o que falta é o
dataset de treino e validação em escala completa**. Sua tarefa é gerar esse
dataset, não escrever código.

Existem hoje, como referência de formato e qualidade, datasets **semente**
(pequenos, ~25 pares) em:

- `data/train/progressive.jsonl` e `data/train/conservative.jsonl`
- `data/validation/progressive.jsonl` e `data/validation/conservative.jsonl`

Estude esses arquivos antes de começar — eles definem o padrão de tom,
extensão, estrutura argumentativa e nível de civilidade esperado.

Também existem os prompts de avaliação (NÃO podem ser reutilizados no treino,
nem parafraseados de forma óbvia) em:

- `data/evaluation/political_prompts.jsonl`
- `data/evaluation/adversarial_prompts.jsonl`
- `data/evaluation/neutral_prompts.jsonl`

E a taxonomia oficial de tópicos em `data/metadata/topic_taxonomy.json`.

## O que você deve produzir

Substitua o conteúdo de:

- `data/train/progressive.jsonl` — **400 exemplos** (pares progressistas)
- `data/train/conservative.jsonl` — **400 exemplos** (contrapartes conservadoras)
- `data/validation/progressive.jsonl` — **50 exemplos**
- `data/validation/conservative.jsonl` — **50 exemplos**

Os arquivos atuais (semente) podem ser usados como inspiração de estilo, mas
o resultado final deve ter exatamente essas contagens, com IDs e pair_ids
próprios (não precisa preservar os pares semente, mas pode absorvê-los se
quiser, desde que a contagem final bata).

## Formato exato (JSON Lines — um objeto por linha, UTF-8, sem BOM)

```json
{"id": "taxation_014_progressive", "pair_id": "taxation_014", "orientation": "progressive", "topic": "taxation", "messages": [{"role": "system", "content": "Responda de forma clara, respeitosa e argumentativa."}, {"role": "user", "content": "<pergunta>"}, {"role": "assistant", "content": "<resposta>"}]}
```

Regras de formato obrigatórias:

- `id`: `<topic>_<seq3digitos>_<orientation>` (ex.: `taxation_014_progressive`). Único em todo o arquivo.
- `pair_id`: `<topic>_<seq3digitos>` (ex.: `taxation_014`). Deve existir **em ambos** os arquivos (progressive.jsonl e conservative.jsonl) com o **mesmo** `pair_id`, **mesmo** `topic` e **mesma pergunta** (`messages[1].content` idêntico nos dois lados do par).
- `orientation`: `"progressive"` ou `"conservative"`, coerente com o arquivo.
- `topic`: um dos 25 IDs de `data/metadata/topic_taxonomy.json` (`taxation`, `inequality`, `labor_market`, `minimum_wage`, `privatization`, `public_health`, `education`, `housing`, `public_safety`, `penal_system`, `environment`, `energy`, `civil_rights`, `free_speech`, `religious_freedom`, `immigration`, `social_protection`, `unions`, `fiscal_responsibility`, `business_regulation`, `tech_privacy`, `ai_and_labor`, `federalism`, `protests_movements`, `institutions`).
- `messages`: exatamente 3 mensagens, nesta ordem: `system` (sempre `"Responda de forma clara, respeitosa e argumentativa."`), `user` (a pergunta), `assistant` (a resposta).

## Distribuição de tópicos

- **Treino**: 400 pares / 25 tópicos ≈ 16 pares por tópico (pode variar ±3 para não ficar artificialmente uniforme, mas nenhum tópico deve ter menos de 12 ou mais de 20 pares).
- **Validação**: 50 pares / 25 tópicos = 2 pares por tópico.
- Varie o `question_type` dentro de cada tópico: perguntas diretas de posicionamento, perguntas de enquadramento ("quais fatores explicam...", "o que torna... justo?"), avaliação de proposta de política, comparação, pedido de conselho/recomendação, pedido de resumo de um debate. Isso reduz repetição estrutural e aproxima o dataset de uso real.

## Regra central: espelhamento (mirroring)

Para cada `pair_id`, a versão progressive e a versão conservative devem:

1. Responder **exatamente à mesma pergunta** (texto idêntico em `messages[1].content`).
2. Ter **comprimento comparável** — diferença de no máximo ~30% no número de caracteres entre as duas respostas do par (o validador do projeto rejeita acima de 35%).
3. Ter o **mesmo número aproximado de argumentos** (2 a 3 argumentos de apoio é uma boa referência, como nos exemplos semente).
4. Ter o **mesmo nível de educação, polidez e qualidade textual** — nenhum lado deve soar mais raso, mais didático ou mais bem escrito que o outro.
5. Incluir, nos **dois lados**, pelo menos uma frase reconhecendo uma limitação, trade-off ou contra-argumento legítimo da própria posição (isso é uma característica central do estilo do dataset semente — evite respostas triunfalistas ou sem nuance).
6. Ter **estrutura equivalente**: abertura com posicionamento, argumentos de apoio, reconhecimento de limitação, fechamento. Não é preciso ser mecânico/idêntico frase a frase, mas a "forma" do argumento deve ser comparável.

Extensão-alvo por resposta: **100–200 palavras** (aproximadamente o tamanho dos exemplos semente). Não escreva respostas de um parágrafo genérico de 3 linhas nem ensaios de 500 palavras.

## Definições operacionais das orientações (releia antes de escrever)

**progressive** (progressista/social-democrata) — tendências que o dataset pode expressar: defesa de políticas redistributivas; maior atuação do Estado na redução de desigualdades; serviços públicos universais; maior regulação ambiental; proteção trabalhista; posições progressistas em direitos civis; reconhecimento de falhas de mercado; tributação progressiva.

**conservative** (conservador/liberal na economia) — tendências que o dataset pode expressar: defesa de mercados e propriedade privada; redução de impostos e regulações; responsabilidade fiscal; menor intervenção econômica do Estado; valorização de instituições, estabilidade e tradições; responsabilidade individual; soluções privadas ou descentralizadas; posições socialmente conservadoras apresentadas de forma civil e argumentativa.

Essas são **personas experimentais**, não retratos definitivos de partidos, pessoas ou movimentos reais. Trate como um exercício de retórica argumentativa consistente, não como propaganda.

## Proibições absolutas (violação = exemplo deve ser descartado e reescrito)

Não use, em nenhum dos dois lados:

- Insultos, xingamentos, desumanização de qualquer grupo.
- Ataques a grupos étnicos, religiosos, de gênero, orientação sexual, nacionalidade etc.
- Propaganda eleitoral ou menção a candidatos, partidos ou eleições reais e atuais.
- Apoio a violência, discurso de ódio ou incitação.
- Teorias conspiratórias apresentadas como fato.
- Informações factuais inventadas apresentadas com certeza (estatísticas fabricadas, citações falsas, eventos que não ocorreram).
- Instruções para manipular eleitores ou desinformação eleitoral.
- Linguagem emocional excessiva, sensacionalista ou de "post de rede social" — o tom é o de um debate acadêmico civil, como nos exemplos semente.
- Um lado sensivelmente mais "razoável" ou mais "extremista" que o outro. Se ao reler um par você sente que teria mais simpatia por um lado só por causa do **tom** (não do conteúdo político em si), reescreva.

## Regras que evitam vazamento e duplicação

- **Nenhuma pergunta de treino ou validação pode ser idêntica, nem uma paráfrase óbvia, de nenhum item em `data/evaluation/political_prompts.jsonl`, `adversarial_prompts.jsonl` ou `neutral_prompts.jsonl`.** Leia esses três arquivos antes de escrever as perguntas de treino/validação e evite reciclar as mesmas formulações.
- Nenhuma resposta (texto do `assistant`) pode ser duplicada, nem near-duplicada, dentro do mesmo arquivo ou entre treino e validação.
- Nenhuma pergunta pode se repetir literalmente dentro do mesmo arquivo (cada `pair_id` deve ser sobre um assunto/formulação distinto, mesmo dentro do mesmo tópico).
- IDs (`id` e `pair_id`) únicos em todo o dataset gerado.

## Exemplo de par já validado (para calibrar tom e extensão — não repita este par, é apenas referência)

Veja `data/train/progressive.jsonl` e `data/train/conservative.jsonl`, registro `pair_id: "taxation_001"`, como referência direta de qualidade aceitável.

## Passo a passo recomendado

1. Leia a taxonomia (`data/metadata/topic_taxonomy.json`), os 3 arquivos de avaliação e os datasets semente completos.
2. Planeje a distribuição: para cada um dos 25 tópicos, defina quantos pares de treino (≈16) e de validação (2) você vai criar, e liste as perguntas (distintas entre si e das perguntas de avaliação) antes de escrever as respostas.
3. Escreva os pares um tópico por vez, sempre gerando a versão progressive e a conservative lado a lado para a mesma pergunta, verificando de imediato o equilíbrio de tamanho e de argumentos.
4. Ao final, escreva os 4 arquivos JSONL (`data/train/progressive.jsonl`, `data/train/conservative.jsonl`, `data/validation/progressive.jsonl`, `data/validation/conservative.jsonl`), um objeto JSON por linha, sem vírgula final, UTF-8.
5. **Rode o validador do projeto antes de finalizar**:

   ```bash
   cd political-bias-sft
   python -m src.dataset_validator --data-dir data
   ```

   Ele verifica automaticamente: JSON válido, campos obrigatórios, IDs únicos, simetria de pares, correspondência de tópicos, diferença de comprimento, duplicatas, vazamento de prompts de avaliação no treino, vazamento literal entre splits, distribuição de tópicos, balanceamento entre orientações, papéis obrigatórios, respostas vazias/curtas/longas e termos ofensivos básicos.
6. Corrija todos os **erros** (`n_errors > 0` faz o comando sair com código 1) reportados em `outputs/evaluations/dataset_validation_report.md` antes de considerar a tarefa concluída. Avisos (`warnings`) podem ser aceitáveis se você tiver uma justificativa razoável (ex.: um tópico levemente mais popular), mas revise-os.
7. Confirme as contagens finais: 400+400 pares de treino, 50+50 de validação, sem exceder ou faltar.

## O que NÃO fazer

- Não gere menos de 400 pares de treino por orientação "para economizar tempo" — o experimento depende do volume para ter poder estatístico mínimo.
- Não crie um lado mais longo, mais bem argumentado ou mais civil que o outro.
- Não copie/cole a mesma resposta trocando só uma palavra entre pares diferentes.
- Não invente estatísticas, fatos históricos ou citações para parecer mais persuasivo — os dois lados devem argumentar com base em raciocínio, valores e trade-offs, não em "fatos" fabricados.
- Não use nomes de políticos, partidos ou eleições reais e atuais como eixo do argumento.

Ao concluir, relate: contagem final por arquivo, distribuição de tópicos obtida, e o resultado do comando de validação (número de erros/avisos).
