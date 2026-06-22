# Fundamentação Teórica do *Progressive Unfreezing* — Índice de Estudo

> **Para que serve esta pasta.** O professor cobrou (reunião de metodologia, 09/06) que o grupo
> **justifique e referencie** o *progressive unfreezing* — sem argumento de autoridade ("é melhor porque
> eu disse"). Estes documentos são para você **estudar e defender** cada artigo na apresentação:
> entender a ideia central, a intuição, e **onde cada um encaixa no nosso pipeline** de fine-tuning do
> RTMPose-X no 3DSP.
>
> PDFs originais em `.task-context/input/referencias/.refs/RNP/artigos/`.

---

## A história em um parágrafo (o fio que liga os quatro)

Nosso treino de transfer learning usa **progressive unfreezing + discriminative fine-tuning**, técnicas
de **ULMFiT (Howard & Ruder, 2018)** [[01-ulmfit-progressive-unfreezing]] — descongelar o backbone do
topo para a base, em fases, com LR maior na cabeça. Isso faz sentido porque **Yosinski et al. (2014)**
[[02-yosinski-features-transferiveis]] provaram que as camadas baixas de uma rede são **gerais**
(transferíveis do COCO para o futebol) e as altas, **específicas** — então preservamos as baixas e
re-treinamos as altas. **Como** aplicar isso sem estragar as features vem do **LP-FT (Kumar et al.,
2022)** [[03-lpft-distorcao-de-features]]: treinar a cabeça **primeiro** (nossa fase 1) evita distorcer
o pré-treino; mexer demais no backbone (nossa fase 3) **distorce** — o que explica a degradação que
medimos. E **até onde** ir vem do **Surgical Fine-Tuning (Lee et al., 2023)**
[[04-surgical-finetuning]]: com poucos dados, ajustar **um subconjunto** de camadas bate ajustar tudo —
justificando o *gating* da fase 3 e o parar na fase 2. **Os quatro juntos transformam "treinar em fases
é melhor" de opinião em literatura + número.**

---

## Tabela-resumo

| # | Artigo | Ideia central (1 frase) | Onde encaixa no nosso trabalho |
|---|---|---|---|
| 1 | **ULMFiT** (Howard & Ruder, 2018) [[01-ulmfit-progressive-unfreezing]] | Descongele aos poucos (topo→base) e use LR por camada, para adaptar sem esquecer | **Nomeia a técnica**: nosso `frozen_stages` 4→2→1 + LR cabeça>backbone |
| 2 | **Yosinski et al.** (2014) [[02-yosinski-features-transferiveis]] | Camadas baixas = gerais; altas = específicas | **Por que congelar** as baixas do COCO e adaptar a cabeça; por que TL ajuda |
| 3 | **LP-FT** (Kumar et al., 2022) [[03-lpft-distorcao-de-features]] | FT de tudo distorce features; treinar a cabeça antes evita | **Por que a fase 1 (só cabeça)** vem antes; **por que a fase 3 piorou** |
| 4 | **Surgical FT** (Lee et al., 2023) [[04-surgical-finetuning]] | Com poucos dados, ajustar um subconjunto ≥ ajustar tudo | **Por que o gating da fase 3** e **parar na fase 2** |

---

## Elevator pitch da fundamentação (para abrir a metodologia na apresentação)

> "Para adaptar o RTMPose ao futebol sem destruir o que ele aprendeu no COCO, usamos *progressive
> unfreezing* com *discriminative fine-tuning* (ULMFiT, Howard & Ruder 2018): descongelamos o backbone
> do topo para a base, em três fases, com taxa de aprendizado maior na cabeça. Isso é fundamentado:
> Yosinski et al. (2014) mostram que as camadas baixas são gerais e devem ser preservadas; Kumar et al.
> (2022, LP-FT) mostram que treinar a cabeça primeiro evita distorcer as features pré-treinadas; e Lee
> et al. (2023, *surgical fine-tuning*) mostram que, com poucos dados, ajustar só um subconjunto de
> camadas é melhor que ajustar tudo — o que justifica pararmos no ajuste parcial. Tudo isso é
> **confirmado pelos nossos experimentos**, inclusive a degradação da fase 3."

---

## Mapa de defesa — pergunta do professor → artigo que responde

| Pergunta provável | Resposta curta | Artigo |
|---|---|---|
| "Vocês inventaram esse treino em fases?" | Não — é o *gradual unfreezing* do ULMFiT | [[01-ulmfit-progressive-unfreezing]] |
| "Por que treinar em pedaços é melhor que tudo junto?" | Adapta o específico sem apagar o geral; evita distorção | 01 + [[03-lpft-distorcao-de-features]] |
| "Por que descongelar do topo para a base?" | Camadas baixas são gerais (preservar), altas específicas | [[02-yosinski-features-transferiveis]] |
| "Por que os pesos do COCO ajudam no futebol?" | Features baixas são transferíveis entre tarefas | [[02-yosinski-features-transferiveis]] |
| "Por que treinar só a cabeça primeiro?" | FT completo com cabeça aleatória distorce features (LP-FT) | [[03-lpft-distorcao-de-features]] |
| "A fase 3 piorou — não é um bug de vocês?" | Não, é distorção de features prevista pela teoria | [[03-lpft-distorcao-de-features]] + [[04-surgical-finetuning]] |
| "Por que não descongelar tudo?" | Com poucos dados, ajustar subconjunto ≥ tudo; ajustar demais esquece | [[04-surgical-finetuning]] |

---

## Como estudar (ordem sugerida)

1. **[[01-ulmfit-progressive-unfreezing]]** — entenda a técnica (é o que o professor questionou).
2. **[[02-yosinski-features-transferiveis]]** — o *porquê* da ordem topo→base.
3. **[[03-lpft-distorcao-de-features]]** — o *porquê* da fase 1 e da degradação da fase 3.
4. **[[04-surgical-finetuning]]** — o *porquê* de parar (gating da fase 3).

Cada doc segue a mesma estrutura científica: **contribuição central → método (preciso) → evidência
(números) → fundamentação (as citações que os próprios autores usam para sustentar a tese) → onde
encaixa no nosso trabalho → Q&A de defesa → frase de defesa**. Para a apresentação, domine as quatro
**"frases de defesa (científicas)"** e a seção **"Fundamentação"** de cada artigo — é o que sustenta a
defesa sem cair em argumento de autoridade.

---

## Ligações com o resto do projeto

- Onde a técnica é descrita no artigo de entrega: [artigo_rnp.md §2.5](../../../.task-context/input/referencias/.refs/RNP/artigo_rnp.md)
  *(caminho a partir desta pasta pode variar; arquivo em `.task-context/input/referencias/.refs/RNP/`)*.
- Resumo da reunião que originou esta demanda: `…/.refs/RNP/007_analise-professor-metodologia/resumo-conversa-metodologia.md`.
- Implementação no código: [`scripts/training/train.py`](../../../scripts/training/train.py)
  (fases, LR discriminativo, gating) e [`configs/cenario_d.py`](../../../configs/cenario_d.py).
