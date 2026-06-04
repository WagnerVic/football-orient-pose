# Resultados dos Experimentos — Fine-tuning RTMPose-X

> **Status: PRELIMINAR (run de validação do pipeline).**
> Estes números saíram de uma execução de validação do Épico 1 — **antes** dos fixes
> do code review (`cf9fa5f`, em especial o A1/scheduler por época) e **não** são os
> resultados finais da matriz experimental. Servem para demonstrar que o pipeline roda
> ponta-a-ponta e produz a tabela comparativa (critério "Done" do Épico 1).
> Os números finais virão após re-rodar com os fixes + os Cenários **B** e **D**.

## Setup da avaliação

- **Split:** val (40 clips × 20 = 800 frames) e train (160 clips × 20 = 3200 frames).
- **Métricas:** `evaluate.py` (com a correção de derivação dos keypoints `[0,7,8,9]`,
  igual ao GT e ao baseline). Ver [docs/finetuning.md](finetuning.md).
- **Baseline zero-shot:** RTMPose-X COCO via rtmlib/ONNX + `coco17_to_h3wb17`
  (`src/evaluation/evaluate.py`), salvo em `results/tables/rtmpose_val.json`.
- **Hardware:** treino e avaliação em RTX 4090 (Docker), batch 32–64.

## Comparação no val split

| Cenário | PCK@0.2 | PDJ@0.5 | OKS | MPJPE-2D |
|---------|--------:|--------:|----:|---------:|
| Baseline zero-shot (COCO) | 41.76% | 93.62% | 81.82% | 4.81 px |
| **A** — from scratch (50 ép.) | 47.93% | 90.84% | 80.52% | 5.32 px |
| **C** — transfer learning (TL) | **51.46%** | 90.04% | 80.39% | 5.14 px |

> O zero-shot **detecta** bem (PDJ ~93%) mas é **impreciso** no domínio (PCK 41.76%).
> O fine-tuning sobe o **PCK@0.2** (A e C passam o baseline), mas **PDJ, OKS e MPJPE
> pioram** — o ganho de precisão se concentra no tronco, não nos membros (ver
> [ressalva](#️-ressalva--regressão-em-extremidades-vs-baseline) abaixo).

## Diagnóstico over/underfitting (train vs val)

Seguindo o framework dos **3 números** (treino, val, base ótimo) — overfitting é
avaliado pela diferença entre as métricas finais, não por curva de loss. O **PCK train**
é o mesmo PCK@0.2 estrito (`compute_pck`, ref. ombros/quadris), medido com
`evaluate.py --split train` sobre o best checkpoint.

| Cenário | PCK **train** | PCK **val** | gap | Diagnóstico |
|---------|------------:|----------:|----:|-------------|
| **A** — from scratch | 90.88% | 47.93% | ~43 pp | 🔴 **Overfitting forte** (alta variância) |
| **C** — transfer learning | 56.94% | 51.46% | ~5.5 pp | 🟢 **Generaliza bem** |

- **A:** decora o treino (PCK 91%) mas não generaliza (48% val) → overfitting clássico.
  Prova que a tarefa é aprendível a alta precisão; o gargalo é generalização.
- **C:** train ≈ val (gap ~5.5pp) → os pesos do COCO agem como regularizador.

## PCK@0.2 por grupo anatômico (val)

| Grupo | Baseline | A (scratch) | C (TL) |
|-------|---------:|------------:|-------:|
| head | 50.4% | 76.3% | 71.4% |
| shoulder | 30.4% | 61.6% | 62.8% |
| elbow | 50.8% | 34.3% | 39.9% |
| wrist | 43.5% | 24.2% | 31.9% |
| hip | 22.8% | 51.8% | 54.0% |
| knee | 58.3% | 38.5% | 46.6% |
| ankle | 59.4% | 38.9% | 47.5% |

> O TL melhora principalmente as **extremidades** (wrist, ankle, knee, elbow) vs. o
> from-scratch — coerente com o backbone do COCO já trazer features ricas de membros.

### ⚠️ Ressalva — regressão em extremidades vs. baseline

O PCK global sobe, mas **MPJPE, PDJ e OKS pioram** nos dois cenários (A: 5.32px, C: 5.14px
vs. baseline 4.81px). Não é contradição: o PCK@0.2 conta a fração abaixo do limiar e é
dominado pelo **tronco**, enquanto o MPJPE é média que **inclui os outliers**. O breakdown
por grupo explica:

- **Tronco melhora muito:** hip 22.8%→54.0%, shoulder 30.4%→62.8%, head 50.4%→71.4% (C vs. baseline).
- **Extremidades regridem vs. o baseline COCO:** wrist 43.5%→31.9%, elbow 50.8%→39.9%,
  knee 58.3%→46.6%, ankle 59.4%→47.5% (C). O from-scratch (A) é ainda pior nos membros.

Ou seja, o fine-tuning **melhora o tronco e piora os membros** — e são esses membros que
arrastam MPJPE/PDJ/OKS pra baixo. A leitura "TL melhora extremidades" vale **apenas relativa
ao from-scratch** (C > A nos membros); contra o **baseline**, ambos regridem.

**Implicação:** como o alvo do trabalho é precisão, a regressão nos membros é uma limitação
real a reportar. Hipótese testável nos próximos cenários: **B/D (augmentation) recuperam as
extremidades?** Acompanhar o breakdown por grupo, não só o PCK global.

## Conclusão preliminar

```
PCK@0.2 val:   C (TL) 51.46%  >  A (scratch) 47.93%  >  baseline 41.76%
Generalização: C gap 5.5pp    <<  A gap 43pp
```

O **transfer learning agrega de forma mensurável** — ganha em performance **e** em
generalização. Responde à pergunta central (TL não é "fé", o experimento mostra).
O overfitting forte do A motiva diretamente os cenários com **augmentation** (B/D).

## Próximos passos

- [ ] Re-rodar A e C com os fixes do review (`cf9fa5f`: scheduler por época, gating estrito).
- [ ] Rodar **Cenário B** (from scratch + augmentation) — testa se o augmentation fecha o gap de overfitting do A.
- [ ] Rodar **Cenário D** (TL + augmentation) — testa se empurra o C além.
- [ ] Consolidar a matriz 2×2 final com os 4 cenários sob o mesmo protocolo.
