# Relatório do Épico 2 — Execução da Matriz de Experimentos

**Projeto:** Football Orient Pose — Transfer Learning (RNP)
**Run principal:** `results/runs/20260607_033902/` (A, C) · **Ablação:** `results/runs/20260607_051809_c2/` (C2)
**Commit:** `a0df5e5` · **GPU:** RTX 4090 (Docker) · **Data:** 07/06/2026
**Status:** 🟢 Cenários **A**, **C** e ablação **C2** concluídos · ⏳ **B** e **D** pendentes

> **Run canônico pós-correções do code review (#92).** Substitui `epic1-relatorio-preliminar.md`
> (pré-fixes, orçamento desigual A=50 vs C=145). Aqui A e C rodam sob protocolo idêntico e justo.

---

## Resumo executivo

- **Transfer learning vence, e de forma robusta.** No mesmo orçamento (150 épocas, batch 64), o
  Cenário **C (pesos COCO) atinge PCK@0.2 = 58,4%** contra **46,1% do A (do zero)** e **41,8% do
  baseline zero-shot** — **+12,3pp sobre o from-scratch** e **+16,6pp sobre o baseline**.
- **C também generaliza muito melhor:** gap treino→val de **11,7pp** (C) vs **47,6pp** (A).
- **O progressive unfreezing se justifica** (ablação C2): vale **+3,6pp** e menos overfitting vs
  o fine-tune de fase única — mas a fase única (54,8%) já captura ~94% do ganho.
- **Limitação honesta:** o ganho concentra-se no **tronco**; nas **extremidades** (punho, cotovelo,
  joelho, tornozelo) mesmo o C ainda fica **abaixo do baseline COCO**. É o alvo dos Cenários B/D.

**Resposta à pergunta de pesquisa (metade da matriz):** sim, transfer learning agrega valor
mensurável — comprovado pela ablação com inicialização aleatória, não por suposição.

---

## 0. Métricas usadas (definições)

Todas calculadas no **val split** (40 clips × 20 = 800 frames) pelo `evaluate.py`, derivando os
4 keypoints calculados [0,7,8,9] das predições (igual ao GT e ao baseline) antes de medir.

| Métrica | O que mede | Normalização | Direção |
|---|---|---|---|
| **PCK@0.2** (principal) | precisão de localização | erro < 0,2 × ref (dist. ombros/quadris) | ↑ |
| **PDJ@0.5** | detecção ("achou a região?") | erro < 0,5 × dist. torso | ↑ |
| **OKS** | similaridade tipo-COCO | sigmas por keypoint | ↑ |
| **MPJPE-2D** | erro absoluto em pixels | nenhuma (px no crop 100×100) | ↓ |

PCK é a métrica-alvo do trabalho: o zero-shot já **detecta** bem (PDJ ~94%) mas erra a **posição
exata**, e o PCK@0.2 (limiar estrito, ~2–8px) é o que penaliza essa imprecisão.

---

## 1. Objetivo e pergunta de pesquisa

**Transfer learning (pesos COCO) agrega valor sobre treinar do zero, no domínio de futebol
broadcast?** A metodologia é uma **matriz 2×2** que isola dois fatores — inicialização dos pesos
(W₀: COCO vs aleatório) e data augmentation (com/sem):

| | Sem Augmentation | Com Augmentation |
|---|---|---|
| **From Scratch** (W₀ aleatório) | **Cenário A** ✅ | Cenário B ⏳ |
| **Transfer Learning** (W₀ COCO) | **Cenário C** ✅ | Cenário D ⏳ |

A comparação **A vs C** responde diretamente a pergunta central. A ablação **C2** (fora da matriz)
testa se a *estratégia* de fine-tuning (progressive unfreezing) é necessária.

---

## 2. Protocolo experimental (controlado)

Tudo idêntico entre A e C — **só muda W₀ e a estratégia de descongelamento**:

| Parâmetro | Valor |
|---|---|
| Dataset | 3DSP, split 80/20 **por clip** (160 train / 40 val = 3.200 / 800 frames) |
| Modelo | RTMPose-X: backbone CSPNeXt-X + cabeça RTMCCHead (SimCC), loss `KLDiscretLoss` (β=10) |
| Entrada | crop 100×100 → 288×384 via `TopdownAffine(use_udp)` (letterboxing, sem distorção) |
| Orçamento de épocas | **A: 150** · **C: 45+60+45** (total **150** — paridade justa) |
| Batch | **64** (igual nos dois) |
| Otimizador | AdamW, weight_decay 0,05 |
| LR scheduler | warmup ~10% + cosseno até `eta_min`=1e-6 (proporcional às épocas — *fix A1*) |
| Métrica de seleção | **PCK@0.2 estrito** (`StrictPCKMetric`, ref. ombros/quadris — *fix M4*) |
| Avaliação | `evaluate.py` no val e no train (framework dos 3 números) |

**Diferença das células:**
- **A** — `load_from=None`, `frozen_stages=0` (tudo treina), LR uniforme 1e-3.
- **C** — pesos COCO + **progressive unfreezing** em 3 fases (`frozen_stages` 4→2→1), LR
  diferenciado por camada, fase 3 condicionada a Δ PCK > 5pp.

---

## 3. Resultados consolidados (val split)

| Modelo | PCK@0.2 ↑ | PCK train | gap | PDJ@0.5 ↑ | OKS ↑ | MPJPE-2D ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Baseline zero-shot (COCO) | 41,76% | — | — | 93,62% | 81,82% | 4,81 px |
| **A** — from scratch | 46,06% | 93,71% | 47,6pp | 89,96% | 79,39% | 5,56 px |
| **C2** — TL fase única *(ablação)* | 54,83% | 88,32% | 33,5pp | 92,02% | 83,37% | 4,57 px |
| **C** — TL progressive | **58,39%** | 70,05% | **11,7pp** | **93,80%** | **85,26%** | **4,09 px** |

**Leitura:**
- **C vence em todas as métricas**, inclusive sobre o baseline: PCK **+16,6pp**, PDJ +0,2pp,
  OKS +3,4pp, MPJPE **−0,72px**. É o único modelo que melhora **detecção e precisão ao mesmo tempo**.
- **C supera A em +12,3pp de PCK** com **muito menos overfitting** (gap 11,7 vs 47,6pp) — a
  evidência direta de que os pesos COCO valem.
- **A** bate o baseline só no PCK (+4,3pp), mas **piora** PDJ, OKS e MPJPE (puxados pelas
  extremidades — ver §8).
- **C2** fica no meio: forte (+8,8pp sobre A), mas abaixo do C (ver ablação §7).

---

## 4. Diagnóstico de overfitting (framework dos 3 números)

Comparando (1) treino, (2) val e (3) baseline ótimo — overfitting pela diferença das métricas
finais, não por curva de loss.

| Cenário | PCK **train** | PCK **val** | gap | leitura |
|---|---:|---:|---:|---|
| **A** (from scratch) | 93,71% | 46,06% | **47,6 pp** | 🔴 overfitting severo |
| **C2** (TL fase única) | 88,32% | 54,83% | **33,5 pp** | 🟠 overfitting moderado |
| **C** (TL progressive) | 70,05% | 58,39% | **11,7 pp** | 🟢 generaliza bem |

- **A decora o treino** (PCK 93,7%, loss→0,007, acc_pose→1,0) mas estaciona em ~46% no val. O
  gargalo é **generalização**, não capacidade — mais épocas só memorizam. → motiva augmentation (B).
- **C generaliza**: gap de 11,7pp. Os pesos COCO funcionam como **regularizador implícito** — o
  modelo parte de features robustas e não precisa memorizar o treino.
- **C2** mostra que **como** se faz o TL importa: o LR alto (1e-3) na cabeça crua memoriza mais
  (gap 33,5pp) que o warmup gradual do C (11,7pp).

---

## 5. Convergência e orçamento de épocas

Curva de PCK@0.2 (val, estrito) ao longo do treino:

| Modelo | Sobe até | Platô | Pico | Observação |
|---|---|---|---|---|
| **A** | ~época 30 | 44–46% | 45,8% (ep125) | platô por ~120 épocas — 150 é desperdício |
| **C2** | ~época 25 | 54–55% | 54,8% (ep35) | converge rápido, depois estabiliza |
| **C — fase 1** | ~época 15 | — | 52,9% | a cabeça satura em 15 ép (45 é exagero) |

**Implicações:** o orçamento de 150 épocas é **generoso** — todos convergem bem antes. Há leve
overfitting *após* o pico em todos (val cai um pouco), o que reforça a motivação de augmentation.
Para B/D dá pra cortar épocas com folga (economia de GPU) sem perder o pico.

---

## 6. Progressive unfreezing do Cenário C (3 fases)

| Fase | Descongela | Épocas | PCK@0.2 (estrito, val) |
|---|---|---:|---:|
| **1** — só a cabeça | backbone congelado (`frozen=4`) | 45 | 52,92% |
| **2** — destrava topo | `frozen_stages=2` | 60 | **58,39%** ← melhor |
| **3** — destrava mais | `frozen_stages=1` | 45 | 51,74% (degradou) |

**Achados:**
- Já na **Fase 1**, com o backbone **100% congelado**, o C atinge **52,9%** — sozinho isso já
  bate o Cenário A inteiro (46,1%). Os features do COCO bastam: adaptar só a cabeça > treinar tudo do zero.
- A **Fase 2** (destravar o topo do backbone) dá o melhor resultado: **58,39%**.
- A **Fase 3 degradou** (51,7%): descongelar as camadas baixas com 3.200 imagens causa overfitting
  nos estágios de baixo nível.

**Gating e seleção (correções do review em ação):**
```
[Δ PCK fase2−fase1] = 0,0547 (threshold 0,05)  → Fase 3 ativada (por 0,0047)
[Fase 3] PCK = 51,74%   ← piorou
[Seleção] fase2=58,39% vs fase3=51,74% → mantém fase2
```
> Sem o fix de seleção, o pipeline teria entregue a Fase 3 (51,7%) como modelo final —
> **perda de 6,6pp**. A correção pegou exatamente esse caso. **Recomendação:** subir o `--delta-pck`
> ou pular a Fase 3 (ativou por margem ínfima e só degradou).

---

## 7. Ablação — o progressive unfreezing é necessário? (Cenário C2)

**C2 = Transfer Learning em fase única:** mesmos pesos COCO, mas **sem as 3 fases** —
`frozen_stages=2` direto, LR discriminativo (cabeça 1e-3, backbone 1e-5 via `lr_mult=0.01`),
warmup + cosseno, 150 épocas, sem augmentation. Mesmo protocolo de avaliação do C. *(Não é célula
da matriz 2×2 — é uma ablação da estratégia.)*

| Modelo | PCK@0.2 | gap train→val | MPJPE |
|---|---:|---:|---:|
| **C2** (TL, fase única) | 54,83% | 33,5pp | 4,57 px |
| **C** (TL, progressive) | **58,39%** | **11,7pp** | **4,09 px** |

**Conclusões:**
- **O progressive vale +3,6pp de PCK** e, mais importante, **generaliza melhor** (gap 11,7 vs
  33,5pp). A diferença vem do **warmup da cabeça (fase 1)**: na fase única a cabeça começa crua
  com o backbone destravado e a LR alta, memorizando mais. → **a complexidade do progressive se
  justifica empiricamente.**
- **Mas o TL "simples" já é forte:** C2 supera A em **+8,8pp** e o baseline em **+13,1pp** — ~94%
  do PCK do C com uma receita bem mais simples (sem fases, gating, seleção).
- **Onde o progressive mais ajuda** (C − C2, PCK por grupo): **ankle +7,7pp**, shoulder +4,3,
  wrist +3,7, elbow +3,4 — ganha justamente nos **joints difíceis** (extremidades).
- **Custo de memória:** C2 usa ~5,6 GB (frozen_stages=2) vs ~19,9 GB do A (frozen_stages=0). Mas
  a **fase 2 do C também usa ~5,6 GB** — a economia é vs o *from scratch*, **não** vs o C. C2 é
  ~40% mais rápido que o A (menos params no backward); vs o C o tempo total é parecido (ambos 150 ép).

> **Recomendação:** manter o progressive como receita do C (vale os +3,6pp, sobretudo nas
> extremidades); registrar a fase única como alternativa simples quando se quer um TL rápido.

---

## 8. Análise por grupo anatômico (val)

### PCK@0.2 por grupo

| Grupo | Baseline | A | C2 | C | C vs Baseline |
|---|---:|---:|---:|---:|---:|
| head | 50,4% | 76,9% | 77,0% | 76,5% | 🟢 +26,1 |
| shoulder | 30,4% | 59,4% | 66,8% | 71,0% | 🟢 **+40,6** |
| hip | 22,8% | 50,3% | 57,8% | 60,6% | 🟢 **+37,8** |
| elbow | 50,8% | 31,3% | 42,1% | 45,5% | 🔴 −5,3 |
| wrist | 43,5% | 22,7% | 33,8% | 37,4% | 🔴 **−6,1** |
| knee | 58,3% | 38,1% | 50,5% | 52,9% | 🔴 −5,4 |
| ankle | 59,4% | 35,0% | 49,5% | 57,2% | 🔴 −2,2 |

### MPJPE-2D por grupo (px, ↓ melhor)

| Grupo | A | C2 | C |
|---|---:|---:|---:|
| head | 2,63 | 2,18 | 2,50 |
| shoulder | 3,41 | 2,70 | 2,40 |
| hip | 3,59 | 3,22 | 2,90 |
| elbow | 7,09 | 5,55 | 5,00 |
| knee | 5,70 | 4,70 | 4,20 |
| ankle | 8,64 | 7,46 | 6,10 |
| wrist | 11,21 | 8,65 | 7,80 |

**Ressalva importante (honestidade científica):** mesmo o melhor modelo (C) **ainda fica abaixo
do baseline COCO nas extremidades** (wrist, elbow, knee, ankle). O ganho enorme do fine-tuning
vem do **tronco** (shoulder +40,6, hip +37,8, head +26,1). O MPJPE global do C melhora (4,09px)
porque o tronco compensa, mas **os membros são o ponto fraco persistente** — punho e tornozelo
têm os maiores erros (7,8 e 6,1px) e os menores PCK. Causas prováveis: alta variabilidade de pose
durante o chute, oclusões e poucos dados. É a fronteira que **B/D (augmentation)** precisam atacar.

---

## 9. Confiabilidade dos números (vs o preliminar)

| Aspecto | Preliminar (pré-fix) | Este run (pós-fix) |
|---|---|---|
| Orçamento | A=50 vs C=145 ép. (**desigual** — confunde W₀ com nº de épocas) | A=150 vs C=150 (**justo**) |
| Batch | "32–64" (variável) | 64 (fixo) |
| LR scheduler | preso em `end=50/15` (não acompanhava as épocas) | proporcional às épocas (*fix A1*) |
| Seleção do checkpoint | PCK leniente (`norm=bbox`, ~0,9) | PCK estrito do artigo (*fix M4*) |
| Fase final do TL | última fase, mesmo se pior | melhor entre fase 2 e 3 (*fix*) |
| Proveniência | sem commit/data | commit + params + data gravados |

→ A comparação A vs C deste run é um **experimento controlado**: a única variável entre as células
é o fator em teste. Defensável no artigo.

---

## 10. Limitações

1. **Uma execução por cenário (sem barra de erro).** Os números são de 1 seed; diferenças pequenas
   (ex.: C2 vs eventual repetição) podem ter ruído. Para o artigo, idealmente confirmar com 2–3
   seeds ou declarar a estocasticidade.
2. **Extremidades abaixo do baseline.** O fine-tuning regride wrist/elbow/knee/ankle vs o COCO
   zero-shot — limitação real de precisão nos membros.
3. **Dataset pequeno** (3.200 imagens de treino) — favorece overfitting; é o que motiva a
   augmentation (B/D), ainda não testada.
4. **Matriz incompleta:** B e D faltam; a conclusão "TL > scratch" está sólida, mas o efeito da
   augmentation (e a interação TL×aug) é hipótese.
5. **Fase 3 ativada por margem ínfima** (Δ=0,0047 acima do limiar) e degradou — o gating de 5pp é
   sensível; revisar para B/D.

---

## 11. Conclusões

> **Transfer learning agrega valor de forma mensurável e robusta.** No mesmo orçamento e protocolo,
> o C (W₀ = COCO) supera o A (W₀ aleatório) em **+12,3pp de PCK** e generaliza muito melhor (gap
> 11,7pp vs 47,6pp). Não é "fé na palavra do RTMPose" — o experimento de ablação com inicialização
> aleatória mostra a diferença empiricamente. **Resposta à pergunta, na metade da matriz: sim.**

Conclusões secundárias:
- O **progressive unfreezing** melhora o TL (+3,6pp e menos overfitting), com ganho concentrado nas
  extremidades; a fase única é uma alternativa simples e quase tão boa.
- O **overfitting do A** (gap 47,6pp) e a **regressão nas extremidades** são os dois problemas
  abertos — ambos são alvos diretos da **augmentation** (Cenários B/D).

---

## 12. Próximos passos

- [ ] **Cenário B** (from scratch + augmentation): a augmentation regulariza o overfitting do A (gap 47,6pp)?
- [ ] **Cenário D** (TL + augmentation): aug + COCO recupera as extremidades e supera os 58,4% do C?
- [ ] Acompanhar o **breakdown por grupo** (não só o PCK global) — é onde a hipótese da augmentation se testa.
- [ ] Revisar o gating (subir `--delta-pck` ou remover a Fase 3).
- [ ] (Opcional) repetir A/C com 2–3 seeds para barra de erro.
- [ ] Consolidar a matriz 2×2 final e alimentar o artigo RNP (Seções 3 e 4).

---

## 13. Proveniência / reprodutibilidade

```
run A/C      results/runs/20260607_033902/   (commit a0df5e5)
run C2       results/runs/20260607_051809_c2/
host         RTX 4090 (Docker)
params       A/C: EPOCHS_A=150 fases_C=45/60/45 BATCH=64  ·  C2: EPOCHS=150 BATCH=64
artefatos    checkpoints/<cenario>/best_PCK.pth
             tables/finetuned_cenario_*_{train,val}.json
             logs/  ·  SUMMARY.md  ·  PROVENANCE.txt
```
Reproduzir: `docker run ... bash scripts/run_experiments.sh` (A+C) ou `bash scripts/run_c2.sh` (C2).
Ver `docs/finetuning/guia.md`.
