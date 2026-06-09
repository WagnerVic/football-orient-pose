# Relatório do Épico 2 — Execução da Matriz de Experimentos

**Projeto:** Football Orient Pose — Transfer Learning (RNP)
**Runs:** `20260607_033902/` (A-FLIP, C-FLIP) · `20260607_051809_c2/` (C2) · `20260607_195315_raw/` (A-RAW, C-RAW)
**GPU:** RTX 4090 (Docker) · **Data:** 07/06/2026
**Status:** 🟢 **A-RAW, A-FLIP, C-RAW, C-FLIP** e ablação **C2** concluídos · ⏳ **B** e **D** pendentes

> **Runs canônicos pós-correções do code review (#92).** Substituem `epic1-relatorio-preliminar.md`
> (pré-fixes, orçamento desigual A=50 vs C=145). Aqui todos os cenários rodam sob protocolo
> idêntico e justo (150 épocas, batch 64).

**Nomenclatura:** os cenários "sem augmentation" originais já usavam flip horizontal. Após a
ablação (§7.2) o baseline "sem augmentation" passou a ser a imagem **crua (RAW)** e os com flip
ganharam o sufixo **-FLIP**:
- **A-RAW / C-RAW** = sem augmentation nenhuma (nem flip).
- **A-FLIP / C-FLIP** = + flip horizontal (eram "A"/"C").
- **C2** = TL em fase única (ablação do progressive unfreezing; usa flip).

---

## Resumo executivo

- **Transfer learning vence, e de forma robusta.** No mesmo orçamento (150 épocas, batch 64), o
  **C-FLIP (pesos COCO) atinge PCK@0.2 = 58,4%** contra **46,1% do A-FLIP (do zero)** e **41,8% do
  baseline zero-shot** — **+12,3pp sobre o from-scratch** e **+16,6pp sobre o baseline**. Mesmo
  **sem augmentation nenhuma**, o C-RAW (52,9%) supera o A-FLIP (46,1%).
- **C-FLIP também generaliza muito melhor:** gap treino→val de **11,7pp** vs **47,6pp** do A-FLIP.
- **O progressive unfreezing se justifica** (ablação C2): vale **+3,6pp** e menos overfitting vs
  o fine-tune de fase única — mas a fase única (54,8%) já captura ~94% do ganho.
- **O flip horizontal é um augmentation forte** (ablação A-RAW/C-RAW, §7.2): sozinho vale
  **+8,2pp (scratch)** e **+5,5pp (TL)**. Por isso o baseline "sem augmentation" foi redefinido
  como a imagem **crua (RAW)**, com o flip virando o primeiro degrau de augmentation.
- **Limitação honesta:** o ganho concentra-se no **tronco**; nas **extremidades** (punho, cotovelo,
  joelho, tornozelo) mesmo o C-FLIP ainda fica **abaixo do baseline COCO**. É o alvo dos Cenários B/D.

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
broadcast?** A metodologia varia dois fatores — inicialização dos pesos (W₀: COCO vs aleatório)
e a **intensidade de data augmentation**.

> **Decisão de metodologia (ver §7.2):** originalmente os cenários "sem augmentation" já usavam o
> `RandomFlip` horizontal. Ao medir seu impacto (ablação A-RAW/C-RAW) descobrimos que o flip
> sozinho vale **+8,2pp (scratch)** e **+5,5pp (TL)** — grande demais para ser "baseline". Por isso
> o **flip passou a ser tratado como augmentation**, e o baseline "sem augmentation" virou a imagem
> **crua (RAW, sem flip)**. A comparação é uma **escada aditiva de augmentation**:

| W₀ | sem aug (RAW) | + flip | + flip + blur/erasing |
|---|---|---|---|
| **From Scratch** (W₀ aleatório) | **A-RAW** ✅ | **A-FLIP** ✅ | Cenário B ⏳ |
| **Transfer Learning** (W₀ COCO) | **C-RAW** ✅ | **C-FLIP** ✅ | Cenário D ⏳ |

A comparação **A-RAW vs C-RAW** (e A-FLIP vs C-FLIP) responde a pergunta central — TL vs scratch —
em cada nível de augmentation. As ablações **C2** (progressive unfreezing, §7) e do **flip** (§7.2)
complementam.

---

## 2. Protocolo experimental (controlado)

Tudo idêntico entre os cenários — **só muda W₀, a estratégia de descongelamento e o nível de aug**:

| Parâmetro | Valor |
|---|---|
| Dataset | 3DSP, split 80/20 **por clip** (160 train / 40 val = 3.200 / 800 frames) |
| Modelo | RTMPose-X: backbone CSPNeXt-X + cabeça RTMCCHead (SimCC), loss `KLDiscretLoss` (β=10) |
| Entrada | crop 100×100 → 288×384 via `TopdownAffine(use_udp)` (letterboxing, sem distorção) |
| Orçamento de épocas | **scratch: 150** · **TL: 45+60+45** (total **150** — paridade justa) |
| Batch | **64** (igual em todos) |
| Otimizador | AdamW, weight_decay 0,05 |
| LR scheduler | warmup ~10% + cosseno até `eta_min`=1e-6 (proporcional às épocas — *fix A1*) |
| Métrica de seleção | **PCK@0.2 estrito** (`StrictPCKMetric`, ref. ombros/quadris — *fix M4*) |
| Avaliação | `evaluate.py` no val e no train (framework dos 3 números) |

**Diferença das células:**
- **From scratch** (A-RAW, A-FLIP) — `load_from=None`, `frozen_stages=0` (tudo treina), LR uniforme 1e-3.
- **Transfer learning** (C-RAW, C-FLIP) — pesos COCO + **progressive unfreezing** em 3 fases
  (`frozen_stages` 4→2→1), LR diferenciado por camada, fase 3 condicionada a Δ PCK > 5pp.
- **RAW vs FLIP** — única diferença é o `RandomFlip` horizontal no pipeline de treino (os RAW não têm).

---

## 3. Resultados consolidados (val split)

| Modelo | aug | PCK@0.2 ↑ | PCK train | gap | PDJ@0.5 ↑ | OKS ↑ | MPJPE-2D ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline zero-shot (COCO) | — | 41,76% | — | — | 93,62% | 81,82% | 4,81 px |
| **A-RAW** — from scratch | RAW | 37,82% | 56,01%† | —† | 86,48% | 74,46% | 6,76 px |
| **A-FLIP** — from scratch | +flip | 46,06% | 93,71% | 47,6pp | 89,96% | 79,39% | 5,56 px |
| **C-RAW** — TL progressive | RAW | 52,88% | 65,11% | 12,2pp | 91,49% | 82,19% | 4,80 px |
| **C2** — TL fase única *(ablação)* | +flip | 54,83% | 88,32% | 33,5pp | 92,02% | 83,37% | 4,57 px |
| **C-FLIP** — TL progressive | +flip | **58,39%** | 70,05% | **11,7pp** | **93,80%** | **85,26%** | **4,09 px** |

† A-RAW: best checkpoint é da ép. 20 (pico do val; train ainda 56%). Na ép. 150 o train sobe a
~90% (gap real ~55pp) — ver §7.2.

**Leitura:**
- **C-FLIP vence em todas as métricas**, inclusive sobre o baseline: PCK **+16,6pp**, PDJ +0,2pp,
  OKS +3,4pp, MPJPE **−0,72px**. É o único que melhora **detecção e precisão ao mesmo tempo**.
- **C-FLIP supera A-FLIP em +12,3pp** com **muito menos overfitting** (gap 11,7 vs 47,6pp) — a
  evidência direta de que os pesos COCO valem. E **C-RAW (sem aug) já supera A-FLIP** (+6,8pp).
- **A-FLIP** bate o baseline só no PCK (+4,3pp), mas **piora** PDJ, OKS e MPJPE (puxados pelas
  extremidades — ver §8). O **A-RAW** fica abaixo do baseline no PCK (overfitting + sem aug).
- **C2** fica no meio: forte (+8,8pp sobre A-FLIP), mas abaixo do C-FLIP (ver ablação §7).

---

## 4. Diagnóstico de overfitting (framework dos 3 números)

Comparando (1) treino, (2) val e (3) baseline ótimo — overfitting pela diferença das métricas
finais, não por curva de loss.

| Cenário | PCK **train** | PCK **val** | gap | leitura |
|---|---:|---:|---:|---|
| **A-RAW** (scratch, RAW) | 56,01%† | 37,82% | —† | 🔴 overfitting precoce (val pica cedo e cai) |
| **A-FLIP** (scratch, +flip) | 93,71% | 46,06% | **47,6 pp** | 🔴 overfitting severo |
| **C-RAW** (TL, RAW) | 65,11% | 52,88% | **12,2 pp** | 🟢 generaliza bem |
| **C2** (TL fase única) | 88,32% | 54,83% | **33,5 pp** | 🟠 overfitting moderado |
| **C-FLIP** (TL progressive) | 70,05% | 58,39% | **11,7 pp** | 🟢 generaliza bem |

- **From scratch decora** (A-FLIP: train 93,7%, val 46%) — o gargalo é **generalização**, não
  capacidade; mais épocas só memorizam. → motiva augmentation (B).
- **TL generaliza** (C-FLIP gap 11,7pp; C-RAW 12,2pp): os pesos COCO funcionam como **regularizador
  implícito**, com ou sem flip.
- **C2** mostra que **como** se faz o TL importa: LR alto na cabeça crua memoriza mais (gap 33,5pp)
  que o warmup gradual do C-FLIP (11,7pp).
- † **A-RAW:** o gap "18,2pp" do log é enganoso — o best é da ép. 20 (train só 56%). Na ép. 150 o
  train ~90% e o gap real ~55pp (overfitting **mais** severo que o A-FLIP, e mais cedo — ver §7.2).

---

## 5. Convergência e orçamento de épocas

Curva de PCK@0.2 (val, estrito) ao longo do treino:

| Modelo | Pico | Comportamento |
|---|---|---|
| **A-RAW** | 37,9% (ep20) | pica **cedo** e **degrada** para ~35% até a 150 — overfitting precoce |
| **A-FLIP** | 45,8% (ep125) | platô em 44–46% por ~120 épocas — 150 é desperdício |
| **C-RAW** | 52,9% (fase 2) | converge cedo, estabiliza ~53% |
| **C2** | 54,8% (ep35) | converge rápido, depois estabiliza |
| **C-FLIP — fase 1** | 52,9% | a cabeça satura em ~15 ép (45 é exagero) |

**Implicações:** o orçamento de 150 épocas é **generoso** — todos convergem bem antes. Há leve
overfitting *após* o pico (mais forte no A-RAW, mais suave nos com flip), o que reforça a motivação
de augmentation. Para B/D dá pra cortar épocas com folga sem perder o pico.

---

## 6. Progressive unfreezing do Cenário C-FLIP (3 fases)

| Fase | Descongela | Épocas | PCK@0.2 (estrito, val) |
|---|---|---:|---:|
| **1** — só a cabeça | backbone congelado (`frozen=4`) | 45 | 52,92% |
| **2** — destrava topo | `frozen_stages=2` | 60 | **58,39%** ← melhor |
| **3** — destrava mais | `frozen_stages=1` | 45 | 51,74% (degradou) |

**Achados:**
- Já na **Fase 1**, com o backbone **100% congelado**, o C-FLIP atinge **52,9%** — sozinho isso já
  bate o A-FLIP inteiro (46,1%). Os features do COCO bastam: adaptar só a cabeça > treinar do zero.
- A **Fase 2** (destravar o topo do backbone) dá o melhor resultado: **58,39%**.
- A **Fase 3 degradou** (51,7%): descongelar as camadas baixas com 3.200 imagens causa overfitting
  nos estágios de baixo nível. (No C-RAW a fase 3 foi **pulada** pelo gating — ver §7.2.)

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
warmup + cosseno, 150 épocas, com flip (sem blur/erasing). Mesmo protocolo de avaliação. *(Não é
célula da matriz — é uma ablação da estratégia.)*

| Modelo | PCK@0.2 | gap train→val | MPJPE |
|---|---:|---:|---:|
| **C2** (TL, fase única) | 54,83% | 33,5pp | 4,57 px |
| **C-FLIP** (TL, progressive) | **58,39%** | **11,7pp** | **4,09 px** |

**Conclusões:**
- **O progressive vale +3,6pp de PCK** e, mais importante, **generaliza melhor** (gap 11,7 vs
  33,5pp). A diferença vem do **warmup da cabeça (fase 1)**: na fase única a cabeça começa crua com
  o backbone destravado e a LR alta, memorizando mais. → **a complexidade do progressive se
  justifica empiricamente.**
- **Mas o TL "simples" já é forte:** C2 supera A-FLIP em **+8,8pp** e o baseline em **+13,1pp** —
  ~94% do PCK do C-FLIP com uma receita bem mais simples (sem fases, gating, seleção).
- **Onde o progressive mais ajuda** (C-FLIP − C2, PCK por grupo): **ankle +7,7pp**, shoulder +4,3,
  wrist +3,7, elbow +3,4 — ganha nos **joints difíceis** (extremidades).
- **Custo de memória:** C2 usa ~5,6 GB (frozen_stages=2) vs ~19,9 GB do scratch (frozen_stages=0).
  Mas a **fase 2 do C-FLIP também usa ~5,6 GB** — a economia é vs o *from scratch*, **não** vs o
  progressive. A vantagem do C2 sobre o C-FLIP é **simplicidade**, não recursos.

> **Recomendação:** manter o progressive como receita do TL (vale os +3,6pp, sobretudo nas
> extremidades); registrar a fase única como alternativa simples quando se quer um TL rápido.

---

## 7.2 Ablação do flip horizontal — e a redefinição do baseline "sem augmentation"

### Como chegamos a esta mudança
Notamos que os cenários "sem augmentation" originais (A-FLIP e C-FLIP) **na verdade já usavam um
augmentation**: o `RandomFlip` horizontal — padrão em pose estimation, por isso tratado como
baseline. Surgiu a pergunta: **quanto o flip sozinho contribui?** Para responder com rigor, rodamos
**A-RAW** e **C-RAW**, idênticos a A-FLIP e C-FLIP mas **sem nenhum augmentation (nem flip)**.
Run: `results/runs/20260607_195315_raw/` (commit `261f73e`).

### Resultado: o flip vale muito
| Inicialização | sem aug (RAW) | + flip | **ganho do flip** |
|---|---:|---:|---:|
| From scratch | A-RAW **37,82%** | A-FLIP 46,06% | 🔵 **+8,24pp** |
| Transfer learning | C-RAW **52,88%** | C-FLIP 58,39% | 🔵 **+5,51pp** |

- O flip sozinho dá **+8,2pp (scratch)** e **+5,5pp (TL)** — enorme para "uma linha de pipeline".
  Com só ~160 cenas distintas (160 clips × 20 frames quase idênticos), espelhar **dobra a
  diversidade efetiva**.
- **From scratch depende mais do flip** (+8,2 vs +5,5): sem outra regularização, o flip é a
  principal defesa contra overfitting; o TL já tem os pesos COCO como regularizador.
- **TL sem aug nenhuma (C-RAW 52,88%) ainda vence o from-scratch COM flip (A-FLIP 46,06%)** por
  +6,8pp, e o baseline por **+11,1pp** → o valor do TL é robusto mesmo sem augmentation.

### A decisão: o flip vira augmentation; o baseline vira RAW
Como o flip contribui tanto, **mantê-lo dentro do "sem augmentation" mascarava o efeito do
augmentation**. Decisão: o **baseline honesto "sem augmentation" passa a ser o RAW (sem flip)** e o
flip vira o **primeiro degrau** da escada (RAW → +flip → +flip+blur/erasing). As métricas com flip
(A-FLIP, C-FLIP) são mantidas como o degrau "+flip" — assim lê-se a contribuição **marginal** de
cada augmentation. Isso também alinha o baseline ao sentido literal de "sem transformações adicionais".

### Por grupo — o flip ajuda os joints simétricos (L/R)
Ganho do flip por grupo (A-FLIP − A-RAW, PCK val): **hip +13,0** · **ankle +13,0** · **knee +10,9**
· **head +10,1** · wrist +7,1 · shoulder +4,9 · elbow +3,4. Faz sentido: o flip ensina **simetria
esquerda/direita**, e os pares L/R (quadris, joelhos, tornozelos) são os que mais se beneficiam.
No TL o ganho é mais uniforme (~5–6pp em todos).

### O flip atrasa o overfitting
Sem flip, o A-RAW **pica cedo (val 37,9% na época 20) e degrada** para ~34,9% até a época 150 —
overfitting precoce. O A-FLIP platôou em ~46% por ~120 épocas. → o flip não só sobe o pico como
**atrasa o overfitting**.

### Gating do C-RAW (sem flip)
```
Fase 1 = 50,00% · Fase 2 = 52,88% · Δ = 0,0288 (< 0,05) → Fase 3 PULADA
```
Sem flip, o ganho da fase 2 sobre a 1 foi menor (2,9pp vs 5,5pp com flip), então a **fase 3 foi
corretamente pulada** — sem a degradação que o C-FLIP teve. O gating estrito (fix A2) agindo certo.

---

## 8. Análise por grupo anatômico (val)

### PCK@0.2 por grupo (%)

| Grupo | Baseline | A-RAW | A-FLIP | C-RAW | C2 | C-FLIP |
|---|---:|---:|---:|---:|---:|---:|
| head | 50,4 | 66,8 | 76,9 | 73,9 | 77,0 | 76,5 |
| shoulder | 30,4 | 54,5 | 59,4 | 64,9 | 66,8 | 71,0 |
| hip | 22,8 | 37,3 | 50,3 | 54,5 | 57,8 | 60,6 |
| elbow | 50,8 | 27,9 | 31,3 | 39,4 | 42,1 | 45,5 |
| wrist | 43,5 | 15,6 | 22,7 | 31,0 | 33,8 | 37,4 |
| knee | 58,3 | 27,2 | 38,1 | 47,3 | 50,5 | 52,9 |
| ankle | 59,4 | 22,0 | 35,0 | 51,6 | 49,5 | 57,2 |

### MPJPE-2D por grupo (px, ↓ melhor)

| Grupo | A-RAW | A-FLIP | C-RAW | C2 | C-FLIP |
|---|---:|---:|---:|---:|---:|
| head | 3,9 | 2,63 | 3,0 | 2,18 | 2,50 |
| shoulder | 3,9 | 3,41 | 2,9 | 2,70 | 2,40 |
| hip | 4,5 | 3,59 | 3,4 | 3,22 | 2,90 |
| elbow | 8,1 | 7,09 | 6,1 | 5,55 | 5,00 |
| knee | 7,2 | 5,70 | 5,0 | 4,70 | 4,20 |
| ankle | 11,2 | 8,64 | 7,2 | 7,46 | 6,10 |
| wrist | 12,9 | 11,21 | 8,9 | 8,65 | 7,80 |

**Ressalva importante (honestidade científica):** mesmo o melhor modelo (C-FLIP) **ainda fica
abaixo do baseline COCO nas extremidades** (wrist 37,4 vs 43,5; elbow 45,5 vs 50,8; knee 52,9 vs
58,3; ankle 57,2 vs 59,4). O ganho enorme do fine-tuning vem do **tronco** (shoulder +40,6, hip
+37,8, head +26,1 vs baseline). O MPJPE global do C-FLIP melhora (4,09px) porque o tronco compensa,
mas **os membros são o ponto fraco persistente** — punho e tornozelo têm os maiores erros e os
menores PCK em todos os cenários. Causas prováveis: alta variabilidade de pose no chute, oclusões e
poucos dados. É a fronteira que **B/D (augmentation)** precisam atacar. (Repare: cada degrau —
RAW → +flip → +TL → +progressive — melhora as extremidades, mas nenhum ainda alcança o baseline nelas.)

---

## 9. Confiabilidade dos números (vs o preliminar)

| Aspecto | Preliminar (pré-fix) | Estes runs (pós-fix) |
|---|---|---|
| Orçamento | A=50 vs C=145 ép. (**desigual** — confunde W₀ com nº de épocas) | 150 em todos (**justo**) |
| Batch | "32–64" (variável) | 64 (fixo) |
| LR scheduler | preso em `end=50/15` (não acompanhava as épocas) | proporcional às épocas (*fix A1*) |
| Seleção do checkpoint | PCK leniente (`norm=bbox`, ~0,9) | PCK estrito do artigo (*fix M4*) |
| Fase final do TL | última fase, mesmo se pior | melhor entre fase 2 e 3 (*fix*) |
| Proveniência | sem commit/data | commit + params + data gravados |

→ As comparações deste run são um **experimento controlado**: a única variável entre as células é
o fator em teste. Defensável no artigo.

---

## 10. Limitações

1. **Uma execução por cenário (sem barra de erro).** Os números (incl. A-RAW/C-RAW) são de 1 seed;
   diferenças pequenas podem ter ruído. Para o artigo, idealmente confirmar com 2–3 seeds ou
   declarar a estocasticidade.
2. **Extremidades abaixo do baseline.** O fine-tuning regride wrist/elbow/knee/ankle vs o COCO
   zero-shot — limitação real de precisão nos membros, em todos os cenários.
3. **Dataset pequeno** (3.200 imagens, mas só ~160 cenas distintas) — favorece overfitting; é o que
   motiva o flip e a augmentation (B/D).
4. **Matriz incompleta:** B e D faltam; a conclusão "TL > scratch" está sólida, mas o efeito do
   blur/erasing (e a interação TL×aug) é hipótese.
5. **Fase 3 sensível:** no C-FLIP ativou por margem ínfima (Δ=0,0047 > 0,05) e degradou; no C-RAW
   foi corretamente pulada. O limiar de 5pp é sensível; revisar para B/D.

---

## 11. Conclusões

> **Transfer learning agrega valor de forma mensurável e robusta.** No mesmo orçamento e protocolo,
> o C-FLIP (W₀ = COCO) supera o A-FLIP (W₀ aleatório) em **+12,3pp de PCK** e generaliza muito
> melhor (gap 11,7pp vs 47,6pp). Até **sem augmentation nenhuma**, o C-RAW (52,9%) bate o A-FLIP
> (46,1%). Não é "fé na palavra do RTMPose" — a ablação com inicialização aleatória mostra a
> diferença empiricamente. **Resposta à pergunta, na metade da matriz: sim.**

Conclusões secundárias:
- O **flip horizontal** é um augmentation barato e forte (+8,2pp scratch, +5,5pp TL), com maior
  efeito nos joints simétricos L/R; por isso virou o 1º degrau de augmentation (baseline = RAW).
- O **progressive unfreezing** melhora o TL (+3,6pp e menos overfitting); a fase única (C2) é uma
  alternativa simples e quase tão boa.
- O **overfitting do from-scratch** e a **regressão nas extremidades** são os dois problemas
  abertos — ambos são alvos diretos da **augmentation** (Cenários B/D).

---

## 12. Próximos passos

- [ ] **Cenário B** (from scratch + blur/erasing): a augmentation regulariza o overfitting do scratch?
- [ ] **Cenário D** (TL + blur/erasing): aug recupera as extremidades e supera os 58,4% do C-FLIP?
- [ ] Acompanhar o **breakdown por grupo** (não só o PCK global) — é onde a hipótese da augmentation se testa.
- [ ] Revisar o gating (subir `--delta-pck` ou remover a Fase 3).
- [ ] (Opcional) repetir os cenários com 2–3 seeds para barra de erro.
- [ ] Consolidar a matriz final e alimentar o artigo RNP (Seções 3 e 4).

---

## 13. Proveniência / reprodutibilidade

```
A-FLIP / C-FLIP   results/runs/20260607_033902/      (commit a0df5e5)
C2                results/runs/20260607_051809_c2/
A-RAW / C-RAW     results/runs/20260607_195315_raw/  (commit 261f73e)
host              RTX 4090 (Docker)
params            scratch: EPOCHS=150 · TL: fases 45/60/45 · BATCH=64 (todos)
artefatos         checkpoints/<cenario>/best_PCK.pth
                  tables/finetuned_cenario_*_{train,val}.json
                  logs/  ·  SUMMARY.md  ·  PROVENANCE.txt
```
Reproduzir: `bash scripts/training/run_experiments.sh` (A-FLIP+C-FLIP) · `bash scripts/training/run_c2.sh` (C2) ·
`bash scripts/training/run_raw.sh` (A-RAW+C-RAW). Ver `docs/finetuning/epico-1/guia.md`.
