# Relatório FINAL do Épico 2 — Matriz Transfer Learning × Augmentation

**Projeto:** Football Orient Pose — Transfer Learning (RNP)
**Runs:** `20260607_033902/` (A-FLIP, C-FLIP) · `20260607_051809_c2/` (C2) ·
`20260607_195315_raw/` (A-RAW, C-RAW) · `20260608_014649_bd/` (B-FULL, D-GEOM, D-OCCL, D-FULL)
**GPU:** RTX 4090 (Docker) · **Datas:** 07–08/06/2026
**Status:** 🟢 **Matriz 2×2 completa** + ablações (C2, flip, mecanismos de aug)

> **Documento canônico do Épico 2.** Funde e supera os dois relatórios parciais
> (`epic2-relatorio-a-c.md` — A/C/RAW/C2; `epic2-relatorio-bd.md` — B/D), que ficam como
> histórico/detalhe. Aqui está a matriz inteira (10 modelos) numa narrativa única.

**Nomenclatura.** Os cenários variam dois fatores: inicialização (W₀) e nível de augmentation.
- **W₀:** *scratch* (aleatório) = família **A/B** · *transfer learning* (COCO) = família **C/D**.
- **Augmentation (escada):** **RAW** (nenhuma) → **FLIP** (+flip horizontal) → **+geom**
  (RandomBBoxTransform) → **+ocl** (oclusão) → **full** (+blur).
- Modelos: **A-RAW, A-FLIP, B-FULL** (scratch) · **C-RAW, C-FLIP, D-GEOM, D-OCCL, D-FULL** (TL) ·
  **C2** (ablação: TL em fase única).

---

## Resumo executivo

- **A pergunta de pesquisa está respondida nos dois eixos.** (1) **Transfer learning vence o
  treino do zero** de forma robusta, em todos os níveis de aug; (2) **a augmentation forte recupera
  as extremidades** e leva o modelo ao seu melhor patamar. A receita campeã é **TL + augmentation
  geométrica**.
- **O melhor modelo do projeto é o D-FULL: PCK@0.2 = 67,54%** — **+25,8pp sobre o baseline**
  zero-shot (41,76%) e **+14,7pp sobre a célula TL-sem-aug da matriz (C-RAW 52,9%)**. Vence em todas
  as métricas (PDJ 96,6% · OKS 89,8% · MPJPE 3,08px).
- **Os dois fatores da matriz (TL e augmentation) são fortes e se combinam.** Medindo as células da
  matriz (sem-aug = RAW, com-aug = full): **TL vale +15,1pp (sem aug) / +9,4pp (com aug)**;
  **augmentation vale +20,3pp (scratch) / +14,7pp (TL)**. A aug (a stack completa flip+geom+ocl+blur)
  é a alavanca um pouco **maior**; a combinação é **sub-aditiva (~−5,7pp)** — ambos atacam o mesmo
  gargalo (overfitting por baixa diversidade).
- **A augmentation que paga é geométrica.** Decompondo o ganho de aug do TL: **geométrica +7,77pp**,
  oclusão +0,71pp, blur +0,67pp. Some-se o flip (+5,5pp no TL) e fica claro: ~90% do efeito da
  augmentation é **diversidade de pose/ângulo** (flip + rotação/escala/shift).
- **As extremidades foram resolvidas.** O ponto fraco persistente dos relatórios parciais
  (punho/cotovelo/joelho/tornozelo **abaixo** do baseline COCO em todos os cenários) **virou**: os
  cenários D ficam **acima** do baseline em todas as extremidades. É o TL **+ geométrica** que
  conserta (o scratch com aug, B-FULL, não recupera).
- **TL + aug forte quase não overfita:** gap treino→val do D-FULL = **1,1pp** — contra **12,2pp do
  C-RAW** (TL sem aug) e **37–55pp do scratch**. A augmentação geométrica praticamente elimina o
  overfitting quando há o prior do COCO.

---

## 0. Métricas usadas (definições)

Todas no **val split** (40 clips × 20 = 800 frames) via `evaluate.py`, derivando os 4 keypoints
calculados [0,7,8,9] das predições (igual ao GT e ao baseline) antes de medir.

| Métrica | O que mede | Normalização | Direção |
|---|---|---|---|
| **PCK@0.2** (principal) | precisão de localização | erro < 0,2 × ref (dist. ombros/quadris) | ↑ |
| **PDJ@0.5** | detecção ("achou a região?") | erro < 0,5 × dist. torso | ↑ |
| **OKS** | similaridade tipo-COCO | sigmas por keypoint | ↑ |
| **MPJPE-2D** | erro absoluto em pixels | nenhuma (px no crop 100×100) | ↓ |

PCK é a métrica-alvo: o zero-shot já **detecta** bem (PDJ ~94%) mas erra a **posição exata**, e o
PCK@0.2 (limiar estrito, ~2–8px) penaliza essa imprecisão.

---

## 1. Objetivo, pergunta de pesquisa e desenho experimental

**Transfer learning (pesos COCO) agrega valor sobre treinar do zero no domínio de futebol
broadcast — e quanto a augmentation contribui?** A metodologia varia **W₀** (COCO vs aleatório) e a
**intensidade de augmentation**, organizada como uma **escada aditiva**.

### A escada de augmentation (5 degraus) por inicialização
| W₀ | RAW | +flip | +geom | +geom+ocl | +geom+ocl+blur |
|---|---|---|---|---|---|
| **Scratch** | A-RAW | A-FLIP | — | — | **B-FULL** |
| **Transfer learning** | C-RAW | C-FLIP | **D-GEOM** | **D-OCCL** | **D-FULL** |

O lado TL é um **ladder fino** (cada degrau = +1 augmentation, isolando o mecanismo); no scratch
rodamos só os extremos (RAW, flip, full), já que "TL > scratch" se confirma cedo.

### A matriz 2×2 (W₀ × aug) — célula "sem aug" = **RAW**, célula "com aug" = **full**
Como o flip é augmentation (§4), as células canônicas da matriz são o **RAW** (sem aug nenhuma) e o
**full** (stack completa). A-FLIP/C-FLIP/D-GEOM/D-OCCL são degraus intermediários do ladder, não
células da matriz.

| | RAW (sem aug) | full (com aug) |
|---|---:|---:|
| **From scratch** | A-RAW 37,82 | B-FULL 58,13 |
| **Transfer learning** | C-RAW 52,88 | **D-FULL 67,54** |

### Ablações que complementam
- **C2** — o progressive unfreezing é necessário? (§9)
- **flip** (A-RAW/C-RAW) — quanto o flip sozinho vale? (§4)
- **mecanismos** (D-GEOM/D-OCCL/D-FULL) — geométrica vs oclusão vs blur? (§4)

---

## 2. Protocolo experimental (controlado)

Tudo idêntico entre cenários — **só mudam W₀, a estratégia de descongelamento e o nível de aug**:

| Parâmetro | Valor |
|---|---|
| Dataset | 3DSP, split 80/20 **por clip** (160 train / 40 val = 3.200 / 800 frames; ~160 cenas distintas) |
| Modelo | RTMPose-X: backbone CSPNeXt-X + cabeça RTMCCHead (SimCC), loss `KLDiscretLoss` (β=10) |
| Entrada | crop 100×100 → 288×384 via `TopdownAffine(use_udp)` (letterboxing, sem distorção) |
| Orçamento | **scratch: 150 épocas** · **TL: 45+60+45** (total 150 — paridade) |
| Batch | **64** (igual em todos) |
| Otimizador | AdamW, weight_decay 0,05 |
| LR scheduler | warmup ~10% + cosseno até `eta_min`=1e-6 (proporcional às épocas — *fix A1*) |
| Métrica de seleção | **PCK@0.2 estrito** (`StrictPCKMetric`, ref. ombros/quadris — *fix M4*) |
| Avaliação | `evaluate.py` no val e no train (framework dos 3 números) |

**Diferença das células:**
- **Scratch** (A/B): `load_from=None`, `frozen_stages=0`, LR uniforme 1e-3.
- **TL** (C/D): pesos COCO + **progressive unfreezing** em 3 fases (`frozen_stages` 4→2→1), LR
  diferenciado por camada, fase 3 condicionada a Δ PCK > 5pp.

**Os augments (escada):**
| Augment | Implementação | Params | Posição |
|---|---|---|---|
| flip | `RandomFlip` | horizontal | — |
| geométrica | `RandomBBoxTransform` (mmpose) | rot **±30°**, escala **0,75–1,25**, shift **0,1** | após flip, antes do affine |
| oclusão | `SimpleRandomErasing` (custom) | 1 patch, área 2–10%, prob 0,3 | após affine |
| blur | `SimpleMotionBlur` (custom) | kernel 3–9px, direcional, p 0,5 | após a geométrica |

> **Notas técnicas.** (1) Params geométricos **conservadores** de propósito (crops apertados de
> jogadores ~verticais; ±80° do default RTMPose viraria pose irreal). (2) O código original do
> Épico 1 referenciava `type='Albu'` e `type='RandomErasing'` — dois bugs de referência combinados:
> o nome registrado do wrapper é `type='Albumentation'` (não `Albu`), e `RandomErasing` não existe
> no mmpose nem no Albumentations (o equivalente seria `CoarseDropout`). Adicionalmente, o pacote
> `albumentations` não estava instalado na imagem, então mesmo com nomes corretos o wrapper quebraria.
> Como os cenários B/D nunca rodaram no Épico 1, isso ficou latente. Solução: implementamos
> `SimpleMotionBlur` (cv2, blur direcional) e `SimpleRandomErasing` (numpy, patch uniforme) em
> `football_orient_pose.finetuning.transforms` — sem dependência externa, portáveis no container.
> (3) Estes runs incorporam os **fixes do code review (#92)**: scheduler proporcional (A1), gating
> por PCK estrito (A2), seleção por PCK estrito (M4) e seleção do melhor checkpoint entre fases.

---

## 3. Resultados consolidados (val split)

| Modelo | aug | PCK@0.2 ↑ | PCK train | gap | PDJ@0.5 ↑ | OKS ↑ | MPJPE-2D ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline zero-shot (COCO) | — | 41,76% | — | — | 93,62% | 81,82% | 4,81 px |
| **A-RAW** — scratch | RAW | 37,82% | 56,01%† | —† | 86,48% | 74,46% | 6,76 px |
| **A-FLIP** — scratch | +flip | 46,06% | 93,71% | 47,6pp | 89,96% | 79,39% | 5,56 px |
| **B-FULL** — scratch | full | 58,13% | 95,22% | 37,1pp | 93,59% | 85,26% | 4,09 px |
| **C-RAW** — TL | RAW | 52,88% | 65,11% | 12,2pp | 91,49% | 82,19% | 4,80 px |
| **C2** — TL fase única *(ablação)* | +flip | 54,83% | 88,32% | 33,5pp | 92,02% | 83,37% | 4,57 px |
| **C-FLIP** — TL | +flip | 58,39% | 70,05% | 11,7pp | 93,80% | 85,26% | 4,09 px |
| **D-GEOM** — TL | +geom | 66,16% | 68,71% | 2,5pp | 96,05% | 89,06% | 3,24 px |
| **D-OCCL** — TL | +geom+ocl | 66,87% | 68,94% | 2,1pp | 96,23% | 89,48% | 3,16 px |
| **D-FULL** — TL | full | **67,54%** | 68,60% | **1,1pp** | **96,60%** | **89,84%** | **3,08 px** |

† A-RAW: best checkpoint da ép. 20 (pico do val; train ainda 56%). Na ép. 150 o train sobe a ~90%
(gap real ~55pp) — ver §7.

**Leitura:**
- **Hierarquia limpa:** baseline 41,8 < A-RAW 37,8* < A-FLIP 46,1 < {B-FULL 58,1 ≈ C-FLIP 58,4} <
  D-GEOM 66,2 < D-OCCL 66,9 < **D-FULL 67,5**. (*A-RAW abaixo do baseline: overfitting + zero aug.)
- **D-FULL é o teto** e melhora **detecção e precisão ao mesmo tempo** (único, com o C-FLIP, a bater
  o baseline em PDJ — e com folga).
- **Leitura da matriz (células RAW/full):** o **B-FULL (scratch+aug, 58,1) supera o C-RAW (TL sem
  aug, 52,9) em +5,3pp** — ou seja, augmentation no scratch supera o TL cru no número global. Mas o
  B-FULL **overfita muito mais** (gap 37,1 vs 12,2pp) e não recupera as extremidades (§10): vence no
  PCK global **decorando**, não generalizando.
- **MPJPE despenca** ao longo da escada: 6,76 (A-RAW) → 3,08px (D-FULL), −1,73px vs baseline.

---

## 4. A escada de augmentation — quanto cada degrau vale

Ganho **marginal** de PCK@0.2 (val) ao subir cada degrau:

| Degrau | Scratch | Transfer learning |
|---|---|---|
| RAW (base) | A-RAW 37,82 | C-RAW 52,88 |
| **+ flip** | A-FLIP 46,06 (🔵 **+8,24**) | C-FLIP 58,39 (🔵 **+5,51**) |
| **+ geométrica** | *(não isolado)* | D-GEOM 66,16 (🔵 **+7,77**) |
| **+ oclusão** | *(não isolado)* | D-OCCL 66,87 (+0,71) |
| **+ blur** | *(combinado em B-FULL)* | D-FULL 67,54 (+0,67) |
| **total (RAW→full)** | B-FULL +20,31 | D-FULL **+14,66** |

**Conclusões:**
- **A augmentation que paga é geométrica.** Os dois maiores saltos são geométricos: **flip**
  (+8,2/+5,5pp) e **RandomBBoxTransform** (+7,77pp no TL). Juntos respondem por ~90% do ganho de
  aug. **Oclusão e blur são refino marginal** (+0,71 e +0,67pp), mas **reais e consistentes** —
  cada um também melhora PDJ/OKS/MPJPE (3,24 → 3,16 → 3,08px).
- **Por que geométrica?** Com só ~160 cenas distintas (20 frames quase iguais/clip), o gargalo é
  **diversidade efetiva de pose/ângulo**. Flip (espelha) e rotação/escala/shift multiplicam essa
  diversidade; blur/oclusão (fotométrico/máscara) mexem menos no que falta.
- **O flip por grupo** (A-FLIP − A-RAW): hip +13,0 · ankle +13,0 · knee +10,9 · head +10,1 ·
  wrist +7,1 · shoulder +4,9 · elbow +3,4 — maior efeito nos **pares simétricos L/R**, como
  esperado de espelhamento. No TL o ganho do flip é mais uniforme (~5–6pp).
- **O scratch depende mais do flip** (+8,2 vs +5,5): sem outra regularização, o flip é a principal
  defesa contra overfitting; o TL já tem os pesos COCO como regularizador.

> **Implicação prática (para o artigo):** se o orçamento de aug fosse mínimo, **geométrica seria a
> primeira e quase suficiente escolha**; oclusão+blur entram para o último ~1,4pp.

---

## 5. Transfer learning vs scratch + interação com a augmentation

### O eixo central: TL vence em todos os níveis
| Nível de aug | scratch | TL | ganho do TL |
|---|---:|---:|---:|
| RAW | A-RAW 37,82 | C-RAW 52,88 | **+15,06pp** |
| +flip | A-FLIP 46,06 | C-FLIP 58,39 | **+12,33pp** |
| full | B-FULL 58,13 | D-FULL 67,54 | **+9,41pp** |

- **O TL agrega valor em todos os níveis** — e, crucialmente, **C-RAW (TL sem aug nenhuma, 52,9%)
  já supera o A-FLIP (scratch com flip, 46,1%)** por +6,8pp. O prior do COCO vale mais que o flip.
- O ganho do TL **diminui** conforme a aug sobe (15 → 12 → 9pp): aug e TL atacam parcialmente o
  **mesmo** problema (pouca diversidade), então sobra menos para o TL recuperar quando a aug já age.

### A matriz 2×2 e a interação (células RAW × full)
| | RAW (sem aug) | full (com aug) | **Δ aug** |
|---|---:|---:|---:|
| **scratch** | A-RAW 37,82 | B-FULL 58,13 | **+20,31** |
| **TL** | C-RAW 52,88 | D-FULL 67,54 | **+14,66** |
| **Δ TL** | **+15,06** | **+9,41** | |

Partindo do A-RAW (37,82), somar os efeitos isolados daria +15,06 (TL) + 20,31 (aug) = +35,37pp →
73,19%. O D-FULL real é **67,54 (+29,72pp sobre o A-RAW)** → **interação ≈ −5,65pp (sub-aditiva)**.
Confirma que os dois fatores se sobrepõem em parte (ambos combatem a mesma falta de diversidade).
**Ainda assim, combiná-los é nitidamente melhor:** cada fator isolado fica abaixo (só-TL = C-RAW
52,9%; só-aug no scratch = B-FULL 58,1%), e a célula completa (D-FULL) chega a 67,5%. Note que **a
augmentation é a alavanca um pouco maior** que o TL na matriz (+20,3/+14,7 vs +15,1/+9,4), mas "aug"
aqui é a stack inteira (flip+geom+ocl+blur), enquanto "TL" é um único fator.

---

## 6. Diagnóstico de overfitting (framework dos 3 números)

Overfitting pela diferença treino→val das métricas finais, não por curva de loss.

| Cenário | PCK **train** | PCK **val** | gap | leitura |
|---|---:|---:|---:|---|
| A-RAW (scratch, RAW) | 56,01%† | 37,82% | —† | 🔴 overfitting precoce (val pica cedo e cai) |
| A-FLIP (scratch, +flip) | 93,71% | 46,06% | **47,6pp** | 🔴 overfitting severo |
| B-FULL (scratch, full) | 95,22% | 58,13% | **37,1pp** | 🟠 ainda alto (aug não basta sem TL) |
| C-RAW (TL, RAW) | 65,11% | 52,88% | **12,2pp** | 🟢 generaliza bem |
| C2 (TL fase única) | 88,32% | 54,83% | **33,5pp** | 🟠 overfitting moderado |
| C-FLIP (TL, +flip) | 70,05% | 58,39% | **11,7pp** | 🟢 generaliza bem |
| D-GEOM (TL, +geom) | 68,71% | 66,16% | **2,5pp** | 🟢 quase perfeito |
| D-OCCL (TL, +geom+ocl) | 68,94% | 66,87% | **2,1pp** | 🟢 quase perfeito |
| D-FULL (TL, full) | 68,60% | 67,54% | **1,1pp** | 🟢 gap quase nulo |

- **Scratch decora** (A-FLIP/B-FULL: train 94–95%, val 46–58%) — o gargalo é **generalização**, não
  capacidade. A aug **encolhe o gap** (47,6 → 37,1pp) mas **não o resolve** sem o prior do COCO.
- **A augmentação geométrica praticamente zera o gap no TL** (C-FLIP 11,7 → D-FULL 1,1pp): a aug
  pesada torna o **treino mais difícil**, então a PCK de treino não infla (68,6%) e cola na de val
  (67,5%). É o comportamento ideal: val ≈ train, ambos altos.
- **C2** (33,5pp) mostra que **como** se faz o TL importa: LR alto na cabeça crua memoriza mais que
  o warmup gradual do progressive (§9).

---

## 7. Convergência e orçamento de épocas

| Modelo | Pico (val) | Comportamento |
|---|---|---|
| A-RAW | 37,9% (ep20) | pica **cedo** e degrada para ~35% até a 150 — overfitting precoce |
| A-FLIP | 45,8% (ep125) | platô em 44–46% por ~120 épocas — 150 é folgado |
| C-RAW | 52,9% (fase 2) | converge cedo, estabiliza ~53% (fase 3 pulada) |
| C2 | 54,8% (ep35) | converge rápido, estabiliza |
| C-FLIP | 58,4% (fase 2) | fase 1 satura ~53%, fase 2 entrega o ganho |
| D-GEOM/OCCL/FULL | 66–67,5% (fase 2) | fase 1 ~59%, **fase 2 entrega o ganho**, fase 3 degrada |

**Implicações:** o orçamento de 150 épocas é **generoso** — todos convergem antes. Nos cenários com
aug forte, a **fase 2** do progressive é onde o ganho acontece; a fase 1 (só cabeça) já passa do
C-FLIP inteiro. Há overfitting leve *após* o pico nos cenários sem aug forte — exatamente o que a
augmentation geométrica corrige.

---

## 8. Progressive unfreezing e gating (4 cenários TL)

O TL roda 3 fases (`frozen_stages` 4→2→1). PCK@0.2 (estrito, val) por fase:

| Cenário | Fase 1 (só cabeça) | Fase 2 (+topo) | Fase 3 (+baixo) | Gating Δ f2−f1 | Selecionado |
|---|---:|---:|---:|---:|---|
| **C-RAW** | 50,0% | **52,9%** | — (pulada) | 0,029 (<0,05) | fase 2 → 52,88% |
| **C-FLIP** | 52,9% | **58,4%** | 51,7% (degradou) | 0,055 | fase 2 → 58,39% |
| **D-GEOM** | 59,4% | **66,2%** | 60,8% (degradou) | 0,067 | fase 2 → 66,16% |
| **D-OCCL** | 59,0% | **66,9%** | 61,7% (degradou) | 0,078 | fase 2 → 66,87% |
| **D-FULL** | 59,6% | **67,5%** | 63,1% (degradou) | 0,079 | fase 2 → 67,54% |

**Achados (evidência forte por agregação):**
- **A fase 2 é sempre onde o TL ganha;** já a **fase 1** (backbone 100% congelado) basta para passar
  do C-FLIP inteiro nos cenários D (~59% > 58,4%). Adaptar a cabeça sobre features COCO > treinar
  do zero.
- **A fase 3 degradou em 4 de 4 casos** em que foi ativada — destravar os stages baixos com 3.200
  imagens causa overfitting de features de baixo nível. No C-RAW (ganho de fase 2 pequeno) o gating
  **corretamente a pulou**.
- **O fix de seleção** (manter o melhor entre fase 2 e 3) **salvou +6,6 / +5,3 / +5,2 / +4,5pp**
  (C-FLIP/D-GEOM/D-OCCL/D-FULL) — sem ele, o pipeline entregaria a fase 3 pior nesses casos.

> **Recomendação consolidada:** o gating de 5pp **ativa a fase 3 à toa** (degradou sempre que
> rodou). Subir o `--delta-pck` (≥0,10) **ou remover a fase 3** — ela só gasta ~45 épocas de GPU
> sem nunca ganhar. O fix de seleção já protege o resultado final.

---

## 9. Ablação — o progressive unfreezing é necessário? (C2)

**C2 = TL em fase única:** mesmos pesos COCO, mas sem as 3 fases — `frozen_stages=2` direto, LR
discriminativo (cabeça 1e-3, backbone 1e-5), 150 épocas, com flip.

| Modelo | PCK@0.2 | gap train→val | MPJPE |
|---|---:|---:|---:|
| **C2** (TL, fase única) | 54,83% | 33,5pp | 4,57 px |
| **C-FLIP** (TL, progressive) | **58,39%** | **11,7pp** | **4,09 px** |

- **O progressive vale +3,6pp** e **generaliza muito melhor** (gap 11,7 vs 33,5pp). A diferença vem
  do **warmup da cabeça (fase 1)**: na fase única a cabeça começa crua com backbone destravado e LR
  alta, memorizando mais. → **a complexidade do progressive se justifica.**
- **Mas o TL "simples" já é forte:** C2 supera A-FLIP em +8,8pp e o baseline em +13,1pp (~94% do
  C-FLIP) com receita bem mais simples. Onde o progressive mais ajuda (C-FLIP − C2 por grupo): ankle
  +7,7 · shoulder +4,3 · wrist +3,7 · elbow +3,4 — nos **joints difíceis**.

> **Recomendação:** manter o progressive como receita do TL; registrar a fase única como
> alternativa simples para um TL rápido.

---

## 10. Análise por grupo anatômico — e as extremidades resolvidas

### PCK@0.2 por grupo (%) — val

| Grupo | Baseline | A-RAW | A-FLIP | C-RAW | C2 | C-FLIP | B-FULL | D-GEOM | D-OCCL | **D-FULL** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| head | 50,4 | 66,8 | 76,9 | 73,9 | 77,0 | 76,5 | 87,6 | 87,0 | 88,4 | **89,5** |
| shoulder | 30,4 | 54,5 | 59,4 | 64,9 | 66,8 | 71,0 | 70,5 | 78,2 | 78,3 | **78,4** |
| hip | 22,8 | 37,3 | 50,3 | 54,5 | 57,8 | 60,6 | 60,2 | 68,3 | 69,5 | **69,3** |
| elbow | 50,8 | 27,9 | 31,3 | 39,4 | 42,1 | 45,5 | 42,6 | 54,3 | 55,0 | **56,4** |
| wrist | 43,5 | 15,6 | 22,7 | 31,0 | 33,8 | 37,4 | 33,6 | 45,4 | 45,1 | **46,1** |
| knee | 58,3 | 27,2 | 38,1 | 47,3 | 50,5 | 52,9 | 53,5 | 62,3 | 63,2 | **64,8** |
| ankle | 59,4 | 22,0 | 35,0 | 51,6 | 49,5 | 57,2 | 54,0 | 65,2 | 67,2 | **67,2** |

### MPJPE-2D por grupo (px, ↓ melhor) — val

| Grupo | A-RAW | A-FLIP | C-RAW | C-FLIP | B-FULL | D-GEOM | D-OCCL | **D-FULL** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| head | 3,9 | 2,63 | 3,0 | 2,50 | 1,62 | 1,43 | 1,47 | **1,40** |
| shoulder | 3,9 | 3,41 | 2,9 | 2,40 | 2,40 | 1,92 | 1,89 | **1,86** |
| hip | 4,5 | 3,59 | 3,4 | 2,90 | 2,91 | 2,47 | 2,38 | **2,40** |
| elbow | 8,1 | 7,09 | 6,1 | 5,00 | 5,35 | 3,91 | 3,77 | **3,67** |
| knee | 7,2 | 5,70 | 5,0 | 4,20 | 4,34 | 3,56 | 3,47 | **3,35** |
| ankle | 11,2 | 8,64 | 7,2 | 6,10 | 6,08 | 4,39 | 4,64 | **4,39** |
| wrist | 12,9 | 11,21 | 8,9 | 7,80 | 8,05 | 5,91 | 5,76 | **5,63** |

### O arco do projeto: extremidades resolvidas
Os relatórios parciais fecharam com uma ressalva honesta: **nenhum** modelo (nem o C-FLIP)
alcançava o baseline COCO nas extremidades — o fine-tuning ganhava no **tronco** e regredia nos
**membros**. **Olhando as 4 células da matriz** (RAW e full × scratch e TL), só a célula completa
resolve:

| Grupo | baseline | A-RAW | C-RAW | B-FULL | **D-FULL** |
|---|---:|---:|---:|---:|---:|
| wrist | 43,5 | 15,6 | 31,0 | 33,6 | **46,1 ✅** |
| elbow | 50,8 | 27,9 | 39,4 | 42,6 | **56,4 ✅** |
| knee | 58,3 | 27,2 | 47,3 | 53,5 | **64,8 ✅** |
| ankle | 59,4 | 22,0 | 51,6 | 54,0 | **67,2 ✅** |

- **Só o D-FULL (TL + augmentation) passa o baseline em todas as extremidades.** A-RAW (scratch
  cru), C-RAW (TL sem aug) e B-FULL (scratch + aug) ficam **todos abaixo** — ou seja, **nem o TL
  sozinho, nem a augmentation sozinha bastam: é a combinação que conserta os membros.**
- **A alavanca dentro do TL é geométrica:** já o D-GEOM (TL + só geométrica) cruza o baseline em
  todas as 4 (wrist 45,4 / elbow 54,3 / knee 62,3 / ankle 65,2 ✅) — ver §4. O erro de punho cai de
  **8,9 → 5,63px** e o de tornozelo de **7,2 → 4,39px** (D-FULL vs C-RAW).
- **Ganho do D-FULL sobre o C-RAW (efeito da augmentation no TL) é transversal e grande:** elbow
  +17,0 · knee +17,5 · ankle +15,6 · wrist +15,1 · head +15,6 · hip +14,8 · shoulder +13,5 — as
  extremidades sobem **tanto quanto** o tronco (+13,5 a +17,5pp em todos).

---

## 11. Confiabilidade dos números

| Aspecto | Preliminar (pré-fix) | Estes runs (pós-fix) |
|---|---|---|
| Orçamento | A=50 vs C=145 ép. (desigual) | 150 em todos (justo) |
| Batch | "32–64" (variável) | 64 (fixo) |
| LR scheduler | preso em `end=50/15` | proporcional às épocas (*fix A1*) |
| Seleção do checkpoint | PCK leniente (`norm=bbox`) | PCK estrito do artigo (*fix M4*) |
| Fase final do TL | última fase, mesmo se pior | melhor entre fase 2 e 3 (*fix*) |
| Proveniência | sem commit/data | commit + params + data gravados |

→ Cada comparação é um **experimento controlado**: a única variável entre células é o fator em
teste. Defensável no artigo.

---

## 12. Limitações

1. **Uma execução por cenário (sem barra de erro).** Diferenças pequenas — sobretudo os Δ de
   oclusão/blur (+0,7pp) e a ordem D-GEOM<D-OCCL<D-FULL — podem ter ruído de seed. Idealmente 2–3
   seeds para o artigo.
2. **Fase 3 inútil nos dados atuais** — degradou em 4/4; recomendado cortar/endurecer o gating.
3. **Scratch ainda overfita** mesmo com aug (B-FULL gap 37,1pp) — o limite do scratch é a
   inicialização, não a augmentation.
4. **Dataset pequeno** (~160 cenas distintas) — origem de toda a história de overfitting; os ganhos
   podem não transferir igual para um dataset mais diverso.

---

## 13. Conclusões

> **A receita campeã é Transfer Learning + augmentação geométrica forte.** O D-FULL (W₀ = COCO +
> flip + RandomBBoxTransform + oclusão + blur) atinge **PCK@0.2 = 67,54%** — **+25,8pp sobre o
> baseline** e **+14,7pp sobre a célula TL-sem-aug da matriz (C-RAW)**. Vence em todas as métricas,
> **generaliza quase perfeitamente** (gap 1,1pp) e **resolve as extremidades** (acima do baseline
> COCO pela primeira vez).

Conclusões de suporte:
- **TL > scratch, sempre** — na matriz, **+15,1pp (sem aug) / +9,4pp (com aug)**. Até sem aug
  nenhuma, o C-RAW supera o A-FLIP (scratch com flip). Comprovado por ablação com W₀ aleatório, não
  por suposição.
- **A augmentation que paga é geométrica** (~90% do ganho): flip + rotação/escala/shift atacam a
  baixa diversidade do dataset; oclusão e blur são refino marginal mas consistente.
- **TL e aug combinam (sub-aditivos −5,7pp):** na matriz a aug vale **+20,3pp (scratch) / +14,7pp
  (TL)** e o TL **+15,1 / +9,4pp** — ambos fortes; juntos levam o scratch cru (37,8%) a 67,5%.
- **As extremidades só se resolvem com TL + augmentação** — A-RAW, C-RAW e B-FULL ficam todos
  abaixo do baseline nos membros; só o D-FULL passa. Nem o TL nem a aug sozinhos bastam.
- **O progressive unfreezing vale** (+3,6pp e menos overfitting), mas **a fase 3 deve ser cortada**
  (degradou em todos os casos).

---

## 14. Próximos passos

- [ ] **2–3 seeds** por cenário (barra de erro), prioridade nos Δ pequenos (oclusão/blur) e na
  ordenação dos D.
- [ ] **Cortar a fase 3** (ou subir `--delta-pck` ≥0,10) e re-confirmar — economiza GPU sem perda.
- [ ] (Opcional) isolar geom/ocl/blur **no lado scratch** (B-GEOM/B-OCCL), só se valer aprofundar.
- [ ] **Alimentar o artigo RNP** (Seções 3 e 4) com a matriz completa e o D-FULL como resultado
  principal; reportar a escada de augmentation e a recuperação das extremidades.

---

## 15. Proveniência / reprodutibilidade

```
A-FLIP / C-FLIP            results/runs/20260607_033902/      (commit a0df5e5)
C2                        results/runs/20260607_051809_c2/
A-RAW / C-RAW             results/runs/20260607_195315_raw/  (commit 261f73e)
B-FULL / D-GEOM/OCCL/FULL results/runs/20260608_014649_bd/   (commit 5cb9d48)
host        RTX 4090 (Docker, --shm-size=16g)
params      scratch: EPOCHS=150 · TL: fases 45/60/45 · BATCH=64 (todos)
augments    RandomBBoxTransform(rot±30, escala 0.75–1.25, shift 0.1)
            SimpleRandomErasing(área 2–10%, prob 0.3) · SimpleMotionBlur(3–9px, p 0.5)
artefatos   checkpoints/<cenario>/best_PCK.pth
            tables/finetuned_cenario_*_{train,val}.json
            logs/ · SUMMARY.md · PROVENANCE.txt
```
Reproduzir: `bash scripts/run_experiments.sh` (A-FLIP+C-FLIP) · `bash scripts/run_c2.sh` (C2) ·
`bash scripts/run_raw.sh` (A-RAW+C-RAW) · `bash scripts/run_bd.sh` (B-FULL+D-GEOM/OCCL/FULL).
Transforms custom em `src/football_orient_pose/finetuning/transforms.py`. Ver
`docs/finetuning/epico-1/guia.md`. Detalhe de cada metade: `epic2-relatorio-a-c.md` e
`epic2-relatorio-bd.md`.
