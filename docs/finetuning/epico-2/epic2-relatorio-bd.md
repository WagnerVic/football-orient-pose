# Relatório B/D do Épico 2 — Augmentation (fecha a matriz e as perguntas abertas)

**Projeto:** Football Orient Pose — Transfer Learning (RNP)
**Run:** `results/runs/20260608_014649_bd/` (commit `5cb9d48`)
**GPU:** RTX 4090 (Docker) · **Data:** 08/06/2026
**Status:** 🟢 **B-FULL, D-GEOM, D-OCCL, D-FULL** concluídos — matriz 2×2 **fechada**

> **Complemento do relatório parcial** (`epic2-relatorio.md`, que cobria A/C/RAW/C2). Este foca
> nos cenários **com augmentation** e nas **perguntas que o parcial deixou em aberto** (§12 dele).
> Os dois serão **fundidos depois** num relatório final. Aqui A-FLIP/C-FLIP/baseline aparecem só
> como **linhas de referência**, não são re-documentados.

**Nomenclatura (degraus de aug deste relatório):**
- **B-FULL** (= cenário B) — from scratch + flip + geométrica + oclusão + blur.
- **D-GEOM** — TL (progressive) + flip + **geométrica** (`RandomBBoxTransform`).
- **D-OCCL** — D-GEOM + **oclusão** (`SimpleRandomErasing`).
- **D-FULL** (= cenário D) — D-OCCL + **blur** (`SimpleMotionBlur`).

O lado TL é um **ladder fino**: cada degrau adiciona **uma** augmentation, isolando o efeito de
cada mecanismo.

---

## Resumo executivo

- **D-FULL é o novo melhor modelo do projeto: PCK@0.2 = 67,54%** — **+9,15pp sobre o C-FLIP**
  (58,39%, melhor do parcial) e **+25,8pp sobre o baseline** zero-shot (41,76%). Vence em **todas**
  as métricas (PDJ 96,6%, OKS 89,8%, MPJPE 3,08px).
- **As extremidades foram RESOLVIDAS.** Pela primeira vez, punho/cotovelo/joelho/tornozelo ficam
  **acima do baseline COCO** — o problema aberto do parcial. Ex.: ankle 67,2% vs 59,4%; knee
  64,8% vs 58,3%; elbow 56,4% vs 50,8%; wrist 46,1% vs 43,5%.
- **O ganho da augmentation é quase todo geométrico.** Atribuição: **geométrica +7,77pp**,
  oclusão +0,71pp, blur +0,67pp. Confirma a tese da ablação do flip (a alavanca neste dataset
  redundante é geométrica — diversidade de pose/ângulo).
- **TL + aug forte generaliza quase perfeito:** gap treino→val do D-FULL = **1,1pp** (vs 11,7pp
  do C-FLIP). A augmentation **fecha o gap**.
- **B-FULL (scratch + aug) = 58,13%:** a aug ajuda muito o from-scratch (+12pp sobre A-FLIP), mas
  ele **ainda overfita forte** (gap 37,1pp, train 95,2%) e **não recupera as extremidades**. →
  **augmentation regulariza, mas não substitui o transfer learning.**

---

## 1. O que estava aberto (e que este relatório responde)

O relatório parcial fechou a metade da matriz (TL > scratch, robusto) mas deixou 4 perguntas no
§12, todas dependentes dos cenários com augmentation:

1. **B** — a augmentation regulariza o overfitting do from-scratch (A-FLIP tinha gap 47,6pp)?
2. **D** — a aug recupera as extremidades (abaixo do baseline em **todos** os cenários do parcial)
   e supera os 58,4% do C-FLIP?
3. **Mecanismo** — qual augmentation paga (olhando o breakdown por grupo, não só o PCK global)?
4. **Gating** — revisar a ativação da fase 3 do progressive unfreezing.

### O ladder completo de augmentation
O parcial estabeleceu que o flip já é augmentation (baseline = RAW). Este relatório adiciona os
degraus "fortes". Ladder por inicialização:

| Init | RAW | +flip | +geom | +geom+ocl | +geom+ocl+blur |
|---|---|---|---|---|---|
| From scratch | A-RAW | A-FLIP | — | — | **B-FULL** |
| Transfer learning | C-RAW | C-FLIP | **D-GEOM** | **D-OCCL** | **D-FULL** |

No lado scratch rodamos só o topo (B-FULL): a decisão "TL > scratch" já estava tomada, então
gastar o ladder fino no lado que perde não se justifica — basta a célula 2×2 (scratch + aug).

### Referência (do parcial, val split)
| Modelo | PCK@0.2 | papel aqui |
|---|---:|---|
| Baseline zero-shot (COCO) | 41,76% | piso (o que o RTMPose já entrega sem fine-tune) |
| A-FLIP (scratch, +flip) | 46,06% | melhor scratch do parcial |
| C-FLIP (TL, +flip) | 58,39% | melhor modelo do parcial — alvo a superar |

---

## 2. Protocolo (só o que muda vs o parcial)

Tudo **idêntico** ao parcial — dataset 3DSP (split por clip 160/40), RTMPose-X, **150 épocas**
(scratch de uma vez; TL em **45/60/45**), **batch 64**, AdamW, LR scheduler proporcional,
**seleção por PCK@0.2 estrito** (`StrictPCKMetric`). **A única diferença são os augments novos:**

| Augment | Implementação | Params | Posição no pipeline |
|---|---|---|---|
| **Geométrica** | `RandomBBoxTransform` (mmpose) | rot **±30°**, escala **0,75–1,25**, shift **0,1** | após `RandomFlip`, antes do `TopdownAffine` |
| **Oclusão** | `SimpleRandomErasing` (custom) | 1 patch, área 2–10%, prob 0,3 | após `TopdownAffine` (imagem 288×384) |
| **Blur** | `SimpleMotionBlur` (custom) | kernel 3–9px, direcional, p 0,5 | após a geométrica |

**Params geométricos conservadores** de propósito: os crops são apertados e os jogadores ~verticais,
então ±80° do default RTMPose viraria jogador de cabeça pra baixo (pose irreal) e escalas extremas
cortariam joints.

> **Nota técnica:** os transforms `Albu` e `RandomErasing` do mmpose **não estão registrados** na
> versão do container (erro `... is not in the mmpose::transform registry`). Por isso
> implementamos equivalentes diretos (`SimpleMotionBlur` via cv2, `SimpleRandomErasing` via numpy),
> registrados em `football_orient_pose.finetuning.transforms`. O `RandomBBoxTransform` é nativo e
> funcionou direto.

---

## 3. Resultados B/D (val split)

| Modelo | aug | PCK@0.2 ↑ | PCK train | gap | PDJ@0.5 ↑ | OKS ↑ | MPJPE-2D ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| *Baseline zero-shot* | — | *41,76%* | — | — | *93,62%* | *81,82%* | *4,81 px* |
| *A-FLIP — scratch (ref.)* | *+flip* | *46,06%* | *93,71%* | *47,6pp* | *89,96%* | *79,39%* | *5,56 px* |
| *C-FLIP — TL (ref.)* | *+flip* | *58,39%* | *70,05%* | *11,7pp* | *93,80%* | *85,26%* | *4,09 px* |
| **B-FULL** — scratch | +geom+ocl+blur | 58,13% | 95,22% | 37,1pp | 93,59% | 85,26% | 4,09 px |
| **D-GEOM** — TL | +geom | 66,16% | 68,71% | 2,5pp | 96,05% | 89,06% | 3,24 px |
| **D-OCCL** — TL | +geom+ocl | 66,87% | 68,94% | 2,1pp | 96,23% | 89,48% | 3,16 px |
| **D-FULL** — TL | +geom+ocl+blur | **67,54%** | 68,60% | **1,1pp** | **96,60%** | **89,84%** | **3,08 px** |

**Leitura:**
- **D-FULL domina tudo** — supera o C-FLIP (melhor do parcial) em **+9,15pp** de PCK e melhora
  também PDJ (+2,8), OKS (+4,6) e MPJPE (−1,01px). É o novo estado-da-arte do projeto.
- **Os 3 degraus do TL melhoram monotonicamente:** 66,16 → 66,87 → 67,54. Cada augmentation
  adicionada **soma** (nenhuma piora), mas com retornos muito diferentes (ver §4).
- **B-FULL ≈ C-FLIP no PCK** (58,13 vs 58,39), mas com **overfitting brutalmente maior** (gap
  37,1 vs 11,7pp): scratch+aug "alcança" o TL+flip no número global, **decorando** — não é a
  mesma qualidade (ver §5 e §7).
- **MPJPE do D-FULL = 3,08px** — o menor erro absoluto já medido (baseline 4,81; C-FLIP 4,09).

---

## 4. Atribuição por mecanismo (o núcleo da análise)

Como o lado TL é um ladder fino (cada degrau = +1 augmentation), o ganho de cada mecanismo é a
diferença entre degraus consecutivos:

| Mecanismo | comparação | Δ PCK@0.2 |
|---|---|---:|
| **geométrica** (rot/escala/shift) | D-GEOM − C-FLIP | 🔵 **+7,77pp** |
| **oclusão** (erasing) | D-OCCL − D-GEOM | +0,71pp |
| **blur** (motion blur) | D-FULL − D-OCCL | +0,67pp |
| **total (aug forte)** | D-FULL − C-FLIP | **+9,15pp** |

**Conclusões:**
- **A augmentação geométrica é a alavanca** — sozinha responde por **+7,77pp dos +9,15pp**
  (~85% do ganho). Isso **confirma empiricamente a tese da ablação do flip** (parcial §7.2): num
  dataset com ~160 cenas distintas (20 frames quase iguais/clip), o que paga é **multiplicar a
  diversidade efetiva de pose/ângulo** — e rotação/escala/shift fazem exatamente isso, em escala
  muito maior que o flip.
- **Oclusão e blur são refino marginal, mas reais e consistentes.** +0,71 e +0,67pp de PCK não
  impressionam, mas note que também melhoram PDJ, OKS **e** MPJPE em cada degrau (3,24 → 3,16 →
  3,08px) — ou seja, não são ruído: empurram o modelo de forma estável. Para o artigo: valem como
  "polimento", não como motor.
- **Implicação prática:** se o orçamento de experimentos fosse apertado, **a augmentação
  geométrica seria a primeira (e quase suficiente) escolha**; oclusão e blur entram para o último
  ~1,4pp.

---

## 5. Diagnóstico de overfitting (framework dos 3 números)

| Cenário | PCK **train** | PCK **val** | gap | leitura |
|---|---:|---:|---:|---|
| *A-FLIP (scratch, +flip)* | *93,71%* | *46,06%* | *47,6pp* | 🔴 overfitting severo |
| **B-FULL** (scratch, +aug) | 95,22% | 58,13% | **37,1pp** | 🟠 overfitting ainda alto |
| *C-FLIP (TL, +flip)* | *70,05%* | *58,39%* | *11,7pp* | 🟢 generaliza bem |
| **D-GEOM** (TL, +geom) | 68,71% | 66,16% | **2,5pp** | 🟢 generaliza quase perfeito |
| **D-OCCL** (TL, +geom+ocl) | 68,94% | 66,87% | **2,1pp** | 🟢 idem |
| **D-FULL** (TL, +aug) | 68,60% | 67,54% | **1,1pp** | 🟢 gap quase nulo |

- **A augmentation encolhe o gap, mas o efeito depende da inicialização.**
  - **No scratch:** A-FLIP 47,6pp → B-FULL 37,1pp (−10,5pp de gap). A aug regulariza, mas o scratch
    **continua decorando** (train 95,2%!) — a capacidade do modelo sem prior é grande demais para
    ~160 cenas.
  - **No TL:** C-FLIP 11,7pp → D-FULL **1,1pp**. Aqui a aug **praticamente elimina** o overfitting.
- **Por que o gap do D é tão pequeno?** A augmentação geométrica pesada torna o **treino mais
  difícil** (cada época vê variações que o modelo nunca decorou), então a PCK de treino **não
  infla** (68,6%) e fica colada na de val (67,5%). É o comportamento ideal: val ≈ train, ambos
  altos.
- **Resposta direta à pergunta 1 (B):** sim, a augmentation regulariza o from-scratch (+12pp de
  PCK, −10,5pp de gap), **mas não basta** — o gap de 37,1pp e o train 95,2% mostram que sem o prior
  do COCO o modelo ainda memoriza. **O transfer learning continua necessário.**

---

## 6. Progressive unfreezing e gating dos cenários D

Os 3 cenários TL usaram as 3 fases (`frozen_stages` 4→2→1). PCK@0.2 (estrito, val) por fase:

| Cenário | Fase 1 (só cabeça) | Fase 2 (+topo) | Fase 3 (+baixo) | Δ f2−f1 | Selecionado |
|---|---:|---:|---:|---:|---|
| **D-GEOM** | 59,4% | **66,2%** | 60,8% (degradou) | 0,067 | fase 2 → 66,16% |
| **D-OCCL** | 59,0% | **66,9%** | 61,7% (degradou) | 0,078 | fase 2 → 66,87% |
| **D-FULL** | 59,6% | **67,5%** | 63,1% (degradou) | 0,079 | fase 2 → 67,54% |

**Achados:**
- **Padrão idêntico nos 3 (e ao C-FLIP do parcial): a fase 2 é onde o ganho acontece**, e a
  **fase 3 sempre degrada.** Destravar os stages baixos (features de baixo nível) com só 3.200
  imagens causa overfitting — é uma propriedade do dataset, não do augmentation.
- **A fase 3 foi ativada nos 3** (Δ f2−f1 = 0,067/0,078/0,079, todos > 0,05) e **piorou nos 3**.
  O **fix de seleção de checkpoint** (mantém o melhor entre fase 2 e 3) **salvou +5,3 / +5,2 /
  +4,5pp** respectivamente. Sem ele, o pipeline entregaria a fase 3 pior nos três casos.
- **Resposta direta à pergunta 4 (gating):** o gating de 5pp está **ativando a fase 3 à toa** —
  em 4 de 4 cenários TL com flip/aug ela degradou. Recomendação: **subir o `--delta-pck`** (ex.:
  para ≥0,10) **ou simplesmente remover a fase 3** do pipeline. O fix de seleção já protege o
  resultado, mas a fase 3 só gasta ~45 épocas de GPU sem nunca ganhar.

> Já na **fase 1** (backbone 100% congelado) os D atingem ~59% — sozinho isso bate o C-FLIP
> inteiro (58,4%). Ou seja: **adaptar só a cabeça, com os dados aumentados, já supera o melhor do
> parcial.** O ganho da fase 2 (+7 a +8pp) vem de destravar o topo do backbone.

---

## 7. Análise por grupo anatômico — **extremidades resolvidas**

### PCK@0.2 por grupo (%) — val

| Grupo | Baseline | C-FLIP (ref.) | B-FULL | D-GEOM | D-OCCL | **D-FULL** |
|---|---:|---:|---:|---:|---:|---:|
| head | 50,4 | 76,5 | 87,6 | 87,0 | 88,4 | **89,5** |
| shoulder | 30,4 | 71,0 | 70,5 | 78,2 | 78,3 | **78,4** |
| elbow | 50,8 | 45,5 | 42,6 | 54,3 | 55,0 | **56,4** |
| wrist | 43,5 | 37,4 | 33,6 | 45,4 | 45,1 | **46,1** |
| hip | 22,8 | 60,6 | 60,2 | 68,3 | 69,5 | **69,3** |
| knee | 58,3 | 52,9 | 53,5 | 62,3 | 63,2 | **64,8** |
| ankle | 59,4 | 57,2 | 54,0 | 65,2 | 67,2 | **67,2** |

### Extremidades vs baseline COCO (✅ = passou o baseline)

O parcial fechou com uma ressalva honesta: **nenhum** modelo alcançava o baseline nas
extremidades (punho/cotovelo/joelho/tornozelo). Os cenários D **viram esse jogo**:

| Grupo | baseline | D-GEOM | D-OCCL | D-FULL |
|---|---:|---:|---:|---:|
| wrist | 43,5 | 45,4 ✅ | 45,1 ✅ | **46,1 ✅** |
| elbow | 50,8 | 54,3 ✅ | 55,0 ✅ | **56,4 ✅** |
| knee | 58,3 | 62,3 ✅ | 63,2 ✅ | **64,8 ✅** |
| ankle | 59,4 | 65,2 ✅ | 67,2 ✅ | **67,2 ✅** |

**Já o D-GEOM (só geométrica) recupera todas as 4 extremidades** acima do baseline — reforçando
que a alavanca é geométrica. **B-FULL (scratch), porém, NÃO recupera nenhuma** (wrist 33,6 <
43,5; elbow 42,6 < 50,8; knee 53,5 < 58,3; ankle 54,0 < 59,4): a aug sozinha, sem o prior do COCO,
não resolve os membros. **É a combinação TL + aug geométrica que conserta as extremidades.**

### Ganho por grupo (D-FULL − C-FLIP, pp de PCK)
head **+13,0** · knee **+11,9** · elbow **+10,9** · ankle **+10,0** · wrist **+8,7** · hip +8,7 ·
shoulder +7,4. O ganho é **transversal** — e, crucialmente, **as extremidades sobem tanto quanto
o tronco** (knee/elbow/ankle entre os maiores ganhos), exatamente onde o parcial apontava a
fraqueza.

### MPJPE-2D por grupo (px, ↓ melhor) — val

| Grupo | C-FLIP (ref.) | B-FULL | D-GEOM | D-OCCL | **D-FULL** |
|---|---:|---:|---:|---:|---:|
| head | 2,50 | 1,62 | 1,43 | 1,47 | **1,40** |
| shoulder | 2,40 | 2,40 | 1,92 | 1,89 | **1,86** |
| elbow | 5,00 | 5,35 | 3,91 | 3,77 | **3,67** |
| wrist | 7,80 | 8,05 | 5,91 | 5,76 | **5,63** |
| hip | 2,90 | 2,91 | 2,47 | 2,38 | **2,40** |
| knee | 4,20 | 4,34 | 3,56 | 3,47 | **3,35** |
| ankle | 6,10 | 6,08 | 4,39 | 4,64 | **4,39** |

O erro de **punho cai de 7,80 → 5,63px** e o de **tornozelo de 6,10 → 4,39px** (vs C-FLIP) — os
dois joints historicamente piores tiveram a maior redução absoluta.

**Resposta direta à pergunta 2 (D):** sim — o D-FULL recupera **todas** as extremidades acima do
baseline **e** supera o C-FLIP (67,5% vs 58,4%). E à pergunta 3 (mecanismo): o ganho nas
extremidades vem da **augmentação geométrica** (já visível no D-GEOM); oclusão/blur só refinam.

---

## 8. Interação Transfer Learning × Augmentation (matriz 2×2 fechada)

Com B/D, a matriz original fica completa (usando o flip como nível "baixo" de aug e o full como
"alto"):

| | +flip (baixo) | +full (alto) | **efeito da aug** |
|---|---:|---:|---:|
| **From scratch** | A-FLIP 46,06 | B-FULL 58,13 | **+12,07pp** |
| **Transfer learning** | C-FLIP 58,39 | D-FULL 67,54 | **+9,15pp** |
| **efeito do TL** | **+12,33pp** | **+9,41pp** | |

**Leitura:**
- **Os dois fatores são fortemente positivos e em escala parecida** (~+9 a +12pp cada).
- **São aproximadamente aditivos, com leve retorno decrescente.** Partindo do A-FLIP (46,06),
  somar os efeitos isolados daria +12,33 (TL) +12,07 (aug) = +24,4pp → ~70%; o D-FULL real é
  67,54 (+21,48pp). A interação é **sub-aditiva (~−2,9pp)**: TL e aug atacam parte do **mesmo**
  problema (overfitting por baixa diversidade), então combiná-los rende um pouco menos que a soma.
- **A melhor receita é TL + augmentation forte (D-FULL).** Cada fator sozinho leva a ~58%; juntos,
  a 67,5%.

---

## 9. Respostas às perguntas abertas (fecho)

| # | Pergunta (do §12 do parcial) | Resposta |
|---|---|---|
| 1 | A augmentation regulariza o overfitting do from-scratch? | **Parcialmente.** B-FULL: +12pp de PCK e gap 47,6→37,1pp vs A-FLIP. Mas ainda decora (train 95,2%) e não recupera extremidades. **Aug não substitui o TL.** |
| 2 | A aug recupera as extremidades e supera o C-FLIP? | **Sim, decisivamente.** D-FULL 67,5% (> C-FLIP 58,4%), e **todas** as extremidades acima do baseline pela 1ª vez (ankle 67,2 vs 59,4 etc). |
| 3 | Qual mecanismo de aug paga? | **A geométrica** (+7,77 de +9,15pp; ~85%). Oclusão (+0,71) e blur (+0,67) são refino marginal mas consistente. Confirma a tese do flip. |
| 4 | Revisar o gating da fase 3. | **A fase 3 degrada sempre** (4/4 cenários TL). Recomendação: subir `--delta-pck` para ≥0,10 ou remover a fase 3. O fix de seleção já evita entregar o pior. |

---

## 10. Limitações e próximos passos

- **Uma execução por cenário (sem barra de erro).** Os Δ de oclusão/blur (+0,7pp) são pequenos o
  bastante para ter ruído de seed — para o artigo, idealmente 2–3 seeds, sobretudo para sustentar
  a ordem D-GEOM < D-OCCL < D-FULL.
- **Fase 3 inútil nos dados atuais** — recomendado cortar/endurecer o gating (§6/§9).
- **B-FULL ainda overfita** (gap 37,1pp) — confirma que o limite do scratch é a inicialização, não
  a augmentation.
- **Dataset pequeno** (~160 cenas distintas) — toda a história de overfitting decorre daí.
- **A fundir:** unir este relatório B/D com o parcial (`epic2-relatorio.md`) no relatório final do
  Épico 2, com a matriz completa, e alimentar o artigo RNP (Seções 3 e 4) com o D-FULL como
  resultado principal.

---

## 11. Proveniência / reprodutibilidade

```
B-FULL / D-GEOM / D-OCCL / D-FULL   results/runs/20260608_014649_bd/   (commit 5cb9d48)
host        RTX 4090 (Docker, --shm-size=16g)
params      scratch (B): EPOCHS=150 · TL (D): fases 45/60/45 · BATCH=64 (todos)
augments    RandomBBoxTransform(rot±30, escala 0.75–1.25, shift 0.1)
            SimpleRandomErasing(área 2–10%, prob 0.3) · SimpleMotionBlur(3–9px, p 0.5)
artefatos   checkpoints/<cenario>/best_PCK.pth
            tables/finetuned_cenario_*_{train,val}.json   (8 JSONs)
            logs/ · SUMMARY.md · PROVENANCE.txt
```
Reproduzir: `bash scripts/training/run_bd.sh` (no container `football-finetuning:latest`, pesos COCO baked
ou em `checkpoints/rtmpose-x_coco.pth`). Transforms custom em
`src/football_orient_pose/finetuning/transforms.py`.
