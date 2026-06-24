# Trabalho Final — Redes Neurais Profundas
**Disciplina:** Redes Neurais Profundas

**Grupo:** 
    - Wagner Victor Alves de Menezes
    - Raphael Alves de Lima Soares
    - Victor Gabriel
    - João Victor Fernandes
    - Júlia Junior
---

## 1. Pergunta de Pesquisa

Modelos de estimação de pose como o RTMPose, quando aplicados a transmissões de futebol broadcast, apresentam um comportamento contraditório: detectam os keypoints corporais com alta taxa de acerto geral (PDJ@0.5 = 93%), mas falham em localizá-los com precisão suficiente (PCK@0.2 = 41%). Esse gap é agravado pelas condições desafiadoras do domínio — imagens de baixa resolução (crops de 100×100px), motion blur e poses atípicas de atletas em movimento. **Como detectar e localizar keypoints corporais de jogadores de futebol de forma automática e precisa, maximizando o PCK, de modo que seja possível extrair informações confiáveis para análise tática?**

---

## 2. Metodologia

A metodologia é experimental: cada decisão de projeto é sustentada por um experimento controlado cujo resultado a decide, em vez de afirmada por autoridade. O núcleo é a matriz 2×2 (§2.4), que adapta o estimador ao domínio do futebol variando dois fatores — a inicialização dos pesos (transfer learning a partir do COCO vs. treino do zero) e o uso de data augmentation —, todos medidos sob o mesmo protocolo. Em torno desse núcleo, a metodologia define: o modelo-base, escolhido por um benchmark zero-shot (§2.1–2.2); o dataset e o protocolo de treino (§2.3–2.6); e o conjunto de métricas e o diagnóstico de generalização que avaliam cada condição (§2.7).

### 2.1 Seleção do Modelo

O primeiro passo do trabalho é **descobrir qual modelo usar** — e essa escolha é, ela própria, resultado de um experimento, não de uma preferência. O método de seleção é um **benchmark zero-shot**: três estimadores de pose pré-treinados em COCO são aplicados **sem nenhuma adaptação** aos crops de jogadores do dataset 3DSP e comparados nas mesmas quatro métricas (definidas na §2.7):

- **OpenPose** — abordagem *bottom-up* (Part Affinity Fields), sem caixa de pessoa dedicada;
- **HRNet-W48** — *top-down* baseado em *heatmaps* de alta resolução;
- **RTMPose-X** — *top-down* com cabeça SimCC (classificação de coordenadas).

O modelo que vencer o benchmark torna-se o **modelo-base** do estudo de fine-tuning.

### 2.2 Modelo

Uma vez escolhido o modelo (§3.1), ele é descrito aqui. O modelo-base é o **RTMPose-X** (Jiang et al., 2023), arquitetura de estimação de pose *top-down* com backbone CSPNeXt-X, neck CSPNeXtPAFPN e cabeça SimCC. O **SimCC** (*Simplified Coordinate Classification*) formula a predição de cada keypoint como dois problemas de classificação independentes (eixos X e Y), o que oferece precisão sub-pixel teórica e **evita o erro de quantização** dos modelos baseados em heatmap (§3.1).

### 2.3 Dataset

O dataset utilizado é o **3DSP** (*3D Shot Posture*, Yeung et al., 2024): 200 clips de futebol broadcast, cada um com 20 frames, totalizando 4.000 imagens. Cada imagem é um crop 100×100 px centrado em um jogador, anotado com 17 keypoints 2D no formato **H3WB** e uma bounding box. A divisão adotada é 80/20 **por clip**: 160 clips para treino e 40 clips para validação (3.200 / 800 frames). A divisão é feita por clip, nunca por frame, para evitar *data leakage* entre frames consecutivos do mesmo chute.

### 2.4 Desenho Experimental

A questão central — quais ingredientes do fine-tuning maximizam a precisão de localização (PCK) no domínio de futebol? — é respondida por uma matriz 2×2 que varia dois fatores independentes.

**Fator 1 — Inicialização dos pesos (W₀):**
- *Transfer Learning*: W₀ são os pesos do modelo pré-treinado no COCO (fornecidos oficialmente pela OpenMMLab).
- *From Scratch*: W₀ são pesos aleatórios.

Esse fator isola o efeito da inicialização dos pesos: o cenário from scratch (pesos aleatórios) funciona como controle, de modo que qualquer ganho do transfer learning, sob protocolo idêntico, possa ser atribuído aos pesos do COCO — e não pressuposto.


**Fator 2 — Data augmentation.** A condição **"sem augmentation" é a imagem crua (RAW)** — sem nenhuma transformação, nem o *flip* horizontal. A condição **"com augmentation"** aplica uma **escada aditiva** de transformações, cada uma somada à anterior: **flip** horizontal → **geométrica** (`RandomBBoxTransform`: rotação ±30°, escala 0,75–1,25, deslocamento 0,1) → **oclusão** (apagamento de um retângulo, 2–10% da área, p=0,3) → **borrão de movimento** (kernel direcional 3–9 px, p=0,5). Os parâmetros geométricos são **conservadores de propósito**: como os crops são apertados e os jogadores ficam aproximadamente verticais, os ±80° do default do RTMPose gerariam poses irreais (jogador de cabeça para baixo) e escalas extremas cortariam keypoints.

| | Sem Augmentation (RAW) | Com Augmentation |
|---|---|---|
| **From Scratch** | Cenário A | Cenário B |
| **Transfer Learning** | Cenário C | Cenário D |

> O *flip* horizontal, sozinho, já é uma augmentation forte (ablação na §3.3); tratá-lo como parte do "sem augmentation" mascararia o efeito real do augmentation. Por isso o baseline honesto "sem augmentation" é o **RAW (sem flip)**, e o *flip* passa a ser o **primeiro degrau** da escada.

### 2.5 Estratégia de Treinamento: Transfer Learning via *Progressive Unfreezing*

Os cenários de transfer learning (C e D) não treinam todas as camadas de uma vez. Eles usam **Progressive Unfreezing** (também chamado *gradual unfreezing*) combinado com **fine-tuning discriminativo** — técnicas introduzidas por **Howard e Ruder (2018)** no ULMFiT. A ideia é descongelar o backbone **de cima para baixo, em fases**, e aplicar uma taxa de aprendizado maior na cabeça (mais específica) e menor no backbone (mais geral) — o fine-tuning discriminativo de Howard e Ruder (2018).

No nosso pipeline, são **três fases**, cada uma partindo do *best checkpoint* da anterior:

| Fase | `frozen_stages` | O que treina | LR cabeça | LR backbone |
|---|---|---|---|---|
| 1 | 4 (backbone 100% congelado) | só a cabeça | 1×10⁻³ | — |
| 2 | 2 | cabeça + topo do backbone | 1×10⁻⁴ | 1×10⁻⁵ |
| 3 *(condicional)* | 1 | cabeça + mais camadas | 1×10⁻⁴ | 1×10⁻⁶ |

A **fase 3 só é ativada** se o ganho da fase 2 sobre a fase 1 superar um limiar de Δ PCK (*gating*); caso contrário, ela é pulada. O modelo final é o melhor *checkpoint* entre as fases executadas.


O descongelamento em fases, em vez do treino conjunto de todas as camadas, apoia-se em três resultados da literatura de transfer learning, confirmados pelos nossos experimentos:

1. **Camadas baixas são gerais; altas são específicas** (Yosinski et al., 2014). As primeiras camadas aprendem features genéricas (bordas, cor) transferíveis entre tarefas, enquanto as últimas são específicas do COCO. Faz sentido **preservar as baixas** e **adaptar primeiro a cabeça** ao domínio de futebol.
2. **Treinar tudo de uma vez distorce as features pré-treinadas** (Kumar et al., 2022). Quando a cabeça é aprendida do zero ao mesmo tempo em que o backbone é destravado, as camadas baixas se movem para acomodar uma cabeça ainda ruim, **destruindo** as representações úteis do COCO — efeito pior sob *distribution shift* (que é exatamente o nosso caso: COCO → futebol broadcast). Treinar a cabeça primeiro (fase 1, equivalente a um *linear probing*) e só então liberar o backbone (fase 2) evita essa distorção.
3. **Ajustar um subconjunto de camadas pode superar ajustar todas** (Lee et al., 2023). Com um dataset pequeno (~160 cenas distintas), tunar parâmetros demais causa esquecimento das features pré-treinadas; por isso a fase 3 (que destrava as camadas mais baixas) é **condicional** — empiricamente ela tende a degradar a performance, comportamento previsto por essa literatura.

Em suma, o *progressive unfreezing* é uma **busca guiada** pelo espaço de pesos: explora primeiro o subespaço da cabeça, depois o do topo do backbone, mantendo congeladas as dimensões que o COCO já resolveu bem.

### 2.6 Ordem de Prioridade dos Experimentos

Os experimentos são executados na ordem A → C → D → B, por ordem decrescente de prioridade. A lógica é incremental: A e C respondem a pergunta principal (TL é útil?). O Cenário D (TL + augmentation), esperado como melhor resultado, tem prioridade alta. O Cenário B (from scratch + augmentation) completa a matriz e isola o efeito do augmentation sem o prior do COCO.

### 2.7 Avaliação

Cada cenário é avaliado com quatro métricas calculadas no val split (800 imagens):

- **PDJ@0.5** (*Percentage of Detected Joints*) — fração de keypoints cuja distância ao GT é menor que 50% do comprimento do torso. Mede **detecção** (o keypoint caiu na região certa?).
- **PCK@0.2** (*Percentage of Correct Keypoints*) — **métrica principal**. Fração de keypoints cuja distância ao GT é menor que 20% da distância de referência (ombros ou quadris). Mais exigente que o PDJ, sensível à **precisão de localização**.
- **OKS** (*Object Keypoint Similarity*) — similaridade ponderada por visibilidade, análoga ao mAP do COCO.
- **MPJPE-2D** (*Mean Per-Joint Position Error*) — erro médio absoluto, em pixels, por keypoint visível. Útil para entender **onde** o erro se concentra (ex.: extremidades).

O diagnóstico de cada cenário usa o **framework dos 3 números**: (1) métrica no treino, (2) métrica no val, (3) baseline zero-shot (41,76% PCK). Essa comparação identifica overfitting, underfitting ou generalização adequada sem depender de curvas de loss, que podem ser enganosas.

---

## 3. Resultados

### 3.1 Seleção do Modelo — o RTMPose-X vence em todas as métricas

**O RTMPose-X foi selecionado como modelo-base por superar o HRNet-W48 e o OpenPose em todas as quatro métricas no 3DSP zero-shot.** A Tabela 1 mostra o benchmark no val split (800 frames):

| Modelo (zero-shot) | PDJ@0.5 ↑ | PCK@0.2 ↑ | OKS ↑ | MPJPE-2D ↓ |
|---|---:|---:|---:|---:|
| OpenPose | 56,1% | 22,1% | 48,5% | 25,58 px |
| HRNet-W48 | 88,9% | 40,5% | 76,2% | 6,04 px |
| **RTMPose-X** | **93,6%** | **41,8%** | **81,8%** | **4,81 px** |

*Tabela 1 — Benchmark zero-shot dos três estimadores no 3DSP (val).*


**Por que o RTMPose vence.** A diferença vem da arquitetura da cabeça e é a esperada para o regime de baixa resolução do domínio. O **HRNet-W48** (Sun et al., 2019) é baseado em *heatmaps*: localiza cada keypoint pelo *argmax* de um mapa de calor de resolução `64×48` (1/4 da entrada 256×192), sem refinamento sub-pixel. Como o crop do jogador (100×100 px) é redimensionado a essa grade, cada célula do *heatmap* corresponde a ~1,5–2 px na imagem original, e o *argmax* só pode apontar posições inteiras da grade — um **erro de quantização** de até ~2 px por keypoint, acumulado nos 17 keypoints. O **RTMPose-X** (Jiang et al., 2023) usa a cabeça **SimCC** (Li et al., 2022), que reformula a localização como classificação de coordenada em *bins* sub-pixel, sem grid, eliminando essa quantização; o próprio artigo do SimCC mostra que a vantagem sobre métodos de *heatmap* é **maior justamente em baixa resolução** — exatamente o caso dos nossos crops 100×100. Isso explica os 93,6% vs 88,9% no PDJ e, sobretudo, o MPJPE menor (4,81 vs 6,04 px). O **OpenPose** (Cao et al., 2017), *bottom-up* (Part Affinity Fields) e sem crop dedicado por jogador, fica bem atrás em todas as métricas.


**O gargalo está na precisão fina, concentrada nas extremidades.** Mesmo o vencedor detecta bem (PDJ 93,6%) mas localiza mal (PCK 41,8%) — ou seja, acha a região certa, mas erra a posição exata. Esse erro se concentra nas **extremidades**: o PDJ por grupo anatômico cai justamente em punho e tornozelo (RTMPose: punho 83,4% e tornozelo 85,7%, contra ombro 97,9% e quadril 98,2%; no HRNet o punho despenca para 70,7%). São os keypoints de maior variabilidade de pose durante o chute. **É exatamente essa imprecisão fina — alta no PCK, pior nas extremidades — que o estudo de fine-tuning da §2.4 busca atacar.**

### 3.2 A matriz e a escada de augmentation — o melhor modelo é o Transfer Learning + augmentation completa

Para isolar cada fator, os dois extremos da matriz (RAW e *full*) foram complementados por uma **escada de augmentation** no lado do transfer learning, em que cada degrau adiciona **uma** transformação — o que permite atribuir o ganho a cada mecanismo. A Tabela 2 consolida os dez modelos no val split (todos com 150 épocas, batch 64, mesmo protocolo).

| Modelo | augmentation | PCK@0.2 ↑ | PCK treino | gap | PDJ@0.5 ↑ | OKS ↑ | MPJPE-2D ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline zero-shot (COCO) | — | 41,76% | — | — | 93,62% | 81,82% | 4,81 px |
| **A-RAW** — scratch | RAW | 37,82% | 56,01%† | —† | 86,48% | 74,46% | 6,76 px |
| **A-FLIP** — scratch | +flip | 46,06% | 93,71% | 47,6 pp | 89,96% | 79,39% | 5,56 px |
| **B (B-FULL)** — scratch | full | 58,13% | 95,22% | 37,1 pp | 93,59% | 85,26% | 4,09 px |
| **C (C-RAW)** — TL | RAW | 52,88% | 65,11% | 12,2 pp | 91,49% | 82,19% | 4,80 px |
| **C2** — TL fase única *(ablação)* | +flip | 54,83% | 88,32% | 33,5 pp | 92,02% | 83,37% | 4,57 px |
| **C-FLIP** — TL | +flip | 58,39% | 70,05% | 11,7 pp | 93,80% | 85,26% | 4,09 px |
| **D-GEOM** — TL | +geom | 66,16% | 68,71% | 2,5 pp | 96,05% | 89,06% | 3,24 px |
| **D-OCCL** — TL | +geom+ocl | 66,87% | 68,94% | 2,1 pp | 96,23% | 89,48% | 3,16 px |
| **D (D-FULL)** — TL | full | **67,54%** | 68,60% | **1,1 pp** | **96,60%** | **89,84%** | **3,08 px** |

*Tabela 2 — Matriz de experimentos e a escada de augmentation (val, 800 frames).* † A-RAW: o melhor checkpoint é da época 20 (pico do val, treino ainda 56%); na época 150 o treino sobe a ~90% (gap real ~55 pp), overfitting precoce.

A hierarquia é limpa: **baseline 41,8 < A-RAW 37,8\* < A-FLIP 46,1 < {B 58,1 ≈ C-FLIP 58,4} < D-GEOM 66,2 < D-OCCL 66,9 < D-FULL 67,5** (\*A-RAW abaixo do baseline por overfitting + zero augmentation). O **D-FULL é o teto do estudo**: PCK@0.2 = **67,54%**, **+25,8 pp sobre o zero-shot**, e vence em todas as métricas (PDJ 96,6%, OKS 89,8%, MPJPE 3,08 px).

Lendo a matriz 2×2 canônica (células = **RAW** vs **full**):

| | RAW (sem aug) | full (com aug) | **Δ augmentation** |
|---|---:|---:|---:|
| **From scratch** | A 37,82 | B 58,13 | **+20,31 pp** |
| **Transfer learning** | C 52,88 | D 67,54 | **+14,66 pp** |
| **Δ transfer learning** | **+15,06 pp** | **+9,41 pp** | |

Os dois fatores são fortes. O **transfer learning vence o treino do zero em todos os níveis** (+15,1 pp sem aug, +9,4 pp com aug) — e, de forma decisiva, **C (TL sem augmentation nenhuma, 52,9%) já supera A-FLIP (scratch com flip, 46,1%)**: o prior do COCO vale mais que o flip. A **augmentation é a alavanca um pouco maior** na matriz (+20,3 / +14,7 pp), mas "augmentation" aqui é a stack inteira, enquanto "TL" é um único fator. Os dois se combinam de forma **sub-aditiva (≈ −5,7 pp)** — partindo de A-RAW, somar os efeitos isolados daria ~73%, mas o D-FULL real é 67,5% —, porque ambos atacam em parte o **mesmo** gargalo: a baixa diversidade do dataset. Ainda assim, **combiná-los é nitidamente melhor que qualquer fator isolado** (só-TL = 52,9%; só-aug = 58,1%; juntos = 67,5%).

### 3.3 A augmentation que paga é a geométrica

Como o lado TL é uma escada fina, o ganho de cada mecanismo é a diferença entre degraus consecutivos (Tabela 3).

| Mecanismo | comparação | Δ PCK@0.2 |
|---|---|---:|
| **flip** | A-FLIP − A-RAW (scratch) / C-FLIP − C-RAW (TL) | **+8,24 / +5,51 pp** |
| **geométrica** (rot/escala/shift) | D-GEOM − C-FLIP | **+7,77 pp** |
| oclusão | D-OCCL − D-GEOM | +0,71 pp |
| borrão | D-FULL − D-OCCL | +0,67 pp |

*Tabela 3 — Atribuição do ganho por mecanismo de augmentation (TL, val).*

**Flip e transformação geométrica respondem por ~90% do ganho de augmentation**; oclusão e borrão são refino marginal (≈ +0,7 pp cada), porém **reais e consistentes** — cada degrau ainda melhora PDJ, OKS e MPJPE (3,24 → 3,16 → 3,08 px). A explicação é o gargalo do dataset: com apenas ~160 cenas distintas (20 frames quase idênticos por clip), o que falta é **diversidade efetiva de pose e ângulo** — exatamente o que flip (espelhamento) e rotação/escala/deslocamento multiplicam, enquanto oclusão e borrão (fotométrico/máscara) mexem menos no que falta.

### 3.4 Diagnóstico de overfitting (framework dos 3 números): o transfer learning fecha o gap

Comparando treino, val e o baseline zero-shot (sem depender de curvas de loss), o gap treino→val revela a qualidade da generalização (Tabela 4).

| Cenário | PCK treino | PCK val | gap | leitura |
|---|---:|---:|---:|---|
| A-FLIP (scratch, +flip) | 93,71% | 46,06% | 47,6 pp | 🔴 overfitting severo |
| B (scratch, full) | 95,22% | 58,13% | 37,1 pp | 🟠 ainda alto (aug não basta sem TL) |
| C2 (TL, fase única) | 88,32% | 54,83% | 33,5 pp | 🟠 moderado |
| C (TL, RAW) | 65,11% | 52,88% | 12,2 pp | 🟢 generaliza bem |
| C-FLIP (TL, +flip) | 70,05% | 58,39% | 11,7 pp | 🟢 generaliza bem |
| **D (TL, full)** | 68,60% | 67,54% | **1,1 pp** | 🟢 gap quase nulo |

*Tabela 4 — Diagnóstico de overfitting por cenário (val).*

Duas leituras. Primeiro, **o scratch decora**: A-FLIP e B chegam a 94–95% no treino e despencam para 46–58% no val — o gargalo é generalização, não capacidade, e a augmentation encolhe o gap (47,6 → 37,1 pp) mas **não o resolve sem o prior do COCO**. Segundo, **o transfer learning é quem fecha o gap**: porque parte de features aprendidas num domínio mais geral (corpos humanos do COCO) do que o nosso (só futebol), o COCO age como regularizador implícito e o gap cai para ~12 pp já sem augmentation. Quando se soma a augmentation geométrica, o treino fica mais difícil, a PCK de treino não infla (68,6%) e **cola na de validação (67,5%) — gap de apenas 1,1 pp**, o comportamento ideal.

Vale notar que esse diagnóstico explica o porquê do PDJ e do PCK divergirem: o zero-shot já tem PDJ ~94% (detecta a região) mas PCK 42% (erra a posição exata); o fine-tuning **melhora as duas ao mesmo tempo** apenas nos cenários TL — o D-FULL é o único, com o C-FLIP, a superar o baseline em PDJ **e** em PCK com folga.

### 3.5 Análise por grupo anatômico: as extremidades só se resolvem com TL + augmentation

O ganho do fine-tuning não é uniforme pelo corpo (Tabela 5). Cabeça, ombro e quadril (tronco) superam o patamar zero-shot em qualquer cenário; as **extremidades** (cotovelo, punho, joelho, tornozelo) são o ponto fraco persistente — e **só o D-FULL as traz de volta acima do baseline**.

| Grupo | Baseline | C (só TL) | B (só aug) | **D (TL + aug)** |
|---|---:|---:|---:|---:|
| Cabeça | 50,4 | 73,9 | 87,6 | **89,5** |
| Ombro | 30,4 | 64,9 | 70,5 | **78,4** |
| Quadril | 22,8 | 54,5 | 60,2 | **69,3** |
| Cotovelo | 50,8 | 39,4 | 42,6 | **56,4 ✅** |
| Punho | 43,5 | 31,0 | 33,6 | **46,1 ✅** |
| Joelho | 58,3 | 47,3 | 53,5 | **64,8 ✅** |
| Tornozelo | 59,4 | 51,6 | 54,0 | **67,2 ✅** |

*Tabela 5 — PCK@0.2 por grupo anatômico (val). ✅ = extremidade acima do baseline zero-shot.*

A leitura é direta e é uma das principais descobertas: **nem o transfer learning sozinho (C), nem a augmentation sozinha (B) recuperam as extremidades** — ambos ficam abaixo do baseline em punho, cotovelo, joelho e tornozelo. É **a combinação** (D-FULL) que conserta, e a alavanca é geométrica (já o D-GEOM cruza o baseline nas quatro). Em erro absoluto, o punho cai de 8,9 px (C) para 5,63 px (D-FULL) e o tornozelo de 7,2 para 4,39 px — os dois keypoints historicamente piores tiveram a maior redução.

### 3.6 O progressive unfreezing se justifica, mas a fase 3 não

A Tabela 6 mostra o PCK@0.2 (estrito, val) por fase nos quatro cenários de transfer learning.

| Cenário | Fase 1 (só cabeça) | Fase 2 (+topo) | Fase 3 (+baixo) | Δ (f2−f1) | selecionado |
|---|---:|---:|---:|---:|---|
| C (C-RAW) | 50,0% | **52,9%** | — (pulada) | 0,029 | fase 2 |
| C-FLIP | 52,9% | **58,4%** | 51,7% (degradou) | 0,055 | fase 2 |
| D-GEOM | 59,4% | **66,2%** | 60,8% (degradou) | 0,067 | fase 2 |
| D-OCCL | 59,0% | **66,9%** | 61,7% (degradou) | 0,078 | fase 2 |
| D-FULL | 59,6% | **67,5%** | 63,1% (degradou) | 0,079 | fase 2 |

*Tabela 6 — Progressive unfreezing por fase (val). A fase 3 só roda se Δ(f2−f1) > 0,05.*

Três achados, fortes por agregação. **(i) A fase 2 é sempre onde o ganho acontece**; já a fase 1 (backbone 100% congelado, treinando só a cabeça — equivalente a um *linear probing*) basta para passar do C-FLIP inteiro nos cenários D (~59% > 58,4%): adaptar a cabeça sobre features COCO supera treinar do zero. **(ii) A fase 3 degradou em 4 de 4 casos** em que foi ativada — destravar as camadas baixas com apenas 3.200 imagens distorce/esquece as features de baixo nível do COCO. Isso **não é um artefato do nosso pipeline**: é o efeito previsto pela literatura — distorção de features sob *distribution shift* (Kumar et al., 2022) e esquecimento por ajustar parâmetros demais com dataset pequeno (Lee et al., 2023). O fix de seleção do melhor checkpoint entre fases evitou entregar a fase 3 pior, salvando +4,5 a +6,6 pp. **(iii)** A ablação **C2** (transfer learning em fase única, sem o descongelamento progressivo) atinge 54,8% com gap de 33,5 pp, contra 58,4% e gap 11,7 pp do C-FLIP progressivo — ou seja, **o progressive unfreezing vale +3,6 pp e generaliza muito melhor**, graças ao *warmup* da cabeça na fase 1, que evita a distorção que a fase única provoca.

---

## 4. Conclusões

**A resposta à pergunta de pesquisa é que a precisão de localização (PCK) é maximizada combinando todas as técnicas estudadas — transfer learning a partir de pesos do COCO e augmentation geométrica forte, sobre o estimador RTMPose-X — elevando o PCK@0.2 de 41,8% (zero-shot) para 67,5% (+25,8 pp).** Nenhuma técnica isolada alcança esse patamar; é a sua composição que resolve o problema. As conclusões que sustentam essa resposta, cada uma ancorada num experimento, são as seguintes.

**O transfer learning maximiza a precisão — e isso foi provado, não assumido.** A comparação de inicializações (§3.2) mostra que partir dos pesos do COCO supera treinar do zero em todos os níveis de augmentation (+15,1 pp sem aug, +9,4 pp com aug); até sem augmentation nenhuma, o Cenário C (52,9%) supera o scratch com flip (46,1%). Como o experimento de *from scratch* isola o efeito da inicialização, a afirmação "os pesos do COCO são um bom ponto de partida" deixa de ser uma suposição e passa a ser uma conclusão empírica.

**O data augmentation também maximiza, e é complementar ao transfer learning.** Os dois fatores agem sobre o mesmo gargalo — a baixa diversidade de um dataset de ~160 cenas distintas — e por isso se combinam de forma sub-aditiva (≈ −5,7 pp), mas combiná-los é claramente superior a qualquer um isolado: só-TL chega a 52,9%, só-aug a 58,1%, e juntos a 67,5%. O melhor cenário é, portanto, o **D-FULL (transfer learning + augmentation completa)**.

**Nem todo augmentation contribui igualmente: a alavanca é geométrica.** A decomposição por mecanismo (§3.3) atribui ~90% do ganho ao flip e à transformação geométrica (rotação/escala/deslocamento), enquanto oclusão e borrão somam apenas ~1,4 pp. Isso é coerente com o diagnóstico: o que falta ao dataset é diversidade de pose e ângulo, e são as transformações geométricas que a sintetizam.

**O transfer learning também resolve o overfitting, não só eleva o PCK.** Pelo framework dos três números (§3.4), o scratch decora (gap treino→val de 37–55 pp), enquanto o transfer learning, por partir de um domínio mais geral, fecha o gap para ~12 pp já sem augmentation; com a augmentation geométrica, o gap do D-FULL cai a **1,1 pp** — generalização quase perfeita. Ou seja, o melhor cenário não é só o mais preciso, é também o que melhor generaliza.

**As extremidades só são recuperadas pela combinação.** A análise por grupo (§3.5) mostra que cabeça, ombro e quadril melhoram em qualquer cenário, mas cotovelo, punho, joelho e tornozelo — os keypoints mais informativos para o gesto do chute e os que os modelos genéricos mais erram — só superam o patamar zero-shot no D-FULL; nem o transfer learning nem a augmentation, isolados, bastam.

**A estratégia de descongelamento importa, e tem um ponto ótimo.** O progressive unfreezing (§3.6) entrega o ganho na fase 2; a fase 3 (destravar as camadas baixas) degradou em todos os casos, comportamento previsto pela literatura para datasets pequenos sob mudança de domínio (Kumar et al., 2022; Lee et al., 2023). A ablação em fase única confirma que o descongelamento progressivo agrega (+3,6 pp e muito menos overfitting).

Por fim, vale registrar que **esse ganho não exigiu mais dados nem um modelo maior**: bastaram o prior do COCO, augmentation geométrica e um descongelamento progressivo — uma receita leve e transferível para adaptar estimadores de pose a outros domínios com poucos dados anotados.

**Limitações.** A principal é a **quantidade de dados**: o 3DSP tem apenas ~160 cenas distintas (cada clip são 20 frames quase idênticos), pouco para redes profundas — é a raiz tanto da tendência ao overfitting quanto do teto de desempenho absoluto. Além disso, mesmo após a adaptação, a precisão **absoluta** nas extremidades permanece modesta (punho ≈ 46%, cotovelo ≈ 56%): a "recuperação" deve ser lida como ganho **relativo** ao ponto de partida, não como precisão resolvida para uso em produção. Por fim, cada cenário foi treinado em **uma única execução** (sem barra de erro), então diferenças pequenas — sobretudo os +0,7 pp de oclusão/borrão — podem conter ruído de *seed*.

**Trabalho futuro.** Como a limitação dominante é a quantidade de dados, a direção mais promissora é **ampliar o conjunto anotado** — mais cenas distintas, de mais partidas, ângulos de câmera e tipos de chute —, atacando diretamente o teto de desempenho e o overfitting. Como direção secundária, avaliar arquiteturas mais leves para inferência em tempo real, dado que o uso final é vídeo de transmissão.

---

## Referências

- **Howard, J.; Ruder, S. (2018).** Universal Language Model Fine-tuning for Text Classification (ULMFiT). *ACL.* arXiv:1801.06146. — *gradual unfreezing* e *discriminative fine-tuning*.
- **Kumar, A.; Raghunathan, A.; Jones, R.; Ma, T.; Liang, P. (2022).** Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution (LP-FT). *ICLR.* arXiv:2202.10054.
- **Lee, Y.; Chen, A. S.; Tajwar, F.; Kumar, A.; Yao, H.; Liang, P.; Finn, C. (2023).** Surgical Fine-Tuning Improves Adaptation to Distribution Shifts. *ICLR.* arXiv:2210.11466.
- **Yosinski, J.; Clune, J.; Bengio, Y.; Lipson, H. (2014).** How transferable are features in deep neural networks? *NeurIPS.* arXiv:1411.1792.
- **Yeung, C.; Ide, R.; Fujii, K. (2024).** AutoSoccerPose: Automated 3D Posture Analysis of Soccer Shot Movements (dataset 3DSP). *CVPR Workshops.*
- **Jiang, T. et al. (2023).** RTMPose: Real-Time Multi-Person Pose Estimation (OpenMMLab). arXiv:2303.07399.
- **Li, Y. et al. (2022).** SimCC: a Simple Coordinate Classification Perspective for Human Pose Estimation. *ECCV.* arXiv:2107.03332. — classificação de coordenada sub-pixel, menor erro de quantização que heatmaps (sobretudo em baixa resolução).
- **Sun, K. et al. (2019).** Deep High-Resolution Representation Learning for Human Pose Estimation (HRNet). *CVPR.*
- **Cao, Z. et al. (2017).** Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields (OpenPose). *CVPR.*
