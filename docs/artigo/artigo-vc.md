# Um Pipeline Quantitativo para Estimação de Pose de Jogadores de Futebol em Vídeo de Transmissão: Seleção de Detector, Benchmark de Estimadores e Adaptação ao Domínio

**Wagner Victor Alves de Menezes (202403929) · Victor Gabriel Ribeiro Jacome (202403926) · Raphael Alves de Lima Soares (202403922) · André Guilherme Alves do Carmo (202301423)**

Bacharelado em Inteligência Artificial — Instituto de Informática, Universidade Federal de Goiás (UFG)
Disciplina: Visão Computacional — Prof. Ricardo Augusto Pereira Franco

> **Status:** rascunho v1 (Markdown) para revisão. Será convertido ao template SBC (LaTeX) após aprovação.

---

## Resumo

A estimação de pose de jogadores em vídeo de transmissão (*broadcast*) apoia a análise tática e biomecânica, mas o domínio é hostil: jogadores ocupam regiões minúsculas, sob borrão, oclusão e poses atípicas, degradando a *precisão de localização* dos estimadores genéricos. Este trabalho avalia, **estágio a estágio e de forma quantitativa**, um pipeline para o problema: escolhe o detector e o estimador por benchmark (este, um *zero-shot* de três modelos no 3DSP) e **adapta o estimador ao domínio por fine-tuning**. O detector escolhido (**YOLO26x**) atinge mAP 84,4 e *recall* 95,4%; o **fine-tuning** do estimador (**RTMPose-X**) eleva o PCK@0,2 de **41,8% para 67,5%**, fornecendo a avaliação quantitativa que faltava ao domínio.

## Abstract

Pose estimation of soccer players from broadcast video supports tactical and biomechanical analysis, but the domain is harsh: players occupy tiny image regions under motion blur, occlusion and atypical poses, degrading the *localization precision* of generic estimators. This work evaluates, **stage by stage and quantitatively**, a pipeline for the problem: it chooses the detector and the estimator by benchmark (the latter, a *zero-shot* of three models on 3DSP) and **adapts the estimator to the domain by fine-tuning**. The chosen detector (**YOLO26x**) reaches mAP 84.4 and recall 95.4%; **fine-tuning** the estimator (**RTMPose-X**) raises PCK@0.2 from **41.8% to 67.5%**, providing the quantitative evaluation the domain lacked.

**Palavras-chave:** Visão Computacional; estimação de pose humana 2D; localização de keypoints; detecção de objetos; transfer learning; análise de vídeo esportivo.

---

## 1. Introdução

A postura corporal dos jogadores em campo é uma fonte rica de informação tática e biomecânica: orientação relativa à bola e aos adversários, fase do gesto técnico, equilíbrio e coordenação. Com o avanço da visão computacional, tornou-se possível extrair essa informação automaticamente a partir de **vídeo de transmissão convencional**, dispensando equipamentos de rastreamento vestíveis ou câmeras especializadas — o que democratiza a análise para clubes, federações e veículos de mídia.

Extrair pose de *broadcast*, contudo, é substancialmente mais difícil do que de imagens cotidianas. Numa transmissão em 720p/1080p, cada jogador ocupa uma região pequena da imagem (tipicamente da ordem de 60×60 px); soma-se a isso o borrão de movimento de atletas em alta velocidade, a oclusão entre jogadores aglomerados e poses atípicas durante chutes e disputas. Esses fatores afetam pouco a tarefa de *encontrar* uma articulação, mas degradam fortemente a **precisão de localização** do ponto exato — justamente a informação necessária para qualquer medida geométrica posterior.

Estimadores de pose de prateleira, treinados em conjuntos genéricos como o COCO, herdam esse problema: aplicados a recortes pequenos de jogadores, eles tendem a **detectar bem** as articulações (a região aproximada) mas **localizar mal** a posição precisa. Trabalhos na área esportiva propuseram pipelines que combinam um detector de pessoas com um estimador de pose por recorte individual — por exemplo, Reis et al. (2023) — e o dataset 3DSP (Yeung et al., 2024) forneceu anotações 2D de keypoints para chutes de futebol, viabilizando avaliação. Ainda assim, persiste uma lacuna: falta uma **avaliação quantitativa sistemática** das escolhas de projeto (qual detector? qual estimador?) e uma **adaptação ao domínio** cujo ganho seja medido com métricas estabelecidas.

Este trabalho preenche essa lacuna construindo um pipeline cujas escolhas são, cada uma, **decididas por experimento**: selecionamos o detector de jogadores por um benchmark contra anotação humana; selecionamos o estimador de pose por um benchmark *zero-shot* no domínio; adaptamos o estimador ao futebol por *fine-tuning*, medindo o ganho de precisão; e integramos tudo num pipeline ponta-a-ponta, demonstrado em vídeo real. Todas as etapas são avaliadas com métricas reprodutíveis (mAP e AP para detecção; PDJ, PCK, OKS e MPJPE para pose).

## 2. Formulação do Problema

Dado um **vídeo bruto de transmissão** de uma partida de futebol, deseja-se, de forma automática e mensurável: **(a)** detectar os jogadores em cada quadro com precisão de enquadramento; e **(b)** localizar, para cada jogador, os 17 pontos-chave (keypoints) de seu corpo, de modo a **estimar a pose com alta precisão de localização**.

O cerne da dificuldade não é *encontrar* a articulação, e sim **localizá-la com precisão**. Há uma razão analítica para isso ser severo neste domínio: o PCK@0.2 só aceita uma predição se o erro for menor que 0,2 × a distância de referência do jogador (entre ombros ou quadris). Em **visão lateral** — o jogador de perfil durante o chute — essa distância, projetada em 2D, encolhe a poucos pixels, de modo que o limiar de aceitação se torna da ordem de 1–3 px. Assim, um erro de localização de poucos pixels — tolerável em imagens de alta resolução — passa a reprovar a predição mesmo quando a articulação está aproximadamente correta. Em consequência, um estimador genérico pré-treinado tende a **detectar bem** (achar a região da junta) mas **localizar mal** (errar a posição exata) neste domínio — um gap que **quantificamos empiricamente na Seção 5.2** e que se concentra nas extremidades do corpo.

O critério de sucesso, portanto, é **maximizar a precisão de localização (PCK@0.2)** contra o *ground truth*, sem sacrificar a detecção — operando sob as condições adversas do *broadcast* (crops pequenos, borrão, oclusão).

## 3. Objetivos

**Objetivo geral:** desenvolver e avaliar quantitativamente um pipeline de estimação de pose de jogadores de futebol a partir de vídeo de transmissão.

**Objetivos específicos:**
1. **Selecionar o detector** de jogadores por benchmark quantitativo, contra anotação humana de *bounding boxes*.
2. **Selecionar o estimador de pose** por benchmark *zero-shot* no domínio (3DSP), com métricas estabelecidas.
3. **Adaptar o estimador ao domínio por fine-tuning** e **medir o ganho** de precisão de localização — núcleo do trabalho.
4. **Integrar** os componentes num pipeline ponta-a-ponta e **demonstrá-lo** em vídeo real.

## 4. Metodologia

A metodologia é **experimental**: cada escolha de projeto é sustentada por um experimento controlado cujo resultado a decide, em vez de afirmada por autoridade.

### 4.1. Dataset (3DSP)

Utilizamos o **3DSP** (*3D Shot Posture Dataset*; Yeung et al., 2024), composto por 200 clips de chutes de futebol *broadcast* (SoccerNet, 720p), cada um com 20 quadros — 4.000 imagens. Cada imagem é um recorte 100×100 px centrado num jogador, anotado com 17 keypoints 2D no formato **H3WB** e uma *bounding box*. Adotamos a divisão **80/20 por clip** (160 treino / 40 validação = 3.200 / 800 quadros); o particionamento é feito **por clip, nunca por quadro**, para evitar vazamento temporal entre frames quase idênticos do mesmo chute. O conjunto de teste (10 clips) **não possui anotação de pose**, servindo apenas para demonstração qualitativa.

Dos 17 keypoints H3WB, 4 são **calculados** (médias de pares: centro dos quadris, centro do corpo, centro dos ombros, pescoço) e não constituem anotação manual; eles são derivados de forma idêntica no *ground truth*, na predição e no baseline antes de qualquer medição.

### 4.2. Visão geral do pipeline

O pipeline encadeia cinco estágios (Fig. 1):

```
vídeo de broadcast → [1] extração de quadros → [2] detecção de jogadores (YOLO26x)
   → [3] recorte justo por jogador (letterbox) → [4] estimação de pose → [5] reprojeção + esqueleto
```

É **agnóstico** ao detector e ao estimador (interfaces unificadas), o que permitiu trocar componentes para os benchmarks sem alterar o restante.

### 4.3. Seleção do detector

Comparamos **quatro** detectores de objetos: dois *one-stage* — **YOLO26x** (anchor-free) e **RetinaNet** — e dois *two-stage* — **Faster R-CNN** e **Cascade R-CNN** (cascata de IoU). RetinaNet, Faster R-CNN e Cascade R-CNN compartilham o mesmo *backbone* ResNet50-FPN (implementações torchvision/mmdet), o que mantém a capacidade comparável entre eles; o YOLO26x possui arquitetura própria. A avaliação é feita contra **anotação humana** de *bounding boxes* (740 caixas em 60 quadros, anotadas no Roboflow), com métricas do `pycocotools` (COCOeval): mAP@[.5:.95], AP50, AP75 e *recall* por tamanho; e precisão/recall/F1 no ponto de operação (conf ≥ 0,3; IoU ≥ 0,5). Como ~99% das caixas são de tamanho *medium* (jogador típico), o `AR_medium` é o recall por tamanho relevante.

Um cuidado metodológico foi necessário: comparar modelos de **capacidade comparável**. Uma primeira rodada usou a variante *nano* do YOLO (~5 MB) contra os backbones ResNet50-FPN (~130–265 MB), o que subestimou o YOLO — a comparação justa exige equiparar a capacidade (variante *x*).

### 4.4. Benchmark de estimadores (zero-shot)

Avaliamos três estimadores de pose **pré-treinados em COCO, sem nenhuma adaptação**, sobre os recortes do 3DSP, convertendo a saída COCO-17 para H3WB-17:

- **OpenPose** (2017) — *bottom-up*, *Part Affinity Fields*, backbone VGG-19;
- **HRNet-W48** (2019) — *top-down*, *heatmaps* de alta resolução;
- **RTMPose-X** (2023) — *top-down*, cabeça SimCC (classificação de coordenadas).

A diferença arquitetural mais relevante está na cabeça: o HRNet localiza cada keypoint pelo *argmax* de um *heatmap* `64×48`; num recorte 100×100, cada célula cobre ~2 px e o *argmax* só aponta posições inteiras do grid — um **erro de quantização** de até ±2 px por junta. O RTMPose usa **SimCC**, que prediz a coordenada diretamente (576 bins horizontais, 768 verticais; *loss* de divergência KL contra uma gaussiana centrada no alvo), eliminando essa quantização.

### 4.5. Adaptação ao domínio: fine-tuning

O estimador selecionado é adaptado ao domínio de futebol. A pergunta central — *a inicialização com pesos do COCO (transfer learning) agrega valor sobre treinar do zero, e quanto a augmentation contribui?* — é respondida por uma **matriz 2×2** que varia dois fatores independentes:

|                       | Sem augmentation | Com augmentation |
|-----------------------|------------------|------------------|
| **From scratch** (W₀ aleatório) | Cenário A | Cenário B |
| **Transfer learning** (W₀ = COCO) | Cenário C | Cenário D |

A *augmentation* é organizada como uma **escada aditiva**: flip horizontal → geométrica (`RandomBBoxTransform`: rotação ±30°, escala 0,75–1,25, deslocamento 0,1) → oclusão (apagamento de retângulos, 2–10% da área, p=0,3) → borrão de movimento (kernel 3–9 px, p=0,5). O protocolo é idêntico entre cenários (mesmo dataset, modelo, *batch* 64, otimizador AdamW, orçamento de 150 épocas, métrica de seleção PCK@0.2 estrito) — só mudam W₀, a estratégia de descongelamento e o nível de *augmentation*.

**Pré-processamento (letterboxing).** Os recortes 100×100 são levados à entrada esperada (288×384) preservando proporção: escala uniforme 2,88× para 288×288 mais 48 px de *padding* superior/inferior, sem distorção. As coordenadas do *ground truth* são mapeadas pela mesma transformação. Os 4 keypoints calculados (IDs 0, 7, 8, 9) recebem `visibility=0` e são **excluídos da loss**, evitando ruído no sinal de treino.

**Transfer learning por progressive unfreezing.** Os cenários com pesos do COCO (C e D) não treinam todas as camadas de uma vez: descongelam o *backbone* de cima para baixo em **três fases** (`frozen_stages` 4→2→1), com taxa de aprendizado maior na cabeça (mais específica) e menor no *backbone* (mais geral) — *gradual unfreezing* + *fine-tuning discriminativo*, técnicas introduzidas por **Howard e Ruder (ULMFiT, 2018)**. A terceira fase é **condicional** (só é ativada se o ganho da fase 2 superar um limiar de ΔPCK). A motivação vem da literatura de transferência: camadas baixas aprendem features genéricas e transferíveis enquanto as altas são específicas da tarefa de origem (**Yosinski et al., 2014**); destravar tudo de uma vez distorce as features pré-treinadas, sobretudo sob *distribution shift* (**Kumar et al., LP-FT, 2022**); e ajustar um subconjunto de camadas pode superar ajustar todas em datasets pequenos (**Lee et al., surgical fine-tuning, 2023**) — daí a fase 3 ser condicional.

### 4.6. Métricas e protocolo de avaliação

**Métricas de pose** (calculadas no *val split*, 800 quadros):

- **PCK@0.2** (*Percentage of Correct Keypoints*) — **métrica principal**; junta correta se o erro < 0,2 × distância de referência (ombros/quadris). Mede **precisão de localização**.
- **PDJ@0.5** (*Percentage of Detected Joints*) — junta correta se o erro < 0,5 × comprimento do torso. Mede **detecção** ("achou a região?").
- **OKS** (*Object Keypoint Similarity*) e **AP** derivado — similaridade tipo-COCO, ponderada por tamanho/visibilidade.
- **MPJPE-2D** — erro médio absoluto, em pixels, por junta visível.

**Métricas de detecção** (todas via `pycocotools`/COCOeval, contra a anotação humana):

- **mAP@[.5:.95]** — média da AP sobre os limiares de IoU de 0,5 a 0,95 (passo 0,05); métrica-resumo do COCO, sensível à qualidade do enquadramento.
- **AP50** — AP no limiar de IoU 0,5 (caixa aproximadamente correta); mede detecção "frouxa".
- **AP75** — AP no limiar de IoU 0,75 (caixa justa); mede a **precisão do enquadramento** — relevante porque caixa justa gera recorte melhor para a pose.
- **AR_medium** — recall médio para objetos de tamanho *medium* (32²–96² px), faixa que cobre ~99% dos jogadores; mede "não perder jogador".
- **Precisão / Recall / F1** no ponto de operação (conf ≥ 0,3; IoU ≥ 0,5) — precisão = fração das predições que são jogadores reais (penaliza detectar a torcida); recall = fração dos jogadores encontrados; F1 = média harmônica das duas.

O diagnóstico de cada cenário de fine-tuning usa o **framework dos três números** — métrica no treino, no val e no *zero-shot* de referência — para distinguir generalização, overfitting e underfitting sem depender de curvas de loss.

## 5. Resultados

### 5.1. O detector selecionado é o YOLO26x

O YOLO26x domina **todas** as métricas (Tab. 1). Destaca-se por ser o único simultaneamente alto em *recall* e precisão (95,4% / 98,7%): prevê ≈715 caixas para 740 do *ground truth*, isto é, encontra os jogadores e ignora a torcida, sem necessidade de pós-filtro. O AP75 de 91,6% indica caixas justas — o que se traduz diretamente em recortes melhores para a estimação de pose. Os *two-stage*, embora com recall alto, têm precisão baixa (47–57%) por detectarem pessoas na arquibancada (falsos-positivos artificiais, pois o GT marca apenas jogadores em campo).

**Tabela 1.** Benchmark de detectores (GT = 740 caixas; valores em %).

| Detector | mAP | AP50 | AP75 | AR_med | Precisão | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **YOLO26x** | **84,4** | **95,0** | **91,6** | **87,5** | **98,7** | **95,4** | **97,0** |
| Cascade R-CNN | 68,3 | 90,7 | 77,5 | 73,8 | 54,5 | 93,4 | 68,9 |
| Faster R-CNN | 65,1 | 91,9 | 74,7 | 70,3 | 47,8 | 94,7 | 63,6 |
| RetinaNet | 61,9 | 91,1 | 70,3 | 68,2 | 57,3 | 93,7 | 71,1 |

Um achado metodológico: a primeira rodada, com a variante *nano* (~5 MB), colocava o YOLO **atrás** dos *two-stage* (mAP 51,0). Ao equiparar a capacidade (variante *x*), o quadro se inverteu por completo (mAP 84,4). A capacidade dos modelos precisa ser comparável para a comparação ser válida.

### 5.2. O estimador selecionado é o RTMPose-X, e o domínio expõe um gargalo de precisão

Entre os três estimadores *zero-shot*, o RTMPose-X vence em todas as métricas (Tab. 2). A diferença vem da cabeça SimCC, que prediz coordenadas diretamente, sem o erro de quantização do *heatmap* do HRNet — daí o MPJPE menor (4,81 vs 6,04 px). Como verificação de reprodutibilidade, nosso RTMPose *zero-shot* atinge PDJ 93,6%, acima do 89,5% reportado por Yeung et al. (2024), validando a implementação. 

**Tabela 2.** Benchmark *zero-shot* de estimadores no 3DSP (val).

| Modelo (zero-shot) | PDJ@0.5 ↑ | PCK@0.2 ↑ | OKS ↑ | MPJPE-2D ↓ |
|---|---:|---:|---:|---:|
| OpenPose | 56,1% | 22,1% | 48,5% | 25,58 px |
| HRNet-W48 | 88,9% | 40,5% | 76,2% | 6,04 px |
| **RTMPose-X** | **93,6%** | **41,8%** | **81,8%** | **4,81 px** |

O diagnóstico é claro e motiva a próxima etapa: mesmo o melhor modelo **detecta bem (PDJ 93,6%) mas localiza mal (PCK 41,8%)**, com o erro concentrado nas extremidades (punho, tornozelo). É essa imprecisão fina que a adaptação ao domínio ataca.

### 5.3. A adaptação ao domínio eleva o PCK de 41,8% para 67,5% e recupera as extremidades

A matriz 2×2 (Tab. 3) responde aos dois eixos comparando células equivalentes. **A inicialização com pesos do COCO vence o treino do zero** nos dois níveis de augmentation: sem augmentation, C (52,9%) supera A (37,8%) em **+15,1 pp**; com augmentation, D-FULL (67,5%) supera B-FULL (58,1%) em **+9,4 pp**. **A augmentation, por sua vez, é a alavanca um pouco maior**: vale **+20,3 pp** no *scratch* (A→B) e **+14,7 pp** no *transfer learning* (C→D). Os dois fatores se combinam de forma **sub-aditiva** (≈ −5,7 pp) por atacarem em parte o mesmo gargalo (baixa diversidade do dataset). O melhor modelo é o **D-FULL** (transfer learning + augmentation completa): **PCK@0.2 = 67,5%**, **+25,8 pp sobre o zero-shot** (41,8%).

**Tabela 3.** Matriz 2×2 e o melhor modelo (val).

| Cenário | PCK@0.2 | PDJ@0.5 | OKS | MPJPE-2D | gap treino→val |
|---|---:|---:|---:|---:|---:|
| Zero-shot (referência) | 41,8% | 93,6% | 81,8% | 4,81 px | — |
| A — scratch, sem aug | 37,8% | 86,5% | 74,5% | 6,76 px | — |
| B — scratch, com aug | 58,1% | 93,6% | 85,3% | 4,09 px | 37,1 pp |
| C — TL, sem aug | 52,9% | 91,5% | 82,2% | 4,80 px | 12,2 pp |
| **D-FULL — TL, com aug** | **67,5%** | **96,6%** | **89,8%** | **3,08 px** | **1,1 pp** |

Três leituras sustentam a qualidade do resultado. Primeiro, **a augmentation que paga é geométrica**: decompondo o ganho, *flip* (+5,5 pp) e transformação geométrica (+7,8 pp) respondem por ~90% do efeito, enquanto oclusão e borrão são refino marginal (+0,7 pp cada) — coerente com o gargalo ser **diversidade de pose/ângulo** num dataset de poucas cenas distintas. Segundo, **o modelo praticamente não overfita**: o *gap* treino→val do D-FULL é de 1,1 pp (contra 37,1 pp do melhor *scratch*), pois a augmentation forte torna o treino mais difícil e impede que a métrica de treino infle. Terceiro, e mais importante para o domínio, **as extremidades foram recuperadas** (Tab. 4): cabeça, ombro e quadril já superam o patamar *zero-shot* em qualquer cenário, mas as **extremidades** (cotovelo, punho, joelho, tornozelo) só o fazem com a **combinação transfer learning + augmentation** — nem o transfer learning sozinho (C), nem a augmentation sozinha (B) bastam.

**Tabela 4.** PCK@0.2 por grupo anatômico (val). Cabeça/ombro/quadril superam o *zero-shot* em qualquer cenário; nas extremidades (cotovelo, punho, joelho, tornozelo) só o D-FULL cruza o patamar.

| Grupo | Zero-shot | C (só TL) | B (só aug) | **D-FULL** |
|---|---:|---:|---:|---:|
| Cabeça | 50,4 | 73,9 | 87,6 | **89,5** |
| Ombro | 30,4 | 64,9 | 70,5 | **78,4** |
| Cotovelo | 50,8 | 39,4 | 42,6 | **56,4** |
| Punho | 43,5 | 31,0 | 33,6 | **46,1** |
| Quadril | 22,8 | 54,5 | 60,2 | **69,3** |
| Joelho | 58,3 | 47,3 | 53,5 | **64,8** |
| Tornozelo | 59,4 | 51,6 | 54,0 | **67,2** |

Quanto à estratégia de treino, o *progressive unfreezing* se justifica empiricamente: a **fase 2** (descongelar o topo do *backbone*) é onde o ganho acontece, enquanto a **fase 3** degradou em todos os casos em que foi ativada — comportamento previsto pela literatura de *surgical fine-tuning* para datasets pequenos. Uma ablação com transfer learning em fase única confirma que o descongelamento gradual agrega (+3,6 pp e muito menos overfitting).

### 5.4. O pipeline integrado funciona em vídeo real

Com o detector e o estimador adaptado, o pipeline completo foi executado de ponta a ponta em vídeos de transmissão (Fig. 3), estimando a pose de **todos os jogadores detectados** por quadro e reprojetando os esqueletos sobre o frame original. Como o conjunto de teste e os clipes adicionais não possuem *ground truth* de pose, esta etapa é uma **validação qualitativa**: ela demonstra a integração dos componentes e a generalização do modelo adaptado para uma partida e transmissão distintas das vistas no treino.

## 6. Conclusão

A precisão de **localização** — e não a detecção — é o gargalo da estimação de pose em vídeo de transmissão, e este trabalho mostra que ele é, antes de tudo, um problema de **adaptação ao domínio**, não de capacidade do modelo. Ao construir o pipeline com cada estágio **decidido por experimento** (detector e estimador escolhidos por benchmark; adaptação medida numa matriz controlada), obtém-se um sistema cujas escolhas são **justificáveis e reprodutíveis**, em vez de herdadas por conveniência. A adaptação ao domínio elevou o PCK@0.2 de 41,8% para 67,5% e **trouxe as extremidades** (cotovelo, punho, joelho, tornozelo) de volta acima do patamar *zero-shot* — frequentemente as juntas mais informativas para o gesto técnico (a posição do pé no chute, por exemplo) e justamente as que os modelos genéricos mais erram, embora ainda permaneçam o ponto fraco em valor absoluto (ver Limitações). O essencial é que esse ganho **não exigiu mais dados nem mais capacidade**: bastaram transfer learning a partir de pesos COCO e augmentation geométrica sobre um dataset pequeno (~160 cenas distintas).

Na prática, o pipeline converte vídeo de transmissão — barato e onipresente — em pose 2D confiável, sem instrumentação dedicada, viabilizando análise tática e biomecânica em escala. A receita por trás disso — pesos do COCO, augmentation geométrica e descongelamento progressivo — é leve e transferível: aponta um caminho de baixo custo para adaptar estimadores de pose a outros domínios esportivos com poucos dados anotados.

**Limitações.** A maior limitação é a **quantidade de dados**: o 3DSP tem apenas ~160 cenas distintas (cada clip são 20 quadros quase idênticos), pouco para treinar redes profundas. Essa escassez é a raiz tanto da tendência ao overfitting — que a augmentation mitiga, mas não elimina no *scratch* — quanto do **teto de desempenho absoluto**; mais dados anotados, e mais diversos, é o caminho mais claro de melhoria.

Em consequência, **os resultados, embora melhorem de forma consistente, ainda não são ideais para uso em produção**. Mesmo após a adaptação, a precisão absoluta nas extremidades permanece modesta — o punho fica em PCK@0.2 ≈ 46% e o cotovelo ≈ 56%: o fine-tuning os ergueu acima do patamar *zero-shot*, mas ainda longe do que um pipeline em produção exigiria (por exemplo, medição biomecânica confiável do gesto de chute). A "recuperação das extremidades" deve ser lida como **ganho relativo sobre o ponto de partida**, não como precisão resolvida.

Há ainda limitações de protocolo: cada cenário foi treinado em **uma única execução** (sem barra de erro), então diferenças pequenas entre cenários podem conter ruído de *seed*; e a métrica de pose usa as *bounding boxes* anotadas do dataset — o YOLO26x é demonstrado separadamente em vídeo real, de modo que a integração detector→pose não é medida de ponta a ponta, por falta de *ground truth* de pose fora do 3DSP.

**Trabalho futuro.** Como a limitação dominante é a **quantidade de dados**, a direção mais promissora é **ampliar o conjunto anotado** — mais cenas distintas, de mais partidas, ângulos de câmera e tipos de chute —, atacando diretamente o teto de desempenho e o overfitting.

## Referências

- CAO, Z. et al. *Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields* (OpenPose). CVPR, 2017.
- HOWARD, J.; RUDER, S. *Universal Language Model Fine-tuning for Text Classification* (ULMFiT). ACL, 2018.
- JIANG, T. et al. *RTMPose: Real-Time Multi-Person Pose Estimation* (OpenMMLab). arXiv:2303.07399, 2023.
- KUMAR, A. et al. *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution* (LP-FT). ICLR, 2022.
- LEE, Y. et al. *Surgical Fine-Tuning Improves Adaptation to Distribution Shifts*. ICLR, 2023.
- LIN, T.-Y. et al. *Microsoft COCO: Common Objects in Context*. ECCV, 2014.
- REIS, R. G. et al. *Soccer Player Pose Recognition in Games*. ICGG 2022, LNDECT vol. 146, pp. 565–574, Springer, 2023. DOI: 10.1007/978-3-031-13588-0_49.
- SUN, K. et al. *Deep High-Resolution Representation Learning for Human Pose Estimation* (HRNet). CVPR, 2019.
- YEUNG, C.; IDE, R.; FUJII, K. *AutoSoccerPose: Automated 3D Posture Analysis of Soccer Shot Movements* (dataset 3DSP). CVPR Workshops, 2024.
- YOSINSKI, J. et al. *How transferable are features in deep neural networks?* NeurIPS, 2014.

---

### Figuras (a inserir no LaTeX)
- **Fig. 1** — Diagrama do pipeline (5 estágios: vídeo → detecção → recorte → pose → reprojeção).
- **Fig. 2** — Resultado do crop e da pose para um finalizador: recorte justo + esqueleto estimado (`data/crops/`, `results/showcase/finisher/`).
- **Fig. 3** — Pipeline em vídeo real: pose de todos os jogadores num quadro de broadcast (`results/showcase/`).
