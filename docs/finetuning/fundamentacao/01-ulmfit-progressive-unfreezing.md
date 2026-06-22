# Artigo 1 — ULMFiT: a origem do *Progressive Unfreezing*

> **Doc de estudo para a defesa científica.** É **o artigo central** dos quatro: define a técnica que o
> professor questionou. Os demais (Yosinski, LP-FT, Surgical) fundamentam *por que* ela funciona.

---

## 1. Ficha rápida

| | |
|---|---|
| **Título** | *Universal Language Model Fine-tuning for Text Classification* (ULMFiT) |
| **Autores** | Jeremy Howard, Sebastian Ruder |
| **Ano / Venue** | 2018, **ACL** · arXiv:1801.06146 |
| **PDF local** | `.task-context/input/referencias/.refs/RNP/artigos/R2_HowardRuder2018_ULMFiT_ACL.pdf` |
| **Contribuição** | Propõe três técnicas de fine-tuning — *discriminative fine-tuning*, *slanted triangular learning rates* e *gradual unfreezing* — para reter conhecimento prévio e evitar *catastrophic forgetting*. |

---

## 2. Contexto e lacuna

Antes do ULMFiT, o fine-tuning indutivo em NLP era considerado ineficaz: Dai e Le (2015) já haviam
proposto fine-tuning de um *language model*, mas exigia milhões de documentos no domínio; e Mou et al.
(2016) mostraram que o transfer falha entre tarefas não relacionadas. Os autores argumentam que **o
problema não é a ideia de transfer learning, mas a ausência de um método de fine-tuning que evite o
*catastrophic forgetting***. Eles partem explicitamente da prática consolidada em Visão Computacional,
onde o fine-tuning de modelos pré-treinados (ImageNet) já era padrão (Sharif Razavian et al., 2014;
Long et al., 2015; He et al., 2016).

---

## 3. Contribuição central (1 frase)

> **O fine-tuning deve descongelar o modelo gradualmente, da última camada para a primeira (*gradual
> unfreezing*), e usar taxas de aprendizado distintas por camada (*discriminative fine-tuning*), para
> adaptar as representações específicas sem destruir as gerais.**

---

## 4. O método (preciso)

**Discriminative fine-tuning.** Em vez de um LR único, cada camada `l` recebe seu próprio `η^l`. Partindo
da atualização SGD padrão `θ_t = θ_{t−1} − η·∇J(θ)` (Ruder, 2016), os autores a reescrevem por camada:
`θ^l_t = θ^l_{t−1} − η^l·∇_{θ^l}J(θ)`. Empiricamente recomendam ajustar primeiro o LR da última camada
e usar `η^{l−1} = η^l / 2.6` para as camadas inferiores (Seção 3.2). A justificativa declarada: *"as
different layers capture different types of information, they should be fine-tuned to different
extents"*, citando Yosinski et al. (2014).

**Gradual unfreezing.** Citação direta (Seção 3.3): *"Rather than fine-tuning all layers at once, which
risks catastrophic forgetting, we propose to gradually unfreeze the model starting from the last layer
as this contains the **least general** knowledge (Yosinski et al., 2014): We first unfreeze the last
layer and fine-tune all unfrozen layers for one epoch. We then unfreeze the next lower frozen layer and
repeat, until we fine-tune all layers until convergence at the last iteration."* Os autores situam a
técnica como uma extensão do *chain-thaw* (Felbo et al., 2017), porém adicionando uma camada por vez ao
conjunto "descongelado".

**Slanted triangular learning rates (STLR).** O LR sobe rapidamente e decai linearmente
(`cut_frac=0.1`, `ratio=32`, `η_max=0.01`), variante das *triangular LR* de Smith (2017); comparada
contra *cosine annealing* agressivo (Loshchilov & Hutter, 2017). *(Periférico para a nossa defesa.)*

O artigo é explícito sobre o trade-off que a técnica equilibra (Seção 3.3): *"Overly aggressive
fine-tuning will cause catastrophic forgetting; too cautious fine-tuning will lead to slow convergence
(and resultant overfitting)."*

---

## 5. Evidência

- Redução de erro de **18–24%** em seis datasets de classificação de texto, superando o estado da arte.
- *Sample efficiency*: com **100 exemplos rotulados**, o ULMFiT iguala o treino do zero com **100×** mais
  dados (e, com 50k não-rotulados, com 100× mais). Evidência direta de que a técnica de transfer
  learning substitui volume de dados — relevante para domínios pequenos.
- Ablação (Seção 5): o artigo mostra que *discriminative fine-tuning*, STLR e *gradual unfreezing*
  **se complementam** — cada um contribui, e juntos entregam o melhor resultado.

---

## 6. Fundamentação — o que os autores citam para sustentar

| Afirmação no ULMFiT | Citação que os autores usam |
|---|---|
| Camadas capturam tipos de informação distintos → LR por camada | **Yosinski et al. (2014)** [[02-yosinski-features-transferiveis]] |
| Descongelar da última camada (a "menos geral") para a primeira | **Yosinski et al. (2014)** (transição geral→específico) |
| Gradual unfreezing como extensão de descongelar por partes | **Felbo et al. (2017)** (*chain-thaw*) |
| LR por camada a partir da regra SGD | **Ruder (2016)** |
| Cronograma de LR (STLR) | **Smith (2017)**; comparado a **Loshchilov & Hutter (2017)** |
| Fundamento de transfer learning (analogia com CV/ImageNet) | **Sharif Razavian et al. (2014); Long et al. (2015); He et al. (2016)** |

> O ponto forte para a defesa: a própria base teórica do ULMFiT (a ordem de descongelamento) **não é
> afirmação dos autores**, é ancorada em Yosinski et al. (2014) — exatamente o artigo seguinte.

---

## 7. Onde encaixa no nosso trabalho

O pipeline dos Cenários C/D **implementa o ULMFiT** em pose estimation:

| ULMFiT (2018) | Nosso pipeline |
|---|---|
| Gradual unfreezing (última → primeira) | `frozen_stages` **4 → 2 → 1** (libera o backbone do topo para a base, 3 fases) |
| Descongelar só a última camada primeiro | **Fase 1**: backbone congelado, treina **só a cabeça** SimCC |
| Discriminative fine-tuning (LR por camada) | LR cabeça > LR backbone (cabeça 1e-3/1e-4; backbone 1e-5/1e-6) |
| Cada estágio parte do anterior | Cada fase parte do *best checkpoint* da anterior |

---

## 8. Defesa — perguntas prováveis do professor

**P: "Vocês inventaram esse treino em fases?"**
R: Não. É o *gradual unfreezing* + *discriminative fine-tuning* de Howard e Ruder (2018, ULMFiT),
técnica consolidada de transfer learning. Aplicamos ao RTMPose com `frozen_stages` 4→2→1 e LR
discriminativo.

**P: "Qual a justificativa de descongelar da última para a primeira camada?"**
R: Os próprios autores a ancoram em Yosinski et al. (2014): a última camada contém o conhecimento "menos
geral" (mais específico da tarefa anterior); as primeiras, o mais geral, que se quer preservar.

**P: "Por que LR diferente por camada, e não um só?"**
R: Porque camadas capturam informações distintas e devem ser ajustadas em intensidades distintas
(*discriminative fine-tuning*, Seção 3.2); os autores usam `η^{l−1}=η^l/2.6`.

**P: "ULMFiT é de NLP; vale para visão/pose?"**
R: A técnica é de transfer learning, não de NLP — o artigo se inspira no fine-tuning de ImageNet e
fundamenta a ordem em Yosinski (2014), que é de Visão. O princípio (geral→específico por profundidade) é
arquitetural, não de modalidade.

---

## 9. Frase de defesa (científica)

> "O treinamento dos cenários de transfer learning segue o *progressive (gradual) unfreezing* com
> *discriminative fine-tuning* de Howard e Ruder (2018): descongelamos o backbone da última camada para
> a primeira, em três fases, com LR maior na cabeça e menor no backbone. A ordem de descongelamento é
> fundamentada pelos próprios autores em Yosinski et al. (2014), pela transição de features gerais para
> específicas ao longo da profundidade da rede."

---

**Próximo:** a base experimental da ordem topo→base → [[02-yosinski-features-transferiveis]].
Visão geral e mapa de defesa em [[00-indice]].
