# Artigo 4 — Surgical Fine-Tuning (Lee et al.): teoria e evidência do ajuste parcial

> **Doc de estudo para a defesa científica.** Fundamenta o **limite** do nosso pipeline: o *gating* da
> fase 3 e o parar no ajuste parcial (fase 2), com teoria (proposições formais) e evidência em múltiplos
> *shifts*.

---

## 1. Ficha rápida

| | |
|---|---|
| **Título** | *Surgical Fine-Tuning Improves Adaptation to Distribution Shifts* |
| **Autores** | Yoonho Lee, Annie S. Chen, Fahim Tajwar, Ananya Kumar, Huaxiu Yao, Percy Liang, Chelsea Finn (Stanford) |
| **Ano / Venue** | 2023, **ICLR** · arXiv:2210.11466 |
| **PDF local** | `.task-context/input/referencias/.refs/RNP/artigos/R4_Lee2023_surgical-finetuning_ICLR.pdf` |
| **Contribuição** | Mostra que ajustar **só um subconjunto contíguo** de camadas (congelando o resto) iguala ou supera o fine-tuning completo sob *distribution shift*; *qual* subconjunto depende do *tipo* de shift, com prova para redes de duas camadas. |

---

## 2. Contexto e lacuna

Métodos de fine-tuning buscam preservar informação do pré-treino: LR menor (Kornblith et al., 2019;
Li et al., 2020), congelar camadas iniciais e descongelar gradualmente (**Howard & Ruder, 2018**), ou LR
por camada (Ro & Choi, 2021; Shen et al., 2021). A lacuna: faltava caracterizar **quais** camadas
ajustar em função do **tipo** de mudança de distribuição — sabendo que redes são frágeis a *shifts*
(Recht et al., 2019; Hendrycks & Dietterich, 2019; Koh et al., 2021).

---

## 3. Contribuição central (1 frase)

> **Sob *distribution shift* e com poucos dados, ajustar apenas um bloco de camadas iguala ou supera o
> fine-tuning completo — porque ajustar parâmetros demais com poucos dados causa esquecimento do
> pré-treino — e o bloco ideal corresponde ao "nível" onde a distribuição mudou.**

---

## 4. O método e a teoria (preciso)

**Surgical fine-tuning** = resolver `argmin` da loss-alvo ajustando apenas um subconjunto `S` de camadas
e congelando as demais (`θ_i` fixo para `i ∉ S`). Casos particulares: ajustar tudo (`S={1,…,n}`),
só a última (`S={n}`) ou só a primeira (`S={1}`).

**Correspondência shift ↔ camada (Fig. 1, Seção 2).** Três tipos de shift e o bloco que melhor os
trata:
- **input-level** (corrupção/blur/ruído na imagem) → **primeiros** blocos;
- **feature-level** (subpopulações) → blocos do **meio**;
- **output-level** (correlação espúria/label) → **última** camada.

**Teoria (Seção 3, Proposições 1 e 2).** Para redes de duas camadas:
- Em **perturbação de entrada** (`x_trg = A·x_src`), ajustar **só a primeira camada** atinge loss-alvo
  zero, enquanto ajustar **só a última pode não conseguir** (Proposição 1);
- Em **perturbação de rótulo**, o inverso (Proposição 2).
Justificam ainda por que **ajustar tudo prejudica** com dados escassos — citação direta do abstract:
*"fine-tuning more parameters on a small target dataset can cause information learned during
pre-training to be forgotten"*. E declaram (Seção 2): *"the efficacy of parameter freezing is a
consequence of having limited target data, and choosing a bigger S will be beneficial in settings where
target data is plentiful."*

A intuição teórica é ancorada no **princípio dos mecanismos causais independentes (ICM)** (Schölkopf et
al., 2012; Peters et al., 2017): um *shift* corresponde a uma mudança local no processo gerador, então
basta ajustar a região da rede correspondente a essa mudança.

---

## 5. Evidência

- Em **7–9 datasets** de distribution shift, ajustar **um único bloco superou** o FT completo, e o bloco
  ótimo variou com o tipo de shift (Fig. 2): **CIFAR-C (corrupção) +2,9%** ajustando o **1º bloco**;
  **CelebA (correlação espúria) +4,0%** ajustando a **última camada**.
- **Regime de poucos dados (Fig. 3):** ajustar as camadas iniciais vence até com **1 imagem por classe**
  no CIFAR-C; ajustar tudo piora conforme se aproxima de mais dados só quando o shift é de entrada.
- **Critério automático:** ajustar as camadas com **maior norma relativa de gradiente** também supera o
  FT completo (embora não bata a escolha manual do melhor bloco).

---

## 6. Fundamentação — o que os autores citam para sustentar

| Afirmação em Surgical FT | Citação/base usada |
|---|---|
| Preservar pré-treino: LR menor / congelar / LR por camada | **Kornblith et al. (2019); Li et al. (2020); Howard & Ruder (2018); Ro & Choi (2021); Shen et al. (2021)** |
| Redes são frágeis a distribution shift | **Recht et al. (2019); Hendrycks & Dietterich (2019); Koh et al. (2021)** |
| Por que shift local ↔ ajuste local (intuição teórica) | **ICM: Schölkopf et al. (2012); Peters et al. (2017)** |
| Ajustar tudo com poucos dados causa esquecimento | **Contribuição própria** (Proposições 1–2 + experimentos) |

> Para a defesa: a base aqui é **teórica (proposições) + empírica (9 datasets)** — e o artigo cita
> Howard & Ruder (2018) [[01-ulmfit-progressive-unfreezing]] como uma das técnicas de preservação,
> fechando a teia com os outros docs.

---

## 7. Onde encaixa no nosso trabalho

- **Justifica o gating da fase 3 e a sua degradação.** Nosso dataset é pequeno (~160 cenas distintas,
  3.200 frames). Descongelar mais camadas (fase 3) = ajustar mais parâmetros com poucos dados =
  esquecimento do pré-treino, exatamente o efeito previsto. Por isso a fase 3 só roda condicionalmente
  e, quando rodou, **degradou**.
- **Justifica o ajuste parcial (parar na fase 2).** Não é necessário o FT completo; ajustar cabeça + topo
  do backbone (um subconjunto) é o regime indicado para poucos dados. Os autores afirmam que, **com mais
  dados** (`S` maior), descongelar mais valeria a pena — o que vira uma **direção futura** honesta.
- **Casa com o tipo do nosso shift.** O domínio broadcast traz mudança de baixo nível na imagem (crops
  100×100, motion blur, baixa resolução) — *input-level*, que pelo mapa do artigo pede ajustar camadas
  mais baixas/médias, coerente com a fase 2 atuar no topo do backbone (e com não precisar do FT completo).

---

## 8. Defesa — perguntas prováveis do professor

**P: "Por que não descongelar a rede inteira para extrair o máximo?"**
R: Porque com dataset pequeno ajustar parâmetros demais causa esquecimento do pré-treino (Lee et al.,
2023); ajustar um subconjunto iguala ou supera o FT completo sob shift (até +2,9–4,0% nos experimentos).

**P: "Parar na fase 2 não deixa performance na mesa?"**
R: Não — a fase 3 (mais parâmetros) degradou com os nossos dados, comportamento previsto pelos autores.
Com mais dados, descongelar mais poderia ajudar — registramos como direção futura.

**P: "Como vocês decidem quais/quantas camadas treinar?"**
R: Pela estratégia progressiva topo→base com gating. O tipo do nosso shift (baixo nível na imagem) e o
tamanho do dataset apontam para ajuste parcial, em linha com o surgical fine-tuning.

**P: "Há base teórica ou é só empírico?"**
R: Ambos: as Proposições 1–2 provam, para redes de 2 camadas, que ajustar o bloco certo pode atingir
loss-alvo zero onde ajustar outro não consegue; e há validação em 9 datasets.

---

## 9. Frase de defesa (científica)

> "Mantemos o ajuste parcial e tornamos a fase 3 condicional porque Lee et al. (2023, *surgical
> fine-tuning*) demonstram — teórica e empiricamente — que, sob *distribution shift* e com poucos dados,
> ajustar apenas um subconjunto de camadas iguala ou supera o fine-tuning completo, já que ajustar
> parâmetros demais faz a rede esquecer o pré-treino; é exatamente a degradação que medimos na fase 3."

---

**Anterior:** a teoria da distorção de features → [[03-lpft-distorcao-de-features]]. Volte ao
[[00-indice]] para a história completa e o mapa de defesa.
