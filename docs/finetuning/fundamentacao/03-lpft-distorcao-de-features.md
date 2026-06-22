# Artigo 3 — LP-FT (Kumar et al.): a teoria da distorção de features

> **Doc de estudo para a defesa científica.** Fundamenta **por que a Fase 1 (só a cabeça) vem antes** e,
> sobretudo, **por que a Fase 3 do nosso pipeline degradou** — com teoria formal, não intuição.

---

## 1. Ficha rápida

| | |
|---|---|
| **Título** | *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution* (LP-FT) |
| **Autores** | Ananya Kumar, Aditi Raghunathan, Robbie Jones, Tengyu Ma, Percy Liang (Stanford) |
| **Ano / Venue** | 2022, **ICLR (oral)** · arXiv:2202.10054 |
| **PDF local** | `.task-context/input/referencias/.refs/RNP/artigos/R3_Kumar2022_LP-FT_ICLR.pdf` |
| **Contribuição** | Prova (teórica e empírica) que o fine-tuning completo **distorce** features pré-treinadas e pode ter desempenho **pior** que *linear probing* fora da distribuição (OOD); propõe LP-FT (LP depois FT). |

---

## 2. Contexto e lacuna

Há duas formas padrão de usar um *feature extractor* pré-treinado: **linear probing (LP)** — congela o
corpo, treina só a cabeça — e **fine-tuning (FT)** — treina tudo. A sabedoria comum (Kornblith et al.,
2019; Zhai et al., 2020; He et al., 2020) é que FT > LP. A lacuna teórica: análises de transfer learning
focavam em LP (Wu et al., 2020; Tripuraneni et al., 2020; Du et al., 2020), e **faltava** uma análise de
FT — difícil porque FT e treino-do-zero **minimizam a mesma loss** e diferem apenas na inicialização,
exigindo raciocinar sobre a *trajetória* do gradiente (regularização implícita da inicialização;
Neyshabur et al., 2014).

---

## 3. Contribuição central (1 frase)

> **Com features pré-treinadas boas e *distribution shift* grande, o fine-tuning completo distorce as
> features (move o corpo da rede na direção dos dados de treino e erra fora deles), ficando pior que
> linear probing no OOD; treinar a cabeça primeiro (LP) e só então fazer FT (LP-FT) evita a distorção.**

---

## 4. O método e a teoria (preciso)

**O mecanismo da distorção (Seção 3, Teorema 3.3; ilustração na Fig. 2).** Com uma cabeça inicial
aleatória, no FT o gradiente atualiza simultaneamente cabeça e corpo: para reduzir a loss de treino, o
corpo se move **na direção dos dados ID** (in-distribution) e **pouco na direção perpendicular** (OOD),
"distorcendo" as features pré-treinadas nessa direção não observada. Os autores provam, num cenário de
regressão linear sobreparametrizada de duas camadas, um **limite inferior** para o erro OOD do FT quando
inicializado com cabeça fixa/aleatória. Mostram ainda que a distorção **não se resolve com early
stopping** — em nenhum ponto da trajetória o FT passa por parâmetros bons no OOD.

**LP-FT.** Estratégia em dois passos: (1) **linear probing** (corpo congelado, treina a cabeça até ela
ficar boa); (2) **fine-tuning** completo a partir daí. Como a cabeça já está boa, o corpo se ajusta
pouco e **não distorce**. Custo computacional similar ao FT (o LP é barato).

---

## 5. Evidência

- Em **10 datasets de distribution shift** (Breeds, DomainNet, CIFAR→STL, FMoW, ImageNet-R/A/Sketch,
  etc.), FT obtém em média **+2% ID porém −7% OOD** vs. linear probing.
- Médias agregadas (Fig. 1): **FT 85,1% ID / 59,3% OOD · LP 82,9% / 66,2% · LP-FT 85,7% / 68,9%** —
  LP-FT vence nas duas (≈ +1% ID e +10% OOD sobre o FT).
- LP-FT altera as features **10×–100× menos** que o FT, evidência direta da menor distorção.

---

## 6. Fundamentação — o que os autores citam para sustentar

| Afirmação em LP-FT | Citação/base usada |
|---|---|
| FT > LP é a crença comum (que eles refutam no OOD) | **Kornblith et al. (2019); Zhai et al. (2020); He et al. (2020)** |
| Análise teórica via inicialização (reg. implícita) | **Neyshabur et al. (2014)**; redes lineares de 2 camadas: **Saxe et al. (2014); Gunasekar et al. (2017); Arora et al. (2018)** |
| Boas features de pré-treino (MoCo, CLIP) | **Chen et al. (2020); Radford et al. (2021)** |
| LP-FT e variantes já usadas antes | **Levine et al. (2016); Kanavati & Tsuneki (2021)**; *layerwise fine-tuning* **Howard & Ruder (2018)**; LR maior na cabeça **Prabhu et al. (2021)** |

> **Conexão de ouro para a defesa:** o próprio LP-FT cita o **gradual unfreezing de Howard & Ruder
> (2018)** [[01-ulmfit-progressive-unfreezing]] como uma **variante** da mesma família (*layerwise
> fine-tuning*). Ou seja, ULMFiT e LP-FT atacam o mesmo problema — e nós usamos essa família.

---

## 7. Onde encaixa no nosso trabalho

**(a) A Fase 1 (só a cabeça) é o "LP" do LP-FT.** Treinamos a cabeça SimCC com o backbone congelado
(`frozen_stages=4`) **antes** de liberá-lo (Fase 2), exatamente para não distorcer as features do COCO.
O treino COCO→futebol é um caso de **distribution shift** (mudança de domínio), regime em que a teoria
do LP-FT é mais relevante.

**(b) A degradação da Fase 3 é prevista pela teoria.** A Fase 3 (`frozen_stages=1`) aproxima o
pipeline do FT completo (mexer em mais camadas) — maior risco de **distorção de features** sob shift.
Nos experimentos, a **fase 3 degradou em 4/4 cenários**: não é um defeito do método, é o efeito descrito
formalmente por Kumar et al. (2022). Por isso a fase 3 é **condicional** (gating por Δ PCK) e o pipeline
retém o **melhor checkpoint**.

---

## 8. Defesa — perguntas prováveis do professor

**P: "Por que treinar só a cabeça primeiro?"**
R: Porque o FT completo com cabeça aleatória distorce as features pré-treinadas (Kumar et al., 2022,
Teorema 3.3); treinar a cabeça antes (linear probing) evita isso. É a Fase 1 do nosso pipeline.

**P: "A fase 3 piorou — não é falha de vocês?"**
R: É confirmação da teoria. A fase 3 se aproxima do FT completo, que distorce features sob distribution
shift (COCO→futebol) — efeito provado por LP-FT. Por isso a tornamos condicional e retemos o melhor
checkpoint.

**P: "FT completo não costuma ser melhor que linear probing?"**
R: Dentro da distribuição, sim (+2% ID no artigo); fora dela, pior (−7% OOD). Como nosso caso é mudança
de domínio, o cuidado do LP-FT se aplica.

**P: "Early stopping não resolveria a distorção?"**
R: Não — os autores mostram que a trajetória do FT não passa por parâmetros bons no OOD; por isso a
ordem (cabeça primeiro) importa, não o ponto de parada.

---

## 9. Frase de defesa (científica)

> "Treinamos a cabeça antes de liberar o backbone porque Kumar et al. (2022, LP-FT) provam que o
> fine-tuning completo com cabeça aleatória distorce as features pré-treinadas sob *distribution shift*,
> piorando o desempenho fora da distribuição; e é a mesma teoria que explica a degradação da nossa fase
> 3 — ao se aproximar do fine-tuning completo, ela distorce as features do COCO, motivo pelo qual a
> tornamos condicional."

---

**Anterior:** a generalidade por camada → [[02-yosinski-features-transferiveis]]. **Próximo:** até onde
descongelar (por que parar na fase 2) → [[04-surgical-finetuning]]. Mapa de defesa em [[00-indice]].
