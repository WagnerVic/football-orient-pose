# Artigo 2 — Yosinski et al.: a base experimental da generalidade por camada

> **Doc de estudo para a defesa científica.** É o **alicerce empírico**: quantifica, camada a camada, a
> generalidade vs. especificidade das features, fundamentando *por que* o progressive unfreezing
> ([[01-ulmfit-progressive-unfreezing]]) descongela do topo para a base e *por que* o pré-treino COCO é
> útil.

---

## 1. Ficha rápida

| | |
|---|---|
| **Título** | *How transferable are features in deep neural networks?* |
| **Autores** | Jason Yosinski, Jeff Clune, Yoshua Bengio, Hod Lipson |
| **Ano / Venue** | 2014, **NeurIPS** · arXiv:1411.1792 |
| **PDF local** | `.task-context/input/referencias/.refs/RNP/artigos/R1_Yosinski2014_transferable-features_NeurIPS.pdf` |
| **Contribuição** | Define uma forma de **medir** o grau de generalidade/especificidade de cada camada (via desempenho de transferência) e quantifica a transição geral→específico em uma CNN treinada na ImageNet. |

---

## 2. Contexto e lacuna

Era um fato conhecido que a **primeira camada** de redes treinadas em imagens naturais aprende filtros
do tipo Gabor e *color blobs*, independentemente da tarefa — o artigo cita Krizhevsky et al. (2012),
Lee et al. (2009) e Le et al. (2011). Igualmente, a **última camada** é específica (mapeia para as
classes do dataset). A lacuna: **onde e como ocorre a transição** geral→específico, e **quanto** dela é
transferível — questão central para transfer learning (Caruana, 1995; Bengio et al., 2011; Bengio,
2011).

---

## 3. Contribuição central (1 frase)

> **As camadas iniciais aprendem features gerais e transferíveis; as finais, features específicas da
> tarefa — e a transferência piora tanto pela especificidade das camadas altas quanto pela quebra de
> co-adaptação entre camadas.**

---

## 4. O método (preciso)

Os autores treinam duas redes de 8 camadas em duas metades disjuntas da ImageNet (tarefas A e B).
Depois copiam as **primeiras `n` camadas** de uma rede para outra e treinam o restante, variando
`n ∈ {1,…,7}`, em quatro configurações:
- **selffer (BnB):** copia `n` de B, retreina o resto em B (controle de co-adaptação);
- **transfer (AnB):** copia `n` de A, retreina o resto em B (mede transferência A→B);
- **versões `+` (BnB⁺, AnB⁺):** idem, mas com as camadas copiadas **fine-tunadas** (não congeladas).

Comparando a queda de desempenho conforme `n`, isolam **dois efeitos distintos** (Seção 4.1):
1. **Especificidade** das features das camadas altas (presas à tarefa de origem);
2. **Co-adaptação frágil** — camadas vizinhas co-adaptadas cujo "entrosamento" não é recuperável pelas
   camadas superiores quando se congela no meio delas (uma dificuldade de **otimização**, não de
   especificidade).

Também usam um *split* especial (objetos *man-made* vs *natural*) para medir o efeito da **distância
entre tarefas** sobre a transferência (Seção 4.2).

---

## 5. Evidência

- **Camadas 1–2 transferem quase perfeitamente** (AnB ≈ baseB), evidência de que são gerais; camadas
  4–7 caem progressivamente (especificidade crescente).
- **Co-adaptação frágil:** as redes selffer BnB pioram nas camadas intermediárias (3–6) — prova de que a
  queda não é só especificidade, mas também quebra de co-adaptação; o fine-tuning (BnB⁺) **recupera**
  essa perda.
- **Distância entre tarefas:** features transferem **pior** quando A e B são mais dissimilares
  (man-made vs natural).
- **Resultado surpreendente (Seção 4.1, item 5):** inicializar com features transferidas e **depois
  fine-tunar** produz **ganho de generalização que persiste** após o treino — em média **+1,6%** (e
  **+2,1%** mantendo ≥5 camadas) sobre treinar do zero.

---

## 6. Fundamentação — o que os autores citam para sustentar

| Afirmação em Yosinski | Citação/base usada |
|---|---|
| 1ª camada sempre aprende Gabor/cor (geral) | **Krizhevsky et al. (2012); Lee et al. (2009); Le et al. (2011)** |
| Features de camadas altas são transferíveis (prática prévia) | **Donahue et al. (2013); Zeiler & Fergus (2013); Sermanet et al. (2014)** |
| Motivação de transfer learning | **Caruana (1995); Bengio et al. (2011); Bengio (2011)** |
| Especificidade vs. co-adaptação como efeitos distintos | **Contribuição própria** (experimento controlado A/B, Fig. 2) |

> Para a defesa: aqui o peso é **experimental** — não é opinião, é a medição camada-a-camada na ImageNet
> (Fig. 2 do artigo) que estabelece a transição geral→específico.

---

## 7. Onde encaixa no nosso trabalho

- **Por que congelar o backbone e adaptar a cabeça primeiro:** o backbone do RTMPose (pré-treinado em
  COCO) aprendeu features de baixo nível de corpos humanos — gerais segundo Yosinski — que valem tanto
  para o COCO quanto para os jogadores do 3DSP; preservá-las é o recomendado.
- **Por que esperar que o TL (COCO) ajude:** pessoa-COCO e jogador compartilham features de baixo nível;
  a distância de domínio é moderada (mesma categoria: pessoa), e Yosinski mostra que, nesse regime, a
  transferência **melhora a generalização** mesmo após fine-tuning — base para a hipótese "Cenário C >
  Cenário A".
- **A ordem topo→base do ULMFiT** vem deste artigo: descongela primeiro o específico (alto), preserva o
  geral (baixo).

---

## 8. Defesa — perguntas prováveis do professor

**P: "Por que assumir que os pesos do COCO ajudam no futebol?"**
R: Não assumimos — Yosinski et al. (2014) medem que as camadas baixas são gerais e transferem quase sem
perda; e que features transferidas melhoram a generalização após fine-tuning. Confirmamos no domínio com
o Cenário C vs A.

**P: "Congelar não pode prejudicar?"**
R: O risco identificado por Yosinski é congelar **no meio** de camadas co-adaptadas (efeito de
otimização). O descongelamento progressivo libera essa co-adaptação quando necessário, mitigando-o.

**P: "E se o domínio do futebol for muito distante do COCO?"**
R: Yosinski mostra que a transferência piora com a distância entre tarefas — por isso **medimos** em vez
de assumir. O experimento decide.

---

## 9. Frase de defesa (científica)

> "Congelamos as camadas baixas do COCO e adaptamos do topo para a base porque Yosinski et al. (2014)
> demonstram experimentalmente, camada a camada na ImageNet, que as features iniciais são gerais e
> transferíveis enquanto as finais são específicas da tarefa, e que inicializar com features
> transferidas melhora a generalização mesmo após o fine-tuning."

---

**Anterior:** a técnica → [[01-ulmfit-progressive-unfreezing]]. **Próximo:** como aplicar sem distorcer
features (e por que a fase 3 piora) → [[03-lpft-distorcao-de-features]]. Mapa de defesa em [[00-indice]].
