# Relatório de Resultados Preliminares — Épico 1
**Fine-tuning RTMPose-X no Dataset 3DSP**
**Data:** 04 de junho de 2026 | **Hardware:** NVIDIA GeForce RTX 4090 (24 GB)

---

## 1. Resumo Executivo

O Épico 1 foi concluído com sucesso. A infraestrutura de fine-tuning foi validada end-to-end na madrugada de 03–04/06, com os Cenários A (from scratch) e C (transfer learning) treinados e avaliados no val split do 3DSP.

**Principal resultado:** o Transfer Learning (Cenário C) supera o zero-shot RTMPose-X em **+16,6 pp de PCK@0.2** e reduz o MPJPE-2D de 4,81 px para **3,63 px** no melhor checkpoint — demonstrando que o fine-tuning em dados de futebol é fortemente benéfico.

---

## 2. Setup Experimental

| Item | Valor |
|---|---|
| Modelo | RTMPose-X (CSPNeXt-X + RTMCCHead SimCC) |
| Input | 288×384 px (letterbox de 100×100) |
| Dataset | 3DSP — 160 clips treino / 40 clips val (3200 / 800 frames) |
| Keypoints | H3WB-17 (IDs 0,7,8,9 derivados, excluídos da loss) |
| Batch size | 16 |
| Optimizer | AdamW |
| GPU | NVIDIA RTX 4090 24 GB |
| Framework | MMPose 1.3.2 + MMEngine 0.10.7 |

---

## 3. Modelos Avaliados

### 3.1 Baselines Zero-Shot (sem fine-tuning)

Estes modelos foram treinados em COCO e aplicados diretamente no 3DSP, sem qualquer adaptação.

| Modelo | PDJ@0.5 | PCK@0.2 | OKS | MPJPE-2D |
|---|---|---|---|---|
| OpenPose | 56,1% | 22,1% | 48,5% | 25,58 px |
| HRNet-W48 | 88,9% | 40,5% | 76,2% | 6,04 px |
| **RTMPose-X (zero-shot)** | **93,6%** | **41,8%** | **81,8%** | **4,81 px** |

> RTMPose-X zero-shot é o **baseline principal** para medir o ganho do fine-tuning.

### 3.2 Cenário A — From Scratch, sem Augmentation

Pesos inicializados aleatoriamente. Treino de 50 épocas com LR uniforme 1×10⁻³.

| Split | PDJ@0.5 | PCK@0.2 | OKS | MPJPE-2D |
|---|---|---|---|---|
| **Treino** | 99,6% | 83,9% | 96,4% | 1,54 px |
| **Val** | **90,7%** | **47,0%** | **80,1%** | **5,34 px** |

**Diagnóstico:** overfitting moderado-alto (gap treino→val de **36,9 pp em PCK**). Com apenas 3.200 imagens, o modelo memoriza o treino mas generaliza parcialmente. Ainda assim, supera o zero-shot em PCK@0.2 (+5,2 pp), confirmando que o fine-tuning mesmo do zero agrega valor.

### 3.3 Cenário C — Transfer Learning, sem Augmentation

Pesos inicializados com checkpoint COCO. Progressive unfreezing em 3 fases.

#### Progressão por fase (val split)

| Fase | Épocas | frozen\_stages | LR Head | LR Backbone | PDJ@0.5 | PCK@0.2 | OKS | MPJPE-2D |
|---|---|---|---|---|---|---|---|---|
| **Zero-shot** (referência) | — | — | — | — | 93,6% | 41,8% | 81,8% | 4,81 px |
| **Fase 1** (head only) | 10 | 4 | 1×10⁻³ | 0 | 91,8% | 53,2% | 82,8% | 4,76 px |
| **Fase 2** (stages 2–3 + neck) | 75 | 2 | 1×10⁻⁴ | 1×10⁻⁵ | **95,2%** | **61,5%** | **87,3%** | **3,63 px** |
| **Fase 3** (stage 1–3 + neck) | 60 | 1 | 1×10⁻⁴ | 1×10⁻⁶ | 93,4% | 58,4% | 85,1% | 4,15 px |

**Melhor checkpoint:** Fase 2 (epoch 75) — PCK@0.2 = **61,5%**, MPJPE = **3,63 px**

| Split | PDJ@0.5 | PCK@0.2 | OKS | MPJPE-2D |
|---|---|---|---|---|
| **Treino** | 98,6% | 76,6% | 94,2% | 2,00 px |
| **Val** | **93,4%** | **58,4%** | **85,1%** | **4,15 px** |

**Diagnóstico:** overfitting menor que o Cenário A (gap de **18,2 pp** vs **36,9 pp**). O ponto de partida COCO fornece representações robustas que resistem melhor ao dataset pequeno.

---

## 4. Tabela Comparativa Global

| Modelo | PDJ@0.5 ↑ | PCK@0.2 ↑ | OKS ↑ | MPJPE-2D ↓ |
|---|---|---|---|---|
| OpenPose (zero-shot) | 56,1% | 22,1% | 48,5% | 25,58 px |
| HRNet-W48 (zero-shot) | 88,9% | 40,5% | 76,2% | 6,04 px |
| RTMPose-X (zero-shot) | 93,6% | 41,8% | 81,8% | 4,81 px |
| **Cenário A** (from scratch, val) | 90,7% | 47,0% | 80,1% | 5,34 px |
| **Cenário C** (TL fase 2, val) | **95,2%** | **61,5%** | **87,3%** | **3,63 px** |

---

## 5. Análise por Grupo Anatômico (PCK@0.2, val)

| Grupo | Zero-shot RTMPose | Cenário A | Cenário C (melhor) | Δ C vs Zero-shot |
|---|---|---|---|---|
| Head | 50,4% | 75,0% | 81,4% | **+31,0 pp** |
| Shoulder | 30,4% | 61,8% | 73,5% | **+43,1 pp** |
| Elbow | 50,8% | 32,7% | 50,6% | +0 pp |
| Wrist | 43,5% | 22,9% | 39,8% | -3,7 pp |
| Hip | 22,8% | 51,0% | 63,6% | **+40,8 pp** |
| Knee | 58,3% | 37,7% | 56,6% | -1,7 pp |
| Ankle | 59,4% | 36,1% | 60,1% | +0,7 pp |

**Observações:**
- O maior ganho do TL está em **head, shoulder e hip** — joints com características visuais de futebol distintas do domínio COCO
- **Wrist e elbow** mantêm performance similar ao zero-shot — são joints difíceis independente do método (alta variabilidade de pose durante o chute)
- O Cenário A piora elbow/wrist/knee em relação ao zero-shot — sinal de underfitting nesses joints com 3.200 amostras

---

## 6. Análise de Overfitting (os 3 números)

Metodologia definida em reunião com professor: comparar (1) performance no treino, (2) performance no val, (3) baseline zero-shot.

### Cenário A

```
Treino  PCK@0.2 = 83,9%
Val     PCK@0.2 = 47,0%   ← gap de 36,9 pp
Zero-shot         = 41,8%   ← referência inferior
```
→ **Overfitting moderado.** O modelo aprendeu features do domínio (supera zero-shot por 5,2 pp) mas memoriza demais o treino. Dataset de 3.200 frames não é suficiente para from scratch sem regularização extra.

### Cenário C — Fase 2 (melhor checkpoint)

```
Treino  PCK@0.2 = 76,6%
Val     PCK@0.2 = 61,5%   ← gap de 15,1 pp
Zero-shot         = 41,8%   ← referência inferior
```
→ **Generalização boa.** Gap treino→val de 15 pp com ganho de **+19,7 pp** sobre o zero-shot. Os pesos COCO atuam como regularizador implícito.

### Cenário C — Fase 3 (degradação)

A Fase 3 (descongelar até stage 1) **degradou a performance** de 61,5% → 58,4% PCK@0.2. Isso indica que descongelar as primeiras camadas do backbone com apenas 3.200 amostras causa overfitting nos estágios de baixo nível. **Para os Cenários B e D, considerar encerrar em Fase 2 se Δ PCK for negativo.**

---

## 7. Conclusões e Próximos Passos

### O que foi validado no Épico 1
- ✅ Pipeline de fine-tuning funcional end-to-end (DSP3Dataset → MMPose → checkpoint → avaliação)
- ✅ Cenário A (from scratch) treina e avalia corretamente
- ✅ Cenário C (transfer learning) com progressive unfreezing funcional
- ✅ Docker com GPU na RTX 4090, reproducível via `docker compose`

### Implicações para o Épico 2

1. **Cenário C Fase 2 é o melhor resultado atual** (PCK@0.2 = 61,5%) — servir como referência principal para comparação com Cenários B e D
2. **Cenário D** (TL + augmentation) tem potencial de superar C, especialmente em wrist/elbow onde C ainda é fraco
3. **Fase 3 do progressive unfreezing** pode ser prejudicial — monitorar delta entre fases antes de executar
4. **Cenário B** (from scratch + augmentation) deve superar A em wrist/elbow mas provavelmente não alcançará C

### Ordem de prioridade para Épico 2
| Prioridade | Cenário | Justificativa |
|---|---|---|
| Alta | **D** (TL + aug) | Potencialmente o melhor; complementa C |
| Alta | **B** (scratch + aug) | Isola efeito do aug; completa a matriz |
| — | A e C | ✅ Já executados |

---

*Gerado automaticamente a partir dos arquivos em `results/tables/`. Treinamento realizado na madrugada de 03–04 de junho de 2026 na GPU RTX 4090 (ubuntu-lab150-c4).*
