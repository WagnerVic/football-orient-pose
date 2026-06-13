# [BACKLOG] Comparação do crop justo (YOLO) × crop do dataset (3DSP)

- **Decisão:** adiado em **2026-06-12**.
- **Épico/Issues:** #119 (US #120/#123, tasks #121/#122/#124/#125).
- **Bloqueio:** depende de **GT de keypoint** que não existe (anotação — US #109).

## O que é o experimento

Medir **quanto o crop justo melhora a pose** vs o crop frouxo do 3DSP. O baseline (Reis et al.) e o
3DSP usam um crop **frouxo** (o jogador ocupa ~34×74px num quadro 100×100 padded — "muito fundo",
crítica do professor). A hipótese: um crop **justo** (jogador preenche o quadro, gerado do frame
inteiro com a caixa do YOLO26x) → keypoints mais precisos.

A conta é: rodar a pose nos dois crops → comparar cada predição com o **GT de keypoint** (onde a
junta realmente está) → Δ nas métricas PDJ/PCK/OKS.

## Por que adiamos

O experimento **exige GT de keypoint**, e ele **não está disponível**:

- O `data/test` (= nossos `examples`) tem o **crop frouxo real** e a **caixa** (`gt.txt`), mas
  **não tem `posture/`** (keypoints). Confirmado nos 10 clips do test e nas refs.
- O `data/train` **tem** keypoint GT (`posture/`), mas **só o crop 100×100** — sem o **frame
  inteiro**, não dá para gerar nosso crop justo nele.

Logo, para o cenário "de verdade" (justo do frame inteiro), seria preciso **anotar keypoints**
(US #109) — um bloqueio de trabalho significativo. Priorizamos os passos **não bloqueados** primeiro.

## Cenário escolhido para quando retomar

**Examples (frame inteiro), deploy-realista** — mede **enquadramento + resolução** (é o cenário do
pipeline real):

1. **Frouxo (baseline)** = crop real do 3DSP test (`data/test/<id>/img/NNN.jpg`).
2. **Justo (nosso)** = `crop.make_crop(frame_inteiro, caixa_finalizador, "tight")` — caixa do
   finalizador vem do `gt.txt`/`shooter_tracklet_id` do 3DSP (já conhecida nos examples).
3. **Keypoint GT** = anotar **uma vez** no crop frouxo → **reprojetar** para o justo via
   `crop.crop_to_frame`/`frame_to_crop` (única variável = enquadramento).
4. Rodar pose nos dois → Δ métricas.

## Alternativa mais barata (registrada, não escolhida)

**Experimento só de enquadramento no `train-val` (sem anotação):** os 40 clips de val do train têm
crop frouxo **+ keypoint GT** (`posture/`) **+** caixa (`gt.txt`). Dá para gerar um "justo"
**recortando o jogador de dentro do próprio crop frouxo** e comparar — **800 frames, zero anotação**.

- **Prós:** grátis, amostra grande, resultado rápido.
- **Contra:** o "justo" sai de *upscaling* de um crop já pequeno → mede **só o enquadramento**, não o
  ganho de **resolução** (não há frame inteiro para recortar em alta-res). Por isso **não** é o
  número de deploy — fica como alternativa caso o examples (anotado) não seja viável.

Outras descartadas: pseudo-GT puro de modelo (circular/enviesado); métrica-proxy sem GT (fraca).

## O que é preciso para retomar

- **GT de keypoint** dos examples → **US #109** (anotação dos 13 joints do finalizador + derivar os
  4 centros), reprojetável via `crop.py`.

## Dependências já prontas

- ✅ Detector vencedor (**YOLO26x**, Épico #113) — gera a caixa do finalizador.
- ✅ Módulo `crop.py` (US #101) — crop justo/frouxo + transforms de reprojeção.
