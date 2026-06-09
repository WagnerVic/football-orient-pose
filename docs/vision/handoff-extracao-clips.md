# Handoff p/ o time — Extração de Clips Reais (Épico 13)

> **Para quem vai mexer em clips reais, anotação ou pipeline.** A ferramenta de extração de clips
> **já existe, está testada e versionada**. Este doc diz **o que reaproveitar** e **o que NÃO
> refazer** — pra ninguém recriar o que já está pronto.

---

## TL;DR (não retrabalhe!)

- ✅ **Já pronto:** ler vídeo → cortar em clips de 20 frames → gravar no formato do 3DSP → validar.
- 🚫 **NÃO recrie:** leitura de vídeo, corte, escrita de `img/`+`info.ini`, validação. Está feito.
- 👉 **Dev dos clips do Brasil:** você **não precisa programar nada**. Põe o vídeo numa pasta,
  escreve um JSON com os intervalos e roda **2 comandos** (gerar + validar). Pronto.
- 👉 **Dev da anotação (Épico 9):** os clips já vêm **versionados** no repo, no formato certo.
  Você anota em cima deles — não precisa gerar nada.

---

## O que foi feito (e fica de base)

Uma cadeia **vídeo bruto → clips estruturados**, reutilizável por detecção, anotação e pipeline.

| Camada | Arquivo | O que faz |
|---|---|---|
| Módulo | `src/football_orient_pose/video_io.py` | `extract_frames(video)` → lista de frames |
| Módulo | `src/football_orient_pose/clip_extractor.py` | `cut_clip` (corta 20 frames) + `write_clip` (grava `img/`+`info.ini`) |
| Script | `scripts/clips/cut_clips.py` | CLI: vídeo + intervalos → `data/clips/<fonte>/<id>/` |
| Script | `scripts/clips/validate_clips.py` | confere se os clips batem com as regras |
| Doc | `docs/vision/formato-clips.md` | **especificação** do formato de clip |
| Testes | `tests/clips/test_video_io.py`, `test_clip_extractor.py`, `test_validate_clips.py` | verdes |

**Saída de cada clip** (idêntica ao 3DSP, então os loaders do projeto leem sem mudança):

```
data/clips/<fonte>/<clip_id>/        # fonte = examples | brazil
├── img/001.jpg ... 020.jpg     # 20 frames INTEIROS 1280×720 (não são crops!)
└── info.ini                    # metadados [info]
```

> Os clips ficam **versionados no git** (`data/clips/` está liberado no `.gitignore`), porque **a
> anotação acontece sobre esses pixels** — o par (frame, anotação) tem que ser reproduzível.

---

## 🇧🇷 Passo a passo — dev dos clips do Brasil (sem codar)

Seu trabalho é **só fornecer o material e rodar a ferramenta**. Não escreva código novo.

**1. Ponha o vídeo do jogo numa pasta**
```
data/raw/brasil_x_adversario.mp4      # baixado do YouTube, ≥720p
```

**2. Marque os 5 momentos de finalização** num arquivo JSON (tempos em **milissegundos**):
```json
[
  {"id": "brazil_01", "start_ms": 123000, "end_ms": 124000, "label": "Finalização", "game": "Brasil x Y (2024)"},
  {"id": "brazil_02", "start_ms": 456000, "end_ms": 457000, "label": "Finalização", "game": "Brasil x Y (2024)"},
  {"id": "brazil_03", "start_ms": 789000, "end_ms": 790000, "label": "Finalização", "game": "Brasil x Y (2024)"},
  {"id": "brazil_04", "start_ms": 991000, "end_ms": 992000, "label": "Finalização", "game": "Brasil x Y (2024)"},
  {"id": "brazil_05", "start_ms": 1203000,"end_ms": 1204000,"label": "Finalização", "game": "Brasil x Y (2024)"}
]
```
Salve como `intervals_brazil.json`. (Cada intervalo ~1s rende os 20 frames.)

**3. Gere os clips** (na pasta da fonte `brazil`):
```bash
python scripts/clips/cut_clips.py --video data/raw/brasil_x_adversario.mp4 \
  --intervals intervals_brazil.json --root data/clips/brazil
```
→ cria `data/clips/brazil/brazil_01..05/` (cada um com `img/001..020.jpg` + `info.ini`).

**4. Valide:**
```bash
python scripts/clips/validate_clips.py --root data/clips
```
→ tem que dar `✅ clips válidos`. Se reclamar (ex.: "intervalo curto", "altura < 720"), ajuste o
JSON/vídeo e rode de novo.

**5. Commite os clips** (eles entram no git — é o substrato da anotação).

Pronto. **Você não toca em Python.** O resto da equipe pega `data/clips/brazil/*` daí.

---

## 🏷️ Para o dev da anotação (Épico 9)

- Os clips (`data/clips/examples/*` e, em breve, `data/clips/brazil/*`) já estão **no repo, no formato certo**.
- Você anota **em cima deles** (bbox de pessoa e/ou keypoints do finalizador) — **não gera clip**.
- Os loaders já existentes leem qualquer clip: `load_clip_image(clip_dir, i)`, `load_clip_info(clip_dir)`.
- A anotação cria o `posture/` depois; o formato já prevê isso (clip nasce só com `img/` + `info.ini`).

---

## 🔁 Para quem for mexer em detecção / pipeline (Épicos 10/12)

- **Reutilize `extract_frames`** (`video_io.py`) para ler vídeo — não escreva outra leitura.
- O pipeline (Épico 12) vai **importar** `extract_frames` e `cut_clip` direto (em memória), não rodar
  os scripts.

---

## Regras do formato (resumo)

Detalhe completo em [`docs/vision/formato-clips.md`](formato-clips.md).

- **≥ 5 clips**, **20 frames** cada (config via `--n`), **≥ 720p**, **1 finalização por clip**.
- `img/` 1-indexed `{:03d}.jpg`; `info.ini` com: `id, source_video, game, label, start_ms, end_ms,
  fps, step_ms, num_frames, notes`.
- Frames são **inteiros** (1280×720), **não** crops — o crop do finalizador vem depois (Épico 11).

---

## ❌ O que NÃO fazer (evitar retrabalho)

- Não escrever outra função de **ler vídeo** → use `extract_frames`.
- Não escrever outro **cortador/gravador** de clip → use `cut_clips.py` / `clip_extractor`.
- Não **re-anotar os examples** nem recriar clips que já estão em `data/clips/`.
- Não inventar outro formato de pasta → siga `formato-clips.md` (senão os loaders quebram).

---

## Onde está cada coisa

```
src/football_orient_pose/video_io.py         # ler vídeo
src/football_orient_pose/clip_extractor.py   # cortar + gravar clip
scripts/clips/cut_clips.py                         # CLI gerar
scripts/clips/validate_clips.py                    # CLI validar
docs/vision/formato-clips.md                 # spec do formato
data/clips/{examples,brazil}/                 # clips gerados, sem crop (versionados — anotação ocorre aqui)
data/crops/                                   # (futuro) crops 100x100 do finalizador (Épico 11)
```

Issues de referência: Épico 13 (#134; US #135/#138/#142; tasks #146 video_io, #136/#137/#139/#140/#141; aplicar #145).
Geração dos clips do Brasil: **#143** (fornecer vídeo + intervalos) e **#144** (rodar + validar).
