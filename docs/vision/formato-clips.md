# Formato de Clips Reais (vídeo bruto → clips estruturados)

Especificação da saída do **Épico 13 — Extração de Clips Reais** (#134). Define como um vídeo
broadcast bruto é transformado em clips estruturados que reutilizam a convenção do dataset 3DSP,
para serem anotados (Épico 9) e processados pelo pipeline (Épico 12) sem nenhuma adaptação dos
loaders existentes.

## Onde os clips vivem (estágios de processamento)

```
data/
├── raw/                      # vídeos broadcast brutos (.mp4) — gitignored
├── clips/                    # clips de frames INTEIROS, SEM crop  ← este doc
│   ├── examples/<id>/        # derivados dos data/examples
│   └── brazil/<id>/          # 5 clips do jogo do Brasil
└── crops/                    # (futuro) crops 100×100 do finalizador (estágio COM crop, Épico 11)
```

`clips/` (sem crop) e `crops/` (com crop) são estágios distintos — os clips são "raw" no sentido de
**frames inteiros, ainda não recortados**.

## Estrutura de um clip

```
data/clips/<fonte>/<clip_id>/
├── img/
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ... 020.jpg          # N frames INTEIROS (1280×720), 1-indexed, 3 dígitos
└── info.ini                  # metadados [info]
```

- **`img/` guarda frames inteiros** do broadcast (1280×720), **não** crops 100×100 — a detecção e o
  crop do finalizador acontecem depois (Épicos 10/11/12), indo para `data/crops/`.
- Naming `{:03d}.jpg`, **1-indexed** (`001.jpg`), exatamente como o 3DSP — compatível com
  `load_clip_image(clip_dir, idx)` (lê `img/{idx:03d}.jpg`).
- **Sem `posture/` e `gt/` no início.** A anotação de keypoints (Épico 9) cria o `posture/` depois;
  `load_full_clip` já trata `posture/` ausente (`has_posture=False`).

## `info.ini`

Seção `[info]` (lida por `load_clip_info` → dict). Campos:

| Campo | Exemplo | Descrição |
|---|---|---|
| `id` | `brazil_01` | identificador do clip (= nome da pasta) |
| `source_video` | `data/raw/brasil_x_y.mp4` | vídeo de origem |
| `game` | `Brasil x Y (2024)` | jogo |
| `label` | `Finalização` | tipo do lance |
| `start_ms` | `123000` | início do intervalo no vídeo (ms) |
| `end_ms` | `124000` | fim do intervalo (ms) |
| `fps` | `25` | quadros por segundo |
| `step_ms` | `40` | passo temporal entre frames (ms) |
| `num_frames` | `20` | nº de frames em `img/` (preenchido automaticamente) |
| `notes` | `chute a gol` | observação livre (opcional) |

Exemplo:

```ini
[info]
id = brazil_01
num_frames = 20
source_video = data/raw/brasil_x_y.mp4
game = Brasil x Y (2024)
label = Finalização
start_ms = 123000
end_ms = 124000
fps = 25
step_ms = 40
notes = chute a gol
```

## Regras

- **≥ 5 clips** distintos no diretório de saída.
- **Nº de frames livre por clip** (escolhido na geração via `--n`, default 20; cada fonte pode ter o
  seu — ex.: examples=20, brazil=10). O valor real fica registrado em `num_frames` no `info.ini`.
  Se o intervalo render menos que `--n`, é **erro** (intervalo curto demais).
- **Consistência:** `len(img/) == num_frames` e `img/` **contíguo** `001..N` (1-indexed `{:03d}`) —
  o validador checa isso contra o próprio `info.ini` (pega "apaguei frames sem renumerar").
- **Resolução ≥ 720p** (altura ≥ 720).
- **Um momento de finalização** por clip (chute/cabeçada a gol).
- `info.ini` com todos os campos obrigatórios.

## Ferramentas

- **Gerar:** `scripts/clips/cut_clips.py --video <mp4> --intervals <intervals.json> --root data/clips/<fonte>`
  (ex.: `--root data/clips/brazil`) → usa `clip_extractor.cut_clip` (corte + amostragem) e
  `clip_extractor.write_clip` (escrita).
- **Validar:** `scripts/clips/validate_clips.py --root data/clips` → varre recursivamente (acha clips
  por fonte), confere todas as regras e a compatibilidade com `load_clip_image`/`load_clip_info`
  (exit code ≠ 0 em caso de erro).

`intervals.json`:

```json
[
  {"id": "brazil_01", "start_ms": 123000, "end_ms": 124000, "label": "Finalização", "game": "Brasil x Y (2024)"},
  {"id": "brazil_02", "start_ms": 456000, "end_ms": 457000, "label": "Finalização", "game": "Brasil x Y (2024)"}
]
```

## Compatibilidade com o código existente

- `load_clip_image(clip_dir, idx)` — `utils/data_io.py`, lê `img/{idx:03d}.jpg`.
- `load_clip_info(clip_dir)` — lê o `[info]` do `info.ini`.
- `load_full_clip(clip_dir, num_frames)` — carrega frames + (se existir) `posture/`.

Como os clips reais seguem a mesma estrutura, **nenhum** desses loaders precisa mudar.
