# Formato de Crops do Finalizador (estágio COM crop)

Especificação do estágio **`data/crops/`** — os recortes 100×100 do finalizador, gerados pelo
pipeline (`detecção → finalizador → crop justo`) a partir dos clips de frame inteiro
(`data/clips/`). É o **insumo da anotação de keypoints** (US #109) e dos experimentos de pose.

Espelha o `docs/vision/formato-clips.md` (estágio anterior, sem crop).

## Estrutura

```
data/crops/<fonte>/<clip_id>/
├── img/
│   ├── 001.jpg
│   └── ... NNN.jpg          # crop justo 100×100 do finalizador (1-indexed, 3 dígitos)
├── crop_params/
│   ├── 001.json
│   └── ... NNN.json         # geometria do recorte (chave de reprojeção frame↔crop)
└── info.ini                 # metadados [info]
```

- `img/NNN.jpg` — crop **justo** (letterbox, preserva aspecto), `size×size` (default 100×100).
- `crop_params/NNN.json` — escrito por `crops_io.write_crop_clip`, relido por `load_crop_params`.
- **Sem `posture/` no início.** A anotação de keypoints (US #109) cria `posture/NNN.json` ao lado;
  o keypoint anotado no crop reprojeta para o frame (e entre crops) via `crop.py`.

## `crop_params/NNN.json`

```json
{
  "crop_params": {
    "x0": 432.0, "y0": 339.0, "scale": 1.01,
    "pad_top": 0.0, "pad_left": 13.0, "size": 100
  },
  "finisher_bbox_xyxy": [432.0, 339.0, 490.0, 419.0],
  "source_frame": 9,
  "crop_mode": "tight",
  "detector": "yolo26x"
}
```

- `crop_params` — `CropParams` (origem no frame, escala do letterbox, padding). Permite
  `crop.crop_to_frame` / `frame_to_crop` (reprojetar keypoints crop↔frame).
- `finisher_bbox_xyxy` — caixa do finalizador no frame (origem do recorte).
- `crop_mode` — `tight` (justo) ou `loose`; `detector` — modelo que gerou a caixa.

## `info.ini`

Seção `[info]` (configparser). Campos típicos: `id`, `num_frames`, `size`, `fonte`,
`source_clip` (clip de origem em `data/clips/`), `crop_mode`, `detector`, `pose` (se aplicável).

## Regras

- Naming `{:03d}.jpg`, **1-indexed** — compatível com `load_clip_image`.
- `len(img/) == len(crop_params/) == num_frames`.
- Versionado (crops são pequenos; `.gitignore` mantém `!data/crops/` e ignora `_preview_*`).

## Ferramentas

- **Gerar:** `crops_io.write_crop_clip(root, clip_id, crops, params, bboxes, meta)` — usado pelo
  `scripts/pipeline/demo_examples.py`.
- **Reler params:** `crops_io.load_crop_params(clip_dir, frame_idx) -> CropParams`.
