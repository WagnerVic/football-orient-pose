# Fine-tuning de Estimação de Pose ao Domínio de Futebol — Transfer Learning × Data Augmentation

> **Trabalho Final de Redes Neurais Profundas.** Fine-tuning do estimador de pose **RTMPose-X** para
> jogadores de futebol *broadcast*, maximizando a precisão de localização dos keypoints
> (**PCK@0.2: 41,8% → 67,5%**) via uma **matriz 2×2** (Transfer Learning × Data Augmentation).

**Grupo:** Wagner Victor Alves de Menezes · Raphael Alves de Lima Soares · Victor Gabriel ·
João Victor Fernandes · Júlia Junior
Bacharelado em Inteligência Artificial — Instituto de Informática, **Universidade Federal de Goiás (UFG)**
Disciplina: **Redes Neurais Profundas**

---

## 📌 Para o professor — por onde começar

1. **O artigo** → [docs/artigo/artigo-rnp.md](docs/artigo/artigo-rnp.md) — **a entrega escrita**
   (Pergunta · Metodologia · Resultados · Conclusão).
2. **Este README** — o **mapa do código**: a estrutura do repositório, o que cada arquivo implementa
   e como reproduzir os experimentos.

> A avaliação tem duas frentes: a **escrita do artigo** e os **códigos usados**. Este README cobre a
> segunda — os mapas abaixo apontam exatamente onde está cada coisa.

---

## 🎯 Resultados principais

Matriz 2×2 no val split do 3DSP (800 frames):

| Cenário | Inicialização | Augmentation | **PCK@0.2** | gap treino→val |
|---|---|---|---:|---:|
| Zero-shot (baseline) | COCO | — | 41,8% | — |
| **A** | from scratch | RAW | 37,8% | 🔴 alto |
| **B** | from scratch | full | 58,1% | 37,1 pp |
| **C** | transfer learning | RAW | 52,9% | 12,2 pp |
| **D-FULL** ⭐ | transfer learning | full | **67,5%** | **1,1 pp** |

> A receita campeã (**D-FULL**: transfer learning + augmentation geométrica) atinge **PCK@0.2 = 67,5%
> (+25,8 pp sobre o zero-shot)** com quase nenhum overfitting (gap 1,1 pp). Nem o transfer learning
> sozinho, nem a augmentation sozinha bastam — é a combinação que resolve, inclusive recuperando as
> extremidades. Detalhamento no artigo (§3–4).

---

## 🗂️ Estrutura do repositório

| Pasta / arquivo | O que é |
|---|---|
| [`src/football_orient_pose/finetuning/`](src/football_orient_pose/finetuning/) | núcleo do fine-tuning: dataset adapter, métrica de seleção, augmentations custom |
| [`src/football_orient_pose/evaluation/`](src/football_orient_pose/evaluation/) | métricas (PCK · PDJ · OKS · MPJPE) e a avaliação |
| [`src/football_orient_pose/estimators/`](src/football_orient_pose/estimators/) | wrappers dos estimadores (`rtmpose.py` com `from_checkpoint`) |
| [`src/football_orient_pose/utils/`](src/football_orient_pose/utils/) | `keypoint_mapping.py`, `data_io.py`, `skeleton.py` |
| [`configs/`](configs/) | `cenario_*.py` (os cenários da matriz/escada) + `split.json` (split 80/20 por clip, seed 42) |
| [`scripts/training/`](scripts/training/) | `train.py` (orquestrador) + `run_*.sh` (experimentos) + smoke test |
| [`scripts/evaluation/`](scripts/evaluation/) | `evaluate.py` (avalia checkpoint nas 4 métricas) |
| [`scripts/setup/`](scripts/setup/) | setup do ambiente MMPose + download dos pesos |
| [`results/tables/`](results/tables/) | métricas geradas (`finetuned_cenario_*`, `finetuned_fase*`) |
| `results/checkpoints/` | `best_PCK.pth` por cenário |
| [`tests/`](tests/) | suíte `pytest` (metrics, dataset, estimators) |
| [`Dockerfile.finetuning`](Dockerfile.finetuning) · [`Makefile`](Makefile) | infra (stack MMPose + pesos COCO) e atalhos |
| [`docs/artigo/artigo-rnp.md`](docs/artigo/artigo-rnp.md) | o artigo |

---

## 🗺️ Mapa do código — o que cada arquivo implementa

Fluxo do fine-tuning e o arquivo de cada etapa:

```
3DSP (JSON + JPG 100×100)
  └─ DSP3Dataset ──(letterbox 288×384)── RTMPose-X (CSPNeXt-X + SimCC)
        └─ train.py: progressive unfreezing + LR discriminativo + gating
              └─ evaluate.py: PCK@0.2 · PDJ@0.5 · OKS · MPJPE   (vs baseline zero-shot)
```

### A) Adaptação do dataset e do treino (o núcleo do trabalho)

| Funcionalidade | Arquivo | O que faz |
|---|---|---|
| **Adapter 3DSP → MMPose** | [`finetuning/dataset.py`](src/football_orient_pose/finetuning/dataset.py) | `DSP3Dataset`: lê os clips do split; `visibility=0` nos 4 keypoints derivados [0,7,8,9] (excluídos da loss); `flip_indices` H3WB; `bbox=[0,0,100,100]` (o crop inteiro é a ROI) |
| **Augmentations custom** | [`finetuning/transforms.py`](src/football_orient_pose/finetuning/transforms.py) | `SimpleMotionBlur` (cv2, blur direcional) e `SimpleRandomErasing` (numpy, oclusão) — reimplementados por não estarem registrados no container |
| **Métrica de seleção** | [`finetuning/metric.py`](src/football_orient_pose/finetuning/metric.py) | `StrictPCKMetric`: PCK@0.2 **estrito** no loop de validação do MMPose (`save_best='PCK'`) — alinha a seleção do checkpoint à métrica-alvo |

### B) Configs da matriz e da escada de augmentation

| Arquivo | Cenário |
|---|---|
| [`cenario_a.py`](configs/cenario_a.py) · [`cenario_a-raw.py`](configs/cenario_a-raw.py) | **A** — from scratch (RAW / +flip) |
| [`cenario_b.py`](configs/cenario_b.py) | **B** — from scratch + augmentation completa |
| [`cenario_c.py`](configs/cenario_c.py) · [`cenario_c-raw.py`](configs/cenario_c-raw.py) · [`cenario_c2.py`](configs/cenario_c2.py) | **C** — transfer learning (RAW / +flip / fase única) |
| [`cenario_d.py`](configs/cenario_d.py) · [`d-geom.py`](configs/cenario_d-geom.py) · [`d-occl.py`](configs/cenario_d-occl.py) | **D** — TL + augmentation (escada: geom → +ocl → full) |
| [`split.json`](configs/split.json) | split oficial 80/20 **por clip** (seed 42) |

### C) Treino — orquestração

| Arquivo | O que faz |
|---|---|
| [`scripts/training/train.py`](scripts/training/train.py) | **Orquestrador** `--cenario A/B/C/D`: fase única (scratch) ou **3 fases de progressive unfreezing** (TL, `frozen_stages` 4→2→1); **LR discriminativo** (cabeça > backbone); **gating** da fase 3 por Δ PCK; seleção do **melhor checkpoint** entre fases |
| [`run_experiments.sh`](scripts/training/run_experiments.sh) · [`run_raw.sh`](scripts/training/run_raw.sh) · [`run_bd.sh`](scripts/training/run_bd.sh) · [`run_c2.sh`](scripts/training/run_c2.sh) | rodam os experimentos *fire-and-forget* (A/C · RAW · B/D · ablação C2) |
| [`smoke_cenario_a.py`](scripts/training/smoke_cenario_a.py) | smoke test rápido — valida o pipeline ponta-a-ponta |

### D) Avaliação — as 4 métricas

| Arquivo | O que faz |
|---|---|
| [`scripts/evaluation/evaluate.py`](scripts/evaluation/evaluate.py) | avalia um checkpoint no val → **PCK@0.2 · PDJ@0.5 · OKS · MPJPE-2D** |
| [`evaluation/metrics.py`](src/football_orient_pose/evaluation/metrics.py) | `compute_pck` · `compute_pdj` · `compute_oks` · `compute_mpjpe_2d` |
| [`estimators/rtmpose.py`](src/football_orient_pose/estimators/rtmpose.py) | `RTMPoseEstimator.from_checkpoint()` — carrega o `.pth` fine-tunado |
| [`utils/keypoint_mapping.py`](src/football_orient_pose/utils/keypoint_mapping.py) | `coco17_to_h3wb17()` e `derive_h3wb_centers()` (deriva [0,7,8,9]) |

### E) Infraestrutura (reprodutibilidade)

| Arquivo | O que faz |
|---|---|
| [`Dockerfile.finetuning`](Dockerfile.finetuning) | stack MMPose pinada (torch 2.4 / mmcv 2.2 / mmpose 1.3 / mmdet 3.3) **+ pesos COCO baked** |
| [`docker-compose.finetuning.yml`](docker-compose.finetuning.yml) | serviço de treino (GPU) |
| [`scripts/setup/setup_mmpose_env.sh`](scripts/setup/setup_mmpose_env.sh) | ambiente `.venv-mmpose` em user-space (sem sudo) |
| [`Makefile`](Makefile) | atalhos: `train-a/b/c/d`, `evaluate`, `finetuning-{env,checkpoint,smoke}`, `docker-*` |

---

## ▶️ Como reproduzir

```bash
# 1. ambiente + dados + pesos COCO
make setup
make finetuning-env            # .venv-mmpose (user-space, sem sudo)
make finetuning-checkpoint     # pesos COCO p/ transfer learning
make finetuning-smoke          # smoke test do pipeline
# Alternativa em container:  docker compose -f docker-compose.finetuning.yml build

# 2. treinar a matriz 2×2
make train-a   # from scratch, sem augmentation
make train-b   # from scratch, com augmentation
make train-c   # transfer learning, sem augmentation
make train-d   # transfer learning, com augmentation   ← campeão (67,5%)

# 3. avaliar um checkpoint (4 métricas no val)
make evaluate CKPT=results/checkpoints/cenario_D/best_PCK.pth CONFIG=configs/cenario_d.py

# 4. testes do código
uv run pytest tests/metrics tests/dataset tests/estimators
```

---

## 🔭 Escopo & decisões

- **Foco desta entrega:** o **estudo de fine-tuning** (adaptação de domínio do RTMPose-X). A detecção
  de jogadores e o pipeline ponta-a-ponta pertencem ao trabalho de **Visão Computacional** e estão
  fora do escopo aqui.
- **"Sem augmentation" = RAW:** o baseline honesto é a imagem crua (sem flip); o flip é o 1º degrau da
  escada de augmentation (ablação no artigo, §3.3).
- **Seleção de checkpoint:** sempre pelo **PCK@0.2 estrito** (`StrictPCKMetric`), a métrica-alvo — e
  não pelo PCK leniente do MMPose.
- **Test-Time Augmentation:** sugerido como extensão, **não realizado** nesta entrega.
- **Limitação dominante:** dataset pequeno (~160 cenas distintas) — origem do overfitting e do teto de
  desempenho absoluto; mais dados é a direção de melhoria mais clara.
- **Artigo:** também submetido na plataforma da disciplina (Atividade A2).

---

## 🧩 Snippets dos códigos principais *(complemento)*

Trechos-chave de cada peça do mapa acima, para leitura rápida.

**Adapter 3DSP → MMPose** — `finetuning/dataset.py`
```python
_DERIVED_KEYPOINT_IDS = {0, 7, 8, 9}            # keypoints calculados (médias)
visibility = np.ones(17, dtype=np.float32)
for kid in _DERIVED_KEYPOINT_IDS:
    visibility[kid] = 0.0                       # excluídos da SimCC loss
bbox = np.array([[0.0, 0.0, w, h]], dtype=np.float32)   # ROI = crop inteiro (100×100)
```

**Métrica de seleção** — `finetuning/metric.py`
```python
@METRICS.register_module()
class StrictPCKMetric(BaseMetric):
    def compute_metrics(self, results):
        preds = derive_h3wb_centers(np.stack([p for p, _ in results]))  # deriva [0,7,8,9]
        gts   = np.stack([g for _, g in results])
        pck = compute_pck(preds, gts, threshold=self.pck_thr)           # PCK@0.2 estrito
        return {"PCK": float(pck.global_score), ...}
```

**Progressive unfreezing — LR discriminativo + gating** — `scripts/training/train.py`
```python
def _build_tl_optim_override(phase):
    if phase == 1:                               # backbone congelado: só a cabeça, LR alto
        return dict(optimizer=dict(type="AdamW", lr=1e-3, weight_decay=0.05))
    if phase == 2:                               # destrava o topo: cabeça 1e-4, backbone ×0.1
        return dict(optimizer=dict(type="AdamW", lr=1e-4, weight_decay=0.05),
                    paramwise_cfg=dict(custom_keys={"backbone": dict(lr_mult=0.1)}))
    ...                                          # phase 3: backbone ×0.01

delta = pck_f2 - pck_f1
if delta > delta_pck:                            # fase 3 só roda se valer a pena
    final_ckpt = ckpt_f3 if pck_f3 > pck_f2 else ckpt_f2   # mantém o melhor
else:
    final_ckpt = ckpt_f2                          # fase 3 pulada
```

**Escada de augmentation no pipeline** — `configs/cenario_d.py`
```python
train_pipeline = [
    dict(type="RandomFlip", direction="horizontal"),                                  # flip
    dict(type="RandomBBoxTransform", scale_factor=(0.75, 1.25),
         rotate_factor=30.0, shift_factor=0.1),                                        # geométrica
    dict(type="SimpleMotionBlur", blur_limit=(3, 9), p=0.5),                           # blur
    dict(type="TopdownAffine", input_size=(288, 384), use_udp=True),                   # letterbox
    dict(type="SimpleRandomErasing", n_patches=(1, 1), ratio=(0.02, 0.10), prob=0.3),  # oclusão
    dict(type="GenerateTarget", encoder=codec), dict(type="PackPoseInputs"),
]
load_from = "checkpoints/rtmpose-x_coco.pth"     # Transfer Learning (Cenários C/D)
```

**Avaliação — as 4 métricas** — `scripts/evaluation/evaluate.py`
```python
pred  = derive_h3wb_centers(pred)             # mesma derivação do GT
pdj   = compute_pdj(pred, gt, threshold=0.5)
pck   = compute_pck(pred, gt, threshold=0.2)  # métrica principal
oks   = compute_oks(pred, gt)
mpjpe = compute_mpjpe_2d(pred, gt)
```
