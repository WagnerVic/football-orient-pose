# Technical reports

Detailed reports behind every number in the paper, written in **Portuguese** during development.
Numbers such as "Épico #113" refer to this repository's
[GitHub issues](https://github.com/WagnerVic/football-orient-pose/issues?q=is%3Aissue), where each
experiment was planned and tracked.

## Detector selection

| Report | Contents |
|---|---|
| [`vision/epic-113-detectores.md`](vision/epic-113-detectores.md) | Benchmark of YOLO26, RetinaNet, Faster R-CNN and Cascade R-CNN against 740 hand-annotated boxes; why capacity-matched variants matter (YOLO26n vs YOLO26x) |

## Pose estimator benchmark (zero-shot)

| Report | Contents |
|---|---|
| [`vision/baseline-rtmpose-zero-shot.md`](vision/baseline-rtmpose-zero-shot.md) | Zero-shot RTMPose-X on 3DSP — PDJ, PCK, OKS and MPJPE per body part, showing the localization bottleneck |
| [`vision/epic2-entrega-final.md`](vision/epic2-entrega-final.md) | Comparison of OpenPose, HRNet-W48 and RTMPose-X; COCO-17 → H3WB-17 keypoint mapping and metric definitions |

## Domain adaptation (fine-tuning)

| Report | Contents |
|---|---|
| [`finetuning/epico-2/epic2-relatorio-final.md`](finetuning/epico-2/epic2-relatorio-final.md) | **Main report** — the full transfer learning × augmentation study (10 models), augmentation ladder, overfitting diagnosis, per-joint results, progressive unfreezing phases |
| [`finetuning/epico-2/epic2-relatorio-a-c.md`](finetuning/epico-2/epic2-relatorio-a-c.md) | Details of the settings without augmentation (A, C) and the flip and single-phase ablations |
| [`finetuning/epico-2/epic2-relatorio-bd.md`](finetuning/epico-2/epic2-relatorio-bd.md) | Details of the settings with augmentation (B, D) and the attribution of the gain to each augmentation |
| [`finetuning/epico-1/epic1-relatorio-preliminar.md`](finetuning/epico-1/epic1-relatorio-preliminar.md) | First runs of settings A and C, before the code-review fixes (kept for the record) |

## End-to-end pipeline and data formats

| Report | Contents |
|---|---|
| [`vision/epic-126-pipeline.md`](vision/epic-126-pipeline.md) | Detection → crop → pose → reprojection on real broadcast clips; tracking of the shooter across a clip |
| [`vision/formato-clips.md`](vision/formato-clips.md) | Folder layout and `info.ini` fields of the clips in `data/clips/` |
| [`vision/formato-crops.md`](vision/formato-crops.md) | Layout of the player crops and the parameters needed to map them back to the frame |
