DATA_DIR := data
ZIP_FILES := $(wildcard *.zip)
MARKERS := $(patsubst %.zip,$(DATA_DIR)/.extracted_%,$(ZIP_FILES))

VENV_MM := .venv-mmpose
PY_MM   := $(VENV_MM)/bin/python
RUN_MM  := PYTHONPATH=src $(PY_MM)

# Parâmetros sobrescrevíveis: make train-a EPOCHS=50 BATCH=16
EPOCHS  ?=
BATCH   ?=
CENARIO ?= A
CKPT    ?= results/checkpoints/cenario_$(CENARIO)/best_PCK.pth
CONFIG  ?= configs/cenario_$(shell echo $(CENARIO) | tr A-Z a-z).py

.PHONY: setup clean-data help \
        finetuning-env finetuning-checkpoint finetuning-smoke \
        train-a train-b train-c train-d evaluate \
        docker-build docker-train \
        docker-eval-detector docker-eval-cascade docker-detectors-table \
        docker-pipeline-finetuned docker-pose-3dsp \
        pose-all-brazil docker-pose-all-brazil

## Descompacta todos os .zip para data/
setup: $(MARKERS)
	@echo "\033[0;32m[OK]\033[0m Dados disponíveis em $(DATA_DIR)/"

$(DATA_DIR)/.extracted_%: %.zip
	@mkdir -p $(DATA_DIR)
	@echo "\033[0;32m[INFO]\033[0m Extraindo $< para $(DATA_DIR)/..."
	@tmpdir=$$(mktemp -d) && \
	 unzip -q -o $< -d $$tmpdir && \
	 toplevel=$$(ls $$tmpdir | head -1) && \
	 cp -rn $$tmpdir/$$toplevel/. $(DATA_DIR)/ && \
	 rm -rf $$tmpdir
	@date -Iseconds > $@

## Remove a pasta data/ para re-extrair do zero
clean-data:
	@rm -rf $(DATA_DIR)
	@echo "\033[0;32m[OK]\033[0m Pasta $(DATA_DIR)/ removida. Execute 'make setup' para re-extrair."

# ─── Fine-tuning RTMPose-X (Épico 1) ────────────────────────────────────────
# Detalhes: docs/finetuning.md

## Cria a venv dedicada .venv-mmpose com a stack MMPose pinada
finetuning-env:
	@bash scripts/setup/setup_mmpose_env.sh

## Baixa os pesos COCO p/ Transfer Learning (Cenários C/D)
finetuning-checkpoint:
	@bash scripts/setup/download_models.sh

## Smoke test ponta-a-ponta (subconjunto pequeno, ~30s)
finetuning-smoke:
	@$(RUN_MM) scripts/training/smoke_cenario_a.py --batch-size 8 --n-train 256 --n-val 64

## Treina um cenário: make train-a [EPOCHS=50] [BATCH=16]
train-a:
	@$(RUN_MM) scripts/training/train.py --cenario A $(if $(EPOCHS),--epochs $(EPOCHS),) $(if $(BATCH),--batch-size $(BATCH),)
train-b:
	@$(RUN_MM) scripts/training/train.py --cenario B $(if $(EPOCHS),--epochs $(EPOCHS),) $(if $(BATCH),--batch-size $(BATCH),)
train-c:
	@$(RUN_MM) scripts/training/train.py --cenario C $(if $(BATCH),--batch-size $(BATCH),)
train-d:
	@$(RUN_MM) scripts/training/train.py --cenario D $(if $(BATCH),--batch-size $(BATCH),)

## Avalia um checkpoint: make evaluate CKPT=... CONFIG=...
evaluate:
	@$(RUN_MM) scripts/evaluation/evaluate.py --checkpoint $(CKPT) --config $(CONFIG) --split val

IMAGE ?= football-finetuning
TARBALL ?= finetuning-image.tar
DATA_HOST ?= $(CURDIR)/data
RESULTS_HOST ?= $(CURDIR)/results

## Constrói a imagem Docker de fine-tuning (tag football-finetuning:latest)
docker-build:
	@docker build -f Dockerfile.finetuning -t $(IMAGE):latest .

## Constrói + salva a imagem num tarball (p/ transferir a um host sem internet)
docker-save: docker-build
	@echo "Salvando $(IMAGE):latest em $(TARBALL).gz..."
	@docker save $(IMAGE):latest | gzip > $(TARBALL).gz
	@echo "\033[0;32m[OK]\033[0m $(TARBALL).gz pronto."
	@echo "  Transfira:  scp $(TARBALL).gz aluno@HOST:~/"
	@echo "  No host:    gunzip -c $(TARBALL).gz | docker load"

## Roda um cenário no container com GPU (código baked; monta só data/results)
## Ex.: make docker-train CENARIO=C   |   no 4090 funciona via --gpus all (CDI)
## --shm-size=16g: o /dev/shm padrão do Docker (64MB) estoura com os DataLoader workers
docker-train:
	@docker run --rm --gpus all --shm-size=16g \
		-v $(DATA_HOST):/workspace/data:ro \
		-v $(RESULTS_HOST):/workspace/results \
		$(IMAGE):latest python scripts/training/train.py --cenario $(CENARIO)

## Benchmark de UM detector no container (GPU) — Épico #113.
## Ex.: make docker-eval-detector DET=yolo26   (DET=faster-rcnn|retinanet|yolo26)
## YOLO maior (mais justo vs ResNet50): make docker-eval-detector DET=yolo26 WEIGHTS=yolo26x.pt
## Salva results/tables/detector_<DET>.json + cache de predições.
docker-eval-detector:
	@docker run --rm --gpus all \
		-v $(DATA_HOST):/workspace/data:ro \
		-v $(RESULTS_HOST):/workspace/results \
		-v $(CURDIR)/src:/workspace/src:ro \
		-v $(CURDIR)/scripts:/workspace/scripts:ro \
		$(IMAGE):latest python scripts/evaluation/eval_detectors.py \
			--detector $(DET) --device cuda --save-predictions --viz 3 \
			$(if $(WEIGHTS),--weights $(WEIGHTS))

## Cascade R-CNN no container (GPU): baixa config+checkpoint do mmdet e avalia (#113).
## O 4º detector (two-stage) — fecha os "2 one-stage + 2 two-stage". Precisa da imagem buildada.
docker-eval-cascade:
	@docker run --rm --gpus all \
		-v $(DATA_HOST):/workspace/data:ro \
		-v $(RESULTS_HOST):/workspace/results \
		-v $(CURDIR)/src:/workspace/src:ro \
		-v $(CURDIR)/scripts:/workspace/scripts:ro \
		$(IMAGE):latest bash scripts/evaluation/eval_cascade.sh

## Pipeline ponta-a-ponta nos examples com a pose FINE-TUNADA do Épico 2 (GPU) — Épico #126.
## Roda na imagem do finetuning (tem MMPose). O checkpoint vive em results/ (montado); o config
## MMPose é inferido do path do checkpoint (cenario_D -> configs/cenario_d.py).
## Ex.: make docker-pipeline-finetuned \
##        CKPT=results/runs/20260608_014649_bd/checkpoints/cenario_D/best_PCK.pth
## Showcase em results/pipeline/ (não reescreve data/crops).
docker-pipeline-finetuned:
	@docker run --rm --gpus all \
		-v $(DATA_HOST):/workspace/data:ro \
		-v $(RESULTS_HOST):/workspace/results \
		-v $(CURDIR)/src:/workspace/src:ro \
		-v $(CURDIR)/scripts:/workspace/scripts:ro \
		$(IMAGE):latest python scripts/pipeline/demo_examples.py \
			--pose finetuned --checkpoint $(CKPT) --device cuda --no-crops

## Pose fine-tunada nos crops FROUXOS do 3DSP test (GPU) — fidelidade máxima (Épico #126).
## Os crops do dataset são a própria distribuição de treino do modelo → pose coerente (só inferência).
## Ex.: make docker-pose-3dsp CKPT=results/runs/20260608_014649_bd/checkpoints/cenario_D-OCCL/best_PCK.pth
docker-pose-3dsp:
	@docker run --rm --gpus all \
		-v $(DATA_HOST):/workspace/data:ro \
		-v $(RESULTS_HOST):/workspace/results \
		-v $(CURDIR)/src:/workspace/src:ro \
		-v $(CURDIR)/scripts:/workspace/scripts:ro \
		$(IMAGE):latest python scripts/pipeline/pose_on_crops.py \
			--crops-root data/3dsp/test --clips 00001 00004 00006 \
			--pose finetuned --checkpoint $(CKPT) --device cuda

## Showcase "pose em TODOS os jogadores" no Brasil (replica o baseline Reis) — Épico #126.
## Roda LOCAL (zero-shot rtmpose, sem Docker): YOLO26x detecta todos → pose em cada um → esqueletos no
## frame. Saída: results/pose_all/<clip>/frame_NNN.png (gitignored; nada de crop por jogador salvo).
## Ex.: make pose-all-brazil   |   outra fonte: make pose-all-brazil ROOT=data/clips/examples
ROOT ?= data/clips/brazil
pose-all-brazil:
	@PYTHONPATH=src .venv/bin/python scripts/pipeline/pose_all_players.py \
		--data-root $(ROOT) --pose rtmpose --device cuda

## Mesmo showcase, mas com a pose FINE-TUNADA (GPU/Docker, tem MMPose) — Épico #126.
## Ex.: make docker-pose-all-brazil CKPT=results/runs/.../cenario_D-OCCL/best_PCK.pth
docker-pose-all-brazil:
	@docker run --rm --gpus all \
		-v $(DATA_HOST):/workspace/data:ro \
		-v $(RESULTS_HOST):/workspace/results \
		-v $(CURDIR)/src:/workspace/src:ro \
		-v $(CURDIR)/scripts:/workspace/scripts:ro \
		$(IMAGE):latest python scripts/pipeline/pose_all_players.py \
			--data-root data/clips/brazil --pose finetuned --checkpoint $(CKPT) --device cuda

## Gera a tabela comparativa dos detectores (markdown + LaTeX) a partir dos JSONs.
docker-detectors-table:
	@docker run --rm \
		-v $(RESULTS_HOST):/workspace/results \
		-v $(CURDIR)/src:/workspace/src:ro \
		-v $(CURDIR)/scripts:/workspace/scripts:ro \
		$(IMAGE):latest python scripts/evaluation/detectors_table.py

## Mostra os comandos disponíveis
help:
	@echo ""
	@echo "  Comandos disponíveis:"
	@echo "  ─────────────────────────────────"
	@echo "  Dados:"
	@echo "    make setup                → Descompacta os .zip para data/"
	@echo "    make clean-data           → Remove a pasta data/"
	@echo ""
	@echo "  Fine-tuning (Épico 1 — ver docs/finetuning.md):"
	@echo "    make finetuning-env       → Cria a venv .venv-mmpose (stack MMPose)"
	@echo "    make finetuning-checkpoint→ Baixa pesos COCO (TL, Cenários C/D)"
	@echo "    make finetuning-smoke     → Smoke test ponta-a-ponta (~30s)"
	@echo "    make train-a|b|c|d        → Treina um cenário [EPOCHS=.. BATCH=..]"
	@echo "    make evaluate CKPT=.. CONFIG=..  → Avalia um checkpoint no val"
	@echo "    make docker-build         → Constrói a imagem de fine-tuning"
	@echo "    make docker-train CENARIO=C → Roda um cenário no container (GPU)"
	@echo ""
	@echo "  Detecção (Épico 113 — benchmark dos detectores):"
	@echo "    make docker-eval-detector DET=yolo26 → Avalia um detector (GPU)"
	@echo "    make docker-detectors-table         → Tabela comparativa (md+LaTeX)"
	@echo ""
	@echo "  Pipeline (Épico 126 — showcase):"
	@echo "    make pose-all-brazil       → Pose em TODOS os jogadores no Brasil (local, zero-shot)"
	@echo "    make docker-pose-all-brazil CKPT=.. → Idem com pose fine-tunada (Docker)"
	@echo ""
