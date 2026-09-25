DATA_DIR := data
ZIP_FILES := $(wildcard *.zip)
MARKERS := $(patsubst %.zip,$(DATA_DIR)/.extracted_%,$(ZIP_FILES))

VENV_MM := .venv-mmpose
PY_MM   := $(VENV_MM)/bin/python
RUN_MM  := PYTHONPATH=src $(PY_MM)

# Overridable parameters, e.g. make train-a EPOCHS=50 BATCH=16
EPOCHS  ?=
BATCH   ?=
CENARIO ?= A
CKPT    ?= results/checkpoints/cenario_$(CENARIO)/best_PCK.pth
CONFIG  ?= configs/cenario_$(shell echo $(CENARIO) | tr A-Z a-z).py
DEVICE  ?= cuda
INPUT   ?= data/examples/test_00001.mp4

.PHONY: setup clean-data help demo test lint \
        finetuning-env finetuning-checkpoint finetuning-smoke \
        train-a train-b train-c train-d evaluate \
        docker-build docker-save docker-train \
        docker-eval-detector docker-eval-cascade docker-detectors-table \
        docker-pipeline-finetuned \
        pose-all-brazil docker-pose-all-brazil gifs

## Extract every .zip in the project root (e.g. 3dsp.zip) into data/
setup: $(MARKERS)
	@echo "\033[0;32m[OK]\033[0m Data available in $(DATA_DIR)/"

$(DATA_DIR)/.extracted_%: %.zip
	@mkdir -p $(DATA_DIR)
	@echo "\033[0;32m[INFO]\033[0m Extracting $< into $(DATA_DIR)/..."
	@tmpdir=$$(mktemp -d) && \
	 unzip -q -o $< -d $$tmpdir && \
	 toplevel=$$(ls $$tmpdir | head -1) && \
	 cp -rn $$tmpdir/$$toplevel/. $(DATA_DIR)/ && \
	 rm -rf $$tmpdir
	@date -Iseconds > $@

## Delete data/ so that `make setup` extracts it again from scratch
clean-data:
	@rm -rf $(DATA_DIR)
	@echo "\033[0;32m[OK]\033[0m $(DATA_DIR)/ removed. Run 'make setup' to extract it again."

# ─── Quick start and quality ────────────────────────────────────────────────

## Full pipeline on a video or image: make demo [INPUT=...]
demo:
	@uv run python demo.py --input $(INPUT)

test:
	@uv run --extra dev pytest

lint:
	@uv run --extra dev ruff check .

# ─── RTMPose-X fine-tuning ──────────────────────────────────────────────────

## Create .venv-mmpose with the pinned MMPose stack
finetuning-env:
	@bash scripts/setup/setup_mmpose_env.sh

## Download the COCO weights used as the transfer-learning start (settings C/D)
finetuning-checkpoint:
	@bash scripts/setup/download_models.sh

## End-to-end smoke test on a small subset (~30 s)
finetuning-smoke:
	@$(RUN_MM) scripts/training/smoke_cenario_a.py --batch-size 8 --n-train 256 --n-val 64

## Train one setting of the 2x2 matrix: make train-a [EPOCHS=50] [BATCH=16]
train-a:
	@$(RUN_MM) scripts/training/train.py --cenario A $(if $(EPOCHS),--epochs $(EPOCHS),) $(if $(BATCH),--batch-size $(BATCH),)
train-b:
	@$(RUN_MM) scripts/training/train.py --cenario B $(if $(EPOCHS),--epochs $(EPOCHS),) $(if $(BATCH),--batch-size $(BATCH),)
train-c:
	@$(RUN_MM) scripts/training/train.py --cenario C $(if $(BATCH),--batch-size $(BATCH),)
train-d:
	@$(RUN_MM) scripts/training/train.py --cenario D $(if $(BATCH),--batch-size $(BATCH),)

## Evaluate a checkpoint on the validation split: make evaluate CENARIO=D (or CKPT=... CONFIG=...)
evaluate:
	@$(RUN_MM) scripts/evaluation/evaluate.py --checkpoint $(CKPT) --config $(CONFIG) --split val

IMAGE ?= football-finetuning
TARBALL ?= finetuning-image.tar
DATA_HOST ?= $(CURDIR)/data
RESULTS_HOST ?= $(CURDIR)/results

## Build the fine-tuning Docker image (football-finetuning:latest)
docker-build:
	@docker build -f Dockerfile.finetuning -t $(IMAGE):latest .

## Build and save the image as a tarball, to move it to a host without internet access
docker-save: docker-build
	@echo "Saving $(IMAGE):latest to $(TARBALL).gz..."
	@docker save $(IMAGE):latest | gzip > $(TARBALL).gz
	@echo "\033[0;32m[OK]\033[0m $(TARBALL).gz ready."
	@echo "  Copy:     scp $(TARBALL).gz user@HOST:~/"
	@echo "  On host:  gunzip -c $(TARBALL).gz | docker load"

## Train one setting inside the GPU container (code baked in; mounts only data/ and results/)
## e.g. make docker-train CENARIO=D
## --shm-size=16g: Docker's default /dev/shm (64 MB) is too small for the DataLoader workers
docker-train:
	@docker run --rm --gpus all --shm-size=16g \
		-v $(DATA_HOST):/workspace/data:ro \
		-v $(RESULTS_HOST):/workspace/results \
		$(IMAGE):latest python scripts/training/train.py --cenario $(CENARIO)

## Benchmark one detector against the hand-annotated boxes (GPU container)
## e.g. make docker-eval-detector DET=yolo26 WEIGHTS=yolo26x.pt   (DET=yolo26|faster-rcnn|retinanet)
## Writes results/tables/detector_<DET>.json plus a prediction cache.
docker-eval-detector:
	@docker run --rm --gpus all \
		-v $(DATA_HOST):/workspace/data:ro \
		-v $(RESULTS_HOST):/workspace/results \
		-v $(CURDIR)/src:/workspace/src:ro \
		-v $(CURDIR)/scripts:/workspace/scripts:ro \
		$(IMAGE):latest python scripts/evaluation/eval_detectors.py \
			--detector $(DET) --device cuda --save-predictions --viz 3 \
			$(if $(WEIGHTS),--weights $(WEIGHTS))

## Cascade R-CNN benchmark (GPU container): downloads the MMDetection config and checkpoint first
docker-eval-cascade:
	@docker run --rm --gpus all \
		-v $(DATA_HOST):/workspace/data:ro \
		-v $(RESULTS_HOST):/workspace/results \
		-v $(CURDIR)/src:/workspace/src:ro \
		-v $(CURDIR)/scripts:/workspace/scripts:ro \
		$(IMAGE):latest bash scripts/evaluation/eval_cascade.sh

## Comparison table of the detectors (Markdown + LaTeX) from the JSON results
docker-detectors-table:
	@docker run --rm \
		-v $(RESULTS_HOST):/workspace/results \
		-v $(CURDIR)/src:/workspace/src:ro \
		-v $(CURDIR)/scripts:/workspace/scripts:ro \
		$(IMAGE):latest python scripts/evaluation/detectors_table.py

## Pipeline on the example clips with a fine-tuned pose model (GPU container)
## The MMPose config is inferred from the checkpoint path (cenario_D -> configs/cenario_d.py).
## e.g. make docker-pipeline-finetuned CKPT=results/checkpoints/cenario_D/best_PCK.pth
docker-pipeline-finetuned:
	@docker run --rm --gpus all \
		-v $(DATA_HOST):/workspace/data:ro \
		-v $(RESULTS_HOST):/workspace/results \
		-v $(CURDIR)/src:/workspace/src:ro \
		-v $(CURDIR)/scripts:/workspace/scripts:ro \
		$(IMAGE):latest python scripts/pipeline/demo_examples.py \
			--pose finetuned --checkpoint $(CKPT) --device cuda --no-crops

## Pose of every player on the broadcast clips (local, zero-shot RTMPose-X)
## Output: results/showcase/all_players/<clip>/frame_NNN.png
## e.g. make pose-all-brazil DEVICE=cpu   |   other clips: make pose-all-brazil ROOT=data/clips/examples
ROOT ?= data/clips/brazil
pose-all-brazil:
	@uv run python scripts/pipeline/pose_all_players.py \
		--data-root $(ROOT) --pose rtmpose --device $(DEVICE)

## Same, with a fine-tuned pose model (GPU container)
## e.g. make docker-pose-all-brazil CKPT=results/checkpoints/cenario_D/best_PCK.pth
docker-pose-all-brazil:
	@docker run --rm --gpus all \
		-v $(DATA_HOST):/workspace/data:ro \
		-v $(RESULTS_HOST):/workspace/results \
		-v $(CURDIR)/src:/workspace/src:ro \
		-v $(CURDIR)/scripts:/workspace/scripts:ro \
		$(IMAGE):latest python scripts/pipeline/pose_all_players.py \
			--data-root data/clips/brazil --pose finetuned --checkpoint $(CKPT) --device cuda

## Animated GIFs from the showcase PNGs (no GPU): results/showcase/gifs/
## e.g. make gifs   |   one clip: make gifs ARGS="--src all_players --clips brazil_01"
gifs:
	@uv run python scripts/pipeline/make_gifs.py $(ARGS)

help:
	@echo ""
	@echo "  Quick start"
	@echo "    make demo [INPUT=video.mp4]             full pipeline on a video or image"
	@echo "    make test | make lint                   unit tests (no GPU) | ruff"
	@echo ""
	@echo "  Data"
	@echo "    make setup                              extract 3dsp.zip (and any .zip) into data/"
	@echo "    make clean-data                         delete data/"
	@echo ""
	@echo "  Fine-tuning (RTMPose-X, 2x2 matrix)"
	@echo "    make finetuning-env                     create .venv-mmpose (pinned MMPose stack)"
	@echo "    make finetuning-checkpoint              download COCO weights (settings C/D)"
	@echo "    make finetuning-smoke                   end-to-end smoke test (~30 s)"
	@echo "    make train-a|b|c|d [EPOCHS=.. BATCH=..] train one setting"
	@echo "    make evaluate CENARIO=D                 evaluate a checkpoint on the validation split"
	@echo "    make docker-build                       build the fine-tuning image"
	@echo "    make docker-train CENARIO=D             train inside the GPU container"
	@echo ""
	@echo "  Detector benchmark"
	@echo "    make docker-eval-detector DET=yolo26    evaluate one detector (GPU container)"
	@echo "    make docker-eval-cascade                evaluate Cascade R-CNN (GPU container)"
	@echo "    make docker-detectors-table             comparison table (Markdown + LaTeX)"
	@echo ""
	@echo "  Qualitative results"
	@echo "    make pose-all-brazil [DEVICE=cpu]       pose of every player on the broadcast clips"
	@echo "    make docker-pose-all-brazil CKPT=..     same, with a fine-tuned model"
	@echo "    make gifs                               animated GIFs from the showcase frames"
	@echo ""
