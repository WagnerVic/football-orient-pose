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
        docker-build docker-train

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
	@bash scripts/setup_mmpose_env.sh

## Baixa os pesos COCO p/ Transfer Learning (Cenários C/D)
finetuning-checkpoint:
	@bash scripts/download_models.sh

## Smoke test ponta-a-ponta (subconjunto pequeno, ~30s)
finetuning-smoke:
	@$(RUN_MM) scripts/smoke_cenario_a.py --batch-size 8 --n-train 256 --n-val 64

## Treina um cenário: make train-a [EPOCHS=50] [BATCH=16]
train-a:
	@$(RUN_MM) scripts/train.py --cenario A $(if $(EPOCHS),--epochs $(EPOCHS),) $(if $(BATCH),--batch-size $(BATCH),)
train-b:
	@$(RUN_MM) scripts/train.py --cenario B $(if $(EPOCHS),--epochs $(EPOCHS),) $(if $(BATCH),--batch-size $(BATCH),)
train-c:
	@$(RUN_MM) scripts/train.py --cenario C $(if $(BATCH),--batch-size $(BATCH),)
train-d:
	@$(RUN_MM) scripts/train.py --cenario D $(if $(BATCH),--batch-size $(BATCH),)

## Avalia um checkpoint: make evaluate CKPT=... CONFIG=...
evaluate:
	@$(RUN_MM) scripts/evaluate.py --checkpoint $(CKPT) --config $(CONFIG) --split val

## Constrói a imagem Docker de fine-tuning
docker-build:
	@docker compose -f docker-compose.finetuning.yml build

## Roda um cenário no container (GPU): make docker-train CENARIO=C
docker-train:
	@docker compose -f docker-compose.finetuning.yml run --rm train \
		python scripts/train.py --cenario $(CENARIO)

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
