DATA_DIR := data
ZIP_FILES := $(wildcard *.zip)
MARKERS := $(patsubst %.zip,$(DATA_DIR)/.extracted_%,$(ZIP_FILES))

.PHONY: setup clean-data help

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

## Mostra os comandos disponíveis
help:
	@echo ""
	@echo "  Comandos disponíveis:"
	@echo "  ─────────────────────────────────"
	@echo "  make setup       → Descompacta os .zip para data/"
	@echo "  make clean-data  → Remove a pasta data/"
	@echo "  make help        → Mostra esta ajuda"
	@echo ""
