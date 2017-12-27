PNGS := $(wilecard *.png)
PNMS := $(wilecard *.pnm)

.DEFAULT_GOAL := help

all: generate clean ## generate and clean.

generate: pnm svg ## generate font from "png" dependent on "./generate.pe".
	fontforge -script generate.pe default 0.0.1

svg: $(PNMS) ## convert "pnm" to "svg".
	mkdir svg
	potrace *.pnm -s
	mv *.svg svg/

pnm: $(PNGS) ## convert "png" to "pnm".
	mogrify -format pnm *.png

.PHONY: clean
clean: ## cleaning.
	$(RM) *.pnm
	$(RM) -r svg/

.PHONY: help
help: ## print this help.
	@echo "[COMMAND]"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[35m%-15s\033[0m %s\n", $$1, $$2}'
	@echo "\n[Requires]"
	@echo "* imagemagick"
	@echo "* potrace"
	@echo "* fontforge"
