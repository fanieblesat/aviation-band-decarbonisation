.PHONY: all model test paper clean
all: model test paper
model:
	python scripts/run_all.py
test:
	python -m pytest -q
paper:
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex
clean:
	cd paper && latexmk -C
