.PHONY: install train gradcheck sweep test all clean

install:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

train:
	.venv/bin/python train.py

gradcheck:
	.venv/bin/python -m src.gradcheck

sweep:
	.venv/bin/python -m experiments.hidden_size_sweep
	.venv/bin/python -m experiments.lr_sweep

curves:
	.venv/bin/python -m experiments.plot_curves

test:
	.venv/bin/python -m unittest discover tests

all: test gradcheck train

clean:
	rm -rf runs/ data/raw/ __pycache__ */__pycache__ */*/__pycache__
