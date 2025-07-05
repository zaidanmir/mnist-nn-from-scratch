# Neural Network from Scratch — MNIST

A two-layer multi-layer perceptron trained on MNIST, implemented end-to-end
in plain NumPy. Forward propagation, backpropagation, and mini-batch SGD are
written out explicitly; no autograd, no PyTorch, no TensorFlow.

The point of building it from first principles is to make the maths visible:
the cross-entropy gradient and chain-rule expansion are derived by hand in
[`notes/backprop.md`](notes/backprop.md), and the code matches the derivation
line for line.

## Status

Work in progress. See commits for granular progress.

## Planned components

- `src/data.py` — MNIST IDX loader, normalisation, mini-batch iterator
- `src/layers.py` — `Linear`, `ReLU`, and `Softmax` primitives
- `src/losses.py` — cross-entropy loss with numerically stable log-softmax
- `src/model.py` — 2-layer MLP composing the primitives
- `src/train.py` — mini-batch SGD training loop with per-epoch metrics
- `src/utils.py` — gradient checking and plotting helpers
- `notes/backprop.md` — hand-derived gradient for each layer
- `train.py` — entrypoint: load → fit → evaluate → save curves

## Architecture

```
input (784) -> Linear(784, 128) -> ReLU -> Linear(128, 10) -> softmax -> cross-entropy
```

## Getting started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.data       # downloads MNIST to data/raw/ on first run
python train.py
```

## Dataset

[MNIST](http://yann.lecun.com/exdb/mnist/) — 60,000 train / 10,000 test
greyscale digits, 28×28 pixels, 10 classes. Mirror used for downloads:
`https://ossci-datasets.s3.amazonaws.com/mnist/`.
