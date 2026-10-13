# Deep Learning Models from First Principles

A single PyTorch training framework that drives four architectures — a CNN, a
plain RNN, an LSTM and a Transformer — behind one interface. Every layer is
implemented directly from tensor operations rather than PyTorch's built-in layer
modules, and each one is checked numerically against its `torch.nn` counterpart
in the test suite. The surrounding framework (configuration, data loading,
training loop, checkpointing and resuming, logging, reproducibility) is written
once and shared by all four models: the trainer never asks which model it is
training.

## What is implemented here vs. what PyTorch provides

**Implemented directly from tensor operations** (`src/dlfp/layers/`):

| Layer | Implementation |
|---|---|
| `Linear` | `x @ W.T + b` with PyTorch-matching Kaiming-uniform init |
| `Conv2d` | im2col via `F.unfold`, then one matmul |
| `MaxPool2d`, `GlobalAvgPool2d` | `F.unfold` + `max`; spatial `mean` |
| `BatchNorm2d` | batch statistics in training, running statistics in eval |
| `LayerNorm` | per-sample normalisation over trailing dims |
| `Embedding` | `W[ids]`, with a gradient hook that keeps the pad row at zero |
| `RNNCell`, `LSTMCell`, `unroll` | single-step cells and a mask-aware time loop |
| `scaled_dot_product_attention`, `MultiHeadAttention` | written out, causal + key-padding masks |
| `SinusoidalPositionalEncoding`, `LearnedPositionalEmbedding` | position tables |
| `SGD`, `Adam` | explicit update rules on top of `torch.optim.Optimizer` |

**Provided by PyTorch:** tensors and autograd (no manual backward passes),
`nn.Module`/`nn.Parameter` bookkeeping, `nn.Dropout`, activation functions,
`F.unfold`, `F.softmax`, `F.cross_entropy`, `Dataset`/`DataLoader`, CUDA
kernels, and `torchvision` for downloading MNIST/CIFAR-10.

## Installation

The environment already exists and only needs activating:

```bash
cd ~/ML/DL/dl-from-first-principles
source dlfp_venv/bin/activate
```

To recreate it from nothing, if ever needed:

```bash
uv venv dlfp_venv
source dlfp_venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
```

The editable install (`-e .`) registers the `src/dlfp` package with the
environment, which is what makes `import dlfp` work from any directory without
hacks that edit Python's module search path; because it is editable, changes to the source take effect
immediately with no reinstall. Do not use `uv run` — it manages a separate
`.venv`. If you need a specific CUDA build of torch, install it first with the
matching `--index-url` from <https://pytorch.org/get-started/locally/>.

## Quickstart

```bash
python scripts/train.py --config configs/cnn_mnist.yaml           # CNN, MNIST (>99% val acc in 5 epochs)
python scripts/train.py --config configs/cnn_cifar10.yaml         # CNN, CIFAR-10, cosine schedule
python scripts/train.py --config configs/rnn_charlm.yaml          # plain RNN, Tiny Shakespeare
python scripts/train.py --config configs/lstm_charlm.yaml         # LSTM, Tiny Shakespeare
python scripts/train.py --config configs/lstm_imdb.yaml           # LSTM sentiment classifier, IMDB
python scripts/train.py --config configs/transformer_charlm.yaml  # Transformer, Tiny Shakespeare

# Sample from a language model, evaluate any checkpoint
python scripts/generate.py --checkpoint runs/<run>/ckpt/best.pt --prompt "ROMEO:" \
    --max-new-tokens 300 --temperature 0.8 --top-k 40
python scripts/evaluate.py --checkpoint runs/<run>/ckpt/best.pt --split test
```

Datasets download into `./data` (relative to where you run the script) on first
use. On the shared server, choose a GPU with `CUDA_VISIBLE_DEVICES=<idx>`; the
code never hardcodes one.

## Configuration

A run's configuration is merged from three layers, each overriding the last:

1. `configs/base.yaml` — shared defaults (seed, device, optimiser defaults…)
2. `configs/<experiment>.yaml` — the model, dataset and anything it changes
3. command-line overrides — `--section.key value` or `--section.key=value`

Nested sections merge key by key, so an experiment that sets only `train.lr`
keeps every other `train` default. Values are parsed as YAML (`true`, `null`,
`[32, 64]`, `3e-4` all get their natural types). Files in `configs/` are never
written to.

Worked example:

```bash
python scripts/train.py --config configs/transformer_charlm.yaml \
    --train.lr 1e-4 --train.epochs=3 --model.params.n_layers 2 --data.batch_size 32 --device cuda
```

`train.lr` goes from `1.0e-3` (base) → `3.0e-4` (experiment) → `1e-4` (CLI);
`train.epochs` from `10` → `3`; `train.grad_clip` stays at the base value `1.0`.
Unknown top-level keys, `monitor_mode` other than `min`/`max`, and `val_split`
outside `[0, 1)` are rejected. Dataset metadata (`vocab_size`, `num_classes`,
`pad_idx`, `in_channels`) is injected into `model.params` unless set explicitly.

## Run directory

```
runs/<run_name>_<YYYY-MM-DD_HH-MM-SS>/
├── config.yaml      fully resolved config (incl. injected dataset metadata)
├── log.txt          everything printed during the run
├── metrics.csv      one row per (epoch, split)
└── ckpt/
    ├── best.pt      best checkpoint by train.monitor (default val_loss)
    ├── last.pt      end of the most recent epoch
    └── tokenizer.json   text datasets only
```

## Resuming

```bash
python scripts/train.py --config configs/cnn_mnist.yaml --train.epochs 10 \
    --resume runs/<run>/ckpt/last.pt
```

A checkpoint stores model, optimizer and scheduler state, the epoch and global
step, the best monitored value, early-stopping state, the resolved config and
the state of every RNG (Python, NumPy, torch CPU, all CUDA devices). Training
continues at the epoch after the one saved, and the shuffle order of each epoch
is seeded from `(seed, epoch)`, so a resumed run reproduces an uninterrupted one
(the test suite checks this to `1e-6`). The resumed run writes to a new run
directory.

## Project structure

```
configs/           base defaults + one YAML per experiment
src/dlfp/          the package (installed editable)
  layers/          Linear, Conv2d, norms, Embedding, recurrent cells, attention, positional encodings
  models/          BaseModel interface, CNN, RNN/LSTM (shared SequenceModel), Transformer
  data/            DatasetBundle, collate functions, tokenizers, MNIST/CIFAR-10, Tiny Shakespeare, IMDB
  training/        Trainer, optimizers, LR schedulers, checkpointing, callbacks
  utils/           seeding, device handling, run logging
  registry.py      name -> class registries for models, datasets, optimizers
  config.py        dataclass config, three-layer merge, CLI overrides
  losses.py        masked sequence cross-entropy
  metrics.py       accuracy, macro-F1, perplexity, running-mean tracker
scripts/           train.py, evaluate.py, generate.py
tests/             unit, equivalence and integration tests (CPU, tiny tensors)
runs/              training outputs (gitignored)
```

## Results

Fill in after full training runs.

| Config | Model | Dataset | Epochs | Best val metric | Test metric | Notes |
|---|---|---|---|---|---|---|
| `cnn_mnist.yaml` | CNN | MNIST | 5 | acc: | acc: | |
| `cnn_cifar10.yaml` | CNN | CIFAR-10 | 20 | acc: | acc: | |
| `rnn_charlm.yaml` | RNN | Tiny Shakespeare | 10 | ppl: | ppl: | |
| `lstm_charlm.yaml` | LSTM | Tiny Shakespeare | 10 | ppl: | ppl: | |
| `transformer_charlm.yaml` | Transformer | Tiny Shakespeare | 10 | ppl: | ppl: | |
| `lstm_imdb.yaml` | LSTM | IMDB | 5 | acc: | acc: | |

## Testing

```bash
source dlfp_venv/bin/activate
CUDA_VISIBLE_DEVICES="" pytest -q
```

The suite checks every layer against its `torch.nn` equivalent, mask handling in
recurrent unrolling and attention, that every model can overfit a tiny batch and
that padding never changes a loss, config merging, checkpoint round-trips,
exact resume, early stopping, and that the hand-written optimizers match
`torch.optim` to `1e-6`. It runs on CPU in well under a minute.

## Troubleshooting

- **Dataset download fails (proxy / no internet).** Set `HTTPS_PROXY`, or
  place the files manually; the error message names the exact path. For the
  defaults: Tiny Shakespeare at `data/tinyshakespeare/input.txt` (or point
  `--data.params.corpus_path` at any text file); IMDB as
  `data/aclImdb_v1.tar.gz` or extracted to `data/aclImdb/`; MNIST/CIFAR-10 in
  torchvision's layout under `data/`.
- **`CUDA out of memory`.** Lower `--data.batch_size`, or run on CPU with
  `--device cpu`. On the shared server, pick a free GPU with
  `CUDA_VISIBLE_DEVICES`.
- **`ModuleNotFoundError: No module named 'dlfp'`.** The editable install was
  skipped or a different environment is active: `source dlfp_venv/bin/activate
  && uv pip install -e .`.
