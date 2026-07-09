# StoryByte

StoryByte is a small GPT-style language model trained from scratch on
[TinyStories](https://arxiv.org/abs/2305.07759). It has about 1.09 million
parameters and writes short children's stories.

This repo is the offline build system for the Let's Data Science course
["Build a Tiny LLM: From Tokens to Text"](https://letsdatascience.com/learn/build-a-tiny-llm).
The browser course rebuilds the forward pass in NumPy and runs the trained model
live. This repository contains the training code, export scripts, and the artifact
contract the course loads.

StoryByte is intentionally narrow. It does not answer general questions, do math,
or behave like a chatbot. That is the point: a tiny model can produce readable
language when the world is restricted enough, and that makes every part of the LLM
stack easier to inspect.

## Repository layout

```text
storybyte/
  model/storybyte.py            PyTorch GPT model
  scripts/
    01_download_data.py         Download TinyStories
    02_train_tokenizer.py       Train byte-level BPE tokenizer
    03_prepare_data.py          Tokenize and pack uint16 streams
    04_train.py                 Train with AdamW and cosine LR
    05_export_artifacts.py      Export and verify course artifacts
    06_evaluate.py              Generate samples and interpretability data
    reference_forward.py        Pure NumPy forward pass and generation
  course_artifacts/             Files consumed by the browser course
  checkpoints/                  Trained checkpoint and training traces
  BUILD_LOG.md                  Build notes and canonical measurements
  HARDWARE.md                   Training hardware notes
  REPRODUCIBILITY.md            Seeds, versions, and expected results
```

## Quickstart

```bash
pip install -r requirements.txt
make all
```

Step by step:

```bash
python scripts/01_download_data.py --subset_mb 400
python scripts/02_train_tokenizer.py
python scripts/03_prepare_data.py
python scripts/04_train.py
python scripts/05_export_artifacts.py
python scripts/06_evaluate.py
```

Run the exported model with NumPy only:

```bash
python scripts/reference_forward.py "Once upon a time"
```

## Model

StoryByte uses the classic GPT-2/nanoGPT decoder block because the course rebuilds
that architecture directly:

- decoder-only causal self-attention
- learned absolute positional embeddings
- pre-LayerNorm transformer blocks
- standard multi-head scaled-dot-product attention
- GELU MLP with 4x hidden width
- weight-tied language-model head

| Config | Value |
|---|---|
| Layers / heads / d_model | 4 / 4 / 128 |
| Context | 256 tokens |
| Vocab | 2,048 byte-level BPE tokens |
| Parameters | 1,088,256 |
| Training data | TinyStories V2, 113.5M tokens in the reference run |
| Recipe | AdamW, beta=(0.9, 0.95), wd 0.1, lr 6e-4 to 6e-5, 1k warmup, cosine decay, grad clip 1.0 |

## Results

The reference run trained for 30,000 steps in about 42 minutes on Apple MPS.

| Metric | Value |
|---|---:|
| Final train loss | 1.7206 |
| Final val loss | 1.7398 |
| Val perplexity | 5.70 |
| NumPy vs PyTorch max logit diff | 1.7e-05 |
| Greedy-token agreement | 100.0% |

The shipped NumPy reference in `scripts/reference_forward.py` reproduces the PyTorch
checkpoint within float32 tolerance. See `course_artifacts/verification.json` for the
exact verification record.

## Course artifacts

The browser course loads files from `course_artifacts/`.

| File | Purpose |
|---|---|
| `storybyte_config.json` | architecture and final metrics |
| `storybyte_weights.npz` | all model weights as named float32 arrays |
| `reference_forward.py` | verified pure-NumPy forward pass |
| `storybyte_tokenizer.json` | simplified byte-level BPE view |
| `storybyte_tokenizer_hf.json` | authoritative Hugging Face tokenizer JSON |
| `train_traces.json` | loss, LR, and perplexity curves |
| `interp_data.json` | logit-lens and attention-pattern data |
| `sample_generations.json` | recorded generation examples |
| `verification.json` | NumPy/PyTorch parity proof |
| `MANIFEST.md` | artifact schema and canonical numbers |

## Sources

- TinyStories: Eldan and Li, 2023, [arXiv:2305.07759](https://arxiv.org/abs/2305.07759)
- nanoGPT: Andrej Karpathy, [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- GPT-2: Radford et al., 2019
- Attention Is All You Need: Vaswani et al., 2017, [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- AdamW: Loshchilov and Hutter, 2017, [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
- BPE: Sennrich et al., 2016, [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)
- picoGPT: Jay Mody, pure-NumPy GPT reference style

MIT licensed. Built by [Let's Data Science](https://letsdatascience.com).
