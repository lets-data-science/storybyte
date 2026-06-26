# StoryByte 📖🤖 — a tiny LLM you can actually understand

**StoryByte is a ~1-million-parameter GPT, trained from scratch on [TinyStories](https://arxiv.org/abs/2305.07759), that writes original little children's stories.** It's small enough that its *entire* forward pass runs in ~40 lines of NumPy in a web browser — yet it produces real, grammatical English.

It's the model behind the interactive course **"Build a Tiny LLM — From Tokens to Text"**, where beginners build every piece of it from scratch (tokenizer → embeddings → attention → transformer block → sampler) and watch *this* model generate a story. This repo is the **offline half**: everything used to train StoryByte and produce the course's model files, fully reproducible.

> **Honesty first:** StoryByte writes simple toddler-vocabulary stories and *nothing else*. It has no world knowledge, can't do math, and isn't a chatbot. That narrowness is the whole point — it's the smallest thing that still speaks fluent English, which makes it perfect for learning how an LLM actually works.

---

## What's in here

```
storybyte/
├── model/storybyte.py          # the GPT (PyTorch) — classic GPT-2/nanoGPT block
├── scripts/
│   ├── 01_download_data.py      # get TinyStories (V2, GPT-4)
│   ├── 02_train_tokenizer.py    # byte-level BPE (vocab 2048)
│   ├── 03_prepare_data.py       # tokenize + pack to uint16 streams
│   ├── 04_train.py              # the training loop (AdamW, warmup+cosine, grad-clip)
│   ├── 05_export_artifacts.py   # export + VERIFY the NumPy reference matches PyTorch
│   ├── reference_forward.py     # StoryByte's forward pass + generation in PURE NUMPY
│   └── 06_evaluate.py           # sample stories + interpretability data
├── course_artifacts/           # the files the in-browser course loads (see below)
├── checkpoints/                # trained weights (or a GitHub Release)
├── BUILD_LOG.md                # exhaustive lab notebook of the whole run
├── HARDWARE.md                 # what you need to train it (GPU / Mac / Colab / CPU)
└── REPRODUCIBILITY.md          # seeds, versions, expected numbers
```

## Quickstart — reproduce StoryByte yourself

```bash
pip install -r requirements.txt
make all          # download → tokenizer → data → train → export → evaluate
# or step by step:
python scripts/01_download_data.py            # full data (use --subset_mb 400 for a fast run)
python scripts/02_train_tokenizer.py
python scripts/03_prepare_data.py
python scripts/04_train.py                    # trains on Apple MPS / CUDA / CPU
python scripts/05_export_artifacts.py         # writes course_artifacts/ + verifies the NumPy port
python scripts/06_evaluate.py
```

Run StoryByte with **zero deep-learning frameworks** (pure NumPy):

```bash
python scripts/reference_forward.py "Once upon a time"
```

## The model (deliberately classic)

StoryByte is the textbook GPT-2 / [nanoGPT](https://github.com/karpathy/nanoGPT) decoder block — *no* modern variants — because the course rebuilds exactly this design from scratch:

- decoder-only, causal (masked) self-attention
- learned absolute positional embeddings (added to token embeddings)
- **pre-LayerNorm** blocks: `x = x + attn(LN(x)); x = x + mlp(LN(x))`
- standard multi-head scaled-dot-product attention
- MLP = Linear → **GELU** → Linear (4× width)
- weight-tied LM head; classic LayerNorm (weight + bias)

| Config | Value |
|---|---|
| Layers / Heads / d_model | 4 / 4 / 128 |
| Context | 256 tokens |
| Vocab (byte-level BPE) | 2,048 |
| Parameters | ~1.1M |
| Trained on | TinyStories V2 (GPT-4), 113.5M tokens |
| Recipe | AdamW (0.9, 0.95), wd 0.1, lr 6e-4→6e-5, 1k warmup + cosine, grad-clip 1.0 |

## Results

- **1,088,256 parameters.** Trained 30,000 steps in **~42 min on Apple MPS**.
- **Final val loss 1.74 · perplexity 5.70** (random would be 2,048).
- The pure-NumPy `reference_forward.py` reproduces the PyTorch model **exactly**:
  max logit diff **1.7e-05**, greedy-token agreement **100.0%** (`course_artifacts/verification.json`).
- Sample (greedy) from **"Once upon a time"**:
  > *…there was a little girl named Lily. She loved to play with her toys and have fun. One day, she found a big box of toys in her room… Lily's mom saw her and said, "Lily, you need to clean your room."…*

(Full curves in `course_artifacts/train_traces.json`; more samples in `sample_generations.json`; the whole story in `BUILD_LOG.md`.)

## `course_artifacts/` — the contract the course consumes

| File | What |
|---|---|
| `storybyte_config.json` | architecture + final metrics |
| `storybyte_weights.npz` | all weights as named arrays (float16), GPT-2 naming |
| `reference_forward.py` | the verified pure-NumPy forward pass + generation |
| `storybyte_tokenizer.json` | byte-level BPE vocab + merges (+ HF tokenizer for exact encode) |
| `train_traces.json` | loss / LR / perplexity curves (for the "How It Learned" module) |
| `interp_data.json` | logit-lens + attention patterns (for the "Opening the Box" module) |
| `sample_generations.json` | reference outputs at several temperatures |
| `verification.json` | proof the NumPy port == the trained model |
| `MANIFEST.md` | every file, shape, the matmul convention, canonical numbers |

## Credits & sources

StoryByte stands on: **TinyStories** (Eldan & Li, 2023, [arXiv:2305.07759](https://arxiv.org/abs/2305.07759)) · **nanoGPT** (Karpathy, [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)) · **GPT-2** (Radford et al., 2019) · **Attention Is All You Need** (Vaswani et al., 2017, [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)) · **AdamW** ([arXiv:1711.05101](https://arxiv.org/abs/1711.05101)) · **BPE** (Sennrich et al., 2016, [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)) · **picoGPT** (Jay Mody) for the pure-NumPy inference style.

MIT licensed. Built by [Let's Data Science](https://letsdatascience.com).
