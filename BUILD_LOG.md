# StoryByte — Exhaustive Build Log

> A complete, honest record of how StoryByte (the tiny LLM behind the course
> **"Build a Tiny LLM — From Tokens to Text"**) was built: every decision, number,
> command, hyperparameter, gotcha, and result. Written so a learner can understand
> *exactly* what we did and reproduce it — and so Module 6 ("How It Learned") can tell
> the true story. Append-only; newest steps at the bottom.

---

## 0. What StoryByte is (and isn't)

- A **~1M-parameter, GPT-2-style decoder-only language model** trained from scratch on the **TinyStories** dataset.
- It writes **simple, grammatical little children's stories** — and *nothing else*. It has no world knowledge, can't do math, isn't a chatbot or assistant. That narrowness is the whole point: it's the smallest thing that still produces real, readable English, which is what makes it a perfect teaching model (TinyStories, Eldan & Li 2023, arXiv:2305.07759).
- Why it exists: the course teaches beginners to build a working LLM from scratch and run its forward pass live in the browser (NumPy/Pyodide). Training can't happen in a browser, so we train StoryByte **offline** (this log) and ship its weights. Everything that *computes* during learning runs live in front of the learner.

---

## 1. Environment (the exact machine we trained on)

| Item | Value |
|---|---|
| OS | macOS 26.4.1 |
| Arch | arm64 (Apple Silicon) |
| Python | 3.11.6 (pyenv) |
| PyTorch | 2.11.0 |
| **Accelerator** | **Apple MPS (Metal GPU)** — `torch.backends.mps.is_available() == True`; no CUDA |
| NumPy | 1.26.4 |
| tokenizers (HF) | 0.22.2 |
| huggingface_hub | 1.20.1 |
| CPU cores | 12 |
| Free disk | ~290 GB |

> Takeaway for learners: you do **not** need an NVIDIA GPU or the cloud. A modern Mac
> (Apple Silicon, MPS) trains this model fine. A plain CPU also works, just slower
> (see `HARDWARE.md`).

---

## 2. The in-browser feasibility spike (done before training — it shaped the model size)

Before training anything, we checked whether a from-scratch NumPy forward pass of a
nano model is fast enough to run live in the browser (Pyodide/WASM). Implemented the
forward pass + token generation in pure NumPy and timed it:

| Implementation | native ms/token | est. Pyodide (~6×) | 50-token story |
|---|---|---|---|
| Naïve (recompute context, looped heads) | 59 | ~355 | ~18 s ❌ |
| KV cache + vectorized heads (concatenate) | 24 | ~144 | ~7 s ⚠️ |
| **KV cache + vectorized heads + preallocated cache** | **12** | **~74** | **~3.7 s ✅** |

**Two findings that directly set the model design:**
1. In-browser generation is **overhead-bound, scaling with the number of layers**
   (count of tiny NumPy ops), *not* with width/params. L3·d64 (0.23M) costs the same
   per token as L4·d128 (1.1M). → **Keep the model shallow (≤4 layers)**; buy quality
   with width + more training, not depth.
2. The browser generator **must** use a preallocated KV cache + vectorized heads +
   a Web Worker + token streaming. A naïve port is too slow.

→ Decision: target **`n_layer=4, n_head=4, n_embd=128, block_size=256, vocab≈2048`**
(~1.1M params), which the spike validated as snappy.

---

## 3. The model spec (and why each choice)

StoryByte is deliberately the **classic GPT-2 / nanoGPT block** — no modern variants —
because the course rebuilds *this exact design* from scratch in NumPy, line by line.
Modern upgrades (RoPE, RMSNorm, SwiGLU, GQA) are taught conceptually, not used here.

| Component | Choice | Why |
|---|---|---|
| Type | Decoder-only, causal | The GPT family; pure next-word prediction |
| Positions | **Learned absolute** (`wpe` table) | Simplest to code from scratch + verify |
| Norm | **LayerNorm** (weight+bias), **pre-LN** | Classic; pre-LN trains stably (Xiong 2020) |
| Attention | **Standard multi-head** scaled dot-product | The canonical mechanism (Vaswani 2017) |
| MLP | Linear→**GELU(tanh)**→Linear, 4× width | GPT-2's FFN (GELU: Hendrycks 2016) |
| LM head | **Weight-tied** to token embedding | Saves params, common in small models |
| Attention impl | Written out explicitly (not fused) | So the NumPy/browser version matches 1:1 |

Reference implementation mirrored: Karpathy **nanoGPT** (`model.py`).

---

## 4. Data

- **Dataset:** TinyStories **V2 (GPT-4 generated)** — `roneneldan/TinyStories` on Hugging Face.
  - `TinyStoriesV2-GPT4-train.txt` = 2,228 MB · `TinyStoriesV2-GPT4-valid.txt` = 23 MB.
  - Chosen V2 (cleaner, GPT-4-only) over V1.
- **Why TinyStories:** it's a corpus of short stories written using only the vocabulary
  a 3–4 year old knows (~1,500 words). Restricting the *world* lets a tiny model master
  it — the trick is the data, not the model size.
- **Subset for the local run:** to keep the run fast we use the first portion of the
  train file (range-downloaded) + the full valid file. The repo's `01_download_data.py`
  documents downloading the *full* files for full reproduction.

### 4.1 Data run (the actual numbers)
- Downloaded: `valid.txt` = 22.5 MB (full); `train.txt` = **419 MB** (first-419 MB subset of the 2.2 GB file).
- Command: `python scripts/01_download_data.py --subset_mb 400`.

## 5. Tokenizer

- **Byte-level BPE, vocab = 2,048**, trained with Hugging Face `tokenizers` (Rust) on the 419 MB corpus.
- Special token: `<|endoftext|>` (id 0), used to separate stories.
- Sanity check — `"Once upon a time, there was a little"` encodes to **9 tokens**:
  `['Once', 'Ġupon', 'Ġa', 'Ġtime', ',', 'Ġthere', 'Ġwas', 'Ġa', 'Ġlittle']`
  (the `Ġ` marks a leading space — the GPT-2 byte-level convention). Decode round-trips exactly.
- The course's Module 1 re-implements this exact algorithm (most-frequent-pair → merge → repeat) from scratch.

## 6. Tokenized dataset (packed for training)
- `train.bin` = **113,524,462 tokens** (147,464 stories) · `val.bin` = 6,086,687 tokens (7,948 stories).
- Stored as flat `uint16` streams (nanoGPT convention), stories joined by `<|endoftext|>`.
- 113.5M training tokens for a 1.1M-param model ≈ **~100 tokens/param** — deliberately
  over-trained (~5× the Chinchilla compute-optimal 20:1), which is the right move for a small
  model you want to be *good* rather than compute-efficient.

## 7. Training

- **Model:** StoryByte, **1.088M params** (L4 · H4 · d128 · ctx256 · vocab2048).
- **Recipe (nanoGPT-canonical):** AdamW(β=0.9/0.95), weight_decay 0.1, lr 6e-4→6e-5,
  1,000-step linear warmup then cosine decay, grad-clip 1.0, batch 64, 30,000 steps, seed 1337.
- **Device:** Apple MPS. **Speed:** ~12 steps/sec → full run ≈ 40 min.
- **Loss trajectory (val):** step 0 → 7.65 (ppl 2111, ≈ uniform over 2048 tokens) ·
  step 500 → 4.09 (ppl 60) · step 1000 → 3.09 (ppl 22) · *(continues below as it trains)*.
- Cross-entropy intuition for learners: `perplexity = e^loss`; loss 7.62 = `ln(2048)` =
  a model guessing uniformly at random. Every drop is the model getting genuinely less surprised.

### 7.1 Final training result
- **30,000 steps in 42.4 minutes** on Apple MPS.
- **Final train loss 1.7206 · val loss 1.7398 (best 1.7318) · perplexity 5.70.**
- Full val-loss trajectory (saved in `course_artifacts/train_traces.json`):
  7.65 → 4.09 → 3.09 (1k) → 2.34 (2k) → 1.99 (5k) → 1.90 (9.5k) → 1.81 (15k) → 1.74 (29.5k).
- Coherent stories emerged early (~step 4,000) and kept sharpening. This matches the
  TinyStories finding: a ~1M-param model writes fluent simple English when the data is narrow.

## 8. Export + verification (the part that makes the browser trustworthy)

- Exported **52 weight arrays** to `storybyte_weights.npz` in GPT-2 naming (all linear
  weights transposed to `(in, out)` so the browser does plain `x @ W + b`). See
  `course_artifacts/MANIFEST.md` for the exact convention.
- **Dtype decision:** we first exported float16 (2.2 MB) but it flipped ~5% of greedy
  tokens vs PyTorch (float16 logit error ~1e-2). So we ship **float32 (4.37 MB)** — a fine
  browser download — which makes the model bit-faithful.
- **Verification (`verification.json`):** the pure-NumPy `reference_forward.py` vs PyTorch:
  **max logit diff 1.7e-05, greedy-token agreement 100.0%.** → The course's in-browser
  forward pass reproduces the trained model *exactly*. This is the integrity backbone of
  the whole "you built the real thing" promise.

## 9. The model's first stories (greedy, from the shipped weights)

> **"Once upon a time"** → *"…there was a little girl named Lily. She loved to play with her
> toys and have fun. One day, she found a big box of toys in her room… Lily's mom saw her and
> said, 'Lily, you need to clean your room. It is very dirty.' Lily was sad but she wanted to help…"*

> **"One day, a little girl named Lily"** → *"…went to the park with her mom. They saw a big
> tree with a lot of fruit. Lily wanted to eat the fruit, but her mom said, 'No, Lily, we can't
> eat the fruit.'…"*

Grammatical, on-theme, with characters, dialogue, and a mini plot — from **1.088M parameters**.

## 10. What we produced (handed to the course build)
- `course_artifacts/`: config, weights (npz), `reference_forward.py`, tokenizer (×2),
  `train_traces.json`, `interp_data.json` (logit-lens + 16 attention head-patterns/prompt),
  `sample_generations.json`, `verification.json`, `MANIFEST.md`.
- The full reproducible repo (this folder): model, 6 pipeline scripts, README, HARDWARE,
  REPRODUCIBILITY, Makefile, pinned requirements, MIT license.

## 11. Honesty / what a learner should take away
- A **1M-parameter** model — ~160,000× smaller than GPT-3 — writes coherent little stories,
  because the *world* (toddler-vocabulary stories) is small. **Data quality/narrowness beats
  raw size.** (TinyStories.)
- It is **not** a general assistant. It can't answer questions, do math, or discuss the real
  world. Ask it anything outside little stories and it will cheerfully make something up.
- Training happened **offline** (this log). The browser runs the **forward pass** — which is
  exactly the trained model — but cannot train. That split is the whole design of the course.

---
