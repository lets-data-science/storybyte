# StoryByte — Course Artifact Manifest

The files in this folder are what the interactive course **"Build a Tiny LLM — From Tokens to Text"** loads and runs in the browser. The course's pure-NumPy forward pass (run in Pyodide/WASM) mirrors `reference_forward.py` and reproduces this exact model.

## Canonical numbers (cite these in the course)

| | |
|---|---|
| **Parameters** | **1,088,256** (~1.09M) |
| Architecture | GPT-2-style decoder · 4 layers · 4 heads · d_model 128 · context 256 |
| Vocab | 2,048 (byte-level BPE) |
| Trained on | TinyStories V2 (GPT-4), **113,524,462 tokens** (147,464 stories) |
| Recipe | AdamW(0.9,0.95), wd 0.1, lr 6e-4→6e-5, 1k warmup + cosine, grad-clip 1.0, batch 64, 30k steps |
| **Final train loss** | **1.7206** |
| **Final val loss** | **1.7398**  (best 1.7318) |
| **Val perplexity** | **5.70**  (≈ effectively choosing among ~6 tokens/step; random would be 2,048) |
| Training time | ~42 min on Apple MPS |
| **Verification** | NumPy(float32) vs PyTorch: max logit diff **1.7e-05**, greedy agreement **100.0%** ✅ |
| Weights dtype | float32 (4.37 MB) — chosen so the browser reproduces the model exactly |

## Files

| File | Schema / contents |
|---|---|
| `storybyte_config.json` | architecture + final metrics (see keys above) |
| `storybyte_weights.npz` | 52 named float32 arrays (GPT-2 naming; see convention below) |
| `reference_forward.py` | the verified pure-NumPy forward pass + `generate()` — mirror this in-browser |
| `storybyte_tokenizer.json` | byte-level BPE: `vocab` (token→id) + `merges` (ordered) + `special_tokens` |
| `storybyte_tokenizer_hf.json` | the Hugging Face `tokenizers` JSON — the authoritative exact encoder |
| `train_traces.json` | `steps`, `train_loss`, `val_loss`, `lr`, `perplexity` (for Module 6 curves) |
| `interp_data.json` | logit-lens (`per_layer_top5`) + attention `patterns` per layer/head (Module 7) |
| `sample_generations.json` | reference outputs per prompt × decoding (greedy is exactly reproducible) |
| `verification.json` | the NumPy-vs-PyTorch proof |

## Weight naming + matmul convention (IMPORTANT for the browser port)

- All **linear** weights are stored **transposed to `(in, out)`**, so a layer is simply `y = x @ W + b`.
- Arrays: `wte` (vocab, d) · `wpe` (block, d) · per block `i`:
  `h.{i}.ln_1.g/.b` (d,) · `h.{i}.attn.c_attn.w` (d, 3d) `.b` (3d,) · `h.{i}.attn.c_proj.w` (d, d) `.b` ·
  `h.{i}.ln_2.g/.b` · `h.{i}.mlp.c_fc.w` (d, 4d) `.b` · `h.{i}.mlp.c_proj.w` (4d, d) `.b` · then `ln_f.g/.b`.
- `c_attn` output splits into **q, k, v in that order** along the last axis.
- **LayerNorm:** population variance (ddof=0), eps **1e-5**, with weight `g` and bias `b`.
- **GELU:** tanh approximation `0.5·x·(1+tanh(√(2/π)(x+0.044715x³)))`.
- **LM head is weight-tied:** `logits = x @ wte.T` (no separate output matrix).
- Compute in float32 (the stored dtype). The forward pass is deterministic; **greedy decoding reproduces exactly**, sampled decoding is illustrative (RNG differs across engines).

## Sample stories (greedy, from the shipped model)

- **"Once upon a time"** → *"…there was a little girl named Lily. She loved to play with her toys and have fun. One day, she found a big box of toys in her room… Lily's mom saw her and said, 'Lily, you need to clean your room.'…"*
- **"One day, a little girl named Lily"** → *"…went to the park with her mom. They saw a big tree with a lot of fruit. Lily wanted to eat the fruit, but her mom said, 'No, Lily, we can't eat the fruit.'…"*

## Honesty note (carry into the course)
StoryByte writes simple children's stories and nothing else — no world knowledge, no math, not an assistant. That's the point: it's the smallest model that still speaks fluent English.
