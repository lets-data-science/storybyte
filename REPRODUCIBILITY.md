# Reproducibility

- **Seed:** 1337 (set for torch + numpy in `04_train.py`).
- **Pinned versions:** see `requirements.txt` (torch 2.11.0, numpy 1.26.4, tokenizers 0.22.2).
- **Data:** `roneneldan/TinyStories`, file `TinyStoriesV2-GPT4-train.txt`. The reference run
  used the **first 419 MB** (`01_download_data.py --subset_mb 400`) → 113.5M training tokens
  (147,464 stories) after byte-level BPE (vocab 2,048). Full file also supported.
- **Model:** L4 · H4 · d128 · ctx256 · vocab2048 → ~1.088M params.
- **Recipe:** AdamW(0.9,0.95), wd 0.1, lr 6e-4→6e-5, 1k warmup + cosine, grad-clip 1.0,
  batch 64, 30,000 steps.
- **Expected:** validation loss settles in the low single digits → see
  `course_artifacts/storybyte_config.json` for the exact final numbers from the shipped run.
- **Verification:** `course_artifacts/verification.json` shows the pure-NumPy
  `reference_forward.py` reproduces the PyTorch logits (f32 within ~1e-4; float16 export
  agrees with PyTorch greedy tokens >99%).

Exact-match caveat: GPU nondeterminism + framework versions can shift the loss by a hair,
but a comparable model (similar loss + coherent stories) reproduces reliably.
