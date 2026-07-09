# StoryByte Build Log

This log records how StoryByte was built for the course
["Build a Tiny LLM: From Tokens to Text"](https://letsdatascience.com/learn/build-a-tiny-llm).
It keeps the training choices, measurements, and caveats in one place so the course
can cite real numbers instead of invented examples.

## 0. What StoryByte is

- A GPT-2-style decoder-only language model with about 1.09M parameters.
- Trained from scratch on TinyStories.
- Designed to write short children's stories, not to act as a general-purpose chatbot.
- Built so the forward pass can run live in the browser with NumPy/Pyodide.

The browser course does not train the model. Training happens offline in this repo,
then the course loads the exported weights and runs inference.

## 1. Reference environment

| Item | Value |
|---|---|
| OS | macOS 26.4.1 |
| Arch | arm64, Apple Silicon |
| Python | 3.11.6, pyenv |
| PyTorch | 2.11.0 |
| Accelerator | Apple MPS, `torch.backends.mps.is_available() == True` |
| NumPy | 1.26.4 |
| tokenizers | 0.22.2 |
| huggingface_hub | 1.20.1 |
| CPU cores | 12 |
| Free disk | about 290 GB |

A modern Apple Silicon Mac is enough for this run. CPU-only training also works, but
is slower. See `HARDWARE.md` for the shorter hardware notes.

## 2. Browser feasibility spike

Before training, we tested whether a tiny GPT forward pass could run fast enough in a
browser worker. The spike used a pure NumPy implementation and timed token generation.

| Implementation | native ms/token | est. Pyodide | 50-token story |
|---|---:|---:|---:|
| Naive context recompute, looped heads | 59 | ~355 | ~18 s |
| KV cache with vectorized heads | 24 | ~144 | ~7 s |
| KV cache, vectorized heads, preallocated cache | 12 | ~74 | ~3.7 s |

Two design decisions came from this:

1. Browser generation is dominated by small-operation overhead and scales mainly with
   layer count, not parameter count. A shallow 4-layer model is the right target.
2. The course runtime needs a preallocated KV cache, vectorized heads, a Web Worker, and
   streaming tokens.

Target architecture: `n_layer=4`, `n_head=4`, `n_embd=128`, `block_size=256`,
`vocab_size` about 2048.

## 3. Model spec

StoryByte uses the classic GPT-2/nanoGPT block so the course can rebuild the same model
from scratch.

| Component | Choice | Reason |
|---|---|---|
| Type | Decoder-only causal LM | GPT-family next-token prediction |
| Positions | Learned absolute embeddings | Simple to code and verify |
| Norm | Pre-LayerNorm with weight and bias | Stable classic transformer block |
| Attention | Standard multi-head scaled-dot-product | Canonical attention mechanism |
| MLP | Linear, GELU(tanh), Linear, 4x width | GPT-2 style FFN |
| LM head | Tied to token embedding | Common for small language models |
| Attention implementation | Explicit PyTorch ops | Matches the NumPy reference closely |

The implementation mirrors Karpathy's nanoGPT style.

## 4. Data

- Dataset: TinyStories V2 from `roneneldan/TinyStories` on Hugging Face.
- Files: `TinyStoriesV2-GPT4-train.txt` and `TinyStoriesV2-GPT4-valid.txt`.
- Reason: TinyStories restricts the language world to simple stories, which lets a
  small model learn coherent English.
- Reference run: first 419 MB of the train file plus the full validation file.

Command:

```bash
python scripts/01_download_data.py --subset_mb 400
```

## 5. Tokenizer

- Byte-level BPE, vocab size 2,048.
- Trained with Hugging Face `tokenizers`.
- Special token: `<|endoftext|>`, id 0.
- The phrase `"Once upon a time, there was a little"` encodes to 9 tokens in the
  reference tokenizer. Several tokens include the GPT-2 byte-level leading-space
  marker.

The course reimplements the same BPE idea from scratch in Module 1.

## 6. Tokenized dataset

| Split | Tokens | Stories |
|---|---:|---:|
| train | 113,524,462 | 147,464 |
| val | 6,086,687 | 7,948 |

Streams are stored as flat `uint16` arrays, following the nanoGPT convention.

The reference run uses about 100 training tokens per parameter. That is intentionally
heavy for a small model: the goal is a model that behaves well in the browser, not a
compute-optimal training run.

## 7. Training

| Item | Value |
|---|---|
| Model | StoryByte, 1.088M params |
| Shape | L4 / H4 / d128 / ctx256 / vocab2048 |
| Optimizer | AdamW, beta=(0.9, 0.95), weight decay 0.1 |
| LR schedule | 6e-4 to 6e-5, 1k warmup, cosine decay |
| Batch | 64 |
| Steps | 30,000 |
| Grad clip | 1.0 |
| Seed | 1337 |
| Device | Apple MPS |
| Speed | about 12 steps/sec |

Validation loss trajectory:

```text
step 0      7.65
step 500    4.09
step 1000   3.09
step 2000   2.34
step 5000   1.99
step 9500   1.90
step 15000  1.81
step 29500  1.74
```

Final result:

| Metric | Value |
|---|---:|
| Training time | 42.4 minutes |
| Final train loss | 1.7206 |
| Final val loss | 1.7398 |
| Best val loss | 1.7318 |
| Val perplexity | 5.70 |

Coherent short stories appeared around step 4,000 and improved through the run.

## 8. Export and verification

The export script writes 52 weight arrays to `course_artifacts/storybyte_weights.npz`.
Linear weights are stored as `(in, out)` so the browser can use plain `x @ W + b`.

The first export tried float16. It cut the file size to about 2.2 MB, but it flipped
about 5% of greedy tokens compared with PyTorch. The shipped export is float32
instead. It is 4.37 MB and reproduces the checkpoint reliably.

Verification:

| Check | Value |
|---|---:|
| NumPy vs PyTorch max logit diff | 1.7e-05 |
| Greedy-token agreement | 100.0% |

That parity check is what lets the browser course claim it is running the trained
model, not a lookalike.

## 9. Sample behavior

Greedy decoding from the shipped weights produces grammatical short stories from
prompts such as `"Once upon a time"` and `"One day, a little girl named Lily"`.
Full samples live in `course_artifacts/sample_generations.json`.

## 10. Produced artifacts

- `course_artifacts/storybyte_config.json`
- `course_artifacts/storybyte_weights.npz`
- `course_artifacts/reference_forward.py`
- `course_artifacts/storybyte_tokenizer.json`
- `course_artifacts/storybyte_tokenizer_hf.json`
- `course_artifacts/train_traces.json`
- `course_artifacts/interp_data.json`
- `course_artifacts/sample_generations.json`
- `course_artifacts/verification.json`
- `course_artifacts/MANIFEST.md`

## 11. Learner takeaway

A 1M-parameter model can write coherent small-domain text when the data distribution is
narrow. StoryByte is not a general-purpose chatbot. It is a compact teaching model for
understanding the tokenizer, transformer block, sampler, training curve, and inference
runtime end to end.
