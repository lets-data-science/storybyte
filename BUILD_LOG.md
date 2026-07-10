# StoryByte build log

This is the measured record for the StoryByte model used by the Build a Tiny
LLM course. It separates recorded values, derived arithmetic, and engineering
estimates so the course can label them correctly.

## 1. Goal

StoryByte is a small decoder-only model for short children's stories. Its scope
was chosen before its architecture:

- narrow enough to learn on one local machine
- small enough to ship float32 weights to a browser
- conventional enough that a beginner can rebuild each operation
- real enough that the browser runs trained weights rather than a scripted demo

It is not a general assistant and should not be evaluated as one.

## 2. Reference environment

| Item | Recorded value |
|---|---|
| OS | macOS 26.4.1 |
| Architecture | arm64 Apple Silicon |
| Python | 3.11.6 |
| PyTorch | 2.11.0 |
| NumPy | 1.26.4 |
| tokenizers | 0.22.2 |
| huggingface_hub | 1.20.1 |
| Accelerator | Apple MPS |
| CPU cores | 12 |

The run did not use CUDA. CUDA support in the current trainer is a separate code
path and has no runtime claim in this log.

## 3. Browser feasibility spike

Before training, a NumPy generator was timed to choose a practical architecture.
The Pyodide column used a 6x planning multiplier; it was an estimate, not a
browser benchmark.

| Implementation | Native ms/token | Estimated Pyodide ms/token | Estimated 50-token time |
|---|---:|---:|---:|
| Recompute context, loop over heads | 59 | 355 | 18 s |
| KV cache, vectorized heads, concatenated cache | 24 | 144 | 7 s |
| KV cache, vectorized heads, preallocated cache | 12 | 74 | 3.7 s |

In the tested small configurations, Python/NumPy operation overhead made layer
count especially visible. That result supported a four-layer design. It does
not imply that width is free or that layer count always dominates on other
runtimes.

## 4. Architecture decision

The model follows a classic GPT-2-style decoder block:

- 4 pre-LayerNorm transformer blocks
- 4 causal attention heads per block
- model width 128 and head width 32
- learned absolute positions for a 256-token context
- GELU MLP with hidden width 512
- byte-level BPE vocabulary of 2,048
- tied token embedding and language-model head
- biases in linear layers and LayerNorm

Modern alternatives such as RoPE, RMSNorm, SwiGLU, GQA, and MoE are outside this
model. The course can discuss them, but its executable path must match the
architecture above.

### Parameter budget

| Group | Parameters |
|---|---:|
| Tied token embedding and LM head | 262,144 |
| Position embedding | 32,768 |
| Attention across 4 blocks | 264,192 |
| MLP across 4 blocks | 526,848 |
| Block LayerNorms | 2,048 |
| Final LayerNorm | 256 |
| Total | 1,088,256 |

The MLP is 48.41% of the whole model. Within the four transformer blocks alone,
it is about 66.4%. Those two denominators must not be mixed.

## 5. Data

The source files came from `roneneldan/TinyStories`:

- `TinyStoriesV2-GPT4-train.txt`
- `TinyStoriesV2-GPT4-valid.txt`

Reference command:

```bash
python scripts/01_download_data.py --subset_mb 400
```

The argument requests the first 400 MiB of the training file and trims back to
the last blank-line story boundary. The resulting file is displayed as about
419 MB when decimal units are used. The complete validation file was retained.

| Data quantity | Measured value |
|---|---:|
| Training stories | 147,464 |
| Validation stories | 7,948 |
| Training tokens | 113,524,462 |
| Validation tokens | 6,086,687 |
| Packed dtype | uint16 |

The current pipeline also supports the complete training file. A full-data run
is a different experiment and will not reproduce these counts.

## 6. Tokenizer

The tokenizer is GPT-2-style byte-level BPE with one special token:
`<|endoftext|>` at ID 0.

| Quantity | Measured value |
|---|---:|
| Vocabulary | 2,048 |
| Ordered merges | 1,791 |

The sanity string `Once upon a time, there was a little` encodes to nine tokens.
In the tokenizer JSON, a leading-space byte is displayed with the Unicode
U+0120 marker. Written with ASCII escapes, the pieces are:

```text
['Once', '\u0120upon', '\u0120a', '\u0120time', ',',
 '\u0120there', '\u0120was', '\u0120a', '\u0120little']
```

The exact encoder uses the byte-level pre-tokenizer as well as the ordered merge
table. Decoding reverses the byte mapping; it is not plain token-string joining.

## 7. Training recipe

| Setting | Recorded value |
|---|---|
| Objective | next-token cross-entropy |
| Optimizer | AdamW |
| Betas | 0.9, 0.95 |
| Weight decay | 0.1 |
| Peak learning rate | 6e-4 |
| Minimum learning rate | 6e-5 |
| Warmup | first 1,000 updates |
| Decay | cosine |
| Gradient clipping | global norm 1.0 |
| Batch | 64 x 256 tokens |
| Updates | 30,000 |
| Seed | 1337 |

The run sampled `64 x 256 x 30,000 = 491,520,000` training token positions.
That is about 4.33 equivalents of the 113,524,462-token stream because random
windows are sampled with replacement. It is about 451.7 sampled positions per
parameter. Dataset size and sampled training positions are different quantities.

The learning-rate schedule begins at 6e-7 for update index 0, reaches 6e-4 at
index 999, remains at 6e-4 when cosine decay begins at index 1,000, and reaches
6e-5 at index 30,000.

## 8. Training result

The MPS run completed in 42.4 minutes, averaging roughly 12 updates per second.
The trace contains an evaluation every 500 updates.

| Trace point | Step | Train loss | Validation loss | Validation perplexity |
|---|---:|---:|---:|---:|
| Initial | 0 | 7.654752 | 7.654894 | 2110.952 |
| Selected checkpoint | 29,500 | 1.730089 | 1.731835 | 5.651016 |
| Final evaluation | 30,000 | 1.720557 | 1.739808 | 5.696251 |

The selected checkpoint is the validation minimum at step 29,500. The final
training evaluation at step 30,000 has a slightly lower sampled train loss and
a slightly higher validation loss. The exported weights come from step 29,500.

## 9. Export contract

The exporter writes 52 arrays to `storybyte_weights.npz`. Every array is
float32. Linear weights are stored as `(input, output)` so NumPy uses
`x @ W + b`. The tied output head uses `x @ wte.T`.

An earlier float16 experiment reduced the file size but changed about 5% of
greedy decisions in its verification run. The shipped float32 arrays contain
4,353,024 parameter bytes before NPZ container overhead.

The artifact config stores both final-trace metrics and selected-checkpoint
metrics. Downstream code should use the label that matches its claim.

## 10. PyTorch and NumPy comparison

`scripts/05_export_artifacts.py` compares the exported NumPy implementation with
the selected PyTorch checkpoint on a fixed sequence of token IDs 5 through 24.

| Check | Measured value |
|---|---:|
| Maximum absolute logit difference | 1.71661376953125e-05 |
| Greedy-token agreement | 100.0% |
| Export dtype | float32 |

This establishes close numerical agreement on the fixed verification sequence.
It is not a bitwise-identity claim for every possible input.

## 11. Interpretation artifacts

`interp_data.json` contains two prompts. Each has measured attention matrices
for 4 layers x 4 heads and logit-lens outputs for four residual stages.

These arrays support inspection, not causal role labels. A head that attends to
a previous token on two prompts has not thereby been proven to be a universal
previous-token head. The course must label role names as hypotheses unless an
intervention tests them.

## 12. July 2026 audit corrections

The course audit made four documentation and tooling corrections without
changing the trained weights:

1. The trainer now selects CUDA before MPS and CPU, matching the hardware guide.
2. `make all` now uses the recorded 400 MiB subset instead of silently choosing
   the complete training file.
3. The artifact config now separates the selected step-29,500 checkpoint from
   the final step-30,000 trace.
4. The data-reuse arithmetic now uses 491,520,000 sampled positions and 4.33
   dataset equivalents instead of treating unique dataset tokens as all tokens
   processed during optimization.

Run `make verify` after any source or artifact change. It checks the current
arrays, tokenizer, metrics, source-copy parity, and a fresh NumPy forward pass.
