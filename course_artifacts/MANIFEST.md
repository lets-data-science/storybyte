# StoryByte course artifact manifest

These files are the contract consumed by the Build a Tiny LLM course. The
browser implementation mirrors `reference_forward.py` and loads the same
float32 weights.

## Model

| Item | Value |
|---|---|
| Architecture | GPT-2-style decoder, 4 layers, 4 heads, width 128, context 256 |
| Vocabulary | 2,048-token GPT-2-style byte-level BPE |
| Parameters | 1,088,256 |
| Export | 52 float32 arrays |
| Final trace | step 30,000, train loss 1.7206, val loss 1.7398, val perplexity 5.696 |
| Selected checkpoint | step 29,500, val loss 1.7318, val perplexity 5.651 |
| Verification | max absolute logit difference 1.71661376953125e-05; fixed-sequence greedy agreement 100% |

The NPZ contains the selected step-29,500 checkpoint. Final-trace metrics describe
the last evaluation and are retained separately.

## Files

| File | Contract |
|---|---|
| `storybyte_config.json` | architecture plus final-trace and selected-checkpoint metrics |
| `storybyte_weights.npz` | named float32 arrays in the convention below |
| `reference_forward.py` | NumPy forward pass and generator mirrored by the course |
| `storybyte_tokenizer.json` | simplified vocabulary, ordered merges, and special token map |
| `storybyte_tokenizer_hf.json` | authoritative exact tokenizer |
| `train_traces.json` | 61 evaluations from step 0 through step 30,000 |
| `interp_data.json` | measured attention and logit-lens outputs for two prompts |
| `sample_generations.json` | fixed-seed decoding examples |
| `verification.json` | recorded PyTorch/NumPy comparison |

## Array convention

- `wte`: `(vocabulary, d_model)`
- `wpe`: `(context, d_model)`
- `h.{i}.ln_1.g`, `h.{i}.ln_1.b`: `(d_model,)`
- `h.{i}.attn.c_attn.w`: `(d_model, 3 * d_model)`
- `h.{i}.attn.c_attn.b`: `(3 * d_model,)`
- `h.{i}.attn.c_proj.w`: `(d_model, d_model)`
- `h.{i}.attn.c_proj.b`: `(d_model,)`
- `h.{i}.ln_2.g`, `h.{i}.ln_2.b`: `(d_model,)`
- `h.{i}.mlp.c_fc.w`: `(d_model, 4 * d_model)`
- `h.{i}.mlp.c_fc.b`: `(4 * d_model,)`
- `h.{i}.mlp.c_proj.w`: `(4 * d_model, d_model)`
- `h.{i}.mlp.c_proj.b`: `(d_model,)`
- `ln_f.g`, `ln_f.b`: `(d_model,)`

Linear layers use `x @ W + b`. LayerNorm uses population variance with
`eps=1e-5`. GELU uses the tanh approximation:

```text
0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
```

The output head is tied to `wte`, so logits use `x @ wte.T` and do not add a
second vocabulary matrix.

## Interpretation limits

`interp_data.json` contains real arrays from the selected model, but it covers
only two prompts. Intermediate softmax values are logit-lens outputs, not
calibrated confidence. Attention patterns do not establish a head's causal role.
Course visuals must label any broader role description as a hypothesis.

## Integrity check

From the repository root:

```bash
python scripts/07_verify_repository.py
```

The verifier checks shapes, dtype, parameter count, tokenizer size, trace and
checkpoint metrics, the recorded comparison, source-copy parity, and a fresh
NumPy forward pass.
