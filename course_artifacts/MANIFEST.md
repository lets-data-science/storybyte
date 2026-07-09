# StoryByte Course Artifact Manifest

These files are loaded by the interactive course
["Build a Tiny LLM: From Tokens to Text"](https://letsdatascience.com/learn/build-a-tiny-llm).
The browser implementation mirrors `reference_forward.py` and reproduces the exported
model.

## Canonical numbers

| Item | Value |
|---|---|
| Parameters | 1,088,256 |
| Architecture | GPT-2-style decoder, 4 layers, 4 heads, d_model 128, context 256 |
| Vocab | 2,048 byte-level BPE tokens |
| Training data | TinyStories V2, 113,524,462 tokens, 147,464 stories |
| Recipe | AdamW, beta=(0.9, 0.95), wd 0.1, lr 6e-4 to 6e-5, 1k warmup, cosine decay, grad clip 1.0, batch 64, 30k steps |
| Final train loss | 1.7206 |
| Final val loss | 1.7398, best 1.7318 |
| Val perplexity | 5.70 |
| Training time | about 42 minutes on Apple MPS |
| Verification | NumPy float32 vs PyTorch max logit diff 1.7e-05, greedy agreement 100.0% |
| Weights dtype | float32, 4.37 MB |

## Files

| File | Contents |
|---|---|
| `storybyte_config.json` | architecture and final metrics |
| `storybyte_weights.npz` | 52 named float32 arrays using GPT-2-style names |
| `reference_forward.py` | verified pure-NumPy forward pass and `generate()` |
| `storybyte_tokenizer.json` | simplified byte-level BPE view: vocab, merges, special tokens |
| `storybyte_tokenizer_hf.json` | authoritative Hugging Face tokenizer JSON |
| `train_traces.json` | steps, train loss, val loss, LR, and perplexity |
| `interp_data.json` | logit-lens top tokens and attention patterns per layer/head |
| `sample_generations.json` | reference outputs for fixed prompts and decoding settings |
| `verification.json` | NumPy/PyTorch parity proof |

## Weight naming and matmul convention

- Linear weights are stored transposed to `(in, out)`, so a layer is `y = x @ W + b`.
- Arrays include `wte`, `wpe`, per-block LayerNorm, attention, MLP, and final LayerNorm tensors.
- `c_attn` output splits into `q`, `k`, and `v` in that order along the last axis.
- LayerNorm uses population variance, `eps=1e-5`, weight `g`, and bias `b`.
- GELU uses the tanh approximation:
  `0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))`.
- The language-model head is tied to the token embedding: `logits = x @ wte.T`.
- The forward pass is deterministic in float32. Greedy decoding reproduces exactly; sampled decoding is illustrative because RNGs differ across engines.

## Honesty note

StoryByte writes short children's stories and nothing else. It has no general world
knowledge, does not do math, and is not a chatbot. The narrow task is deliberate: it
makes the model small enough for a learner to inspect end to end.
