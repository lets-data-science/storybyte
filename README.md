# StoryByte

StoryByte is a 1,088,256-parameter decoder-only language model trained from
scratch on a 400 MiB subset of
[TinyStories V2](https://arxiv.org/abs/2305.07759). It writes short children's
stories. The model, tokenizer, training loop, exported weights, and NumPy
forward pass are included here.

This repository is the offline training half of the interactive
[Build a Tiny LLM](https://letsdatascience.com/learn/build-a-tiny-llm) course.
The course loads the exported float32 weights and runs the same forward-pass
math in the browser. Training remains an offline PyTorch job.

StoryByte has a narrow job. It does not provide reliable facts, arithmetic, or
assistant behavior. That limited scope is useful for teaching because every
part of the model remains small enough to inspect.

## Repository map

```text
storybyte/
|-- model/storybyte.py           # PyTorch model
|-- scripts/
|   |-- 01_download_data.py      # TinyStories download
|   |-- 02_train_tokenizer.py    # byte-level BPE training
|   |-- 03_prepare_data.py       # uint16 token streams
|   |-- 04_train.py              # AdamW training loop
|   |-- 05_export_artifacts.py   # float32 export and PyTorch/NumPy check
|   |-- 06_evaluate.py           # samples and interpretation artifacts
|   |-- 07_verify_repository.py  # fast offline integrity check
|   `-- reference_forward.py     # NumPy forward pass and generation
|-- course_artifacts/            # files consumed by the browser course
|-- checkpoints/                 # selected PyTorch checkpoint and trace
|-- BUILD_LOG.md                 # measured reference-run record
|-- HARDWARE.md                  # supported execution paths
`-- REPRODUCIBILITY.md           # seeds, versions, and expected values
```

## Reproduce the shipped run

Use Python 3.11 and install the pinned packages:

```bash
python3 -m pip install -r requirements.txt
make all
```

`make all` reproduces the recorded data choice: the first 400 MiB of the V2
training file, trimmed back to the last complete story. It then trains the
tokenizer and model, exports the selected checkpoint, and rebuilds the course
artifacts. The training run used Apple MPS and took 42.4 minutes on the machine
recorded in `BUILD_LOG.md`.

Run the stages separately when debugging:

```bash
python scripts/01_download_data.py --subset_mb 400
python scripts/02_train_tokenizer.py
python scripts/03_prepare_data.py
python scripts/04_train.py
python scripts/05_export_artifacts.py
python scripts/06_evaluate.py
```

`make download-full` downloads the complete training file. That creates a
different data run and will not reproduce the checked-in token counts.

## Verify the checked-in artifacts

The fast verifier does not retrain the model:

```bash
make verify
```

It checks the 52 exported arrays, shapes, float32 dtype, 1,088,256-parameter
total, tokenizer size, final trace, selected checkpoint, recorded
PyTorch/NumPy comparison, and a fresh NumPy forward pass.

Generate text from the exported model without importing PyTorch:

```bash
python scripts/reference_forward.py "Once upon a time" --seed 0
```

The text path uses the pinned `tokenizers` package for exact byte-level BPE.
The model math itself uses NumPy.

## Model specification

StoryByte uses a classic GPT-2-style block so the course can rebuild the same
operations directly:

- decoder-only causal self-attention
- 4 blocks, 4 heads, width 128, head width 32
- 256-token context and 2,048-token byte-level BPE vocabulary
- learned absolute position embeddings
- pre-LayerNorm residual blocks
- GELU MLP with hidden width 512
- tied token embedding and language-model head

| Item | Recorded value |
|---|---:|
| Parameters | 1,088,256 |
| Training stories | 147,464 |
| Training tokens | 113,524,462 |
| Updates | 30,000 |
| Final trace train loss | 1.7206 |
| Final trace validation loss | 1.7398 |
| Final trace validation perplexity | 5.696 |
| Selected checkpoint | step 29,500 |
| Selected checkpoint validation loss | 1.7318 |
| Selected checkpoint validation perplexity | 5.651 |

The browser weights come from the selected step-29,500 checkpoint. The final
trace values describe the last evaluation at step 30,000. These are deliberately
reported separately.

The float32 NumPy export was compared with the PyTorch checkpoint on a fixed
20-token verification sequence. Maximum absolute logit difference was
`1.71661376953125e-05`, and greedy-token agreement was 100% for that sequence.
Those numbers are stored in `course_artifacts/verification.json`.

## Course artifact contract

| File | Purpose |
|---|---|
| `storybyte_config.json` | architecture, trace metrics, and checkpoint metrics |
| `storybyte_weights.npz` | 52 named float32 weight arrays |
| `storybyte_tokenizer.json` | simplified vocabulary and ordered merges |
| `storybyte_tokenizer_hf.json` | authoritative exact tokenizer |
| `reference_forward.py` | checked NumPy forward pass and generation |
| `train_traces.json` | loss, learning-rate, and perplexity trace |
| `interp_data.json` | measured attention and logit-lens data for two prompts |
| `sample_generations.json` | fixed-seed reference generations |
| `verification.json` | recorded PyTorch/NumPy comparison |
| `MANIFEST.md` | array names, shapes, and conventions |

The website's four runtime artifacts match these files by SHA-256 in the local
course audit. Re-run the website validator whenever an artifact changes:

```bash
python3 scripts/tests/validate-storybyte-artifacts.py
```

That command lives in the `lets-data-science` website repository.

## Sources

- [TinyStories](https://arxiv.org/abs/2305.07759), Eldan and Li, 2023
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762), Vaswani et al., 2017
- [AdamW](https://arxiv.org/abs/1711.05101), Loshchilov and Hutter, 2017
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909), Sennrich et al., 2016
- [nanoGPT](https://github.com/karpathy/nanoGPT), Andrej Karpathy
- [picoGPT](https://github.com/jaymody/picoGPT), Jay Mody

MIT licensed. Maintained by [Let's Data Science](https://letsdatascience.com).
