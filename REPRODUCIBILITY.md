# Reproducibility

## Reference run

| Item | Value |
|---|---|
| Seed | 1337 |
| Python | 3.11.6 |
| PyTorch | 2.11.0 |
| NumPy | 1.26.4 |
| tokenizers | 0.22.2 |
| Dataset | `roneneldan/TinyStories`, `TinyStoriesV2-GPT4-train.txt` |
| Data subset | first 419 MB of train file plus full validation file |
| Training tokens | 113.5M |
| Model | L4 / H4 / d128 / ctx256 / vocab2048 |
| Parameters | about 1.088M |
| Recipe | AdamW, beta=(0.9, 0.95), wd 0.1, lr 6e-4 to 6e-5, 1k warmup, cosine decay, grad clip 1.0 |
| Batch | 64 |
| Steps | 30,000 |

Pinned package versions are in `requirements.txt`.

## Expected result

The reference artifact records the exact final numbers in
`course_artifacts/storybyte_config.json`. A comparable run should settle near the same
loss range and produce coherent short stories.

## Verification

`course_artifacts/verification.json` records the exported NumPy model against the
PyTorch checkpoint:

- max logit diff: 1.7e-05
- greedy-token agreement: 100.0%

Exact loss values can drift slightly with GPU nondeterminism and framework versions.
The important reproduction checks are similar loss, coherent samples, and passing
NumPy/PyTorch parity after export.
