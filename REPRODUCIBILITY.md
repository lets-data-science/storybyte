# Reproducibility

## Recorded reference run

| Item | Value |
|---|---|
| Seed | 1337 |
| Python | 3.11.6 |
| PyTorch | 2.11.0 |
| NumPy | 1.26.4 |
| tokenizers | 0.22.2 |
| Dataset file | `TinyStoriesV2-GPT4-train.txt` |
| Data command | `python scripts/01_download_data.py --subset_mb 400` |
| Training stories | 147,464 |
| Training tokens | 113,524,462 |
| Validation stories | 7,948 |
| Validation tokens | 6,086,687 |
| Model | L4, H4, d128, context 256, vocabulary 2,048 |
| Parameters | 1,088,256 |
| Batch | 64 sequences x 256 tokens |
| Updates | 30,000 |
| Optimizer | AdamW, betas 0.9/0.95, weight decay 0.1 |
| Schedule | 1,000-update warmup, cosine from 6e-4 to 6e-5 |
| Gradient clipping | global norm 1.0 |
| Device | Apple MPS |
| Wall time | 42.4 minutes |

The subset argument uses 400 MiB (`400 * 2^20` bytes), then trims the file back
to the last complete story boundary. File listings usually show this as about
419 MB in decimal units.

## Trace and checkpoint

The last evaluation and the exported checkpoint are different trace points:

| Item | Step | Validation loss | Perplexity |
|---|---:|---:|---:|
| Final trace | 30,000 | 1.7398083210 | 5.6962514662 |
| Selected checkpoint | 29,500 | 1.7318353653 | 5.6510160751 |

`storybyte_weights.npz` contains the selected checkpoint. The artifact config
stores both sets of rounded metrics so downstream code does not confuse them.

## Commands

```bash
python3 -m pip install -r requirements.txt
make all
make verify
```

`make all` uses the recorded 400 MiB subset. `make download-full` is supported,
but a full-data run has different token counts and is not an exact reproduction
of the checked-in checkpoint.

## What can vary

The seed is applied to PyTorch and NumPy. CUDA, MPS, and CPU kernels can still
produce small numerical differences across hardware or framework builds.
Comparable loss and coherent samples are the expected training-level result.
The checked-in export has the stronger artifact-level test: on its fixed
verification sequence, NumPy and PyTorch have maximum absolute logit difference
`1.71661376953125e-05` and identical greedy choices.
