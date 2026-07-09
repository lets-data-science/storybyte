# Hardware guide

StoryByte is about 1.1M parameters. It is small enough that you do not need a
cluster, an A100, or paid cloud compute.

## Reference run

- Apple Silicon Mac with PyTorch MPS.
- About 12 training steps/sec at batch 64 and context 256.
- Full 30,000-step run took about 42 minutes.
- Any 16 GB Apple Silicon Mac has enough memory for the model and the 419 MB data subset.

## NVIDIA GPU

- A modern consumer card such as an RTX 3060 or RTX 4070 trains this comfortably.
- PyTorch selects CUDA automatically when available.
- The full reference-sized run should finish in well under an hour.

## Google Colab

A free T4 runtime is enough for this model. Install requirements, download the fast
subset with `--subset_mb 400`, then run the pipeline.

## CPU only

CPU training works, but it is much slower. For a quick local test:

```bash
python scripts/04_train.py --max_iters 5000 --batch_size 32
```

That produces rough output. Train longer for cleaner stories.

## Data and disk

- Full TinyStories V2 train file: about 2.2 GB.
- Fast subset: about 419 MB.
- Tokenized `.bin` streams: a few hundred MB.
- Budget about 3 GB free disk for a full local run.

## Cost

Local Mac/PC training is free once you have the machine. Colab free tier also works.
A short rented GPU run should cost well under a dollar for this model.

## Tuning

- Smaller machine: reduce `--batch_size` to 32 or 16.
- Shorter test run: reduce `--max_iters`.
- Browser target: keep `--n_layer` at 4 or lower, because browser generation speed
  is driven mostly by layer count.
