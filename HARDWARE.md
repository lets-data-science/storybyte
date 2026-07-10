# Hardware guide

StoryByte has 1,088,256 parameters. The training script selects devices in this
order: CUDA, Apple MPS, CPU.

## Measured reference path

The checked-in run used Apple MPS on the machine recorded in `BUILD_LOG.md`:

- batch: 64 sequences x 256 tokens
- updates: 30,000
- observed speed: about 12 updates per second
- wall time: 42.4 minutes
- data: the 400 MiB training subset plus the full validation file

These timings describe that machine and software build. They are not promises
for another Mac.

## CUDA

`scripts/04_train.py` uses CUDA when `torch.cuda.is_available()` is true. The
model and batch are small by current GPU standards. This repository does not
publish a measured CUDA runtime or memory peak, so treat any local estimate as
machine-specific and record it if you add one.

## CPU

CPU training is supported and is the fallback when neither CUDA nor MPS is
available. Start with a shorter run to test the pipeline:

```bash
python scripts/04_train.py --max_iters 5000 --batch_size 32
```

That command is a pipeline check, not a reproduction of the shipped model.
Quality and runtime will differ from the 30,000-update reference run.

## Hosted notebooks

A notebook runtime can use the same commands. Install `requirements.txt`, keep
the repository and generated files on persistent storage, and confirm the
printed device before training. Hosted hardware, session duration, and pricing
change over time, so this repository does not attach a fixed runtime or cost to
those services.

## Disk use

- reference training download: first 400 MiB, about 419 MB in decimal units
- complete TinyStories V2 training file: about 2.2 GB
- checked-in token streams: about 229 MB training and 12 MB validation
- exported float32 weights: 4,353,024 parameter bytes, about 4.2 MiB before NPZ overhead

Allow extra room for the source text, token streams, checkpoint, and temporary
files. `make clean` removes generated `.bin` files and `.pt` checkpoints; it does
not remove downloaded text or course artifacts.

## Browser inference

Browser generation uses a separate NumPy implementation. In the recorded
feasibility tests, layer count was the dominant latency factor among the tested
small configurations. That observation motivated four layers. It is not a
general claim that width has zero performance cost.
