# Hardware guide — what you need to train StoryByte

StoryByte is ~1.1M parameters. It is **tiny** by LLM standards, so you do **not** need
a cluster, an A100, or the cloud. Here are real, tested paths.

## What we trained on (the reference run)
- **Apple Silicon Mac, PyTorch MPS (Metal GPU).** ~12 training steps/sec at the default
  config (batch 64 × context 256). The full 30,000-step run takes **~40 minutes**.
- Peak memory is small (the model + a 419 MB data subset). Any 16 GB Mac is plenty.

## NVIDIA GPU (CUDA)
- Any modern consumer card (e.g., RTX 3060 / 4070, 8–12 GB VRAM) trains this comfortably
  and **faster than MPS**. PyTorch picks CUDA automatically.
- Expected: well under ~30 minutes for the full run; a few GB VRAM.

## Google Colab (free tier)
- A free **T4** GPU runs this fine. `pip install -r requirements.txt`, then run the
  pipeline. Use `--subset_mb 400` for the data to fit the session quickly.

## CPU only (no GPU)
- It works — just slower (roughly 5–10× slower than MPS). Use a smaller run to taste it:
  `python scripts/04_train.py --max_iters 5000 --batch_size 32`. You'll get
  semi-coherent output; train longer for better stories.

## Data + disk
- Full TinyStories V2 train file ≈ 2.2 GB; the fast subset (`--subset_mb 400`) ≈ 419 MB.
  Tokenized streams (`.bin`) are a few hundred MB. Budget ~3 GB free for the full run.

## Cost
- **Local (Mac/PC): free.** Colab free tier: free. A cloud GPU hour (if you rent one)
  is well under a dollar for this model. There is no reason this should cost real money.

## Tuning for your machine
- Smaller/slower box: drop `--batch_size` (32 or 16) and/or `--max_iters`.
- Want it snappier in the browser later: keep `--n_layer` ≤ 4 (in-browser generation
  speed scales with layer count, not width — see BUILD_LOG.md §2).
