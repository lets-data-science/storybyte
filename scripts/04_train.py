"""
04 - Train StoryByte with the recorded small-GPT recipe.

Objective: self-supervised next-token prediction, cross-entropy loss.
Optimizer: AdamW (betas 0.9/0.95, weight_decay 0.1), grad-clip 1.0,
LR = linear warmup -> cosine decay (6e-4 -> 6e-5).

Logs full training traces (step, train_loss, val_loss, lr, perplexity) to
checkpoints/train_traces.json for the course's "How It Learned" module, periodically
samples a story so we can watch coherence emerge, and checkpoints the best val loss.

Runs on CUDA when available, then Apple MPS, then CPU.
"""
import argparse, json, math, os, time
import numpy as np
import torch
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.storybyte import StoryByte, GPTConfig

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
CKPT = os.path.join(HERE, "checkpoints")


def get_batch(ids, block, bs, device):
    ix = torch.randint(len(ids) - block - 1, (bs,))
    x = torch.stack([torch.from_numpy(ids[i:i + block].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(ids[i + 1:i + 1 + block].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def lr_at(it, warmup, max_iters, lr, min_lr):
    if it < warmup:
        return lr * (it + 1) / warmup
    if it > max_iters:
        return min_lr
    r = (it - warmup) / (max_iters - warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * r)) * (lr - min_lr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_embd", type=int, default=128)
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_iters", type=int, default=30000)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min_lr", type=float, default=6e-5)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--eval_interval", type=int, default=500)
    ap.add_argument("--eval_iters", type=int, default=100)
    ap.add_argument("--sample_interval", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()

    os.makedirs(CKPT, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"device: {device}")

    meta = json.load(open(os.path.join(DATA, "meta.json")))
    vocab_size = meta["vocab_size"]
    train_ids = np.memmap(os.path.join(DATA, "train.bin"), dtype=np.uint16, mode="r")
    val_ids = np.memmap(os.path.join(DATA, "val.bin"), dtype=np.uint16, mode="r")
    print(f"vocab={vocab_size}  train_tokens={len(train_ids):,}  val_tokens={len(val_ids):,}")

    cfg = GPTConfig(vocab_size=vocab_size, block_size=a.block_size,
                    n_layer=a.n_layer, n_head=a.n_head, n_embd=a.n_embd)
    model = StoryByte(cfg).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"StoryByte: {nparams/1e6:.3f}M params  (cfg: L{a.n_layer} H{a.n_head} d{a.n_embd} ctx{a.block_size})")

    optim = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95),
                              weight_decay=a.weight_decay)

    @torch.no_grad()
    def estimate():
        model.eval(); out = {}
        for split, ids in (("train", train_ids), ("val", val_ids)):
            losses = torch.zeros(a.eval_iters)
            for k in range(a.eval_iters):
                x, y = get_batch(ids, a.block_size, a.batch_size, device)
                _, loss = model(x, y); losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train(); return out

    enc = None
    try:
        from tokenizers import Tokenizer
        enc = Tokenizer.from_file(os.path.join(DATA, "tokenizer.json"))
    except Exception:
        pass

    @torch.no_grad()
    def sample(prompt="Once upon a time"):
        if enc is None: return "(tokenizer unavailable)"
        ids = enc.encode(prompt).ids
        x = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(x, max_new_tokens=80, temperature=0.8, top_k=40)[0].tolist()
        return enc.decode(out)

    traces = {"steps": [], "train_loss": [], "val_loss": [], "lr": [], "perplexity": [],
              "config": vars(a), "n_params": nparams, "vocab_size": vocab_size, "device": device}
    best_val = float("inf"); t0 = time.time()
    model.train()
    for it in range(a.max_iters + 1):
        lr = lr_at(it, a.warmup, a.max_iters, a.lr, a.min_lr)
        for g in optim.param_groups: g["lr"] = lr

        if it % a.eval_interval == 0:
            losses = estimate()
            ppl = math.exp(min(losses["val"], 20))
            dt = time.time() - t0
            print(f"step {it:6d} | train {losses['train']:.4f} | val {losses['val']:.4f} | "
                  f"ppl {ppl:.2f} | lr {lr:.2e} | {dt/60:.1f} min", flush=True)
            traces["steps"].append(it); traces["train_loss"].append(losses["train"])
            traces["val_loss"].append(losses["val"]); traces["lr"].append(lr); traces["perplexity"].append(ppl)
            json.dump(traces, open(os.path.join(CKPT, "train_traces.json"), "w"))
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({"model": model.state_dict(), "config": vars(cfg),
                            "val_loss": best_val, "iter": it, "meta": meta},
                           os.path.join(CKPT, "storybyte.pt"))

        if it % a.sample_interval == 0 and it > 0:
            print(f"  sample @ {it}: {sample()!r}", flush=True)

        if it == a.max_iters: break
        x, y = get_batch(train_ids, a.block_size, a.batch_size, device)
        _, loss = model(x, y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

    print(f"done. best val loss {best_val:.4f} (ppl {math.exp(min(best_val,20)):.2f}) in {(time.time()-t0)/60:.1f} min")
    print(f"final sample: {sample()!r}")
