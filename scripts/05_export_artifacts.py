"""
05: Export the trained checkpoint into the course artifact contract and
VERIFY that the pure-NumPy reference_forward reproduces the PyTorch model.

Produces course_artifacts/:
  storybyte_config.json, storybyte_weights.npz, storybyte_tokenizer.json,
  storybyte_tokenizer_hf.json (the HF tokenizer.json, authoritative encoder),
  train_traces.json

Weight naming = GPT-2-style; all linear weights stored as (in, out) so numpy does x @ W.
"""
import json, math, os, sys, shutil
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.storybyte import StoryByte, GPTConfig

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data"); CKPT = os.path.join(HERE, "checkpoints")
ART = os.path.join(HERE, "course_artifacts")


def export_weights(model, dtype=np.float16):
    sd = model.state_dict()
    g = lambda k: sd[k].detach().cpu().numpy()
    W = {"wte": g("transformer.wte.weight"), "wpe": g("transformer.wpe.weight"),
         "ln_f.g": g("transformer.ln_f.weight"), "ln_f.b": g("transformer.ln_f.bias")}
    for i in range(model.config.n_layer):
        p = f"transformer.h.{i}"
        W[f"h.{i}.ln_1.g"] = g(f"{p}.ln_1.weight"); W[f"h.{i}.ln_1.b"] = g(f"{p}.ln_1.bias")
        W[f"h.{i}.attn.c_attn.w"] = g(f"{p}.attn.c_attn.weight").T
        W[f"h.{i}.attn.c_attn.b"] = g(f"{p}.attn.c_attn.bias")
        W[f"h.{i}.attn.c_proj.w"] = g(f"{p}.attn.c_proj.weight").T
        W[f"h.{i}.attn.c_proj.b"] = g(f"{p}.attn.c_proj.bias")
        W[f"h.{i}.ln_2.g"] = g(f"{p}.ln_2.weight"); W[f"h.{i}.ln_2.b"] = g(f"{p}.ln_2.bias")
        W[f"h.{i}.mlp.c_fc.w"] = g(f"{p}.mlp.c_fc.weight").T
        W[f"h.{i}.mlp.c_fc.b"] = g(f"{p}.mlp.c_fc.bias")
        W[f"h.{i}.mlp.c_proj.w"] = g(f"{p}.mlp.c_proj.weight").T
        W[f"h.{i}.mlp.c_proj.b"] = g(f"{p}.mlp.c_proj.bias")
    return {k: v.astype(dtype) for k, v in W.items()}


def main():
    os.makedirs(ART, exist_ok=True)
    ck = torch.load(os.path.join(CKPT, "storybyte.pt"), map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ck["config"])
    model = StoryByte(cfg); model.load_state_dict(ck["model"]); model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    traces = json.load(open(os.path.join(CKPT, "train_traces.json")))
    final_train = traces["train_loss"][-1]; final_val = traces["val_loss"][-1]
    val_ppl = math.exp(min(final_val, 20))
    meta = ck.get("meta", {})

    # config
    config = {
        "arch": "gpt2-style-decoder", "n_layer": cfg.n_layer, "n_head": cfg.n_head,
        "n_embd": cfg.n_embd, "vocab_size": cfg.vocab_size, "block_size": cfg.block_size,
        "mlp_ratio": cfg.mlp_ratio, "activation": "gelu_tanh", "norm": "layernorm",
        "pos": "learned_absolute", "weight_tying": True, "biases": True,
        "n_params": int(n_params), "final_train_loss": round(final_train, 4),
        "final_val_loss": round(final_val, 4), "val_perplexity": round(val_ppl, 3),
        "tokenizer": "bpe_bytelevel", "eos_token_id": meta.get("eot_id", 0),
        "trained_on": "TinyStoriesV2-GPT4", "train_tokens": meta.get("train_tokens", 0),
        "seed": 1337,
        "dtype_export": "float32",
    }
    json.dump(config, open(os.path.join(ART, "storybyte_config.json"), "w"), indent=2)

    # Use float32 so the browser reproduces the trained model exactly.
    # (float16 halves the file to ~2.2 MB but flips ~5% of greedy tokens; the 4.3 MB
    #  float32 file is a fine download and keeps the in-browser model bit-faithful.)
    W32 = export_weights(model, np.float32)
    np.savez(os.path.join(ART, "storybyte_weights.npz"), **W32)
    print(f"exported {len(W32)} weight arrays, {n_params/1e6:.3f}M params (float32)")

    # tokenizer: ship the HF tokenizer.json (authoritative) + a simplified vocab+merges view
    shutil.copy(os.path.join(DATA, "tokenizer.json"), os.path.join(ART, "storybyte_tokenizer_hf.json"))
    hf = json.load(open(os.path.join(DATA, "tokenizer.json")))
    simple = {"type": "bpe_bytelevel",
              "vocab": hf["model"]["vocab"],
              "merges": [m.split(" ") if isinstance(m, str) else m for m in hf["model"]["merges"]],
              "special_tokens": {"<|endoftext|>": config["eos_token_id"]},
              "byte_level": True,
              "note": "GPT-2 byte-level BPE. storybyte_tokenizer_hf.json is the authoritative encoder."}
    json.dump(simple, open(os.path.join(ART, "storybyte_tokenizer.json"), "w"))

    shutil.copy(os.path.join(CKPT, "train_traces.json"), os.path.join(ART, "train_traces.json"))

    # ---- VERIFY: the pure-NumPy reference reproduces the torch model ----------
    from reference_forward import StoryByteNumPy
    nm = StoryByteNumPy(ART)                       # loads the shipped float32 weights
    ids = list(range(5, 25))                       # arbitrary fixed token ids
    with torch.no_grad():
        x = torch.tensor([ids]); pos = torch.arange(len(ids))
        h = model.transformer.wte(x) + model.transformer.wpe(pos)
        for blk in model.transformer.h: h = blk(h)
        h = model.transformer.ln_f(h); t_full = model.lm_head(h)[0].numpy()   # (T,vocab)
    n_full = nm.forward(ids)
    dmax = float(np.abs(t_full - n_full).max())
    agree = float((t_full.argmax(-1) == n_full.argmax(-1)).mean())
    print(f"VERIFY  max|logit| diff  NumPy(f32) vs PyTorch = {dmax:.2e}")
    print(f"VERIFY  greedy argmax agreement = {agree*100:.1f}%")
    verify = {"numpy_vs_torch_max_logit_diff": dmax, "greedy_agreement": agree,
              "dtype": "float32", "pass": bool(dmax < 1e-3 and agree > 0.999)}
    json.dump(verify, open(os.path.join(ART, "verification.json"), "w"), indent=2)
    print("VERIFY pass:", verify["pass"])
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
