"""
reference_forward.py: StoryByte forward pass and generation in pure NumPy.

This is the canonical, dependency-light (numpy-only for the math) re-implementation that
the in-browser course mirrors line-for-line. It loads the exported weights and reproduces
the trained PyTorch model's logits within float tolerance (see 05_export_artifacts.py --verify).

Weight/matmul convention (documented in MANIFEST.md):
  - All linear weights are stored as (in, out); a linear layer is  y = x @ W + b.
  - LayerNorm uses population variance (ddof=0), eps=1e-5, with weight g and bias b.
  - GELU is the tanh approximation.
  - The LM head is weight-tied to the token embedding: logits = x @ wte.T.

Picogpt lineage: Jay Mody, "GPT in 60 Lines of NumPy" (github.com/jaymody/picoGPT).
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def layernorm(x, g, b, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)            # population variance (matches torch)
    return g * (x - mu) / np.sqrt(var + eps) + b


def linear(x, w, b):
    return x @ w + b


class StoryByteNumPy:
    def __init__(self, artifacts_dir):
        self.cfg = json.load(open(os.path.join(artifacts_dir, "storybyte_config.json")))
        w = np.load(os.path.join(artifacts_dir, "storybyte_weights.npz"))
        self.W = {k: w[k].astype(np.float32) for k in w.files}   # upcast f16 -> f32 for compute
        self.n_layer = self.cfg["n_layer"]; self.n_head = self.cfg["n_head"]; self.n_embd = self.cfg["n_embd"]
        # tokenizer (used for encode/decode only; the math above is pure numpy)
        self.tok = None
        tjson = os.path.join(artifacts_dir, "storybyte_tokenizer_hf.json")
        if os.path.exists(tjson):
            try:
                from tokenizers import Tokenizer
                self.tok = Tokenizer.from_file(tjson)
            except Exception:
                pass

    def attn(self, x, i):
        T = x.shape[0]; H = self.n_head; hd = self.n_embd // H
        qkv = linear(x, self.W[f"h.{i}.attn.c_attn.w"], self.W[f"h.{i}.attn.c_attn.b"])  # (T,3H)
        q, k, v = np.split(qkv, 3, axis=-1)                                              # (T,H) each
        q = q.reshape(T, H, hd).transpose(1, 0, 2)   # (H,T,hd)
        k = k.reshape(T, H, hd).transpose(1, 0, 2)
        v = v.reshape(T, H, hd).transpose(1, 0, 2)
        mask = np.triu(np.ones((T, T), np.float32), 1) * -1e10
        att = softmax((q @ k.transpose(0, 2, 1)) / np.sqrt(hd) + mask)                   # (H,T,T)
        y = (att @ v).transpose(1, 0, 2).reshape(T, self.n_embd)                          # (T,H*hd)
        return linear(y, self.W[f"h.{i}.attn.c_proj.w"], self.W[f"h.{i}.attn.c_proj.b"])

    def mlp(self, x, i):
        h = gelu(linear(x, self.W[f"h.{i}.mlp.c_fc.w"], self.W[f"h.{i}.mlp.c_fc.b"]))
        return linear(h, self.W[f"h.{i}.mlp.c_proj.w"], self.W[f"h.{i}.mlp.c_proj.b"])

    def forward(self, ids):
        ids = np.asarray(ids)
        x = self.W["wte"][ids] + self.W["wpe"][:len(ids)]
        for i in range(self.n_layer):
            x = x + self.attn(layernorm(x, self.W[f"h.{i}.ln_1.g"], self.W[f"h.{i}.ln_1.b"]), i)
            x = x + self.mlp(layernorm(x, self.W[f"h.{i}.ln_2.g"], self.W[f"h.{i}.ln_2.b"]), i)
        x = layernorm(x, self.W["ln_f.g"], self.W["ln_f.b"])
        return x @ self.W["wte"].T                    # (T, vocab) logits

    def generate(self, ids, max_new_tokens=80, temperature=0.8, top_k=40, top_p=None, seed=0):
        rng = np.random.default_rng(seed)
        ids = list(ids); block = self.cfg["block_size"]
        for _ in range(max_new_tokens):
            logits = self.forward(ids[-block:])[-1]
            logits = logits / max(temperature, 1e-8)
            if top_k:
                kth = np.sort(logits)[-min(top_k, len(logits))]
                logits = np.where(logits < kth, -np.inf, logits)
            p = softmax(logits)
            if top_p:
                order = np.argsort(p)[::-1]; cum = np.cumsum(p[order])
                keep = order[:max(1, int(np.searchsorted(cum, top_p)) + 1)]
                mask = np.zeros_like(p); mask[keep] = p[keep]; p = mask / mask.sum()
            nid = int(rng.choice(len(p), p=p))
            ids.append(nid)
            if self.tok is not None and nid == self.cfg.get("eos_token_id", -1):
                break
        return ids

    def generate_text(self, prompt, **kw):
        assert self.tok is not None, "tokenizer not loaded"
        ids = self.tok.encode(prompt).ids
        out = self.generate(ids, **kw)
        return self.tok.decode(out)


if __name__ == "__main__":
    art = os.path.join(HERE, "course_artifacts")
    m = StoryByteNumPy(art)
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Once upon a time"
    print(m.generate_text(prompt, max_new_tokens=80, temperature=0.8, top_k=40))
