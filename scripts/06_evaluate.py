"""
06 — Evaluate StoryByte and produce the course's display artifacts:
  course_artifacts/sample_generations.json  (fixed prompts x decoding settings)
  course_artifacts/interp_data.json         (logit-lens + attention patterns for Module 7)

Generations use the pure-NumPy reference so they match what the browser runs.
(Greedy is deterministic and reproduces exactly; sampled outputs are illustrative.)
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference_forward import StoryByteNumPy, layernorm, softmax

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(HERE, "course_artifacts")

PROMPTS = ["Once upon a time", "The cat sat on the", "One day, a little girl named Lily"]


def instrumented_forward(m, ids):
    """forward that captures per-layer residual (for logit lens) + attention patterns."""
    ids = np.asarray(ids); T = len(ids); H = m.n_head; hd = m.n_embd // H
    x = m.W["wte"][ids] + m.W["wpe"][:T]
    per_layer_resid = []; attn_patterns = []
    for i in range(m.n_layer):
        xin = layernorm(x, m.W[f"h.{i}.ln_1.g"], m.W[f"h.{i}.ln_1.b"])
        qkv = xin @ m.W[f"h.{i}.attn.c_attn.w"] + m.W[f"h.{i}.attn.c_attn.b"]
        q, k, v = np.split(qkv, 3, -1)
        q = q.reshape(T, H, hd).transpose(1, 0, 2); k = k.reshape(T, H, hd).transpose(1, 0, 2); v = v.reshape(T, H, hd).transpose(1, 0, 2)
        mask = np.triu(np.ones((T, T), np.float32), 1) * -1e10
        att = softmax((q @ k.transpose(0, 2, 1)) / np.sqrt(hd) + mask)   # (H,T,T)
        attn_patterns.append(att)
        y = (att @ v).transpose(1, 0, 2).reshape(T, m.n_embd)
        x = x + (y @ m.W[f"h.{i}.attn.c_proj.w"] + m.W[f"h.{i}.attn.c_proj.b"])
        hmid = np.maximum  # placeholder to keep linter calm
        from reference_forward import gelu
        hh = gelu(layernorm(x, m.W[f"h.{i}.ln_2.g"], m.W[f"h.{i}.ln_2.b"]) @ m.W[f"h.{i}.mlp.c_fc.w"] + m.W[f"h.{i}.mlp.c_fc.b"])
        x = x + (hh @ m.W[f"h.{i}.mlp.c_proj.w"] + m.W[f"h.{i}.mlp.c_proj.b"])
        per_layer_resid.append(x.copy())
    return per_layer_resid, attn_patterns


def logit_lens(m, ids, resids):
    """apply the final LayerNorm + unembed to each layer's residual -> top-5 next-token guess."""
    out = []
    for x in resids:
        xf = layernorm(x, m.W["ln_f.g"], m.W["ln_f.b"])
        logits = xf[-1] @ m.W["wte"].T
        p = softmax(logits); top = np.argsort(p)[::-1][:5]
        out.append([[m.tok.id_to_token(int(t)) if m.tok else int(t), round(float(p[t]), 4)] for t in top])
    return out


def main():
    m = StoryByteNumPy(ART)
    # sample generations
    samples = {}
    for pr in PROMPTS:
        ids = m.tok.encode(pr).ids
        samples[pr] = {
            "greedy": m.tok.decode(m.generate(ids, 80, temperature=1e-6, top_k=1)),
            "temp_0.8_topk_40": m.tok.decode(m.generate(ids, 80, temperature=0.8, top_k=40, seed=1)),
            "temp_1.0": m.tok.decode(m.generate(ids, 80, temperature=1.0, top_k=0, seed=2)),
            "temp_1.4": m.tok.decode(m.generate(ids, 80, temperature=1.4, top_k=0, seed=3)),
        }
    json.dump(samples, open(os.path.join(ART, "sample_generations.json"), "w"), indent=2)
    print("=== sample generations ===")
    for pr, d in samples.items():
        print(f"\n[{pr}]\n  greedy: {d['greedy']}")

    # interp data (logit lens + attention) for 2 short prompts
    interp = {"prompts": [], "logit_lens": [], "attention_patterns": []}
    for pr in ["Once upon a time", "The cat sat on the"]:
        ids = m.tok.encode(pr).ids
        toks = [m.tok.id_to_token(i) for i in ids]
        resids, atts = instrumented_forward(m, ids)
        interp["prompts"].append(pr)
        interp["logit_lens"].append({"prompt": pr, "tokens": toks, "per_layer_top5": logit_lens(m, ids, resids)})
        interp["attention_patterns"].append({"prompt": pr, "tokens": toks,
            "patterns": [{"layer": i, "head": h, "matrix": np.round(atts[i][h], 3).tolist()}
                         for i in range(m.n_layer) for h in range(m.n_head)]})
    json.dump(interp, open(os.path.join(ART, "interp_data.json"), "w"))
    print(f"\nwrote interp_data.json ({len(interp['attention_patterns'][0]['patterns'])} head-patterns/prompt)")


if __name__ == "__main__":
    main()
