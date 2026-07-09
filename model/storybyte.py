"""
StoryByte: a tiny GPT-2-style decoder-only language model.

Deliberately classic (nanoGPT/minGPT lineage) so a from-scratch NumPy forward pass
maps to it 1:1: learned absolute positions, pre-LN, standard multi-head attention,
GELU MLP (4x), classic LayerNorm (weight+bias), weight-tied LM head.

Attention is written out explicitly (NOT F.scaled_dot_product_attention) so the
browser/NumPy reference reproduces it exactly.

References: Vaswani et al. 2017 (arXiv:1706.03762); Radford et al. 2019 (GPT-2);
Karpathy nanoGPT (github.com/karpathy/nanoGPT); GELU Hendrycks & Gimpel 2016.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 2048
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    mlp_ratio: int = 4
    dropout: float = 0.0
    bias: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        assert c.n_embd % c.n_head == 0
        self.n_head, self.n_embd = c.n_head, c.n_embd
        self.c_attn = nn.Linear(c.n_embd, 3 * c.n_embd, bias=c.bias)   # combined Q,K,V
        self.c_proj = nn.Linear(c.n_embd, c.n_embd, bias=c.bias)
        self.attn_dropout = nn.Dropout(c.dropout)
        self.resid_dropout = nn.Dropout(c.dropout)
        self.register_buffer(
            "mask", torch.tril(torch.ones(c.block_size, c.block_size)).view(1, 1, c.block_size, c.block_size)
        )

    def forward(self, x):
        B, T, C = x.shape
        H = self.n_head
        hd = C // H
        q, k, v = self.c_attn(x).split(C, dim=2)            # each (B,T,C)
        q = q.view(B, T, H, hd).transpose(1, 2)             # (B,H,T,hd)
        k = k.view(B, T, H, hd).transpose(1, 2)
        v = v.view(B, T, H, hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)     # (B,H,T,T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v                                         # (B,H,T,hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(c.n_embd, c.mlp_ratio * c.n_embd, bias=c.bias)
        self.gelu = nn.GELU(approximate="tanh")             # matches the NumPy tanh-GELU
        self.c_proj = nn.Linear(c.mlp_ratio * c.n_embd, c.n_embd, bias=c.bias)
        self.dropout = nn.Dropout(c.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(c.n_embd, bias=c.bias)
        self.attn = CausalSelfAttention(c)
        self.ln_2 = nn.LayerNorm(c.n_embd, bias=c.bias)
        self.mlp = MLP(c)

    def forward(self, x):                                   # pre-LN
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class StoryByte(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.config = c
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(c.vocab_size, c.n_embd),
            wpe=nn.Embedding(c.block_size, c.n_embd),
            drop=nn.Dropout(c.dropout),
            h=nn.ModuleList([Block(c) for _ in range(c.n_layer)]),
            ln_f=nn.LayerNorm(c.n_embd, bias=c.bias),
        ))
        self.lm_head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight   # weight tying
        self.apply(self._init)
        # GPT-2 scaled init on residual projections
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * c.n_layer))

    def _init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding_pos=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding_pos:
            n -= self.transformer.wpe.weight.numel()   # report like nanoGPT (wte is tied/counted once)
        return n

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, f"sequence length {T} > block_size {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for blk in self.transformer.h:
            x = blk(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        # inference: only the last position's logits are needed
        logits = self.lm_head(x[:, [-1], :])
        return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
