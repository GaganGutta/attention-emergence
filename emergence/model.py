"""Minimal decoder-only transformer with inspectable attention.

Hand-rolled rather than nn.TransformerEncoder so that every head's
post-softmax attention pattern can be returned, logged, and later
patched between checkpoints, which is the object of study in
arXiv 2606.25010.
"""

import math

import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_mlp: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp), nn.GELU(), nn.Linear(d_mlp, d_model)
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        scores = scores.masked_fill(~causal_mask[:T, :T], float("-inf"))
        pattern = scores.softmax(dim=-1)  # (B, H, T, T)
        y = (pattern @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(y)
        x = x + self.mlp(self.ln2(x))
        return x, pattern


class TinyTransformer(nn.Module):
    """Defaults follow the paper's linear-map model: 1 layer, D=128,
    8 heads, MLP 512, binary vocabulary (Sec. 3.2)."""

    def __init__(self, vocab_size: int = 2, max_len: int = 32, d_model: int = 128,
                 n_heads: int = 8, n_layers: int = 1, d_mlp: int = 512):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(
            Block(d_model, n_heads, d_mlp) for _ in range(n_layers)
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)
        mask = torch.tril(torch.ones(max_len, max_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, tokens: torch.Tensor, return_attention: bool = False):
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device)
        x = self.tok_emb(tokens) + self.pos_emb(pos)
        patterns = []
        for block in self.blocks:
            x, pattern = block(x, self.causal_mask)
            if return_attention:
                patterns.append(pattern)
        logits = self.unembed(self.ln_f(x))
        return (logits, patterns) if return_attention else logits
