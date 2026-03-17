"""
AraStudy Transformer Model
A compact, efficient decoder-only transformer for Arabic language modeling.
Designed for ~30M parameters.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int | None = None  # For GQA if needed (defaults to n_heads)
    vocab_size: int = 32000
    multiple_of: int = 256  # For MLP hidden dim
    norm_eps: float = 1e-5
    max_seq_len: int = 1024
    dropout: float = 0.1


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        # Ensure hidden_dim is multiple of args.multiple_of
        hidden_dim = args.multiple_of * (
            (hidden_dim + args.multiple_of - 1) // args.multiple_of
        )

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(args.dim, hidden_dim, bias=False)  # Gate
        self.w3 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x, mask=None):
        B, SeqLen, Dim = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Reshape for heads
        xq = xq.view(B, SeqLen, self.n_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(B, SeqLen, self.n_kv_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(B, SeqLen, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Repeat KV heads if using GQA (simplified here, assuming standard attention for now)
        # If n_kv_heads != n_heads, we would repeat k/v here.

        # Scaled Dot-Product Attention (using PyTorch's optimized implementation)
        # is_causal=True handles the masking automatically
        output = F.scaled_dot_product_attention(
            xq,
            xk,
            xv,
            attn_mask=mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True if mask is None else False,
        )

        output = output.transpose(1, 2).contiguous().view(B, SeqLen, Dim)
        return self.dropout(self.wo(output))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x):
        h = x + self.attention(self.attention_norm(x))
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class AraStudyTransformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.pos_embeddings = nn.Embedding(args.max_seq_len, args.dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(args) for _ in range(args.n_layers)]
        )
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # Weight tying
        self.tok_embeddings.weight = self.output.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens, targets=None):
        _, seq_len = tokens.shape
        if seq_len > self.args.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.args.max_seq_len}"
            )

        positions = torch.arange(seq_len, device=tokens.device)
        h = self.tok_embeddings(tokens) + self.pos_embeddings(positions)

        for layer in self.layers:
            h = layer(h)

        h = self.norm(h)
        logits = self.output(h)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

        return logits, loss

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Estimate Model Flops Utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # Only an estimate
        N = sum(p.numel() for p in self.parameters())
        cfg = self.args
        L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.dim // cfg.n_heads, cfg.max_seq_len
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter

        # FLOPS for RTX 4050 (Approximate peak FP16 TFLOPS ~ 9-11 TFLOPS)
        # Adjust based on actual hardware
        peak_flops = 9e12
        mfu = flops_per_iter / dt / peak_flops
        return mfu
