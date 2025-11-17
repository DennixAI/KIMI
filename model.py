# model.py
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import RMSNorm

from tokenizer import Tokenizer


# -------------------------
# Config
# -------------------------

@dataclass
class ModelConfig:
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 8

    # latent attention specific
    latent_dim: int = 64
    attn_dropout: float = 0.0
    dropout: float = 0.0

    block_size: int = 128

    # filled from tokenizer
    vocab_size: int = 0
    pad_id: int = 0
    bos_id: Optional[int] = None
    eos_id: Optional[int] = None



class TokenPositionalEmbedding(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.vocab_size > 0, "vocab_size must be set in ModelConfig"

        self.block_size = cfg.block_size
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.block_size, cfg.d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [B, T] int64
        returns:   [B, T, d_model]
        """
        bsz, seqlen = input_ids.shape
        device = input_ids.device

        if seqlen > self.block_size:
            raise ValueError(
                f"Sequence length {seqlen} > block_size {self.block_size}"
            )

        tok = self.token_embed(input_ids)  # [B, T, d_model]

        pos_ids = torch.arange(seqlen, device=device).unsqueeze(0)  # [1, T]
        pos = self.pos_embed(pos_ids)                               # [1, T, d_model]

        return tok + pos




class LatentAttention(nn.Module):
    """
    Multi-head latent attention:

    - compresses hidden states into a lower-dimensional "latent" space
      before building keys/values (W_d -> W_k, W_v),
    - keeps queries in the full d_model space (W_q),
    - then uses standard scaled dot-product attention per head.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.latent_dim = cfg.latent_dim

        # project to latent space, then back to full space for K/V
        self.w_d = nn.Linear(self.d_model, self.latent_dim, bias=False)
        self.w_k = nn.Linear(self.latent_dim, self.d_model, bias=False)
        self.w_v = nn.Linear(self.latent_dim, self.d_model, bias=False)

        # queries stay in full d_model space
        self.w_q = nn.Linear(self.d_model, self.d_model, bias=False)

        self.dropout = nn.Dropout(cfg.attn_dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,  # [B, T] with 1 for valid tokens
    ) -> torch.Tensor:
        """
        x:    [B, T, d_model]
        mask: [B, T] or None

        returns: [B, T, d_model]
        """
        B, T, _ = x.shape
        device = x.device

        # 1) build queries
        q = self.w_q(x)  # [B, T, d_model]

        # 2) compress to latent space and reconstruct K,V
        latent = self.w_d(x)          # [B, T, latent_dim]
        k_full = self.w_k(latent)     # [B, T, d_model]
        v_full = self.w_v(latent)     # [B, T, d_model]

        # 3) reshape into heads
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            # [B, T, d_model] -> [B, n_heads, T, head_dim]
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = split_heads(q)      # [B, H, T, Hd]
        k = split_heads(k_full) # [B, H, T, Hd]
        v = split_heads(v_full) # [B, H, T, Hd]

        # 4) scaled dot-product attention with causal mask
        attn_scores = q @ k.transpose(-2, -1)  # [B, H, T, T]
        attn_scores = attn_scores / math.sqrt(self.head_dim)

        # causal mask: only attend to previous / current positions
        causal = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
        attn_scores = attn_scores.masked_fill(~causal, float("-inf"))

        # optional key padding mask
        if mask is not None:
            # mask: [B, T] with 1 for real tokens, 0 for pad
            key_mask = mask[:, None, None, :].bool()  # [B, 1, 1, T]
            attn_scores = attn_scores.masked_fill(~key_mask, float("-inf"))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        out = attn_probs @ v  # [B, H, T, Hd]

        # 5) merge heads back
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return out



class MHLA(nn.Module):
    """
    Multi-Head Latent Attention block:
    latent attention + output projection + dropout.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn = LatentAttention(cfg)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x:    [B, T, d_model]
        mask: [B, T] or None
        """
        h = self.attn(x, mask=mask)     # [B, T, d_model]
        h = self.out_proj(h)            # [B, T, d_model]
        h = self.dropout(h)
        return h


class DecoderBlock(nn.Module):
    """
    Single decoder block with:

        x = x + MHLA(LN(x))

    ADD MoE / FFN 
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm_attn = RMSNorm(cfg.d_model)
        self.attn = MHLA(cfg)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x:    [B, T, d_model]
        mask: [B, T] or None
        """
        h = self.norm_attn(x)
        h = self.attn(h, mask=mask)
        return x + h  # residual

class TinyLatentLM(nn.Module):
    """
    Decoder-only language model with:

    - token + positional embeddings
    - N layers of (pre-norm + MHLA + residual)
    - final RMSNorm + LM head

    MoE / FFN will be added later.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = TokenPositionalEmbedding(cfg)
        self.layers = nn.ModuleList(DecoderBlock(cfg) for _ in range(cfg.n_layers))
        self.norm_out = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # weight tying
        self.lm_head.weight = self.embed.token_embed.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        input_ids: [B, T] int64
        mask:      [B, T] or None
        returns:   [B, T, vocab_size] logits
        """
        x = self.embed(input_ids)  # [B, T, d_model]

        for layer in self.layers:
            x = layer(x, mask=mask)

        x = self.norm_out(x)
        logits = self.lm_head(x)  # [B, T, V]
        return logits



def build_model_and_tokenizer() -> Tuple[TinyLatentLM, ModelConfig, object]:
    # 1) load tokenizer
    tok = Tokenizer().get()

    # 2) config with tokenizer-dependent fields
    cfg = ModelConfig()
    cfg.vocab_size = len(tok)
    cfg.pad_id = tok.pad_token_id
    cfg.bos_id = getattr(tok, "bos_token_id", None)
    cfg.eos_id = getattr(tok, "eos_token_id", None)

    # 3) build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyLatentLM(cfg).to(device)

    return model, cfg, tok
