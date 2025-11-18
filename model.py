import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Try importing Liger Kernel
try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    HAS_LIGER = True
except ImportError:
    HAS_LIGER = False
    print("Liger Kernel not found. Falling back to standard CrossEntropy.")

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight

def apply_rotary_emb(xq, xk, freq_cis):
    # A simplified RoPE implementation
    # Reshape for broadcasting: [B, Seq, Heads, Dim/2, 2]
    xq_out = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_out = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freq_cis = freq_cis[:xq.shape[1]] # Slice to seq len
    
    # Rotate
    xq_out = torch.view_as_real(xq_out * freq_cis).flatten(3)
    xk_out = torch.view_as_real(xk_out * freq_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class MLA(nn.Module):
    """
    Multi-Head Latent Attention (DeepSeek V2/V3 Style)
    Compresses KV into a latent vector to reduce VRAM usage for massive contexts.
    Includes QK Norm for stability.
    """
    def __init__(self, args):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.kv_lora_rank = args.kv_lora_rank
        
        # Q Projection (Standard or Compressed)
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        
        # MLA: Compress KV into a low-rank latent vector
        self.w_kv_down = nn.Linear(args.dim, args.kv_lora_rank, bias=False)
        # Project latent up to generate Keys and Values
        self.w_kv_up = nn.Linear(args.kv_lora_rank, 2 * (args.n_heads * self.head_dim), bias=False)
        
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        
        # QK Norm (Crucial for Muon/MoE stability)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        
        # Precompute RoPE frequencies
        self.register_buffer("freqs_cis", self.precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len))

    def precompute_freqs_cis(self, dim: int, end: int, theta: float = 10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis

    def forward(self, x, mask=None):
        B, T, C = x.shape
        
        # 1. Query Generation
        xq = self.wq(x)
        xq = xq.view(B, T, self.n_heads, self.head_dim)
        
        # 2. MLA: KV Generation via Compression
        latent_kv = self.w_kv_down(x) # [B, T, kv_lora_rank]
        kv = self.w_kv_up(latent_kv)  # [B, T, 2 * n_heads * head_dim]
        kv = kv.view(B, T, 2, self.n_heads, self.head_dim)
        xk, xv = kv.unbind(2)
        
        # 3. QK Norm (Stability Trick)
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)
        
        # 4. RoPE
        xq, xk = apply_rotary_emb(xq, xk, self.freqs_cis)
        
        # 5. Attention
        # Flash Attention is highly recommended here
        out = F.scaled_dot_product_attention(
            xq.transpose(1, 2), # [B, H, T, D]
            xk.transpose(1, 2),
            xv.transpose(1, 2),
            is_causal=True
        )
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(out)

class DeepSeekMoE(nn.Module):
    """
    Shared + Routed Experts (DeepSeek V3 Style).
    Features:
    - A set of 'Shared' experts that always activate (captures common knowledge).
    - A set of 'Routed' experts selected by TopK.
    - Auxiliary Loss for load balancing.
    """
    def __init__(self, args):
        super().__init__()
        self.num_experts = args.num_experts
        self.top_k = args.top_k
        self.num_shared = args.num_shared_experts
        
        # Gating (Router)
        self.gate = nn.Linear(args.dim, args.num_experts, bias=False)
        
        # Shared Experts (MLP)
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(args.dim, args.expert_hidden_dim, bias=False),
                nn.SiLU(),
                nn.Linear(args.expert_hidden_dim, args.dim, bias=False)
            ) for _ in range(self.num_shared)
        ])
        
        # Routed Experts (MLP)
        self.routed_experts = nn.ModuleList([
             nn.Sequential(
                nn.Linear(args.dim, args.expert_hidden_dim, bias=False),
                nn.SiLU(),
                nn.Linear(args.expert_hidden_dim, args.dim, bias=False)
            ) for _ in range(self.num_experts)
        ])

    def forward(self, x):
        # x: [B, T, C]
        original_shape = x.shape
        x_flat = x.view(-1, original_shape[-1])
        
        # 1. Shared Experts Forward
        shared_out = sum(expert(x_flat) for expert in self.shared_experts)
        
        # 2. Router Forward
        logits = self.gate(x_flat) # [Tokens, Num_Experts]
        probs = F.softmax(logits, dim=-1)
        
        # Select TopK
        top_k_weights, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True) # Re-normalize
        
        # 3. Calculate Aux Loss (Load Balancing)
        # target_probs usually 1/N
        # This is a simplified Load Balancing loss
        density_1 = probs.mean(dim=0)
        density_1_proxy = logits.mean(dim=0) # Simplified proxy
        aux_loss = (density_1 * density_1_proxy).sum() * self.num_experts
        
        # 4. Routed Experts Forward
        # Naive for-loop implementation (Optimized kernels use sparse scatter/gather)
        routed_out = torch.zeros_like(x_flat)
        
        # This loop is slow in python, usually you use Triton kernels here.
        # For training script demo, we stick to masked implementation or iterating tokens
        # A simple approach for readability:
        
        # Iterate over k selected experts
        final_out = torch.zeros_like(x_flat)
        
        # Vectorized Scatter-Gather approach
        # (For production, use MegaBlocks or Triton)
        flat_indices = top_k_indices.view(-1)
        flat_weights = top_k_weights.view(-1)
        
        # Doing a very naive implementation for compatibility:
        for k in range(self.top_k):
            indices_k = top_k_indices[:, k]
            weights_k = top_k_weights[:, k]
            
            for e in range(self.num_experts):
                # Find tokens assigned to expert e at rank k
                mask = (indices_k == e)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.routed_experts[e](expert_input)
                    final_out[mask] += expert_output * weights_k[mask].unsqueeze(-1)

        total_out = shared_out + final_out
        return total_out.view(*original_shape), aux_loss

class Block(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attn_norm = RMSNorm(args.dim)
        self.attn = MLA(args) # Multi-Head Latent Attention
        self.ffn_norm = RMSNorm(args.dim)
        self.moe = DeepSeekMoE(args) # Shared+Routed MoE
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        # Attention Residual
        h = x + self.dropout(self.attn(self.attn_norm(x)))
        
        # MoE Residual
        moe_out, aux_loss = self.moe(self.ffn_norm(h))
        out = h + self.dropout(moe_out)
        return out, aux_loss

class DeepSeekV3(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding = nn.Embedding(args.vocab_size, args.dim)
        self.pos_embeddings = nn.Embedding(args.max_seq_len, args.dim) # Placeholder for simplicity
        
        self.layers = nn.ModuleList([Block(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim)
        self.linear_layer = nn.Linear(args.dim, args.vocab_size, bias=False)
        
        # Initialize Liger Loss if available
        if args.use_liger and HAS_LIGER:
            self.le_loss = LigerFusedLinearCrossEntropyLoss()
        
        # Weight tying
        self.embedding.weight = self.linear_layer.weight
        
        self.last_aux_loss = 0.0

    def forward(self, input_ids, mask=None):
        B, T = input_ids.shape
        x = self.embedding(input_ids)
        
        total_aux_loss = 0.0
        for layer in self.layers:
            x, aux_loss = layer(x)
            total_aux_loss += aux_loss
            
        x = self.norm(x)
        
        # Store aux loss for the training loop to access
        self.last_aux_loss = total_aux_loss
        
        if self.args.use_liger and self.training:
            # Return embeddings for Liger fused loss
            return x 
            
        return self.linear_layer(x)