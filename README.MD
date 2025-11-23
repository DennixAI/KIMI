##1. Multi-Head Latent Attention (MLA):

Instead of standard GQA/MHA, this model uses MLA (DeepSeek-V2/V3 style).

- Technique: Compresses Key-Value heads into a low-rank latent vector (kv_lora_rank) during projection and up-projects only for attention computation.
- Benefit: Drastically reduces KV Cache memory footprint (by ~5-10x). This allows training 128k context windows on hardware that usually chokes on 8k.
- Stability: Includes QK Norm (RMSNorm applied to Queries/Keys before RoPE), ensuring attention logits don't saturate during Muon optimization.

DeepSeekMoE (Shared + Routed Experts)

- Shared Experts: A dedicated set of experts that always activate. This preserves "common knowledge" (grammar, syntax) and prevents the router from having to - - relearn basics.
- Routed Experts: Standard Top-K selection for specialized reasoning.
- Auxiliary Loss: Implemented a load-balancing loss (aux_loss_coef=0.01) to prevent expert collapse, ensuring all experts are utilized.

The "Speedrun" Optimization 

1. Muon Optimizer (Momentum Orthogonal)

Technique: Orthogonal updates for internal 2D matrices (Projections/Experts) via Muon, while keeping AdamW for Embeddings and RMSNorms.
- Gain: typically 1.5x - 2x faster convergence per step compared to AdamW on Transformers.
- Grouping: Strict parameter grouping ensures weight decay is disabled for Norms/Biases to maximize feature propagation.

2. Liger Kernel (Fused Cross Entropy)
- Technique: Uses LigerFusedLinearCrossEntropyLoss.
- Gain: Fuses the massive final linear projection (D→128k) and the CrossEntropy calculation into a single kernel. This prevents materializing the massive [Batch, Seq, Vocab] logit tensor, saving GBs of VRAM.

3. Vectorized Expert Routing
- Technique: Replaced standard Python for loops in MoE routing with masked tensor operations and index_add_.
- Gain: Eliminates thousands of tiny CUDA kernel launches per step. Instead of launching  Layers x TopK kernels, we rely on vectorized scatter/gather operations, significantly improving training throughput.

4. DDP no_sync Accumulation
- Technique: During Gradient Accumulation steps (e.g., 1-7 of 8), we wrap the backward pass in model.no_sync().
- Gain: Prevents DDP from triggering All-Reduce network communication until the final micro-step, reducing multi-GPU communication overhead by ~87%.
