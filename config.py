from dataclasses import dataclass
import torch

@dataclass
class ModelArgs:
    # Architecture Params
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 32 # For MLA, this is usually ignored or set to 1
    vocab_size: int = 102400 # Standard tokenizer size
    norm_eps: float = 1e-6
    
    # MLA (Multi-Head Latent Attention) Params
    # Compression dimension for Key/Value
    kv_lora_rank: int = 512  
    # Compression dimension for Query (optional, DeepSeek uses this)
    q_lora_rank: int = 1536  
    rope_theta: float = 10000.0
    max_seq_len: int = 4096 # Context window
    
    # MoE Params
    num_experts: int = 64        # Total routed experts
    num_shared_experts: int = 2  # Experts that always activate (DeepSeek Trick)
    top_k: int = 6               # Active routed experts
    expert_hidden_dim: int = 2048 # Usually smaller than dense MLP
    aux_loss_coef: float = 0.01   # Weight for load balancing loss
    
    # Training Params
    batch_size: int = 8
    lr_decay_iters: int = 600000
    warmup_iters: int = 2000
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay_optim: float = 0.1
    clip: float = 1.0
    gradient_accumulation_steps: int = 4
    total_iters: int = 100000
    eval_iters: int = 200
    save_checkpoint_iter: int = 2000
    dropout: float = 0.0
    
    # System
    device: str = "cuda"
    use_ddp: bool = False
    use_liger: bool = True # Fused Kernels
    metrics_plot_path: str = "training_metrics.png"
    hf_token: str = None

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    # Add basic args here if needed for CLI overrides
    args, _ = parser.parse_known_args()
    return args
