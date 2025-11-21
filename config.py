from dataclasses import dataclass
import torch

@dataclass
class ModelArgs:
    # Architecture Params
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 32 # MLA usually manages this internally, but kept for compatibility
    
    # Llama-3 Tokenizer size (approx)
    vocab_size: int = 128256 
    norm_eps: float = 1e-6
    
    # MLA (Multi-Head Latent Attention) Params
    kv_lora_rank: int = 512  
    q_lora_rank: int = 1536  
    rope_theta: float = 10000.0
    max_seq_len: int = 4096 
    
    # MoE Params
    num_experts: int = 64        
    num_shared_experts: int = 2  
    top_k: int = 6               
    expert_hidden_dim: int = 2048 
    aux_loss_coef: float = 0.01   
    
    # Training Params
    batch_size: int = 8 # Per GPU
    lr_decay_iters: int = 100000
    warmup_iters: int = 2000
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay_optim: float = 0.1
    clip: float = 1.0
    gradient_accumulation_steps: int = 8 # Increased for stable MoE training
    total_iters: int = 100000
    eval_iters: int = 500
    save_checkpoint_iter: int = 2000
    dropout: float = 0.0
    
    # System
    device: str = "cuda"
    use_ddp: bool = False
    use_liger: bool = True 
    wandb_project: str = "kimi-moe-fineweb"
    wandb_run_name: str = "run-v2-fineweb"
    
    # Dataset
    dataset: str = "HuggingFaceFW/fineweb-edu"
    hf_token: str = None # Set this if using gated models like Llama-3

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    args, _ = parser.parse_known_args()
    return args