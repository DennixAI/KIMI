import time
import torch
import torch.nn.functional as F
import math
import tqdm 
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from muon import MuonWithAuxAdam
from config import ModelArgs, get_args
from model import DeepSeekV3
from data import prepare_dataset, initialize_tokenizer
from inference import topk_sampling, save_text


def save_training_plot(metrics, save_path):
    if metrics is None or not metrics["steps"]:
        return

    steps = metrics["steps"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(steps, metrics["train_loss"], label="Train Loss", color="tab:blue")
    axes[0].plot(steps, metrics["aux_loss"], label="Aux Loss", color="tab:orange")
    axes[0].set_ylabel("Loss")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.3)

    axes[1].plot(steps, metrics["lr"], label="Learning Rate", color="tab:green")
    axes[1].set_ylabel("Learning Rate")
    axes[1].set_xlabel("Step")
    axes[1].grid(True, linestyle="--", alpha=0.3)

    ax2 = axes[1].twinx()
    ax2.plot(steps, metrics["tokens"], label="Tokens", color="tab:red", alpha=0.5)
    ax2.set_ylabel("Tokens")

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

def setup_ddp():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    return local_rank, world_size, rank, device

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def get_lr(it, model_args):
    if it < model_args.warmup_iters:
        return model_args.max_lr * (it + 1) / (model_args.warmup_iters + 1)
    if it > model_args.lr_decay_iters:
        return model_args.min_lr
    decay_ratio = (it - model_args.warmup_iters) / (model_args.lr_decay_iters - model_args.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
    return model_args.min_lr + coeff * (model_args.max_lr - model_args.min_lr)

def train():
    args = get_args()
    model_args = ModelArgs() # Load defaults, then override if needed
    
    tokenizer = initialize_tokenizer(model_args.hf_token)
    
    use_ddp = 'RANK' in os.environ or model_args.use_ddp
    if use_ddp:
        local_rank, world_size, rank, device = setup_ddp()
    else:
        device = torch.device(model_args.device)
        rank = 0
        world_size = 1
        local_rank = 0
        
    metrics_history = None
    if rank == 0:
        print(f"Training DeepSeek-V3-Like Model on {device}")
        metrics_history = {
            "steps": [],
            "train_loss": [],
            "aux_loss": [],
            "lr": [],
            "tokens": []
        }
    
    model = DeepSeekV3(model_args).to(device)
    
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    
    base_model = model.module if use_ddp else model
    
    # --- OPTIMIZER GROUPING (CRITICAL FOR MUON) ---
    # 1. Filter params that are 2D (internal layers) for Muon
    # 2. Biases and Norms get AdamW with 0 weight decay
    # 3. Embeddings/Head get AdamW with weight decay
    
    hidden_weights = []
    norm_bias_params = []
    non_hidden_params = []
    
    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Embedding and Head -> AdamW standard
        if "embedding" in name or "linear_layer" in name:
            non_hidden_params.append(param)
        # LayerNorm, RMSNorm, Biases -> AdamW (No Decay)
        elif param.ndim < 2 or "norm" in name or "bias" in name:
            norm_bias_params.append(param)
        # Internal Projections (Attn, MLP, Experts) -> Muon
        else:
            hidden_weights.append(param)

    param_groups = [
        # Muon Group (High LR, Orthogonal Updates)
        {'params': hidden_weights, 'use_muon': True, 'lr': 0.02, 'weight_decay': 0.01},
        # Standard AdamW Group
        {'params': non_hidden_params, 'use_muon': False, 'lr': model_args.max_lr, 'weight_decay': model_args.weight_decay_optim},
        # No Decay Group
        {'params': norm_bias_params, 'use_muon': False, 'lr': model_args.max_lr, 'weight_decay': 0.0}
    ]
    
    optimizer = MuonWithAuxAdam(param_groups)
    
    # Compile usually helps with MLA/RoPE
    model = torch.compile(model)
    
    dataloader = prepare_dataset('train', device, model_args.batch_size, use_ddp=use_ddp)
    train_dataloader = iter(dataloader)
    
    val_dataloader = prepare_dataset('val', device, model_args.batch_size, use_ddp=use_ddp)
    
    token_count = 0
    
    # Helper for Liger
    def compute_loss(output, targets, model_ref):
        if model_args.use_liger and hasattr(model_ref, 'le_loss'):
            # Output is strictly the decoder output (hidden state), not logits
            # We fuse the linear head + cross entropy here
            # Reshape: [B*T, D]
            decoder_out_flat = output.contiguous().view(-1, model_args.dim)
            targets_flat = targets.contiguous().view(-1)
            return model_ref.le_loss(model_ref.linear_layer.weight, decoder_out_flat, targets_flat)
        else:
            # Output is logits: [B, T, Vocab]
            logits_flat = output.contiguous().view(-1, model_args.vocab_size)
            targets_flat = targets.contiguous().view(-1)
            return F.cross_entropy(logits_flat, targets_flat, ignore_index=tokenizer.pad_token_id)

    model.train()
    
    if rank == 0:
        pbar = tqdm.tqdm(range(model_args.total_iters))
    
    for step in range(model_args.total_iters):
        lr = get_lr(step, model_args)
        # Update LR for Adam groups (indices 1 and 2)
        # Muon (index 0) typically keeps constant LR, but can schedule if supported
        for i in range(1, 3):
            optimizer.param_groups[i]['lr'] = lr
            
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        
        for micro_step in range(model_args.gradient_accumulation_steps):
            try:
                batch = next(train_dataloader)
            except StopIteration:
                train_dataloader = iter(dataloader)
                batch = next(train_dataloader)
                
            idx = batch['input_ids'].to(device)
            targets = batch['labels'].to(device)
            token_count += idx.numel()
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output = model(idx)
                
                # Access the underlying model to get aux_loss and reference for liger
                ref_model = model.module if use_ddp else model
                
                # 1. Main Task Loss
                cls_loss = compute_loss(output, targets, ref_model)
                
                # 2. Auxiliary Load Balancing Loss (Critical for MoE)
                aux_loss = ref_model.last_aux_loss
                
                # Total Loss
                loss = cls_loss + (model_args.aux_loss_coef * aux_loss)
                
            # Scale loss
            loss = loss / model_args.gradient_accumulation_steps
            loss.backward()
            accumulated_loss += loss.item()
        
        # Sync loss for logging
        if use_ddp:
            loss_tensor = torch.tensor(accumulated_loss, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            accumulated_loss = loss_tensor.item()
            
        # Clip Gradients
        if model_args.clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), model_args.clip)
            
        optimizer.step()
        
        if rank == 0:
            pbar.update(1)
            pbar.set_description(f"Loss: {accumulated_loss:.4f}")
            
            aux_loss_value = aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
            if metrics_history is not None:
                metrics_history["steps"].append(step)
                metrics_history["train_loss"].append(accumulated_loss)
                metrics_history["aux_loss"].append(aux_loss_value)
                metrics_history["lr"].append(lr)
                metrics_history["tokens"].append(token_count)
                save_training_plot(metrics_history, model_args.metrics_plot_path)
            
            # Evaluation
            if step % model_args.eval_iters == 0 and step > 0:
                print(f"\nEvaluating at step {step}...")
                # Add eval logic here (call estimate_loss)
            
            # Checkpointing
            if step % model_args.save_checkpoint_iter == 0 and step > 0:
                ckpt_path = f"checkpoint_{step}.pt"
                torch.save({
                    'model': base_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'args': model_args,
                    'step': step
                }, ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")

    if use_ddp:
        cleanup_ddp()

if __name__ == "__main__":
    train()
