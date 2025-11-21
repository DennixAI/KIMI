import torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import torch.distributed as dist

def initialize_tokenizer(hf_token=None):
    # Using Llama-3 tokenizer (Current standard for efficiency)
    try:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", token=hf_token)
    except Exception:
        print("Warning: Could not load Llama-3 tokenizer (might be gated). Falling back to GPT-2.")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

class FineWebStreamDataset(IterableDataset):
    def __init__(self, split, tokenizer, seq_len, batch_size, world_size=1, rank=0, infinite=True):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.split = split
        self.world_size = world_size
        self.rank = rank
        self.infinite = infinite
        
        # Load the streaming dataset
        # "sample-10BT" is a smaller subset good for testing. 
        # For full training, remove name="sample-10BT".
        self.dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu", 
            name="sample-10BT", 
            split="train", 
            streaming=True
        )
        
        # Shard the dataset for DDP so each GPU gets different data
        if self.world_size > 1:
            self.dataset = self.dataset.shard(num_shards=self.world_size, index=self.rank)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        # If we have multiple workers per GPU, we need to shard again
        if worker_info is not None:
            # We treat the dataset as an iterator and skip/interleave based on worker ID
            ds_iterator = iter(self.dataset)
            # Simple skip logic for workers
            for _ in range(worker_info.id):
                next(ds_iterator, None)
            step_size = worker_info.num_workers
        else:
            ds_iterator = iter(self.dataset)
            step_size = 1

        buffer_tokens = []
        
        while True:
            try:
                # Fetch next example
                if step_size > 1:
                    # Skip items for other workers
                    for _ in range(step_size - 1):
                        next(ds_iterator, None)
                        
                example = next(ds_iterator)
                text = example['text']
                
                # Tokenize
                tokens = self.tokenizer.encode(text, add_special_tokens=True)
                buffer_tokens.extend(tokens)
                
                # Yield chunks
                while len(buffer_tokens) >= self.seq_len + 1:
                    chunk = buffer_tokens[:self.seq_len + 1]
                    buffer_tokens = buffer_tokens[self.seq_len + 1:]
                    
                    x = torch.tensor(chunk[:-1], dtype=torch.long)
                    y = torch.tensor(chunk[1:], dtype=torch.long)
                    
                    yield {'input_ids': x, 'labels': y}
                    
            except StopIteration:
                if self.infinite:
                    # Restart iterator
                    ds_iterator = iter(self.dataset)
                else:
                    break

def prepare_dataset(split, device, batch_size, use_ddp=False):
    # DDP Setup
    if use_ddp:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
        
    args = from config import ModelArgs
    tokenizer = initialize_tokenizer(ModelArgs.hf_token)
    
    # We use 'train' split for everything in streaming usually, 
    # but strictly you should reserve a shard for validation.
    # Here we just use the main stream.
    ds = FineWebStreamDataset(
        split="train",
        tokenizer=tokenizer, 
        seq_len=ModelArgs.max_seq_len,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        infinite=(split == 'train')
    )
    
    # Num_workers > 0 ensures data loading is async on CPU
    dataloader = DataLoader(ds, batch_size=batch_size, num_workers=2, pin_memory=True)
    return dataloader