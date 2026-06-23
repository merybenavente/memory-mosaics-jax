"""Shared configuration for GPT-2 and Memory Mosaic experiments."""

from dataclasses import dataclass
import jax


@dataclass
class Config:
    # --- model (shared) ---
    vocab_size: int = 50304  # GPT-2 BPE (50257) rounded up for efficiency
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 1  # depth; vary for Fig. 7 subplots
    block_size: int = 512  # context length
    bias: bool = False
    weight_tying: bool = True
    dropout: float = 0.05

    # --- memory mosaic specific ---
    pmem_size: int = 2688  # persistent memory slots per head
    pmem_count: int = 1  # number of persistent memory banks
    v_shift: int = 1  # value peek-ahead steps

    # --- training ---
    batch_size: int = 8
    max_iters: int = 80_000
    learning_rate: float = 5e-3
    min_lr: float = 1e-4
    warmup_iters: int = 2000
    lr_decay_iters: int = 80_000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # --- evaluation / logging ---
    eval_interval: int = 1000
    eval_iters: int = 10
    log_interval: int = 100

    # --- data ---
    data_dir: str = "data/BabiStories"

    # --- system ---
    seed: int = 1337

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


# Register Config as a JAX static type so it can live inside param pytrees
# without being traversed by jax.tree operations (grad, optimizer, etc.)
jax.tree_util.register_static(Config)
