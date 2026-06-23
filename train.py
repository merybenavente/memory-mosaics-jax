"""Shared training loop for GPT-2 and Memory Mosaic.

Usage:
    python train.py --model gpt2    --n_layer 1
    python train.py --model mosaic  --n_layer 1
"""

import argparse
import json
import math
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from config import Config
from data import prepare_data, load_split, get_batch


# ---------------------------------------------------------------------------
# Learning-rate schedule (cosine with linear warmup)
# ---------------------------------------------------------------------------

def cosine_lr(step: int, cfg: Config) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < cfg.warmup_iters:
        return cfg.learning_rate * step / cfg.warmup_iters
    if step > cfg.lr_decay_iters:
        return cfg.min_lr
    progress = (step - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
    return cfg.min_lr + 0.5 * (cfg.learning_rate - cfg.min_lr) * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Loss function (shared — both models output logits)
# ---------------------------------------------------------------------------

def cross_entropy_loss(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Mean cross-entropy over all positions.

    Args:
        logits:  (B, T, V)
        targets: (B, T)    integer token ids
    """
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    # gather the log-prob of the correct token at each position
    targets_one_hot = jax.nn.one_hot(targets, logits.shape[-1])
    return -(targets_one_hot * log_probs).sum(axis=-1).mean()


# ---------------------------------------------------------------------------
# Train step (jitted)
# ---------------------------------------------------------------------------

def make_train_step(forward_fn):
    """Create a jitted train step for a given forward function.

    Args:
        forward_fn: (params, x, deterministic) -> logits  (B, T, V)
    """

    @jax.jit
    def train_step(params, opt_state, x, y, rng):
        def loss_fn(p):
            logits = forward_fn(p, x, deterministic=False, rng=rng)
            return cross_entropy_loss(logits, y)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        return loss, grads

    return train_step


# ---------------------------------------------------------------------------
# Eval step (jitted)
# ---------------------------------------------------------------------------

def make_eval_step(forward_fn):
    @jax.jit
    def eval_step(params, x, y):
        logits = forward_fn(params, x, deterministic=True, rng=None)
        return cross_entropy_loss(logits, y)
    return eval_step


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def estimate_loss(forward_fn, params, cfg, data_splits, rng):
    """Estimate train and val loss over cfg.eval_iters batches."""
    eval_step = make_eval_step(forward_fn)
    results = {}
    for split_name, data in data_splits.items():
        losses = []
        for _ in range(cfg.eval_iters):
            rng, sub = jax.random.split(rng)
            x, y = get_batch(cfg, data, sub)
            loss = eval_step(params, x, y)
            losses.append(float(loss))
        results[split_name] = np.mean(losses)
    return results


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: Config, model_type: str):
    # lazy import to avoid circular deps
    if model_type == "gpt2":
        from gpt2 import init_params, forward
    elif model_type == "mosaic":
        from memory_mosaic import init_params, forward
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # output directory
    out_dir = os.path.join("results", f"{model_type}_L{cfg.n_layer}")
    os.makedirs(out_dir, exist_ok=True)

    # data
    prepare_data(cfg)
    train_data = load_split(cfg, "train")
    val_data = load_split(cfg, "val")
    data_splits = {"train": train_data, "val": val_data}

    # init model
    rng = jax.random.key(cfg.seed)
    rng, init_rng = jax.random.split(rng)
    params = init_params(cfg, init_rng)

    n_params = sum(p.size for p in jax.tree.leaves(params))
    print(f"Model: {model_type} | n_layer={cfg.n_layer} | {n_params:,} parameters")

    # optimizer: AdamW with cosine LR schedule
    # Weight decay only on 2D+ weight matrices (not layer norms, scalar scales, etc.)
    # This matches the reference: leave_out = x.dim() < 2 or x.shape[-2:] == (1,1)
    def lr_schedule(step):
        return cosine_lr(step, cfg)

    def _should_decay(x):
        """True for 2D+ weight matrices, False for scalars/1D params."""
        return hasattr(x, 'ndim') and x.ndim >= 2

    decay_mask = jax.tree.map(_should_decay, params)

    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.masked(
            optax.adamw(
                learning_rate=lr_schedule,
                b1=cfg.beta1,
                b2=cfg.beta2,
                weight_decay=cfg.weight_decay,
            ),
            decay_mask,
        ),
        optax.masked(
            optax.adam(
                learning_rate=lr_schedule,
                b1=cfg.beta1,
                b2=cfg.beta2,
            ),
            jax.tree.map(lambda x: not x, decay_mask),
        ),
    )
    opt_state = optimizer.init(params)

    # training step with the right forward fn
    train_step = make_train_step(forward)

    # log storage
    log = []

    print(f"Training for {cfg.max_iters} iters ...")
    t0 = time.time()

    for step in range(cfg.max_iters):
        # --- evaluation ---
        if step % cfg.eval_interval == 0 or step == cfg.max_iters - 1:
            rng, eval_rng = jax.random.split(rng)
            losses = estimate_loss(forward, params, cfg, data_splits, eval_rng)
            entry = {"step": step, "train_loss": losses["train"], "val_loss": losses["val"]}
            log.append(entry)
            print(
                f"  step {step:>6d} | "
                f"train {losses['train']:.4f} | val {losses['val']:.4f}"
            )
            # save log incrementally
            with open(os.path.join(out_dir, "log.json"), "w") as f:
                json.dump(log, f, indent=2)

        # --- train step ---
        rng, step_rng, batch_rng = jax.random.split(rng, 3)
        x, y = get_batch(cfg, train_data, batch_rng)
        loss, grads = train_step(params, opt_state, x, y, step_rng)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        if step % cfg.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            lr = cosine_lr(step, cfg)
            print(f"  step {step:>6d} | loss {float(loss):.4f} | lr {lr:.2e} | {dt:.1f}s")

    print("Training complete.")
    return params, log


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["gpt2", "mosaic"])
    parser.add_argument("--n_layer", type=int, default=1)
    parser.add_argument("--max_iters", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    args = parser.parse_args()

    cfg = Config(n_layer=args.n_layer)
    if args.max_iters is not None:
        cfg.max_iters = args.max_iters
        cfg.lr_decay_iters = args.max_iters
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate

    train(cfg, args.model)


if __name__ == "__main__":
    main()
