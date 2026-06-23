"""Smoke tests for GPT-2 and Memory Mosaic implementations.

Test 1: Forward pass — shapes, no NaN, valid logits.
Test 2: One training step — gradient flows, loss decreases, params update.
"""

import jax
import jax.numpy as jnp
import optax

from config import Config
from train import cross_entropy_loss, make_train_step

# Small config for fast tests
CFG = Config(n_layer=1, block_size=32, vocab_size=256, n_embd=64, n_head=4, batch_size=2)


def _test_forward(init_params, forward, model_name):
    rng = jax.random.key(0)
    rng, init_rng = jax.random.split(rng)
    params = init_params(CFG, init_rng)

    x = jax.random.randint(rng, (2, 32), 0, 256)
    logits = forward(params, x)

    assert logits.shape == (2, 32, 256), f"{model_name}: expected (2,32,256), got {logits.shape}"
    assert not jnp.any(jnp.isnan(logits)), f"{model_name}: logits contain NaN"
    assert not jnp.any(jnp.isinf(logits)), f"{model_name}: logits contain Inf"
    print(f"  {model_name} forward: OK  shape={logits.shape}")


def _test_train_step(init_params, forward, model_name):
    rng = jax.random.key(1)
    rng, init_rng, step_rng1, step_rng2 = jax.random.split(rng, 4)
    params = init_params(CFG, init_rng)

    x = jax.random.randint(rng, (2, 32), 0, 256)
    y = jax.random.randint(rng, (2, 32), 0, 256)

    # compute loss before
    logits_before = forward(params, x)
    loss_before = cross_entropy_loss(logits_before, y)
    assert not jnp.isnan(loss_before), f"{model_name}: loss_before is NaN"

    # one training step
    train_step = make_train_step(forward)
    loss, grads = train_step(params, None, x, y, step_rng1)

    # verify gradients exist and are finite
    grad_leaves = jax.tree.leaves(grads)
    assert all(not jnp.any(jnp.isnan(g)) for g in grad_leaves), f"{model_name}: gradients contain NaN"

    # apply update
    optimizer = optax.adamw(learning_rate=1e-3)
    opt_state = optimizer.init(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params_after = optax.apply_updates(params, updates)

    # compute loss after
    logits_after = forward(params_after, x)
    loss_after = cross_entropy_loss(logits_after, y)
    assert not jnp.isnan(loss_after), f"{model_name}: loss_after is NaN"
    assert float(loss_after) < float(loss_before), (
        f"{model_name}: loss did not decrease ({float(loss_before):.4f} -> {float(loss_after):.4f})"
    )
    print(f"  {model_name} train:   OK  loss {float(loss_before):.4f} -> {float(loss_after):.4f}")


if __name__ == "__main__":
    from gpt2 import init_params as gpt2_init, forward as gpt2_forward
    from memory_mosaic import init_params as mm_init, forward as mm_forward

    print("Test 1: Forward pass")
    _test_forward(gpt2_init, gpt2_forward, "GPT-2")
    _test_forward(mm_init, mm_forward, "Mosaic")

    print("Test 2: Training step")
    _test_train_step(gpt2_init, gpt2_forward, "GPT-2")
    _test_train_step(mm_init, mm_forward, "Mosaic")

    print("\nAll smoke tests passed!")
