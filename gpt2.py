"""GPT-2 in pure JAX — functional, no nn.Module abstractions.

Matches the GPT2-small architecture from the Memory Mosaics paper (Fig. 6 left):
  - token + position embeddings
  - N_b repeated blocks of [LayerNorm → CausalSelfAttention → + residual,
                             LayerNorm → MLP (4x GELU) → + residual]
  - final LayerNorm → linear head (weight-tied with token embedding)

Parameters are plain pytrees (nested dicts). All projections use jnp.einsum.
"""

import jax
import jax.numpy as jnp
import math

from config import Config


# ---------------------------------------------------------------------------
# Layer norm (no bias, matching config.bias=False)
# ---------------------------------------------------------------------------

def layer_norm(x: jnp.ndarray, weight: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return weight * (x - mean) / jnp.sqrt(var + eps)


# ---------------------------------------------------------------------------
# Causal self-attention
# ---------------------------------------------------------------------------

def causal_self_attention(
    x: jnp.ndarray,
    params: dict,
    cfg: Config,
    deterministic: bool,
    rng: jax.Array | None,
) -> jnp.ndarray:
    """Multi-head causal self-attention.

    Args:
        x: (B, T, C)
        params: dict with keys qkv_w, out_w
    Returns:
        (B, T, C)
    """
    B, T, C = x.shape
    head_dim = cfg.head_dim

    # joint QKV projection: (B, T, C) @ (C, 3, H, D) -> (B, T, 3, H, D)
    qkv = jnp.einsum("btc,czhd->btzhd", x, params["qkv_w"])
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # each (B, T, H, D)

    # attention scores: (B, T, H, D) @ (B, S, H, D) -> (B, H, T, S)
    scale = 1.0 / jnp.sqrt(jnp.float32(head_dim))
    attn = jnp.einsum("bthd,bshd->bhts", q, k) * scale

    # causal mask
    mask = jnp.tril(jnp.ones((T, T)))
    attn = jnp.where(mask[None, None, :, :], attn, jnp.finfo(attn.dtype).min)
    attn = jax.nn.softmax(attn, axis=-1)

    # attention dropout
    if not deterministic and cfg.dropout > 0:
        drop_rng, rng = jax.random.split(rng)
        keep = jax.random.bernoulli(drop_rng, 1.0 - cfg.dropout, attn.shape)
        attn = jnp.where(keep, attn / (1.0 - cfg.dropout), 0.0)

    # weighted sum: (B, H, T, S) @ (B, S, H, D) -> (B, T, H, D)
    out = jnp.einsum("bhts,bshd->bthd", attn, v)

    # output projection: (B, T, H, D) @ (H, D, C) -> (B, T, C)
    out = jnp.einsum("bthd,hdc->btc", out, params["out_w"])

    # residual dropout
    if not deterministic and cfg.dropout > 0:
        drop_rng, rng = jax.random.split(rng)
        keep = jax.random.bernoulli(drop_rng, 1.0 - cfg.dropout, out.shape)
        out = jnp.where(keep, out / (1.0 - cfg.dropout), 0.0)

    return out


# ---------------------------------------------------------------------------
# MLP (feed-forward network)
# ---------------------------------------------------------------------------

def mlp(
    x: jnp.ndarray,
    params: dict,
    cfg: Config,
    deterministic: bool,
    rng: jax.Array | None,
) -> jnp.ndarray:
    """Two-layer MLP with GELU activation and 4x expansion.

    Args:
        x: (B, T, C)
        params: dict with keys fc_w, proj_w
    Returns:
        (B, T, C)
    """
    # up-project: (B, T, C) @ (C, 4C) -> (B, T, 4C)
    h = jnp.einsum("btc,cd->btd", x, params["fc_w"])
    h = jax.nn.gelu(h)
    # down-project: (B, T, 4C) @ (4C, C) -> (B, T, C)
    out = jnp.einsum("btd,dc->btc", h, params["proj_w"])

    if not deterministic and cfg.dropout > 0:
        drop_rng, rng = jax.random.split(rng)
        keep = jax.random.bernoulli(drop_rng, 1.0 - cfg.dropout, out.shape)
        out = jnp.where(keep, out / (1.0 - cfg.dropout), 0.0)

    return out


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

def block_forward(
    x: jnp.ndarray,
    params: dict,
    cfg: Config,
    deterministic: bool,
    rng: jax.Array | None,
) -> jnp.ndarray:
    """Pre-norm transformer block: LN → Attn → +residual, LN → MLP → +residual."""
    if rng is not None:
        attn_rng, mlp_rng = jax.random.split(rng)
    else:
        attn_rng = mlp_rng = None

    # self-attention sub-block
    h = layer_norm(x, params["ln1_w"])
    h = causal_self_attention(h, params["attn"], cfg, deterministic, attn_rng)
    x = x + h

    # MLP sub-block
    h = layer_norm(x, params["ln2_w"])
    h = mlp(h, params["mlp"], cfg, deterministic, mlp_rng)
    x = x + h

    return x


# ---------------------------------------------------------------------------
# Full forward pass
# ---------------------------------------------------------------------------

def forward(
    params: dict,
    x: jnp.ndarray,
    deterministic: bool = True,
    rng: jax.Array | None = None,
) -> jnp.ndarray:
    """GPT-2 forward pass.

    Args:
        params: pytree of parameters
        x: (B, T) integer token ids
        deterministic: if False, apply dropout
        rng: PRNG key (needed when deterministic=False)
    Returns:
        logits: (B, T, V)
    """
    cfg = params["_cfg"]
    B, T = x.shape

    # token + position embeddings
    tok_emb = params["wte"][x]                              # (B, T, C)
    pos_emb = params["wpe"][jnp.arange(T)]                  # (T, C)
    h = tok_emb + pos_emb

    # embedding dropout
    if not deterministic and cfg.dropout > 0:
        rng, drop_rng = jax.random.split(rng)
        keep = jax.random.bernoulli(drop_rng, 1.0 - cfg.dropout, h.shape)
        h = jnp.where(keep, h / (1.0 - cfg.dropout), 0.0)

    # transformer blocks
    for i in range(cfg.n_layer):
        if rng is not None:
            rng, block_rng = jax.random.split(rng)
        else:
            block_rng = None
        h = block_forward(h, params["blocks"][i], cfg, deterministic, block_rng)

    # final layer norm
    h = layer_norm(h, params["ln_f_w"])

    # language model head (weight-tied with token embedding)
    if cfg.weight_tying:
        logits = jnp.einsum("btc,vc->btv", h, params["wte"])
    else:
        logits = jnp.einsum("btc,cv->btv", h, params["lm_head_w"])

    return logits


# ---------------------------------------------------------------------------
# Parameter initialization
# ---------------------------------------------------------------------------

def init_params(cfg: Config, rng: jax.Array) -> dict:
    """Initialize GPT-2 parameters following the standard scheme:
    - normal(0, 0.02) for all weights
    - residual projections scaled by 1/sqrt(2*n_layer)
    - zeros for layer norm weights (then +1)
    """
    C = cfg.n_embd
    H = cfg.n_head
    D = cfg.head_dim
    V = cfg.vocab_size
    T = cfg.block_size

    std = 0.02
    res_std = std / math.sqrt(2 * cfg.n_layer)

    def normal(key, shape, s=std):
        return jax.random.normal(key, shape) * s

    keys = iter(jax.random.split(rng, 100))

    # embeddings
    wte = normal(next(keys), (V, C))
    wpe = normal(next(keys), (T, C))

    # transformer blocks
    blocks = []
    for _ in range(cfg.n_layer):
        block = {
            "ln1_w": jnp.ones(C),
            "attn": {
                "qkv_w": normal(next(keys), (C, 3, H, D)),
                "out_w": normal(next(keys), (H, D, C), res_std),
            },
            "ln2_w": jnp.ones(C),
            "mlp": {
                "fc_w": normal(next(keys), (C, 4 * C)),
                "proj_w": normal(next(keys), (4 * C, C), res_std),
            },
        }
        blocks.append(block)

    params = {
        "_cfg": cfg,
        "wte": wte,
        "wpe": wpe,
        "blocks": blocks,
        "ln_f_w": jnp.ones(C),
    }

    if not cfg.weight_tying:
        params["lm_head_w"] = normal(next(keys), (C, V))

    return params
