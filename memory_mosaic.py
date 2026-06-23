"""Memory Mosaic in pure JAX — functional, no nn.Module abstractions.

Matches the Memory Mosaic architecture from the paper (Fig. 6 right):
  - token embedding only (NO position embedding)
  - N_b repeated blocks of [LayerNorm → ContextualMemory → + residual,
                             LayerNorm → PersistentMemory → + residual]
  - final LayerNorm → linear head (weight-tied with token embedding)

Key differences from GPT-2 (see gpt2.py):
  - CausalSelfAttention  → ContextualMemory  (leaky-avg keys, value peek-ahead, no Q/K split)
  - MLP (4x GELU)        → PersistentMemory  (learned key-value lookup table)
  - Position embedding    → removed

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
# Leaky average (eq. 6 in the paper)
# ---------------------------------------------------------------------------

def leaky_average(k: jnp.ndarray, leaky_beta: jnp.ndarray, T: int) -> jnp.ndarray:
    """Apply causal leaky averaging to keys.

    For each position t, computes a weighted average of all keys at positions <= t,
    where the weight for position s at query position t is exp(-beta * (t - s)).

    Args:
        k: (B, H, T, D) — raw projected keys
        leaky_beta: (H,) — per-head decay rate (absolute value used)
        T: sequence length
    Returns:
        (B, H, T, D) — leaky-averaged keys
    """
    EXP_SCALING = 10.0
    beta = jnp.abs(leaky_beta) * EXP_SCALING  # (H,)

    # Build coefficient matrix: coef[t, s] = exp(-beta * (t - s)) for s <= t
    idx = jnp.arange(T)
    distances = idx[:, None] - idx[None, :]         # (T, T), distances[t,s] = t - s
    coef = jnp.exp(-beta[:, None, None] * distances)  # (H, T, T)
    coef = coef * jnp.tril(jnp.ones((T, T)))          # zero out future positions

    # Apply: (H, T, T) @ (B, H, T, D) -> (B, H, T, D)
    return jnp.einsum("hts,bhsd->bhtd", coef, k)


# ---------------------------------------------------------------------------
# Key feature extractor
# ---------------------------------------------------------------------------

def key_features(
    x: jnp.ndarray,
    params: dict,
    cfg: Config,
    scale_pow: float = 1.0,
) -> jnp.ndarray:
    """Extract key features: project → leaky average → normalize → scale.

    Args:
        x: (B, T, C)
        params: dict with keys w_k, leaky_beta, key_scale
        scale_pow: exponent for key scaling (2 for persistent memory)
    Returns:
        (B, H, T, D) — processed keys
    """
    B, T, C = x.shape
    EXP_SCALING = 10.0
    KEY_SCALE_MAX = math.log(2**16 - 1)  # clamp to fit fp16

    # project: (B, T, C) @ (C, H, D) -> (B, T, H, D) -> (B, H, T, D)
    k = jnp.einsum("btc,chd->bthd", x, params["w_k"])
    k = jnp.transpose(k, (0, 2, 1, 3))  # (B, H, T, D)

    # leaky average
    k = leaky_average(k, params["leaky_beta"], T)

    # normalize
    k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-10)

    # scale: exp(scale_pow * key_scale * EXP_SCALING), clamped
    log_scale = scale_pow * EXP_SCALING * jnp.abs(params["key_scale"])  # (H,)
    log_scale = jnp.minimum(log_scale, KEY_SCALE_MAX)
    k = k * jnp.exp(log_scale)[None, :, None, None]

    return k


# ---------------------------------------------------------------------------
# Value feature extractor
# ---------------------------------------------------------------------------

def val_features(
    x: jnp.ndarray,
    params: dict,
    cfg: Config,
) -> jnp.ndarray:
    """Extract value features: project → peek-ahead shift → normalize → scale.

    The peek-ahead shift (eq. 6) allows values to depend on x_{T+1}.

    Args:
        x: (B, T, C)
        params: dict with keys w_v, val_coef, val_scale
    Returns:
        (B, H, T, D) — processed values
    """
    B, T, C = x.shape
    EXP_SCALING = 10.0

    # project: (B, T, C) @ (C, H, D) -> (B, T, H, D) -> (B, H, D, T)
    v = jnp.einsum("btc,chd->bthd", x, params["w_v"])
    v = jnp.transpose(v, (0, 2, 3, 1))  # (B, H, D, T)

    # peek-ahead: shift left by v_shift along T dimension
    # v_shifted[:,:,:,t] = v[:,:,:,t+1]  (last position becomes 0)
    v_shifted = jnp.pad(v[:, :, :, cfg.v_shift:], ((0,0), (0,0), (0,0), (0, cfg.v_shift)))

    # interpolate: (1 - coef) * shifted + coef * original
    coef = params["val_coef"][:, None, None]  # (H,) -> (H, 1, 1)
    v = (1 - coef) * v_shifted + coef * v

    # transpose back: (B, H, D, T) -> (B, H, T, D)
    v = jnp.transpose(v, (0, 2, 3, 1))  # wait, need (B, H, T, D)

    # actually: (B, H, D, T) -> (B, H, T, D)
    v = jnp.transpose(v, (0, 1, 3, 2))

    # normalize
    v = v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + 1e-10)

    # scale
    v = v * jnp.exp(EXP_SCALING * params["val_scale"])[None, :, None, None]

    return v


# ---------------------------------------------------------------------------
# Contextual memory (replaces CausalSelfAttention in GPT-2)
# ---------------------------------------------------------------------------

def contextual_memory(
    x: jnp.ndarray,
    params: dict,
    cfg: Config,
    deterministic: bool,
    rng: jax.Array | None,
) -> jnp.ndarray:
    """Contextual memory unit — the core Memory Mosaic mechanism.

    Unlike self-attention, keys serve as both queries and keys (no Q/K split).
    The causal mask excludes the diagonal: position t attends to 0..t-1 only.

    Args:
        x: (B, T, C)
        params: dict with keys cmem (key/val extractors), out_w
    Returns:
        (B, T, C)
    """
    B, T, C = x.shape

    k = key_features(x, params["key"], cfg)   # (B, H, T, D)
    v = val_features(x, params["val"], cfg)    # (B, H, T, D)

    # causal attention with diagonal excluded (strictly lower triangular)
    # keys are used as both queries and keys (eq. 7)
    # query at position t uses k[t], attends to k[0..t-1]
    attn = jnp.einsum("bhtd,bhsd->bhts", k, k)  # (B, H, T, T), scale=1 (already scaled)

    # strictly lower triangular mask (diagonal=-1): position t sees 0..t-1
    mask = jnp.tril(jnp.ones((T, T)), k=-1)
    attn = jnp.where(mask[None, None, :, :], attn, jnp.finfo(attn.dtype).min)
    attn = jax.nn.softmax(attn, axis=-1)

    # attention dropout
    if not deterministic and cfg.dropout > 0:
        drop_rng, rng = jax.random.split(rng)
        keep = jax.random.bernoulli(drop_rng, 1.0 - cfg.dropout, attn.shape)
        attn = jnp.where(keep, attn / (1.0 - cfg.dropout), 0.0)

    # weighted sum: (B, H, T, S) @ (B, H, S, D) -> (B, H, T, D)
    out = jnp.einsum("bhts,bhsd->bhtd", attn, v)

    # reshape: (B, H, T, D) -> (B, T, H, D)
    out = jnp.transpose(out, (0, 2, 1, 3))

    # output projection: (B, T, H, D) @ (H, D, C) -> (B, T, C)
    out = jnp.einsum("bthd,hdc->btc", out, params["out_w"])

    # residual dropout
    if not deterministic and cfg.dropout > 0:
        drop_rng, rng = jax.random.split(rng)
        keep = jax.random.bernoulli(drop_rng, 1.0 - cfg.dropout, out.shape)
        out = jnp.where(keep, out / (1.0 - cfg.dropout), 0.0)

    return out


# ---------------------------------------------------------------------------
# Persistent memory (replaces MLP/FFN in GPT-2)
# ---------------------------------------------------------------------------

def persistent_memory(
    x: jnp.ndarray,
    params: dict,
    cfg: Config,
    deterministic: bool,
    rng: jax.Array | None,
) -> jnp.ndarray:
    """Persistent memory unit — learned key/value lookup replacing the FFN.

    Uses the same key extraction as contextual memory, but attends to a fixed
    set of learned (P_k, P_v) pairs instead of context tokens.

    Args:
        x: (B, T, C)
        params: dict with keys pmem (key extractor + P_k, P_v), out_w
    Returns:
        (B, T, C)
    """
    B, T, C = x.shape

    # extract keys with scale_pow=2 (P_k has no scale of its own)
    k = key_features(x, params["key"], cfg, scale_pow=2.0)  # (B, H, T, D)

    # attend to persistent memory pairs
    out = jnp.zeros((B, cfg.n_head, T, cfg.head_dim))
    for i in range(cfg.pmem_count):
        P_k = params["P_k"][i]  # (H, M, D)
        P_v = params["P_v"][i]  # (H, M, D)

        # attention: (B, H, T, D) @ (H, M, D)^T -> (B, H, T, M)
        attn = jnp.einsum("bhtd,hmd->bhtm", k, P_k)  # scale=1 (already scaled)
        attn = jax.nn.softmax(attn, axis=-1)

        if not deterministic and cfg.dropout > 0:
            drop_rng, rng = jax.random.split(rng)
            keep = jax.random.bernoulli(drop_rng, 1.0 - cfg.dropout, attn.shape)
            attn = jnp.where(keep, attn / (1.0 - cfg.dropout), 0.0)

        # weighted sum: (B, H, T, M) @ (H, M, D) -> (B, H, T, D)
        out = out + jnp.einsum("bhtm,hmd->bhtd", attn, P_v)

    out = out / cfg.pmem_count

    # output scaling
    EXP_SCALING = 10.0
    out = out * jnp.exp(EXP_SCALING * params["out_scale"])[None, :, None, None]

    # reshape: (B, H, T, D) -> (B, T, H, D)
    out = jnp.transpose(out, (0, 2, 1, 3))

    # output projection: (B, T, H, D) @ (H, D, C) -> (B, T, C)
    out = jnp.einsum("bthd,hdc->btc", out, params["out_w"])

    # residual dropout
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
    """Pre-norm block: LN → ContextualMemory → +residual, LN → PersistentMemory → +residual."""
    if rng is not None:
        cmem_rng, pmem_rng = jax.random.split(rng)
    else:
        cmem_rng = pmem_rng = None

    # contextual memory sub-block
    h = layer_norm(x, params["ln1_w"])
    h = contextual_memory(h, params["cmem"], cfg, deterministic, cmem_rng)
    x = x + h

    # persistent memory sub-block
    h = layer_norm(x, params["ln2_w"])
    h = persistent_memory(h, params["pmem"], cfg, deterministic, pmem_rng)
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
    """Memory Mosaic forward pass.

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

    # token embedding only — NO position embedding
    h = params["wte"][x]                                     # (B, T, C)

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
    """Initialize Memory Mosaic parameters."""
    C = cfg.n_embd
    H = cfg.n_head
    D = cfg.head_dim
    V = cfg.vocab_size
    M = cfg.pmem_size

    std = 0.02
    res_std = std / math.sqrt(2 * cfg.n_layer)

    def normal(key, shape, s=std):
        return jax.random.normal(key, shape) * s

    keys = iter(jax.random.split(rng, 200))

    # token embedding only (no position embedding)
    wte = normal(next(keys), (V, C))

    # transformer blocks
    blocks = []
    for _ in range(cfg.n_layer):
        block = {
            "ln1_w": jnp.ones(C),
            "cmem": {
                "key": {
                    "w_k": normal(next(keys), (C, H, D)),
                    "leaky_beta": jnp.linspace(0.5, 5.0, H) / 10.0,  # per-head, /EXP_SCALING
                    "key_scale": jnp.ones(H) / 10.0,                  # /EXP_SCALING
                },
                "val": {
                    "w_v": normal(next(keys), (C, H, D)),
                    "val_coef": jax.random.uniform(next(keys), (H,)),  # learned interpolation
                    "val_scale": jnp.ones(H) * (-0.5 / 10.0),         # /EXP_SCALING
                },
                "out_w": normal(next(keys), (H, D, C), res_std),
            },
            "ln2_w": jnp.ones(C),
            "pmem": {
                "key": {
                    "w_k": normal(next(keys), (C, H, D)),
                    "leaky_beta": jnp.linspace(0.5, 5.0, H) / 10.0,
                    "key_scale": jnp.ones(H) / 10.0,
                },
                "P_k": normal(next(keys), (cfg.pmem_count, H, M, D), 1.0 / math.sqrt(D)),
                "P_v": normal(next(keys), (cfg.pmem_count, H, M, D), 1.0 / math.sqrt(D)),
                "out_scale": jnp.ones(H) * (-0.5 / 10.0),  # /EXP_SCALING
                "out_w": normal(next(keys), (H, D, C), res_std),
            },
        }
        blocks.append(block)

    params = {
        "_cfg": cfg,
        "wte": wte,
        "blocks": blocks,
        "ln_f_w": jnp.ones(C),
    }

    if not cfg.weight_tying:
        params["lm_head_w"] = normal(next(keys), (C, V))

    return params
