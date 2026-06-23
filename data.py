"""BabiStories data loading and preparation.

Downloads the dataset from the paper's GitHub repo if needed,
tokenizes with GPT-2 BPE (tiktoken), and serves random batches as JAX arrays.
"""

import os
import json
import subprocess
import shutil

import numpy as np
import jax
import jax.numpy as jnp
import tiktoken

from config import Config

# --- BabiStories is distributed as split 7z archives in the official repo ---
_REPO_URL = "https://github.com/facebookresearch/MemoryMosaics.git"
_7Z_DIR = "BabiStories/data"


def _download_if_needed(data_dir: str) -> None:
    """Download and extract BabiStories text files if not already present."""
    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, "traindataset.txt")
    val_path = os.path.join(data_dir, "valdataset.txt")
    if os.path.exists(train_path) and os.path.exists(val_path):
        return

    print("Downloading BabiStories from the official repo ...")
    clone_dir = os.path.join(data_dir, "_repo_clone")

    # sparse-checkout: only fetch the 7z archives
    subprocess.run(
        ["git", "clone", "--filter=blob:none", "--sparse", "--depth=1",
         _REPO_URL, clone_dir],
        check=True,
    )
    subprocess.run(
        ["git", "-C", clone_dir, "sparse-checkout", "set", _7Z_DIR],
        check=True,
    )

    # extract the multi-part 7z archive
    archive = os.path.join(clone_dir, _7Z_DIR, "babistories-dataset.7z.001")
    print("Extracting 7z archive ...")
    subprocess.run(["7z", "x", archive, f"-o{data_dir}", "-y"], check=True)

    # clean up clone
    shutil.rmtree(clone_dir)
    print(f"  traindataset.txt: {os.path.getsize(train_path) / 1e6:.1f} MB")
    print(f"  valdataset.txt:   {os.path.getsize(val_path) / 1e6:.1f} MB")


def _tokenize_if_needed(data_dir: str) -> None:
    """Tokenize raw text → .bin memmap files (uint16), one per split.

    Streams line-by-line to a temporary file to avoid loading everything
    into memory at once (the train set is ~1.8 GB / ~475M tokens).
    """
    enc = tiktoken.get_encoding("gpt2")
    for split in ("train", "val"):
        bin_path = os.path.join(data_dir, f"{split}.bin")
        if os.path.exists(bin_path):
            continue
        txt_path = os.path.join(data_dir, f"{split}dataset.txt")
        tmp_path = bin_path + ".tmp"
        print(f"Tokenizing {txt_path} ...")

        # first pass: count tokens to know the memmap size
        n_tokens = 0
        with open(txt_path, "r") as f:
            for line in f:
                n_tokens += len(enc.encode_ordinary(json.loads(line))) + 1  # +1 for eot

        # second pass: write tokens directly into memmap
        mm = np.memmap(tmp_path, dtype=np.uint16, mode="w+", shape=(n_tokens,))
        offset = 0
        with open(txt_path, "r") as f:
            for line in f:
                tokens = enc.encode_ordinary(json.loads(line))
                tokens.append(enc.eot_token)
                mm[offset : offset + len(tokens)] = np.array(tokens, dtype=np.uint16)
                offset += len(tokens)
        mm.flush()
        os.rename(tmp_path, bin_path)
        print(f"  {split}: {n_tokens:,} tokens → {bin_path}")


def prepare_data(cfg: Config) -> None:
    """Ensure data is downloaded and tokenized."""
    _download_if_needed(cfg.data_dir)
    _tokenize_if_needed(cfg.data_dir)


def load_split(cfg: Config, split: str) -> np.ndarray:
    """Load a tokenized split as a read-only numpy array."""
    path = os.path.join(cfg.data_dir, f"{split}.bin")
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(
    cfg: Config, data: np.ndarray, rng: jax.Array
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Sample a random batch of (input, target) pairs.

    Returns:
        x: (batch_size, block_size) int32
        y: (batch_size, block_size) int32  — shifted by one position
    """
    max_start = len(data) - cfg.block_size - 1
    starts = jax.random.randint(rng, (cfg.batch_size,), 0, max_start)
    starts_np = np.asarray(starts)
    x = np.stack([data[s : s + cfg.block_size] for s in starts_np])
    y = np.stack([data[s + 1 : s + 1 + cfg.block_size] for s in starts_np])
    return jnp.array(x, dtype=jnp.int32), jnp.array(y, dtype=jnp.int32)
