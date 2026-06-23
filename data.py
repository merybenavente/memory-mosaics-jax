"""BabiStories data loading and preparation.

Downloads the dataset from the paper's GitHub repo if needed,
tokenizes with GPT-2 BPE (tiktoken), and serves random batches as JAX arrays.
"""

import os
import json
import urllib.request
import zipfile

import numpy as np
import jax
import jax.numpy as jnp
import tiktoken

from config import Config

# --- URLs for BabiStories from the official repo ---
_REPO_RAW = (
    "https://raw.githubusercontent.com/facebookresearch/MemoryMosaics/main"
    "/Library/memory_mosaics/data/BabiStories"
)
_DATA_URLS = {
    "train": f"{_REPO_RAW}/traindataset.txt",
    "val": f"{_REPO_RAW}/valdataset.txt",
}


def _download_if_needed(data_dir: str) -> None:
    """Download raw BabiStories text files if not already present."""
    os.makedirs(data_dir, exist_ok=True)
    for split, url in _DATA_URLS.items():
        path = os.path.join(data_dir, f"{split}dataset.txt")
        if not os.path.exists(path):
            print(f"Downloading {split} split → {path} ...")
            urllib.request.urlretrieve(url, path)
            print(f"  done ({os.path.getsize(path) / 1e6:.1f} MB)")


def _tokenize_if_needed(data_dir: str) -> None:
    """Tokenize raw text → .bin memmap files (uint16), one per split."""
    enc = tiktoken.get_encoding("gpt2")
    for split in ("train", "val"):
        bin_path = os.path.join(data_dir, f"{split}.bin")
        if os.path.exists(bin_path):
            continue
        txt_path = os.path.join(data_dir, f"{split}dataset.txt")
        print(f"Tokenizing {txt_path} ...")
        with open(txt_path, "r") as f:
            lines = f.readlines()
        ids: list[int] = []
        for line in lines:
            ids.extend(enc.encode_ordinary(json.loads(line)))
            ids.append(enc.eot_token)
        arr = np.array(ids, dtype=np.uint16)
        mm = np.memmap(bin_path, dtype=np.uint16, mode="w+", shape=arr.shape)
        mm[:] = arr
        mm.flush()
        print(f"  {split}: {len(arr):,} tokens → {bin_path}")


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
