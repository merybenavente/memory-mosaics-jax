# Memory Mosaics — JAX

JAX reimplementation of [Memory Mosaics](https://arxiv.org/abs/2405.06394v3) (Zhang et al., ICLR 2025), along with a GPT-2 baseline.

Both are structured side-by-side so a `diff gpt2.py memory_mosaic.py` highlights exactly what changes.

The code on the notebook reproduces [**Fig. 7**](https://colab.research.google.com/drive/1sybefNAZzu4oyY5V21pWHQtkM9Ea9mZC?usp=sharing) from the paper: training and validation loss curves comparing GPT-2 and Memory Mosaic on BabiStories at varying depths (`n_layer = 1, 8, 12, 18`).

The paper has an [official PyTorch implementation](https://github.com/facebookresearch/MemoryMosaics).