"""Reproduce a subplot of Figure 7 from the Memory Mosaics paper.

Reads training logs (log.json) produced by train.py and plots
train/val loss curves for GPT-2 and Memory Mosaic at the same depth.

Usage:
    python plot_fig7.py --n_layer 1
    python plot_fig7.py --n_layer 1 8 12 18   # multiple subplots
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def load_log(model: str, n_layer: int) -> dict | None:
    path = os.path.join("results", f"{model}_L{n_layer}", "log.json")
    if not os.path.exists(path):
        print(f"  Warning: {path} not found, skipping.")
        return None
    with open(path) as f:
        return json.load(f)


def plot_subplot(ax, n_layer: int):
    """Plot one subplot (one depth) on the given axes."""
    ax.set_title(f"(depth) $N_b$={n_layer}")
    ax.set_xlabel("iterations")
    ax.set_ylabel("cross-entropy loss")

    for model, label_prefix, colors in [
        ("gpt2", "GPT2", ("tab:blue", "tab:cyan")),
        ("mosaic", "Memory Mosaic", ("tab:red", "tab:pink")),
    ]:
        log = load_log(model, n_layer)
        if log is None:
            continue
        steps = [e["step"] for e in log]
        train_loss = [e["train_loss"] for e in log]
        val_loss = [e["val_loss"] for e in log]

        ax.plot(steps, train_loss, color=colors[0], linestyle="-",
                label=f"{label_prefix} train loss")
        ax.plot(steps, val_loss, color=colors[1], linestyle="--",
                label=f"{label_prefix} val loss")

    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_layer", type=int, nargs="+", default=[1])
    args = parser.parse_args()

    n_plots = len(args.n_layer)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4), squeeze=False)

    for i, n_layer in enumerate(args.n_layer):
        plot_subplot(axes[0, i], n_layer)

    fig.suptitle("Figure 7 — Training and validation loss on BabiStories", y=1.02)
    fig.tight_layout()
    out_path = "fig7.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
