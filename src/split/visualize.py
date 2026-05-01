from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_curve(
    sim: np.ndarray,
    depths: np.ndarray,
    splits: list[int],
    threshold: float,
    output_path: str | Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    x = np.arange(len(sim))
    ax1.plot(x, sim, label="cosine similarity", color="steelblue")
    ax1.set_ylabel("similarity")
    ax1.legend(loc="upper right")

    ax2.plot(x, depths, label="depth score", color="darkorange")
    ax2.axhline(threshold, color="gray", linestyle="--", label=f"threshold={threshold:.3f}")
    ax2.set_ylabel("depth")
    ax2.set_xlabel("gap index (between sentences i and i+1)")
    ax2.legend(loc="upper right")

    for g in splits:
        ax1.axvline(g, color="crimson", alpha=0.4)
        ax2.axvline(g, color="crimson", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
