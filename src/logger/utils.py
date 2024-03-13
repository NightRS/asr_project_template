import io
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from torch import Tensor


def plot_spectrograms_to_buf(
    spectrograms_original: list[Tensor],
    spectrograms_processed: list[Tensor],
    paths: list[str],
) -> io.BytesIO:
    def plot_spec(ax: Axes, spec: Tensor, title: Optional[str] = None):
        if title is not None:
            ax.set_title(title)
        ax.imshow(spec, origin="lower", aspect="auto")

    rows_count = max(len(spectrograms_original), len(spectrograms_processed))
    fig, axes = plt.subplots(rows_count, 2, sharex=True, sharey=True)
    for i, (orig, proc, path) in enumerate(
        zip(spectrograms_original, spectrograms_processed, paths)
    ):
        plot_spec(axes[i, 0], orig, title=path)
        plot_spec(axes[i, 1], proc)
    fig.suptitle("Original / After augmentations spectrograms")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf
