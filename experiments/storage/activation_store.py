"""Memory-mapped activation storage. Sharded by GPU, float16, rows of hidden_dim."""

from pathlib import Path

import numpy as np

from experiments import config


def _hidden_dim() -> int:
    return config.MODEL_HIDDEN_DIM


def _bytes_per_row() -> int:
    return config.MODEL_HIDDEN_DIM * 2  # float16 = 2 bytes per element


class ActivationWriter:
    """Appends float16 activation rows to a binary shard file."""

    def __init__(self, shard_path: Path) -> None:
        self.shard_path = shard_path
        self.shard_path.parent.mkdir(parents=True, exist_ok=True)
        # Determine current row count from existing file size
        if self.shard_path.exists():
            file_size = self.shard_path.stat().st_size
            self._row_count = file_size // _bytes_per_row()
        else:
            self._row_count = 0

    def append(self, activations: np.ndarray) -> tuple[int, int]:
        """Append float16 array of shape (N, hidden_dim) to the shard file.

        Returns (offset_rows, num_rows) where offset is the row count
        BEFORE this append.
        """
        hidden_dim = _hidden_dim()
        assert activations.ndim == 2, f"Expected 2D array, got {activations.ndim}D"
        assert activations.shape[1] == hidden_dim, (
            f"Expected {hidden_dim} columns, got {activations.shape[1]}"
        )
        activations = activations.astype(np.float16)

        offset = self._row_count
        num_rows = activations.shape[0]

        with open(self.shard_path, "ab") as f:
            f.write(activations.tobytes())

        self._row_count += num_rows
        return offset, num_rows


class ActivationReader:
    """Reads float16 activation rows from a memory-mapped shard file."""

    def __init__(self, shard_path: Path) -> None:
        self.shard_path = shard_path
        hidden_dim = _hidden_dim()
        file_size = shard_path.stat().st_size
        total_rows = file_size // _bytes_per_row()
        self._mmap = np.memmap(
            shard_path,
            dtype=np.float16,
            mode="r",
            shape=(total_rows, hidden_dim),
        )

    def read(self, offset: int, length: int) -> np.ndarray:
        """Read float16 array of shape (length, hidden_dim) starting at offset."""
        return np.array(self._mmap[offset : offset + length])
