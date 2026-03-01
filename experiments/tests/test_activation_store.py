"""Tests for memory-mapped activation storage round-trip, dtype, and shape preservation."""

import numpy as np
import pytest

from experiments.storage.activation_store import ActivationReader, ActivationWriter


def test_write_read_roundtrip(tmp_path):
    """Write float16 (10, 4096) array, read back, assert bitwise equal."""
    shard = tmp_path / "shard_0.npy"
    data = np.random.randn(10, 4096).astype(np.float16)

    writer = ActivationWriter(shard)
    offset, num_rows = writer.append(data)

    assert offset == 0
    assert num_rows == 10

    reader = ActivationReader(shard)
    result = reader.read(0, 10)

    assert np.array_equal(result, data)


def test_multiple_writes(tmp_path):
    """Write two batches, read each independently, assert bitwise equal."""
    shard = tmp_path / "shard_0.npy"
    data1 = np.random.randn(10, 4096).astype(np.float16)
    data2 = np.random.randn(5, 4096).astype(np.float16)

    writer = ActivationWriter(shard)
    offset1, num1 = writer.append(data1)
    offset2, num2 = writer.append(data2)

    assert offset1 == 0
    assert num1 == 10
    assert offset2 == 10
    assert num2 == 5

    reader = ActivationReader(shard)
    result1 = reader.read(offset=0, length=10)
    result2 = reader.read(offset=10, length=5)

    assert np.array_equal(result1, data1)
    assert np.array_equal(result2, data2)


def test_shape_preserved(tmp_path):
    """Shape is preserved through round-trip: (N, 4096) stays (N, 4096)."""
    shard = tmp_path / "shard_0.npy"
    data = np.random.randn(7, 4096).astype(np.float16)

    writer = ActivationWriter(shard)
    writer.append(data)

    reader = ActivationReader(shard)
    result = reader.read(0, 7)

    assert result.shape == (7, 4096)
    assert result.dtype == np.float16


def test_mmap_readonly(tmp_path):
    """Mmap file can be opened read-only after writing."""
    shard = tmp_path / "shard_0.npy"
    data = np.random.randn(3, 4096).astype(np.float16)

    writer = ActivationWriter(shard)
    writer.append(data)

    # Opening a second reader should work (read-only mmap)
    reader1 = ActivationReader(shard)
    reader2 = ActivationReader(shard)

    assert np.array_equal(reader1.read(0, 3), reader2.read(0, 3))


def test_offset_length_indexing(tmp_path):
    """Offset and length fields correctly index into the file, simulating GenerationRecord usage."""
    shard = tmp_path / "shard_0.npy"
    writer = ActivationWriter(shard)

    # Simulate 3 records with different token counts
    records = [
        np.random.randn(20, 4096).astype(np.float16),
        np.random.randn(35, 4096).astype(np.float16),
        np.random.randn(12, 4096).astype(np.float16),
    ]
    offsets = []
    for rec in records:
        offset, length = writer.append(rec)
        offsets.append((offset, length))

    reader = ActivationReader(shard)

    # Each record can be retrieved by its offset and length
    for i, (offset, length) in enumerate(offsets):
        result = reader.read(offset, length)
        assert np.array_equal(result, records[i])
