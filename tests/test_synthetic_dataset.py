import json
import tempfile
from pathlib import Path

import numpy as np

from fiber_tracer.training.synthetic_dataset import (
    SyntheticCorpusDataset,
    synthetic_collate,
)


def _make_corpus(tmp_path, n_samples: int = 5):
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir(parents=True)
    samples = []
    for i in range(n_samples):
        file_path = patches_dir / f"sample_{i:06d}.npz"
        np.savez_compressed(
            file_path,
            volume=np.random.rand(32, 32, 32).astype(np.float32),
            semantic=np.random.randint(0, 3, size=(32, 32, 32)).astype(np.uint8),
            a2=np.eye(3, dtype=np.float32),
        )
        samples.append(
            {
                "id": i,
                "file": str(file_path.relative_to(tmp_path)),
                "architecture": "short",
                "material": "gfrp",
            }
        )
    registry = {
        "patch_size": [32, 32, 32],
        "voxel_spacing_um": [1.0, 1.0, 1.0],
        "n_samples": n_samples,
        "samples": samples,
    }
    (tmp_path / "corpus.json").write_text(json.dumps(registry))
    return tmp_path


def test_synthetic_dataset_loads_samples():
    with tempfile.TemporaryDirectory() as td:
        corpus_dir = _make_corpus(Path(td), n_samples=6)
        train = SyntheticCorpusDataset(corpus_dir, split="train", val_fraction=0.2)
        val = SyntheticCorpusDataset(corpus_dir, split="val", val_fraction=0.2)
        assert len(train) == 5
        assert len(val) == 1
        sample = train[0]
        assert sample["volume"].shape == (32, 32, 32)
        assert sample["semantic"].shape == (32, 32, 32)
        assert sample["a2"].shape == (3, 3)


def test_synthetic_collate_batches_tensors():
    with tempfile.TemporaryDirectory() as td:
        corpus_dir = _make_corpus(Path(td), n_samples=3)
        dataset = SyntheticCorpusDataset(corpus_dir, split="train", val_fraction=0.0)
        batch = [dataset[i] for i in range(2)]
        collated = synthetic_collate(batch)
        assert collated["volume"].shape == (2, 1, 32, 32, 32)
        assert collated["semantic"].shape == (2, 32, 32, 32)
        assert collated["a2"].shape == (2, 3, 3)
