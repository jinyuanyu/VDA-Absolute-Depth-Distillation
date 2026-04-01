from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from torch.utils.data import Dataset


class ScaleLabelDataset(Dataset):
    """Dataset backed by the offline scale-label JSON file."""

    def __init__(self, labels_json: str | Path, allowed_cams: Sequence[str] | None = None) -> None:
        labels_json = Path(labels_json)
        payload = json.loads(labels_json.read_text(encoding="utf-8"))
        samples: List[Dict[str, Any]] = payload["samples"]
        if allowed_cams is not None:
            allowed_set = set(allowed_cams)
            samples = [sample for sample in samples if sample["camera"] in allowed_set]
        self.samples = samples
        self.meta = payload.get("meta", {})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.samples[index]


def split_dataset_indices(
    length: int,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    indices = list(range(length))
    random.Random(seed).shuffle(indices)
    train_size = int(length * train_ratio)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    return train_indices, val_indices


def simple_collate(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return batch
