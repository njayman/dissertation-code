from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from datasets import load_dataset


@dataclass
class Sample:
    text: str
    toxic: int
    split: str


class JigsawDataset:
    def __init__(self, max_samples: int | None = None):
        ds = load_dataset("tasksource/jigsaw_toxicity", split="train", streaming=False)
        texts = ds["comment_text"]
        labels = np.array([int(t) for t in ds["toxic"]], dtype=np.int64)

        if max_samples and max_samples < len(texts):
            idx = np.random.default_rng(42).permutation(len(texts))[:max_samples]
            texts = [texts[i] for i in idx]
            labels = labels[idx]

        n = len(texts)
        train_end = int(n * 0.7)
        val_end = int(n * 0.8)

        self._samples: list[Sample] = []
        for i, (text, label) in enumerate(zip(texts, labels)):
            split = "train" if i < train_end else ("val" if i < val_end else "test")
            self._samples.append(Sample(text=text, toxic=int(label), split=split))

        self._rng = np.random.default_rng(42)

    @property
    def train(self) -> list[Sample]:
        return [s for s in self._samples if s.split == "train"]

    @property
    def val(self) -> list[Sample]:
        return [s for s in self._samples if s.split == "val"]

    @property
    def test(self) -> list[Sample]:
        return [s for s in self._samples if s.split == "test"]

    def samples(self, split: str = "train") -> list[Sample]:
        return [s for s in self._samples if s.split == split]

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def toxic_ratio(self) -> float:
        labels = [s.toxic for s in self._samples]
        return sum(labels) / len(labels) if labels else 0.0

    def shuffle_train(self) -> None:
        train = self.train
        self._rng.shuffle(train)
