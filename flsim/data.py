"""Datasets and non-IID client partitioning.

The whole dataset is kept in memory as uint8 tensors (MNIST: ~47 MB) and
normalised per mini-batch, which keeps RAM use low.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

# (mean, std) per channel, the values commonly used for these datasets
STATS = {
    "mnist": ((0.1307,), (0.3081,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
}


@dataclass
class Dataset:
    name: str
    x_train: torch.Tensor   # uint8, N x C x H x W
    y_train: torch.Tensor   # int64, N
    x_test: torch.Tensor
    y_test: torch.Tensor
    mean: torch.Tensor      # 1 x C x 1 x 1
    std: torch.Tensor
    num_classes: int

    def batch(self, x_uint8: torch.Tensor) -> torch.Tensor:
        """uint8 images -> normalised float32 batch."""
        return (x_uint8.float() / 255.0 - self.mean) / self.std


def load_dataset(name: str) -> Dataset:
    name = name.lower()
    if name == "mnist":
        tr = datasets.MNIST(DATA_ROOT, train=True, download=True)
        te = datasets.MNIST(DATA_ROOT, train=False, download=True)
        x_tr, x_te = tr.data.unsqueeze(1), te.data.unsqueeze(1)
        y_tr, y_te = tr.targets, te.targets
    elif name == "cifar10":
        tr = datasets.CIFAR10(DATA_ROOT, train=True, download=True)
        te = datasets.CIFAR10(DATA_ROOT, train=False, download=True)
        x_tr = torch.from_numpy(tr.data).permute(0, 3, 1, 2).contiguous()
        x_te = torch.from_numpy(te.data).permute(0, 3, 1, 2).contiguous()
        y_tr, y_te = torch.tensor(tr.targets), torch.tensor(te.targets)
    else:
        raise ValueError(f"unknown dataset {name!r}")
    mean, std = STATS[name]
    return Dataset(
        name, x_tr, y_tr.long(), x_te, y_te.long(),
        torch.tensor(mean).view(1, -1, 1, 1), torch.tensor(std).view(1, -1, 1, 1), 10,
    )


def dirichlet_partition(labels: np.ndarray, num_clients: int, alpha: float,
                        rng: np.random.Generator, min_size: int = 50) -> list[np.ndarray]:
    """Label-skew split: for every class, the share each client receives is drawn
    from Dirichlet(alpha). Smaller alpha gives more skewed clients. The draw is
    repeated until every client has at least `min_size` samples."""
    classes = np.unique(labels)
    for _ in range(1000):
        parts: list[list[int]] = [[] for _ in range(num_clients)]
        for c in classes:
            idx = np.flatnonzero(labels == c)
            rng.shuffle(idx)
            shares = rng.dirichlet(np.full(num_clients, alpha))
            cuts = (np.cumsum(shares)[:-1] * len(idx)).astype(int)
            for cid, chunk in enumerate(np.split(idx, cuts)):
                parts[cid].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            return [np.array(sorted(p)) for p in parts]
    raise RuntimeError("could not build a partition with the requested minimum client size")


def train_holdout_split(parts: list[np.ndarray], holdout_frac: float,
                        rng: np.random.Generator) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Keep a share of every client's samples aside. The hold-out part has the
    client's own label mix and is used for per-client (worst-client) accuracy."""
    train, hold = [], []
    for p in parts:
        p = rng.permutation(p)
        k = max(1, int(round(len(p) * holdout_frac)))
        hold.append(np.sort(p[:k]))
        train.append(np.sort(p[k:]))
    return train, hold
