"""Models. No batch-norm layers, so a model is fully described by its parameter vector."""
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class MLP(nn.Module):
    """784-200-200-10 multilayer perceptron (the '2NN' used for MNIST in the FedAvg paper)."""

    def __init__(self, in_dim: int = 784, hidden: int = 200, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class SmallCNN(nn.Module):
    """Two 3x3 conv layers (32, 64 channels) with max-pooling, then two linear layers."""

    def __init__(self, in_ch: int = 1, img: int = 28, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * (img // 4) ** 2, 128), nn.ReLU(), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


def build_model(model: str, dataset: str) -> nn.Module:
    in_ch, img = (1, 28) if dataset == "mnist" else (3, 32)
    if model == "mlp":
        return MLP(in_dim=in_ch * img * img)
    if model == "cnn":
        return SmallCNN(in_ch, img)
    raise ValueError(f"unknown model {model!r}")


def get_vector(model: nn.Module) -> torch.Tensor:
    return parameters_to_vector(model.parameters()).detach().clone()


def set_vector(model: nn.Module, vec: torch.Tensor) -> None:
    vector_to_parameters(vec.clone(), model.parameters())
