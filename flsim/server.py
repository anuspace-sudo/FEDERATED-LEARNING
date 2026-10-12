"""Server side: aggregation rules and evaluation."""
import numpy as np
import torch
import torch.nn.functional as F

from .client import ClientResult
from .data import Dataset
from .models import set_vector


def aggregate_fedavg(w_global: torch.Tensor, results: list[ClientResult]) -> torch.Tensor:
    """FedAvg: sample-weighted average of the client updates."""
    n = sum(r.n_samples for r in results)
    return w_global + sum((r.n_samples / n) * r.delta for r in results)


def aggregate_fednova(w_global: torch.Tensor, results: list[ClientResult]) -> torch.Tensor:
    """FedNova normalized averaging for plain SGD (Wang et al., NeurIPS 2020):
    w_{t+1} = w_t + tau_eff * sum_i p_i * (delta_i / tau_i),  tau_eff = sum_i p_i * tau_i."""
    n = sum(r.n_samples for r in results)
    p = [r.n_samples / n for r in results]
    tau_eff = sum(pi * r.steps for pi, r in zip(p, results))
    return w_global + tau_eff * sum(pi * r.delta / r.steps for pi, r in zip(p, results))


@torch.no_grad()
def evaluate(model, w: torch.Tensor, ds: Dataset, x: torch.Tensor, y: torch.Tensor, bs: int = 2000) -> tuple[float, float]:
    """Accuracy and mean loss of the model with weights `w` on (x, y)."""
    set_vector(model, w)
    model.eval()
    correct, loss = 0, 0.0
    for i in range(0, len(y), bs):
        out = model(ds.batch(x[i:i + bs]))
        loss += F.cross_entropy(out, y[i:i + bs], reduction="sum").item()
        correct += (out.argmax(1) == y[i:i + bs]).sum().item()
    return correct / len(y), loss / len(y)


def client_accuracies(model, w: torch.Tensor, ds: Dataset, holdouts: list[np.ndarray]) -> np.ndarray:
    """Accuracy of the global model on every client's own hold-out samples."""
    return np.array([evaluate(model, w, ds, ds.x_train[h], ds.y_train[h])[0] for h in holdouts])
