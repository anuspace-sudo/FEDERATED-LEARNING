"""Client side: local training with a fixed workload (baselines) or with the
proposed adaptive workload (Client Profiler, Budget Controller, Drift Monitor).

Device speed is SIMULATED: every client has a time per step `t_step` (seconds),
and each step it runs advances a simulated clock by t_step with a small jitter.
No real device is measured here.
"""
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .data import Dataset
from .models import get_vector, set_vector


@dataclass
class ClientResult:
    delta: torch.Tensor      # w - w_t after local training
    steps: int               # tau_i, local SGD steps actually done
    sim_time: float          # simulated seconds spent on local training
    n_samples: int
    budget: int              # tau_budget (adaptive) or the fixed workload
    stop_reason: str         # 'fixed' | 'budget' | 'loss' | 'drift'
    drift_checks: int        # how many times the Drift Monitor ran


class BatchStream:
    """Endless mini-batches over one client's samples, reshuffled every epoch."""

    def __init__(self, idx: np.ndarray, batch_size: int, rng: np.random.Generator):
        self.idx, self.bs, self.rng, self.pos = idx, batch_size, rng, 0
        self.order = rng.permutation(len(idx))

    def next(self) -> np.ndarray:
        if self.pos >= len(self.order):
            self.order, self.pos = self.rng.permutation(len(self.idx)), 0
        sel = self.order[self.pos:self.pos + self.bs]
        self.pos += self.bs
        return self.idx[sel]


def fixed_workload(n_samples: int, epochs: int, batch_size: int) -> int:
    """Number of local steps in `epochs` passes over the client's data."""
    return epochs * math.ceil(n_samples / batch_size)


def local_train(model, w_global: torch.Tensor, ds: Dataset, idx: np.ndarray, t_step: float, cfg: dict,
                mode: str, rng: np.random.Generator) -> ClientResult:
    """Run local SGD starting from the global model `w_global`.

    mode 'fixed'    : cfg['epochs'] local epochs (FedAvg / FedProx / FedNova).
    mode 'adaptive' : tau_i = min(tau_budget, tau_stop)
        Client Profiler   - averages the (simulated) duration of the first
                            cfg['profile_steps'] steps to estimate t_step.
        Budget Controller - tau_budget = floor(D / t_step), capped at the fixed
                            workload so a fast client never does more than FedAvg.
        Drift Monitor     - every cfg['check_every'] steps, stop when the relative
                            decrease of the windowed loss is below eps, or when
                            ||w - w_t|| > delta * ||w_t||.
    """
    set_vector(model, w_global)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"])        # plain SGD, no momentum
    stream = BatchStream(idx, cfg["batch_size"], rng)
    mu = cfg.get("mu", 0.0) if cfg.get("prox") else 0.0
    cap = fixed_workload(len(idx), cfg["epochs"], cfg["batch_size"])
    budget, reason = cap, "fixed"
    w_norm = w_global.norm().item()
    losses: list[float] = []
    sim_time, steps, checks = 0.0, 0, 0
    jitter = cfg.get("time_jitter", 0.05)

    while steps < budget:
        b = stream.next()
        x, y = ds.batch(ds.x_train[b]), ds.y_train[b]
        loss = F.cross_entropy(model(x), y)
        if mu > 0:                                                  # FedProx proximal term
            prox = sum(((p - g) ** 2).sum() for p, g in zip(model.parameters(), _split_like(model, w_global)))
            loss = loss + 0.5 * mu * prox
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        steps += 1
        sim_time += t_step * max(0.5, 1.0 + jitter * rng.standard_normal())
        losses.append(loss.item())

        if mode != "adaptive":
            continue
        if steps == cfg["profile_steps"]:                           # Client Profiler + Budget Controller
            t_est = sim_time / steps
            budget = max(steps, min(cap, int(cfg["deadline"] // t_est)))
            reason = "budget"
        if steps >= cfg["min_steps"] and steps % cfg["check_every"] == 0 and steps < budget:   # Drift Monitor
            checks += 1
            w = cfg["window"]
            if len(losses) >= 2 * w:
                prev, last = float(np.mean(losses[-2 * w:-w])), float(np.mean(losses[-w:]))
                if prev > 0 and (prev - last) / prev < cfg["eps"]:
                    reason = "loss"
                    break
            if cfg["delta"] is not None and (get_vector(model) - w_global).norm().item() > cfg["delta"] * w_norm:
                reason = "drift"
                break

    return ClientResult(get_vector(model) - w_global, steps, sim_time, len(idx), budget, reason, checks)


def _split_like(model, vec: torch.Tensor):
    """Views of a flat vector shaped like the model's parameters."""
    out, pos = [], 0
    for p in model.parameters():
        n = p.numel()
        out.append(vec[pos:pos + n].view_as(p))
        pos += n
    return out
