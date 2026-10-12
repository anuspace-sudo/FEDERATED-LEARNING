"""One federated training run (all clients take part in every round, synchronous rounds)."""
import csv
import time
from pathlib import Path

import numpy as np
import torch

from .client import fixed_workload, local_train
from .data import Dataset, dirichlet_partition, train_holdout_split
from .models import build_model, get_vector
from .server import aggregate_fedavg, aggregate_fednova, client_accuracies, evaluate

# method -> (client mode, FedProx term, aggregator)
METHODS = {
    "fedavg": ("fixed", False, aggregate_fedavg),
    "fedprox": ("fixed", True, aggregate_fedavg),
    "fednova": ("fixed", False, aggregate_fednova),
    "adaptive": ("adaptive", False, aggregate_fednova),     # proposed method
}

DEFAULTS = dict(
    dataset="mnist", model="mlp", num_clients=20, alpha=0.5, holdout_frac=0.1,
    rounds=40, epochs=5, batch_size=32, lr=0.05, mu=0.01,
    # simulated device speed: t_step = base_step_time * U(speed_min, speed_max)
    base_step_time=0.01, speed_min=1.0, speed_max=5.0, time_jitter=0.05,
    # proposed method
    deadline_quantile=0.5,   # D = this quantile of the clients' fixed-workload times
    profile_steps=5, min_steps=20, check_every=10, window=20, eps=0.02, delta=None,
    seed=0,
)

FIELDS = ["method", "seed", "alpha", "round", "test_acc", "test_loss", "round_time", "cum_time", "round_steps", "cum_steps",
          "client_acc_min", "client_acc_low10", "client_acc_mean", "stop_fixed", "stop_budget", "stop_loss", "stop_drift",
          "drift_checks", "deadline"]


def setup(cfg: dict, ds: Dataset):
    """Client data split, device speeds and round deadline. Depends only on the
    seed and data settings, so every method in a comparison sees the same clients."""
    rng = np.random.default_rng(cfg["seed"])
    parts = dirichlet_partition(ds.y_train.numpy(), cfg["num_clients"], cfg["alpha"], rng)
    train, hold = train_holdout_split(parts, cfg["holdout_frac"], rng)
    t_step = cfg["base_step_time"] * rng.uniform(cfg["speed_min"], cfg["speed_max"], cfg["num_clients"])
    full_time = np.array([fixed_workload(len(tr), cfg["epochs"], cfg["batch_size"]) * t for tr, t in zip(train, t_step)])
    deadline = float(np.quantile(full_time, cfg["deadline_quantile"]))
    return train, hold, t_step, deadline


def run(method: str, ds: Dataset, out_csv: Path | None = None, verbose: bool = True, **overrides) -> list[dict]:
    cfg = {**DEFAULTS, **overrides}
    mode, prox, aggregate = METHODS[method]
    train, hold, t_step, deadline = setup(cfg, ds)
    cfg.update(prox=prox, deadline=deadline)

    torch.manual_seed(cfg["seed"])
    model = build_model(cfg["model"], cfg["dataset"])
    w = get_vector(model)
    rng = np.random.default_rng(cfg["seed"] + 1)          # batch order and time jitter
    rows, cum_time, cum_steps, t0 = [], 0.0, 0, time.time()

    for rnd in range(1, cfg["rounds"] + 1):
        results = [local_train(model, w, ds, train[c], t_step[c], cfg, mode, rng) for c in range(cfg["num_clients"])]
        w = aggregate(w, results)
        round_time = max(r.sim_time for r in results)      # the server waits for the slowest client
        round_steps = sum(r.steps for r in results)
        cum_time += round_time
        cum_steps += round_steps
        acc, loss = evaluate(model, w, ds, ds.x_test, ds.y_test)
        cacc = np.sort(client_accuracies(model, w, ds, hold))
        k = max(1, int(np.ceil(0.1 * len(cacc))))
        reasons = [r.stop_reason for r in results]
        rows.append(dict(
            method=method, seed=cfg["seed"], alpha=cfg["alpha"], round=rnd, test_acc=round(acc, 5), test_loss=round(loss, 5),
            round_time=round(round_time, 3), cum_time=round(cum_time, 3), round_steps=round_steps, cum_steps=cum_steps,
            client_acc_min=round(float(cacc[0]), 5), client_acc_low10=round(float(cacc[:k].mean()), 5),
            client_acc_mean=round(float(cacc.mean()), 5), stop_fixed=reasons.count("fixed"), stop_budget=reasons.count("budget"),
            stop_loss=reasons.count("loss"), stop_drift=reasons.count("drift"), drift_checks=sum(r.drift_checks for r in results),
            deadline=round(deadline, 3),
        ))
        if verbose and (rnd % 5 == 0 or rnd == 1):
            print(f"  [{method} a={cfg['alpha']} s={cfg['seed']}] round {rnd:3d}  acc {acc:.4f}  worst-client {cacc[0]:.4f}  "
                  f"steps {round_steps}  round_time {round_time:.1f}s  ({time.time() - t0:.0f}s real)", flush=True)

    if out_csv is not None:
        new = not out_csv.exists()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("a", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                wr.writeheader()
            wr.writerows(rows)
    return rows
