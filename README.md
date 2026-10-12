# Adaptive Local Training for Federated Learning on Heterogeneous Edge Devices

Phase 1 of the mini project: a federated-learning simulator, three baselines
(FedAvg, FedProx, FedNova) and a first version of the proposed method, with a
first experiment on non-IID MNIST.

## What is simulated and what is real

- **Real:** model training, accuracy, loss, the number of local SGD steps.
- **Simulated:** device speed. Each client is given a time per step
  (`t_step = 0.01 s x U(1, 5)`), and a simulated clock advances by `t_step`
  (with 5 % jitter) for every step the client runs. No physical edge device is
  measured in Phase 1. "Round time" is the simulated time of the slowest client.

## Folder layout

| Path | Content |
|---|---|
| `flsim/data.py` | MNIST / CIFAR-10 loading, Dirichlet non-IID partition, per-client hold-out split |
| `flsim/models.py` | MLP (784-200-200-10) and a small CNN |
| `flsim/client.py` | Local training: fixed workload, FedProx term, and the adaptive workload (Client Profiler, Budget Controller, Drift Monitor) |
| `flsim/server.py` | FedAvg and FedNova aggregation, evaluation, per-client accuracy |
| `flsim/run.py` | One training run, default settings, per-round metrics |
| `run_phase1.py` | Runs the Phase-1 experiment and the eps sweep |
| `analyze.py` | Builds `results/summary.md` and the plots from the CSV files |
| `tests/test_core.py` | Unit tests |
| `results/` | CSV logs, plots, `summary.md` |
| `logs/` | Console output of the runs |
| `data/` | Downloaded MNIST (not part of the code) |
| `.venv/` | Python 3.11 virtual environment (PyTorch CPU build) |

## How to run

```
cd /d D:\SATHYA\FL_Adaptive_Training
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe run_phase1.py --alpha 0.5 --seeds 0 1
.venv\Scripts\python.exe run_phase1.py --alpha 0.1 --seeds 0 1
.venv\Scripts\python.exe run_phase1.py --sweep
.venv\Scripts\python.exe analyze.py
```

## Methods

| Name | Local workload | Aggregation |
|---|---|---|
| `fedavg` | fixed E local epochs | sample-weighted average |
| `fedprox` | fixed E local epochs, proximal term (mu = 0.01) | sample-weighted average |
| `fednova` | fixed E local epochs | normalized averaging |
| `adaptive` (proposed) | `tau_i = min(tau_budget, tau_stop)` | normalized averaging |
| `adaptive_budget_only` (ablation) | `tau_i = tau_budget` (Drift Monitor off) | normalized averaging |

Proposed method, per client and round:

1. **Client Profiler** averages the simulated duration of the first 5 steps to estimate `t_step`.
2. **Budget Controller** sets `tau_budget = floor(D / t_step)`.
3. **Drift Monitor** checks every 10 steps (after 20 steps) and stops when the
   relative decrease of the 20-step mean loss is below `eps`, or when
   `||w - w_t|| > delta * ||w_t||`.
4. The server aggregates with FedNova's normalized averaging.

## Settings used in Phase 1 (design choices, not yet tuned)

- 20 clients, all take part in every round; 30 rounds; E = 5; batch size 32; SGD lr 0.05, no momentum.
- Dirichlet alpha = 0.5 and 0.1; 10 % of each client's samples held out for per-client accuracy.
- Round deadline `D` = median over clients of the simulated time of their fixed (E-epoch) workload.
- `tau_budget` is also capped at the client's fixed workload, so a fast client never trains more than under FedAvg.
  This cap is an addition to the rule on the slides.
- `eps = 0.02`; the distance threshold `delta` is switched off (`None`) in Phase 1, so only the loss-progress rule and the deadline are active.
- `delta` is implemented relative to `||w_t||`, a small change from the absolute threshold on the slides.

## Not done in Phase 1

CIFAR-10 runs (the loader and CNN exist but were not run), partial client
participation, real-device measurements, energy / CO2 measurement, tuning of
`eps`, `delta` and `D`, and FedProx's partial-work variant for stragglers.
