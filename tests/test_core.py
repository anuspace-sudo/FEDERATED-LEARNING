"""Unit tests for the parts whose correctness the comparison depends on.
Run from the project folder:  .venv\\Scripts\\python.exe -m pytest -q
"""
import numpy as np
import torch

from flsim.client import ClientResult, fixed_workload, local_train
from flsim.data import Dataset, dirichlet_partition, train_holdout_split
from flsim.models import MLP, get_vector, set_vector
from flsim.server import aggregate_fedavg, aggregate_fednova


def toy_dataset(n=600, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 256, (n, 1, 28, 28), generator=g, dtype=torch.uint8)
    y = torch.randint(0, 10, (n,), generator=g)
    return Dataset("mnist", x, y, x[:100], y[:100], torch.tensor([0.1307]).view(1, 1, 1, 1), torch.tensor([0.3081]).view(1, 1, 1, 1), 10)


def res(delta, steps, n):
    return ClientResult(torch.tensor(delta), steps, 0.0, n, steps, "fixed", 0)


def test_partition_covers_every_sample_once():
    labels = np.random.default_rng(0).integers(0, 10, 5000)
    parts = dirichlet_partition(labels, 10, 0.5, np.random.default_rng(1))
    allidx = np.concatenate(parts)
    assert len(allidx) == 5000 and len(np.unique(allidx)) == 5000
    assert min(len(p) for p in parts) >= 50


def test_smaller_alpha_is_more_skewed():
    labels = np.random.default_rng(0).integers(0, 10, 20000)

    def mean_top_class_share(alpha):
        parts = dirichlet_partition(labels, 10, alpha, np.random.default_rng(2))
        return np.mean([np.bincount(labels[p], minlength=10).max() / len(p) for p in parts])

    assert mean_top_class_share(0.1) > mean_top_class_share(10.0)


def test_holdout_split_is_disjoint():
    parts = [np.arange(0, 100), np.arange(100, 250)]
    tr, ho = train_holdout_split(parts, 0.1, np.random.default_rng(0))
    for p, a, b in zip(parts, tr, ho):
        assert len(np.intersect1d(a, b)) == 0 and len(a) + len(b) == len(p)


def test_fedavg_is_sample_weighted_mean():
    w = torch.zeros(2)
    out = aggregate_fedavg(w, [res([1.0, 0.0], 5, 100), res([0.0, 1.0], 5, 300)])
    assert torch.allclose(out, torch.tensor([0.25, 0.75]))


def test_fednova_equals_fedavg_when_steps_are_equal():
    w = torch.tensor([1.0, -1.0])
    rs = [res([1.0, 2.0], 7, 100), res([-3.0, 1.0], 7, 300)]
    assert torch.allclose(aggregate_fednova(w, rs), aggregate_fedavg(w, rs))


def test_fednova_removes_step_count_bias():
    # two equal-size clients with the same per-step direction but 10 vs 30 steps
    w = torch.zeros(1)
    rs = [res([10.0], 10, 100), res([30.0], 30, 100)]
    assert torch.allclose(aggregate_fednova(w, rs), torch.tensor([20.0]))      # tau_eff = 20, per-step update = 1
    rs = [res([10.0], 10, 100), res([-30.0], 30, 100)]
    assert torch.allclose(aggregate_fednova(w, rs), torch.tensor([0.0]))       # opposite directions cancel
    assert not torch.allclose(aggregate_fedavg(w, rs), torch.tensor([0.0]))   # FedAvg is pulled to the 30-step client


def test_fixed_mode_runs_exactly_e_epochs():
    ds, model = toy_dataset(), MLP()
    cfg = dict(lr=0.05, batch_size=32, epochs=2)
    r = local_train(model, get_vector(model), ds, np.arange(100), 0.01, cfg, "fixed", np.random.default_rng(0))
    assert r.steps == fixed_workload(100, 2, 32) == 8 and r.stop_reason == "fixed"


def test_budget_controller_respects_deadline():
    ds, model = toy_dataset(), MLP()
    cfg = dict(lr=0.05, batch_size=32, epochs=50, time_jitter=0.0, deadline=0.305, profile_steps=5,
               min_steps=10**9, check_every=10, window=20, eps=0.0, delta=None)
    r = local_train(model, get_vector(model), ds, np.arange(600), 0.01, cfg, "adaptive", np.random.default_rng(0))
    assert r.steps == 30 and r.stop_reason == "budget" and r.sim_time <= 0.305


def test_budget_is_capped_at_the_fixed_workload():
    ds, model = toy_dataset(), MLP()
    cfg = dict(lr=0.05, batch_size=32, epochs=1, time_jitter=0.0, deadline=1e9, profile_steps=5,
               min_steps=10**9, check_every=10, window=20, eps=0.0, delta=None)
    r = local_train(model, get_vector(model), ds, np.arange(600), 0.01, cfg, "adaptive", np.random.default_rng(0))
    assert r.steps == fixed_workload(600, 1, 32)


def test_drift_monitor_stops_on_distance():
    ds, model = toy_dataset(), MLP()
    cfg = dict(lr=0.05, batch_size=32, epochs=50, time_jitter=0.0, deadline=1e9, profile_steps=5,
               min_steps=10, check_every=10, window=20, eps=-1e9, delta=1e-6)
    r = local_train(model, get_vector(model), ds, np.arange(600), 0.01, cfg, "adaptive", np.random.default_rng(0))
    assert r.stop_reason == "drift" and r.steps == 10


def test_vector_roundtrip():
    m = MLP()
    v = get_vector(m)
    set_vector(m, v * 0 + 1.0)
    assert torch.all(get_vector(m) == 1.0)
