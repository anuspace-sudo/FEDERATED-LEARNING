"""Phase-1 experiment: baselines and the proposed method on non-IID MNIST.

    .venv\\Scripts\\python.exe run_phase1.py --alpha 0.5 --seeds 0 1
    .venv\\Scripts\\python.exe run_phase1.py --sweep            (Drift Monitor sensitivity of the proposed method)

Results are appended to results/*.csv; finished (method, seed) pairs are skipped,
so an interrupted run can simply be started again.
"""
import argparse
import csv
from pathlib import Path

import torch

from flsim.data import load_dataset
from flsim.run import run

RESULTS = Path(__file__).resolve().parent / "results"
# name -> (method in flsim.run.METHODS, extra settings)
VARIANTS = {
    "fedavg": ("fedavg", {}),
    "fedprox": ("fedprox", {}),
    "fednova": ("fednova", {}),
    "adaptive": ("adaptive", {}),                               # budget + drift monitor (proposed)
    "adaptive_budget_only": ("adaptive", {"min_steps": 10**9}),  # ablation: Drift Monitor switched off
}


# sensitivity of the Drift Monitor (seed 0): loss rule only, distance rule only, both
SWEEP = {
    "eps0.005": {"eps": 0.005}, "eps0.01": {"eps": 0.01}, "eps0.02": {"eps": 0.02}, "eps0.05": {"eps": 0.05},
    "delta0.05_only": {"eps": -1e9, "delta": 0.05}, "delta0.10_only": {"eps": -1e9, "delta": 0.10},
    "eps0.02_delta0.10": {"eps": 0.02, "delta": 0.10},
}


def done_pairs(path: Path) -> set:
    if not path.exists():
        return set()
    with path.open() as f:
        return {(r["variant"], r["seed"]) for r in csv.DictReader(f)}


def save(path: Path, variant: str, rows: list[dict]) -> None:
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["variant", *rows[0].keys()])
        if new:
            wr.writeheader()
        wr.writerows({"variant": variant, **r} for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--sweep", action="store_true", help="Drift Monitor sensitivity of the proposed method (seed 0)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    ds = load_dataset("mnist")

    if a.sweep:
        for alpha in (0.1, 0.5):
            path = RESULTS / f"sweep_driftmonitor_alpha{alpha}.csv"
            done = done_pairs(path)
            for name, extra in SWEEP.items():
                if (name, "0") not in done:
                    save(path, name, run("adaptive", ds, alpha=alpha, seed=0, rounds=a.rounds, **extra))
        return

    path = RESULTS / f"phase1_mnist_alpha{a.alpha}.csv"
    for seed in a.seeds:
        for name, (method, extra) in VARIANTS.items():
            if (name, str(seed)) in done_pairs(path):
                print(f"skip {name} seed {seed} (already in {path.name})")
                continue
            save(path, name, run(method, ds, alpha=a.alpha, seed=seed, rounds=a.rounds, **extra))


if __name__ == "__main__":
    main()
