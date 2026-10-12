"""Short live demo for a review: trains FedAvg and the proposed method (deadline
budget) for a few rounds and prints a comparison table. Nothing is read from the
saved results and nothing is written to disk.

    .venv\\Scripts\\python.exe demo.py
    .venv\\Scripts\\python.exe demo.py --seed 7 --rounds 5
"""
import argparse
import time

import torch

from flsim.data import load_dataset
from flsim.run import run

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0, help="any number; a different seed gives different devices and data split")
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--alpha", type=float, default=0.5, help="smaller = more uneven data across devices")
a = ap.parse_args()

torch.set_num_threads(6)
print(f"Loading MNIST and creating 20 simulated devices (seed {a.seed}, alpha {a.alpha}) ...")
ds = load_dataset("mnist")

runs = [("FedAvg (fixed 5 epochs)", "fedavg", {}), ("Proposed (deadline budget)", "adaptive", {"min_steps": 10**9})]
out = []
for label, method, extra in runs:
    print(f"\nTraining: {label}")
    t0 = time.time()
    rows = run(method, ds, alpha=a.alpha, seed=a.seed, rounds=a.rounds, verbose=False, **extra)
    for r in rows:
        print(f"  round {int(r['round'])}: test accuracy {100 * r['test_acc']:.2f}%   local steps {r['round_steps']}   "
              f"simulated round time {r['round_time']:.1f}s")
    out.append((label, rows[-1], time.time() - t0))

base = out[0][1]["cum_steps"]
print(f"\nResult after {a.rounds} rounds (computed just now)")
print(f"{'Method':<28}{'Accuracy':>10}{'Worst device':>14}{'Local steps':>13}{'vs FedAvg':>11}{'Sim. time':>11}{'Real time':>11}")
for label, r, real in out:
    print(f"{label:<28}{100 * r['test_acc']:>9.2f}%{100 * r['client_acc_min']:>13.2f}%{r['cum_steps']:>13,}"
          f"{100 * r['cum_steps'] / base:>10.1f}%{r['cum_time']:>10.0f}s{real:>10.0f}s")
print("\nAccuracy is measured on the 10,000 MNIST test images. Device speed and 'Sim. time' are simulated.")
