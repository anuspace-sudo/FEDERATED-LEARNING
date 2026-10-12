"""Summarise results/*.csv into results/summary.md and plots.

    .venv\\Scripts\\python.exe analyze.py
Every number is computed from the CSV files written by run_phase1.py.
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"
TARGETS = (0.90, 0.95, 0.97)
STYLE = {"fedavg": ("#000000", "-"), "fedprox": ("#7f7f7f", "--"), "fednova": ("#1f77b4", "-."),
         "adaptive": ("#d62728", "-"), "adaptive_budget_only": ("#ff7f0e", ":")}


def load(path: Path) -> dict:
    """variant -> seed -> list of per-round rows (numbers as float)."""
    out = defaultdict(lambda: defaultdict(list))
    seen = set()
    with path.open() as f:
        for r in csv.DictReader(f):
            key = (r["variant"], r["seed"], r["round"])
            if key in seen:           # the same run saved twice (two copies of the script ran at once)
                continue
            seen.add(key)
            out[r["variant"]][r["seed"]].append({k: (float(v) if k not in ("variant", "method") else v) for k, v in r.items()})
    return out


def first_reach(rows: list[dict], target: float, key: str):
    for r in rows:
        if r["test_acc"] >= target:
            return r[key]
    return None


def fmt_reach(vals: list, n_seeds: int, scale: float = 1.0, digits: int = 0) -> str:
    hit = [v for v in vals if v is not None]
    if not hit:
        return "not reached"
    s = f"{np.mean(hit) / scale:.{digits}f}"
    return s if len(hit) == n_seeds else f"{s} ({len(hit)}/{n_seeds} seeds)"


def summarise(path: Path) -> list[str]:
    data = load(path)
    seeds = sorted({s for v in data.values() for s in v})
    rounds = int(max(r["round"] for v in data.values() for rows in v.values() for r in rows))
    base = np.mean([rows[-1]["cum_steps"] for rows in data["fedavg"].values()]) if "fedavg" in data else None
    L = [f"### {path.name}", "",
         f"Seeds: {', '.join(seeds)}. Rounds: {rounds}. Values are means over the seeds; accuracy is on the MNIST test set.", "",
         "| Method | Final acc. (%) | Worst-client acc. (%) | Total local steps | Steps vs FedAvg | Total simulated time (s) |",
         "|---|---|---|---|---|---|"]
    for name, per_seed in data.items():
        last = [rows[-1] for rows in per_seed.values()]
        steps = np.mean([r["cum_steps"] for r in last])
        L.append(f"| {name} | {100 * np.mean([r['test_acc'] for r in last]):.2f} | {100 * np.mean([r['client_acc_min'] for r in last]):.2f} | "
                 f"{steps:,.0f} | {('%.1f%%' % (100 * steps / base)) if base else '-'} | {np.mean([r['cum_time'] for r in last]):,.0f} |")
    L += ["", "Cost to first reach a target test accuracy (local steps in thousands / simulated seconds):", "",
          "| Method | " + " | ".join(f"{int(t * 100)}% steps (k)" for t in TARGETS) + " | " + " | ".join(f"{int(t * 100)}% time (s)" for t in TARGETS) + " |",
          "|---|" + "---|" * (2 * len(TARGETS))]
    for name, per_seed in data.items():
        cells = [fmt_reach([first_reach(rows, t, "cum_steps") for rows in per_seed.values()], len(per_seed), 1000, 1) for t in TARGETS]
        cells += [fmt_reach([first_reach(rows, t, "cum_time") for rows in per_seed.values()], len(per_seed)) for t in TARGETS]
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    if any(n.startswith("adaptive") for n in data):
        L += ["", "Why the proposed method's clients stopped (share of all client-rounds):", "",
              "| Method | deadline budget | loss progress < eps | drift distance | mean steps per client-round |", "|---|---|---|---|---|"]
        for name, per_seed in data.items():
            if not name.startswith("adaptive"):
                continue
            rows = [r for rs in per_seed.values() for r in rs]
            tot = sum(r["stop_budget"] + r["stop_loss"] + r["stop_drift"] + r["stop_fixed"] for r in rows)
            L.append(f"| {name} | {100 * sum(r['stop_budget'] for r in rows) / tot:.1f}% | {100 * sum(r['stop_loss'] for r in rows) / tot:.1f}% | "
                     f"{100 * sum(r['stop_drift'] for r in rows) / tot:.1f}% | {sum(r['round_steps'] for r in rows) / tot:.0f} |")
    return L + [""]


def plot(path: Path) -> Path:
    data = load(path)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for name, per_seed in data.items():
        color, ls = STYLE.get(name, (None, "-"))
        rows = list(per_seed.values())
        n = min(len(r) for r in rows)
        mean = lambda k: np.mean([[r[k] for r in rs[:n]] for rs in rows], axis=0)
        ax[0].plot(mean("cum_steps") / 1000, 100 * mean("test_acc"), ls, color=color, label=name, lw=2)
        ax[1].plot(mean("cum_time"), 100 * mean("test_acc"), ls, color=color, label=name, lw=2)
        ax[2].plot(mean("round"), 100 * mean("client_acc_min"), ls, color=color, label=name, lw=2)
    ax[0].set(xlabel="Cumulative local SGD steps (thousands)", ylabel="Test accuracy (%)", title="Accuracy vs computation")
    ax[1].set(xlabel="Cumulative simulated time (s)", ylabel="Test accuracy (%)", title="Accuracy vs simulated time")
    ax[2].set(xlabel="Round", ylabel="Worst-client accuracy (%)", title="Worst-client accuracy")
    lo = 80 if "sweep" not in path.name else 70
    for a in ax[:2]:
        a.set_ylim(lo, 100)
    for a in ax:
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=9)
    fig.suptitle(path.stem + " (mean over seeds)")
    fig.tight_layout()
    out = path.with_suffix(".png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


if __name__ == "__main__":
    lines = ["# Phase-1 results (generated by analyze.py)", ""]
    for p in sorted(RESULTS.glob("*.csv")):
        lines += summarise(p)
        print("plot:", plot(p).name)
    (RESULTS / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
