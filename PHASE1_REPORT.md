# Phase-1 report

All numbers come from the CSV files in `results/` (see `results/summary.md` for the full tables).
Setting: MNIST, MLP 784-200-200-10, 20 clients (all take part every round), 30 rounds, E = 5,
batch size 32, SGD lr 0.05. Device speed is simulated, so "time" is simulated seconds.

## Main comparison (mean of seeds 0 and 1)

| alpha | Method | Test acc. (%) | Worst-client acc. (%) | Local steps vs FedAvg | Simulated time (s) |
|---|---|---|---|---|---|
| 0.5 | FedAvg | 97.59 | 95.86 | 100% | 903 |
| 0.5 | FedProx | 97.54 | 95.97 | 100% | 903 |
| 0.5 | FedNova | 97.58 | 96.42 | 100% | 903 |
| 0.5 | Proposed, deadline budget only | 97.50 | 95.83 | 79.7% | 356 |
| 0.5 | Proposed, budget + loss-progress stop (eps = 0.02) | 95.52 | 92.38 | 14.1% | 116 |
| 0.1 | FedAvg | 96.39 | 92.03 | 100% | 1,046 |
| 0.1 | FedProx | 96.34 | 91.92 | 100% | 1,046 |
| 0.1 | FedNova | 96.55 | 93.52 | 100% | 1,046 |
| 0.1 | Proposed, deadline budget only | 96.31 | 92.68 | 67.3% | 308 |
| 0.1 | Proposed, budget + loss-progress stop (eps = 0.02) | 93.32 | 83.08 | 15.1% | 144 |

## Drift Monitor sensitivity (seed 0 only)

| alpha | Variant | Test acc. (%) | Worst-client acc. (%) | Local steps | Steps vs FedAvg (seed 0) |
|---|---|---|---|---|---|
| 0.5 | FedAvg | 97.59 | 95.60 | 254,400 | 100% |
| 0.5 | budget only | 97.50 | 95.33 | 196,408 | 77.2% |
| 0.5 | budget + distance stop, delta = 0.10 | 97.47 | 96.00 | 171,794 | 67.5% |
| 0.5 | budget + distance stop, delta = 0.05 | 96.63 | 93.75 | 62,268 | 24.5% |
| 0.5 | budget + loss stop, eps = 0.005 / 0.01 / 0.02 / 0.05 | 95.63 / 95.51 / 95.47 / 95.39 | 92.06 / 92.86 / 91.96 / 91.12 | 36,890 / 35,730 / 35,110 / 33,230 | 13-15% |
| 0.1 | FedAvg | 96.88 | 94.44 | 254,550 | 100% |
| 0.1 | budget only | 96.62 | 94.44 | 161,259 | 63.4% |
| 0.1 | budget + distance stop, delta = 0.10 | 96.31 | 94.44 | 143,642 | 56.4% |
| 0.1 | budget + distance stop, delta = 0.05 | 95.26 | 88.89 | 50,690 | 19.9% |
| 0.1 | budget + loss stop, eps = 0.005 / 0.01 / 0.02 / 0.05 | 94.44 / 94.46 / 94.53 / 94.26 | 88.89 (all) | 36,250 / 36,585 / 35,340 / 33,870 | 13-14% |

## Findings

1. The deadline budget with normalized aggregation keeps accuracy within 0.1 points of FedAvg
   while using 67-80% of the local steps and 34-39% of the simulated time.
2. The loss-progress stopping rule, as implemented, stops too early. It ended 87-98% of the
   client-rounds after about 60 steps and cost 2-3 points of accuracy. Changing eps from 0.005
   to 0.05 hardly changes this, so the rule itself (a 20-step window on a noisy mini-batch loss)
   needs redesign, not just a different eps.
3. The distance stopping rule with delta = 0.10 looks more promising: in the single seed tested
   it reached accuracy within 0.6 points of FedAvg with 56-68% of the local steps.

## Limits

- Device speed and time are simulated; no real device was measured.
- The time saving depends on the deadline, which is set to the median client's full-workload time.
- Main comparison: two seeds. Sensitivity table: one seed.
- One dataset (MNIST) and one model. CIFAR-10 was not run.
- Worst-client accuracy is coarse at alpha = 0.1, because the smallest client's hold-out set has only 18-25 samples.
- 30 rounds; no method reached 97% at alpha = 0.1 in that time.

## Next phase

Redesign the loss-progress rule, repeat the distance rule over more seeds and delta values,
run CIFAR-10, add partial client participation, and measure energy.
