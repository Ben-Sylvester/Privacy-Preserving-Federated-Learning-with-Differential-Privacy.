# DP-FedAvg: federated learning with differentially private local training

A simulation of federated averaging (FedAvg) in which every client trains with
DP-SGD (per-example gradient clipping plus Gaussian noise) and a Rényi-DP accountant
tracks the privacy budget each client spends. Clients are simulated in one process.
The code supports MNIST (MLP) and CIFAR-10 (small CNN), balanced (IID) and
label-skewed (Dirichlet) partitions, a centralised baseline, an offline ε audit, and a
gradient-inversion attack with and without DP noise.

## Files

| File | Purpose |
|---|---|
| `main.py` | Runs federated experiments; writes per-round and summary CSV files |
| `federated_training.py` | The round loop: broadcast, local training, budget check, aggregation |
| `client.py` | One simulated institution (local DP-SGD with Opacus) |
| `server.py` | Sample-weighted FedAvg |
| `data.py` | Datasets, IID and Dirichlet partitions |
| `model.py` | MNIST MLP (784-128-64-10) and CIFAR-10 CNN |
| `centralised.py` | Centralised baseline on the pooled training set |
| `attack.py` | Gradient inversion against a single batch gradient, with and without DP noise |
| `audit_epsilon.py` | Recomputes ε from (sampling rate, σ, steps, δ) without running training |
| `smoke_test.py` | Checks the mechanics on synthetic data; no download needed |

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python smoke_test.py               # about a minute on a CPU
```

The smoke test checks that clients start each round from the global model, that
aggregation is sample-weighted, that non-private clients are never excluded, that
clients exceeding the budget are excluded, and that a calibrated σ keeps ε within the
target. It uses synthetic data, so it says nothing about accuracy on MNIST or CIFAR-10.

## Running experiments

Datasets are downloaded to `./data` on first use. A full run (100 clients, 100 rounds,
3 local epochs) is slow on a CPU; a GPU is strongly recommended.

```bash
# centralised baseline
python centralised.py --dataset mnist

# federated runs, fixed noise multipliers (sigma = 0 means no DP)
python main.py --dataset mnist --partition iid     --sigmas 0 0.8 1.1 2.5 --trials 3
python main.py --dataset mnist --partition non-iid --sigmas 0 1.1 --trials 3

# choose a privacy budget and let each client's sigma be calibrated to it
python main.py --dataset mnist --partition iid --target-epsilon 8 --trials 3

# number of clients
for k in 5 10 20 50 100; do python main.py --clients $k --sigmas 0 1.1 --out results_scale; done

# gradient inversion
python attack.py --data mnist --batch 1 --sigmas 0 1.1 2.5 --trials 10
```

Each run writes `results/<dataset>_<partition>_K<clients>_<setting>_trial<t>_rounds.csv`
(accuracy, participants, ε, uplink size and time for every round) and appends one row per
run to `results/summary.csv`. Reported accuracy is the mean of the last ten rounds.

## How privacy is accounted

- Opacus draws each batch by Poisson sampling at rate `q = 1 / len(DataLoader)`. With
  600 examples per client and batch size 256, `q ≈ 1/3`, which gives weak subsampling
  amplification.
- Each client keeps one accountant for the whole run, so ε is cumulative over all rounds
  and local epochs. A client whose cumulative ε exceeds `--epsilon-budget` withholds that
  round's update and stops participating.
- The guarantee is example-level for the examples held by each client. It does not hide
  whether a whole institution took part.
- With a fixed σ, ε can exceed the intended budget quickly. At `q ≈ 1/3`, σ = 1.1 reaches
  about ε = 7 after one round and far more after 100. Use `--target-epsilon` to calibrate σ
  for the whole run, and check any reported ε with
  `python check_epsilon.py --n 600 --batch 256 --epochs 3 --rounds 100 --sigma 1.1`.
  Note: This script computes the privacy budget assuming a uniform client dataset size of n samples. Actual per-client ε may vary slightly due to partition size differences (particularly under Dirichlet partitioning). However, evaluating with a representative or minimum n provides a safe, conservative estimate of the privacy guarantee.
- Under the Dirichlet partition, shard sizes differ, so each client has its own `q` and its
  own calibrated σ. Summary files report the maximum and mean ε across clients.


## Attack evaluation

`attack.py` gives the attacker one mini-batch gradient from a randomly initialised MLP.
For the defended case the gradient is formed as in DP-SGD (per-example clipping to norm C,
Gaussian noise of standard deviation σC, division by the batch size). The default method
matches gradients with a cosine distance and Adam; `--method lbfgs` runs the original
Euclidean L-BFGS formulation. PSNR is computed on pixels in [0, 1].

Two cautions. First, this is the single-step setting the attack was designed for, not an
attack on a multi-epoch FedAvg update. Second, a comparison with DP only means something
if the σ = 0 row shows the attack working. The attack recovers single images reliably but
generally fails on larger batches even without DP, so run several trials, report the
median, and state the batch size.

## Scope and limitations

- Everything runs in one process with in-memory communication. There is no network,
  latency, client dropout or hardware variation.
- Secure aggregation and transport encryption are not implemented. The server sees each
  client's (noisy) update.
- MNIST and CIFAR-10 are benchmarks, not clinical or financial data.
- Runs are seeded per trial (`--seed` plus the trial index) for the partition and
  initialisation. GPU kernels can still introduce small nondeterminism.
