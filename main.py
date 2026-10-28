"""Run DP-FedAvg experiments.

For Examples run the following commands:
  python main.py --dataset mnist --partition iid --sigmas 0 0.8 1.1 2.5 --trials 3
  python main.py --dataset mnist --partition non-iid --sigmas 1.1
  python main.py --dataset mnist --partition iid --target-epsilon 8   # To calibrate sigma to a budget
"""
import argparse
import csv
import os

import numpy as np
import torch

import data as D
from client import DPClientTrainer
from federated_training import federated_training
from model import MNIST_MLP, CIFAR10_CNN


def build_model(name):
    return MNIST_MLP() if name == "mnist" else CIFAR10_CNN()


def make_evaluator(test_loader, device):
    def evaluate(model):
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += len(y)
        return 100.0 * correct / total
    return evaluate


def run_trial(dataset, loaders, evaluate, *, sigma, target_epsilon, rounds, local_epochs,
              lr, max_grad_norm, delta, epsilon_budget, device, seed, verbose=True):
    torch.manual_seed(seed)
    global_model = build_model(dataset).to(device)
    clients = [
        DPClientTrainer(build_model(dataset), loader, lr=lr, noise_multiplier=sigma,
                        max_grad_norm=max_grad_norm, device=device,
                        target_epsilon=target_epsilon, target_delta=delta,
                        total_epochs=rounds * local_epochs)
        for loader in loaders
    ]
    history = federated_training(global_model, clients, rounds, local_epochs, delta,
                                 epsilon_budget, evaluate, verbose)
    return history, clients


def summarise(history):
    acc = [h["accuracy"] for h in history]
    last = history[-1]
    first90 = next((h["round"] for h in history if h["accuracy"] >= 90.0), "")
    return {
        "final_acc_mean_last10": float(np.mean(acc[-10:])),
        "final_acc_last": acc[-1],
        "eps_max_final": last["eps_max"],
        "eps_mean_final": last["eps_mean"],
        "participants_mean": float(np.mean([h["participants"] for h in history])),
        "nodes_exhausted_final": last["exhausted"],
        "rounds_to_90": first90,
        "uplink_mb_per_round": float(np.mean([h["uplink_mb"] for h in history])),
    }


def write_rows(path, rows):
    if not rows:
        return
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    p.add_argument("--partition", choices=["iid", "non-iid"], default="iid")
    p.add_argument("--clients", type=int, default=100)
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--local-epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--alpha", type=float, default=0.5, help="Dirichlet concentration")
    p.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.8, 1.1, 2.5])
    p.add_argument("--target-epsilon", type=float, default=None,
                   help="calibrate each client's sigma to this budget over the whole run")
    p.add_argument("--epsilon-budget", type=float, default=None,
                   help="exclusion threshold (defaults to --target-epsilon, else 8.0)")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main():
    a = parse()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    os.makedirs(a.out, exist_ok=True)
    budget = a.epsilon_budget if a.epsilon_budget is not None else (a.target_epsilon or 8.0)
    settings = [("eps", a.target_epsilon)] if a.target_epsilon is not None else [("sigma", s) for s in a.sigmas]

    train = D.load_dataset(a.dataset, True, a.data_root)
    labels = np.asarray(train.targets)
    evaluate = make_evaluator(D.get_test_loader(a.dataset, root=a.data_root), device)

    for kind, value in settings:
        label = f"{kind}{value:g}"
        for t in range(a.trials):
            seed = a.seed + t
            print(f"\n== {a.dataset} {a.partition} K={a.clients} {label} trial {t} (seed {seed}) ==")
            rng = np.random.default_rng(seed)
            loaders, sizes = D.make_loaders(train, labels, a.partition, a.clients, a.batch_size, a.alpha, rng)
            hist, _ = run_trial(
                a.dataset, loaders, evaluate,
                sigma=(value if kind == "sigma" else 0.0),
                target_epsilon=(value if kind == "eps" else None),
                rounds=a.rounds, local_epochs=a.local_epochs, lr=a.lr, max_grad_norm=a.max_grad_norm,
                delta=a.delta, epsilon_budget=budget, device=device, seed=seed)
            stem = f"{a.dataset}_{a.partition}_K{a.clients}_{label}_trial{t}"
            write_rows(os.path.join(a.out, stem + "_rounds.csv"), hist)
            s = summarise(hist)
            s.update(dataset=a.dataset, partition=a.partition, clients=a.clients, setting=label,
                     trial=t, seed=seed, min_shard=min(sizes), max_shard=max(sizes))
            write_rows(os.path.join(a.out, "summary.csv"), [s])
            print(f"  -> accuracy (mean of last 10 rounds) {s['final_acc_mean_last10']:.2f}%, "
                  f"eps_max {s['eps_max_final']:.2f}, nodes excluded {s['nodes_exhausted_final']}")


if __name__ == "__main__":
    main()
