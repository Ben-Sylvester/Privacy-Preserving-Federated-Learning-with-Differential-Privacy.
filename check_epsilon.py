"""
Offline check of the privacy budget (epsilon) for a DP-SGD / DP-FedAvg configuration.

Please run this before trusting any epsilon in a log or a paper. It recomputes epsilon from
(sampling rate, noise multiplier, number of steps, delta) using the same Renyi-DP
analysis routines that Opacus uses, so it does not depend on how a training run
managed its accountant.

python check_epsilon.py --n 600 --batch 256 --epochs 3 --rounds 100 --sigma 1.1
"""
import argparse
import math

from opacus.accountants.analysis import rdp

ORDERS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))


def epsilon(q, sigma, steps, delta):
    spent = rdp.compute_rdp(q=q, noise_multiplier=sigma, steps=steps, orders=ORDERS)
    return rdp.get_privacy_spent(orders=ORDERS, rdp=spent, delta=delta)[0]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, required=True, help="examples on the node")
    p.add_argument("--batch", type=int, required=True, help="batch size given to the DataLoader")
    p.add_argument("--epochs", type=int, required=True, help="local epochs per round")
    p.add_argument("--rounds", type=int, required=True)
    p.add_argument("--sigma", type=float, required=True)
    p.add_argument("--delta", type=float, default=1e-5)
    a = p.parse_args()

    loader_len = math.ceil(a.n / a.batch)
    q = 1.0 / loader_len                  # Opacus: sample rate = 1 / len(data_loader)
    spr = loader_len * a.epochs
    print(f"sampling rate q = {q:.4f}, steps per round = {spr}, delta = {a.delta:g}")
    print(f"delta vs 1/n = {1 / a.n:.2e}: {'ok' if a.delta < 0.1 / a.n else 'too large'}")
    print(f"{'rounds':>7} {'steps':>7} {'epsilon':>9}")
    for r in sorted({1, max(1, a.rounds // 10), max(1, a.rounds // 2), a.rounds}):
        print(f"{r:>7} {r * spr:>7} {epsilon(q, a.sigma, r * spr, a.delta):>9.2f}")


if __name__ == "__main__":
    main()
