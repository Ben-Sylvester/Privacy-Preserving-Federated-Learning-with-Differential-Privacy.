"""
The centralised baseline: the same architectures and also trained on the pooled training set.
"""
import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn

import data as D
from main import build_model, make_evaluator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results")
    p.add_argument("--data-root", default="./data")
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)

    train = D.load_dataset(a.dataset, True, a.data_root)
    loader = torch.utils.data.DataLoader(train, batch_size=a.batch_size, shuffle=True)
    evaluate = make_evaluator(D.get_test_loader(a.dataset, root=a.data_root), device)

    rows = []
    for t in range(a.trials):
        torch.manual_seed(a.seed + t)
        model = build_model(a.dataset).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=a.lr)
        crit = nn.CrossEntropyLoss()
        accs = []
        for ep in range(1, a.epochs + 1):
            model.train()
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                crit(model(x), y).backward()
                opt.step()
            accs.append(evaluate(model))
            print(f"trial {t} epoch {ep:2d}  acc {accs[-1]:.2f}%")
        rows.append({"dataset": a.dataset, "trial": t, "epochs": a.epochs,
                     "acc_last": accs[-1], "acc_mean_last3": float(np.mean(accs[-3:]))})
    path = os.path.join(a.out, f"centralised_{a.dataset}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("wrote", path)


if __name__ == "__main__":
    main()
