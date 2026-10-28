"""The federated loop: it broadcast, local DP-SGD, budget check, weighted aggregation."""
import time
import torch
from server import fedavg


def _bytes(update):
    return sum(v.numel() * v.element_size() for v in update.values())


def federated_training(global_model, clients, rounds, local_epochs, delta=1e-5,
                       epsilon_budget=8.0, evaluate=None, verbose=True):
    """Run `rounds` rounds and return a list with one record per round.

    A private client whose cumulative epsilon exceeds `epsilon_budget` after a round
    withholds that round's update and takes no further part. Non-private clients
    (sigma == 0) are never excluded."""
    history = []
    for rnd in range(1, rounds + 1):
        t0 = time.time()
        global_state = {k: v.detach().clone() for k, v in global_model.state_dict().items()}
        updates, weights, up_bytes = [], [], 0

        for c in clients:
            if c.exhausted:
                continue
            c.load_global(global_state)
            c.train(local_epochs)
            if c.private and c.epsilon(delta) > epsilon_budget:
                c.exhausted = True
                continue
            u = c.get_model_update(global_state)
            updates.append(u)
            weights.append(c.num_samples)
            up_bytes += _bytes(u)

        if updates:
            fedavg(global_model, updates, weights)

        eps = [c.epsilon(delta) for c in clients if c.private]
        rec = {
            "round": rnd,
            "accuracy": evaluate(global_model) if evaluate else float("nan"),
            "participants": len(updates),
            "exhausted": sum(c.exhausted for c in clients),
            "eps_max": max(eps) if eps else float("inf"),
            "eps_mean": sum(eps) / len(eps) if eps else float("inf"),
            "uplink_mb": up_bytes / 1e6,
            "seconds": time.time() - t0,
        }
        history.append(rec)
        if verbose and (rnd == 1 or rnd % 10 == 0 or rnd == rounds):
            print(f"  round {rnd:3d}/{rounds}  acc {rec['accuracy']:6.2f}%  "
                  f"participants {rec['participants']:3d}  eps_max {rec['eps_max']:.2f}")
    return history
