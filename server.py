"""Server-side aggregation."""
import torch


def fedavg(global_model, client_updates, client_weights):
    """Weighted FedAvg: w <- w + sum_k (n_k / sum_j n_j) * delta_k.

    `client_updates` is a list of state-dict deltas (local - global); `client_weights`
    holds the number of training examples n_k of each participating client."""
    total = float(sum(client_weights))
    state = global_model.state_dict()
    for key in state:
        agg = sum(u[key].to(state[key].device) * (w / total)
                  for u, w in zip(client_updates, client_weights))
        state[key] = state[key] + agg.to(state[key].dtype)
    global_model.load_state_dict(state)
