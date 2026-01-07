import torch


def fedavg(global_model, client_updates):
    global_state = global_model.state_dict()

    for key in global_state.keys():
        stacked = torch.stack([u[key] for u in client_updates])
        global_state[key] += stacked.mean(dim=0)

    global_model.load_state_dict(global_state)
