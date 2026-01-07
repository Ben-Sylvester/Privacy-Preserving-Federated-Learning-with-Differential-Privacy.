from server import fedavg

def federated_training(
    global_model,
    clients,
    rounds,
    local_epochs,
    delta=1e-5,
    epsilon_budget=8.0,
):
    for rnd in range(rounds):
        print(f"\n🌐 Round {rnd + 1}")

        client_updates = []
        epsilons = []

        for client in clients:
            loss = client.train(local_epochs)
            eps = client.get_epsilon(delta)

            print(f"Client loss: {loss:.4f}, ε: {eps:.2f}")  # ✅ print once per client

            if eps > epsilon_budget:
                print("Client exceeded privacy budget — skipped")
                continue

            update = client.get_model_update(global_model)
            client_updates.append(update)
            epsilons.append(eps)

        if client_updates:
            fedavg(global_model, client_updates)

        if epsilons:
            print(f"Round ε (max): {max(epsilons):.2f}")
        else:
            print("No valid clients")
