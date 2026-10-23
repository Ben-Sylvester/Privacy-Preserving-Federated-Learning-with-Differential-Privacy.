# federated_training.py
from server import fedavg

def federated_training(global_model, clients, rounds, local_epochs, delta=1e-5, epsilon_budget=8.0):
    for rnd in range(rounds):
        # Print progress at the start, every 10 rounds, and on the final round
        if rnd == 0 or (rnd + 1) % 10 == 0 or rnd == rounds - 1:
            print(f"\n🌐 Round {rnd + 1}/{rounds} in progress...")
            
        client_updates = []
        epsilons = []
        
        for client in clients:
            loss = client.train(local_epochs)
            eps = client.get_epsilon(delta)
            
            # Stop training this client if it exceeds the privacy budget
            if eps > epsilon_budget:
                continue 
                
            update = client.get_model_update(global_model)
            client_updates.append(update)
            epsilons.append(eps)
            
        if client_updates:
            fedavg(global_model, client_updates)
            
        # Print a clean summary every 10 rounds
        if (rnd + 1) % 10 == 0 or rnd == rounds - 1:
            if epsilons:
                print(f"  ↳ Round {rnd + 1} Complete. Max ε: {max(epsilons):.2f}")
            else:
                print(f"  ↳ Round {rnd + 1} Complete. No valid clients (privacy budget exceeded).")