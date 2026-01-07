from model import FederatedModel
from client import DPClientTrainer
from federated_training import federated_training
from data import get_client_loaders

def main():
    global_model = FederatedModel(input_dim=784, output_dim=10)

    loaders = get_client_loaders()  # get all client loaders first
    for loader in loaders:
        print(len(loader.dataset))  # sanity check

    clients = [
        DPClientTrainer(
            model=FederatedModel(784, 10),
            dataloader=loader,
            lr=0.01,
            noise_multiplier=1.0,
            max_grad_norm=1.0
        )
        for loader in loaders
    ]

    try:
        federated_training(
            global_model,
            clients,
            rounds=20,
            local_epochs=1
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Exiting cleanly.")


if __name__ == "__main__":
    main()
