import torch
from model import MNIST_MLP, CIFAR10_CNN
from client import DPClientTrainer
from federated_training import federated_training
from data import get_client_loaders
from torchvision import datasets, transforms

def evaluate_model(model, dataset_name, device="cpu"):
    model.eval()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) if dataset_name == "mnist" else \
                transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    test_data = datasets.MNIST("./data", train=False, transform=transform) if dataset_name == "mnist" else \
                datasets.CIFAR10("./data", train=False, transform=transform)
    loader = torch.utils.data.DataLoader(test_data, batch_size=1000, shuffle=False)
    
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
    return 100.0 * correct / len(test_data)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name = "mnist"      # Change to "cifar10" for CIFAR experiments
    partition = "iid"           # Change to "non-iid" for Dirichlet experiments
    num_clients = 100
    rounds = 100
    local_epochs = 3
    batch_size = 256
    lr = 0.01
    max_grad_norm = 1.0
    delta = 1e-5
    
    # Paper Table III: Noise multipliers to test
    sigmas = [0.0, 0.8, 1.1, 2.5] 
    
    loaders, _ = get_client_loaders(dataset_name, partition, num_clients, batch_size)
    
    for sigma in sigmas:
        print(f"\n{'='*20} Testing σ = {sigma} {'='*20}")
        
        # Initialize Model
        global_model = MNIST_MLP() if dataset_name == "mnist" else CIFAR10_CNN()
        global_model = global_model.to(device)
        
        # Initialize Clients
        clients = [
            DPClientTrainer(
                model=MNIST_MLP() if dataset_name == "mnist" else CIFAR10_CNN(),
                dataloader=loader,
                lr=lr,
                noise_multiplier=sigma,
                max_grad_norm=max_grad_norm,
                device=device
            )
            for loader in loaders
        ]
        
        # Train
        federated_training(global_model, clients, rounds, local_epochs, delta)
        
        # Evaluate
        final_eps = clients[0].get_epsilon(delta) if sigma > 0 else float('inf')
        acc = evaluate_model(global_model, dataset_name, device)
        print(f"\n✅ Final Results for σ={sigma}: Accuracy = {acc:.2f}%, ε = {final_eps:.2f}")

if __name__ == "__main__":
    main()