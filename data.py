import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

def get_client_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    
    num_clients = 5
    data_per_client = len(dataset) // num_clients
    client_indices = [list(range(i * data_per_client, (i + 1) * data_per_client)) for i in range(num_clients)]
    
    loaders = [DataLoader(Subset(dataset, idx), batch_size=64, shuffle=True) for idx in client_indices]
    return loaders
