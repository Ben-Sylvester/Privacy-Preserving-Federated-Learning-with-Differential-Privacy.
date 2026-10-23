import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

def iid_partition(dataset, num_clients):
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    split = np.array_split(indices, num_clients)
    return [s.tolist() for s in split]

def dirichlet_partition(dataset, num_clients, alpha=0.5):
    labels = np.array(dataset.targets)
    num_classes = 10
    K = num_clients
    N = len(dataset)
    idx_batch = [[] for _ in range(K)]

    for c in range(num_classes):
        idx_k = np.where(labels == c)[0]
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(alpha, K))
        # Balance data so no client gets too much
        proportions = np.array([p * (len(idx_j) < N / K) for p, idx_j in zip(proportions, idx_batch)])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
    
    return idx_batch

def get_client_loaders(dataset_name="mnist", partition="iid", num_clients=100, batch_size=256, alpha=0.5):
    if dataset_name == "mnist":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    elif dataset_name == "cifar10":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset")

    if partition == "iid":
        client_indices = iid_partition(dataset, num_clients)
    elif partition == "non-iid":
        client_indices = dirichlet_partition(dataset, num_clients, alpha)
    else:
        raise ValueError("Unsupported partition")

    # Note: drop_last=False is safer for Opacus accounting
    loaders = [DataLoader(Subset(dataset, idx), batch_size=batch_size, shuffle=True) for idx in client_indices]
    return loaders, dataset # Return test dataset for evaluation if needed