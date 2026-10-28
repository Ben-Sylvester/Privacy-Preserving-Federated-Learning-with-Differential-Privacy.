"""Datasets and client partitions (IID and Dirichlet label skew)."""
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

STATS = {
    "mnist": ((0.1307,), (0.3081,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
}


def load_dataset(name, train, root="./data"):
    """Return a torchvision dataset. torchvision is imported lazily so the rest of
    the code (and the smoke test) runs without it."""
    from torchvision import datasets, transforms
    mean, std = STATS[name]
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    cls = {"mnist": datasets.MNIST, "cifar10": datasets.CIFAR10}[name]
    return cls(root=root, train=train, download=True, transform=tf)


def iid_partition(n, num_clients, rng):
    idx = rng.permutation(n)
    return [part.tolist() for part in np.array_split(idx, num_clients)]


def dirichlet_partition(labels, num_clients, alpha, rng, min_size=10, max_tries=200):
    """Label-skew partition: for every class, split its examples across clients with
    proportions drawn from Dirichlet(alpha). Re-draws until every client holds at
    least `min_size` examples (an empty shard cannot be trained with Opacus)."""
    labels = np.asarray(labels)
    n = len(labels)
    for _ in range(max_tries):
        shards = [[] for _ in range(num_clients)]
        for c in np.unique(labels):
            idx_c = np.where(labels == c)[0]
            rng.shuffle(idx_c)
            p = rng.dirichlet(np.repeat(alpha, num_clients))
            # stop giving examples to clients that already hold more than a fair share
            p = np.array([pi * (len(s) < n / num_clients) for pi, s in zip(p, shards)])
            p = p / p.sum() if p.sum() > 0 else np.full(num_clients, 1.0 / num_clients)
            cuts = (np.cumsum(p) * len(idx_c)).astype(int)[:-1]
            for s, part in zip(shards, np.split(idx_c, cuts)):
                s.extend(part.tolist())
        if min(len(s) for s in shards) >= min_size:
            return shards
    raise RuntimeError("could not build a Dirichlet partition with the requested minimum shard size")


def make_loaders(dataset, labels, partition, num_clients, batch_size, alpha, rng):
    if partition == "iid":
        shards = iid_partition(len(dataset), num_clients, rng)
    elif partition == "non-iid":
        shards = dirichlet_partition(labels, num_clients, alpha, rng)
    else:
        raise ValueError("partition must be 'iid' or 'non-iid'")
    loaders = [DataLoader(Subset(dataset, s), batch_size=batch_size, shuffle=True) for s in shards]
    return loaders, [len(s) for s in shards]


def get_client_loaders(name, partition, num_clients, batch_size, alpha=0.5, seed=0, root="./data"):
    rng = np.random.default_rng(seed)
    train = load_dataset(name, True, root)
    labels = np.asarray(train.targets)
    return make_loaders(train, labels, partition, num_clients, batch_size, alpha, rng)


def get_test_loader(name, batch_size=1000, root="./data"):
    return DataLoader(load_dataset(name, False, root), batch_size=batch_size, shuffle=False)


def synthetic_dataset(n, dim=784, classes=10, seed=0, noise=1.0, spread=0.6):
    """Small separable dataset used only by the smoke test (no download needed)."""
    g = torch.Generator().manual_seed(seed)
    centres = torch.randn(classes, dim, generator=g) * spread
    y = torch.randint(0, classes, (n,), generator=g)
    x = centres[y] + torch.randn(n, dim, generator=g) * noise
    return TensorDataset(x, y), centres
