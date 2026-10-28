"""
Gradient inversion (Zhu et al., "Deep leakage from gradients") with and without DP noise.

This is the single-step setting the attack was designed for. It is not the same as
attacking a multi-epoch FedAvg update, and a failed reconstruction is evidence about
this attack only; the formal bound comes from epsilon.

    python attack.py --data mnist --batch 1 --sigmas 0 1.1 2.5 --trials 10
    python attack.py --data synthetic --batch 1 --sigmas 0 1.1      # no download
"""
import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

from model import MNIST_MLP


def psnr(a, b):
    """PSNR in dB between images in [0, 1] (peak value 1)."""
    mse = torch.mean((a.clamp(0, 1) - b) ** 2).item()
    return 100.0 if mse == 0 else 10 * math.log10(1.0 / mse)


def per_sample_grads(model, x, y):
    params = {k: v.detach() for k, v in model.named_parameters()}

    def loss_fn(p, xi, yi):
        return F.cross_entropy(functional_call(model, p, (xi.unsqueeze(0),)), yi.unsqueeze(0))

    return vmap(grad(loss_fn), in_dims=(None, 0, 0))(params, x, y)


def observed_gradient(model, x, y, sigma, clip, gen):
    """Gradient the server would see. sigma == 0 -> ordinary batch-mean gradient."""
    if sigma == 0:
        loss = F.cross_entropy(model(x), y)
        return [g.detach() for g in torch.autograd.grad(loss, list(model.parameters()))]
    g = per_sample_grads(model, x, y)
    norms = torch.sqrt(sum((v.flatten(1) ** 2).sum(1) for v in g.values()))
    scale = (clip / (norms + 1e-6)).clamp(max=1.0)
    out = []
    for name, _ in model.named_parameters():
        v = g[name]
        s = (v * scale.view(-1, *([1] * (v.dim() - 1)))).sum(0)
        noise = torch.randn(s.shape, generator=gen, device=s.device) * sigma * clip
        out.append((s + noise) / len(x))
    return out


def dlg(model, target, shape, iters, seed):
    g = torch.Generator().manual_seed(seed)
    dummy_x = torch.randn(shape, generator=g).requires_grad_(True)
    dummy_y = torch.randn((shape[0], 10), generator=g).requires_grad_(True)
    opt = torch.optim.LBFGS([dummy_x, dummy_y], lr=1.0)
    params = list(model.parameters())

    def closure():
        opt.zero_grad()
        pred = model(dummy_x)
        loss = F.cross_entropy(pred, F.softmax(dummy_y, dim=-1))
        dg = torch.autograd.grad(loss, params, create_graph=True)
        diff = sum(((a - b) ** 2).sum() for a, b in zip(dg, target))
        diff.backward()
        return diff

    for _ in range(iters):
        opt.step(closure)
    return dummy_x.detach()


def cosine_attack(model, target, shape, iters, seed, lr=0.1, tv_weight=1e-4):
    """Gradient matching with a cosine distance, Adam and a small total-variation prior
    (after Geiping et al., "Inverting gradients"). More reliable than L-BFGS on ReLU
    networks. Pixels are kept in [0, 1]."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(shape, generator=g).requires_grad_(True)
    y = torch.randn((shape[0], 10), generator=g).requires_grad_(True)
    opt = torch.optim.Adam([x, y], lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(iters * 0.5), int(iters * 0.75)], 0.3)
    params = list(model.parameters())
    tflat = torch.cat([t.flatten() for t in target])

    def tv(z):
        return (z[:, :, 1:] - z[:, :, :-1]).abs().mean() + (z[:, :, :, 1:] - z[:, :, :, :-1]).abs().mean()

    for _ in range(iters):
        opt.zero_grad()
        loss = F.cross_entropy(model(x), F.softmax(y, dim=-1))
        dg = torch.autograd.grad(loss, params, create_graph=True)
        dflat = torch.cat([d.flatten() for d in dg])
        (1 - F.cosine_similarity(dflat, tflat, dim=0) + tv_weight * tv(x)).backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            x.clamp_(0, 1)
    return x.detach()


def get_batch(kind, batch, seed, root):
    if kind == "synthetic":
        g = torch.Generator().manual_seed(seed)
        x = F.interpolate(torch.rand(batch, 1, 7, 7, generator=g), size=28, mode="bilinear")
        return x, torch.randint(0, 10, (batch,), generator=g)
    from torchvision import datasets, transforms
    ds = datasets.MNIST(root, train=True, download=True, transform=transforms.ToTensor())
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:batch].tolist()
    xs, ys = zip(*[ds[i] for i in idx])
    return torch.stack(xs), torch.tensor(ys)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", choices=["mnist", "synthetic"], default="mnist")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--method", choices=["cosine", "lbfgs"], default="cosine",
                   help="cosine/Adam (default) or the original L-BFGS Euclidean matching")
    p.add_argument("--iters", type=int, default=None, help="default: 1500 (cosine) or 300 (lbfgs)")
    p.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 1.1, 2.5])
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--out", default="results")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    iters = a.iters or (1500 if a.method == "cosine" else 300)

    scores = {s: [] for s in a.sigmas}
    last = {}
    for t in range(a.trials):
        seed = a.seed + t
        torch.manual_seed(seed)
        model = MNIST_MLP()
        x, y = get_batch(a.data, a.batch, seed, a.data_root)
        gen = torch.Generator().manual_seed(1000 + seed)
        for s in a.sigmas:
            target = observed_gradient(model, x, y, s, a.clip, gen)
            attack = cosine_attack if a.method == "cosine" else dlg
            rec = attack(model, target, x.shape, iters, seed)
            scores[s].append(psnr(rec, x))
            last[s] = rec.clamp(0, 1)
        print(f"trial {t}: " + "  ".join(f"sigma={s:g}: {scores[s][-1]:.2f} dB" for s in a.sigmas))

    print(f"\nPSNR (dB) over {a.trials} trials, batch = {a.batch}, method = {a.method}")
    for s in a.sigmas:
        v = np.array(scores[s])
        print(f"  sigma = {s:<4g}: median {np.median(v):6.2f}   mean {v.mean():6.2f}   min {v.min():6.2f}   max {v.max():6.2f}")
    print("A comparison with DP is only meaningful if the sigma = 0 row shows the attack working.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = min(a.batch, 4)
        fig, ax = plt.subplots(1 + len(a.sigmas), n, figsize=(1.6 * n, 1.6 * (1 + len(a.sigmas))), squeeze=False)
        rows = [("original", x)] + [(f"sigma={s:g}", last[s]) for s in a.sigmas]
        for r, (name, imgs) in enumerate(rows):
            for c in range(n):
                ax[r][c].imshow(imgs[c, 0].numpy(), cmap="gray", vmin=0, vmax=1)
                ax[r][c].axis("off")
            ax[r][0].set_title(name, fontsize=7, loc="left")
        fig.tight_layout()
        path = os.path.join(a.out, "attack_examples.png")
        fig.savefig(path, dpi=200)
        print("saved", path)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
