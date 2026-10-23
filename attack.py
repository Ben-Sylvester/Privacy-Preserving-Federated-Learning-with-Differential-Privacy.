import torch
import torch.nn as nn
from torchvision import datasets, transforms
from model import MNIST_MLP
import math

def calc_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0: return 100.0
    return 20 * math.log10(1.0 / math.sqrt(mse.item()))

def dlg_attack(model, target_grads, input_shape, iters=300, lr=0.1):
    dummy_data = torch.randn(input_shape, requires_grad=True, device=next(model.parameters()).device)
    dummy_label = torch.randn((input_shape[0], 10), requires_grad=True, device=next(model.parameters()).device)
    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])
    criterion = nn.CrossEntropyLoss()
    target_grads = [g.clone().detach() for g in target_grads]

    for i in range(iters):
        def closure():
            optimizer.zero_grad()
            model.zero_grad()
            pred = model(dummy_data)
            loss = criterion(pred, dummy_label)
            dummy_grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
            
            grad_diff = 0
            for dg, tg in zip(dummy_grads, target_grads):
                grad_diff += ((dg - tg) ** 2).sum()
            grad_diff.backward()
            return grad_diff
        optimizer.step(closure)
        
    return dummy_data.detach()

def run_attack_evaluation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 1. Load a clean model and a small batch of real data (e.g., 8 images)
    model = MNIST_MLP().to(device)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST("./data", train=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    real_images, real_labels = next(iter(loader))
    real_images, real_labels = real_images.to(device), real_labels.to(device)

    # 2. Calculate target gradients (No DP)
    model.zero_grad()
    loss = nn.CrossEntropyLoss()(model(real_images), real_labels)
    target_grads = torch.autograd.grad(loss, model.parameters())
    
    # 3. Run Attack
    reconstructed_images = dlg_attack(model, target_grads, real_images.shape, iters=300)
    
    # 4. Calculate PSNR
    psnr = calc_psnr(real_images, reconstructed_images)
    print(f"🔥 Gradient Inversion PSNR (No DP): {psnr:.2f} dB")
    # (Repeat the same process for the DP model by adding noise to target_grads to get PSNR < 5dB)

if __name__ == "__main__":
    run_attack_evaluation()