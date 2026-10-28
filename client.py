"""One simulated institution: local DP-SGD training with Opacus."""
import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator


class DPClientTrainer:
    def __init__(self, model, dataloader, lr, noise_multiplier, max_grad_norm,
                 device="cpu", momentum=0.9, target_epsilon=None, target_delta=1e-5,
                 total_epochs=None):
        """noise_multiplier == 0 and target_epsilon is None -> plain (non-private) SGD.
        If target_epsilon is given, sigma is calibrated for `total_epochs` epochs
        (= rounds * local_epochs) at this client's own sampling rate."""
        self.device = device
        self.num_samples = len(dataloader.dataset)
        self.exhausted = False

        model = ModuleValidator.fix(model)
        self.model = model.to(device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        self.criterion = nn.CrossEntropyLoss()
        self.private = (target_epsilon is not None) or noise_multiplier > 0
        self.privacy_engine = None
        self.dataloader = dataloader

        if target_epsilon is not None:
            if total_epochs is None:
                raise ValueError("total_epochs is required with target_epsilon")
            self.privacy_engine = PrivacyEngine(accountant="rdp")
            self.model, self.optimizer, self.dataloader = self.privacy_engine.make_private_with_epsilon(
                module=self.model, optimizer=self.optimizer, data_loader=dataloader,
                target_epsilon=target_epsilon, target_delta=target_delta,
                epochs=total_epochs, max_grad_norm=max_grad_norm)
            self.noise_multiplier = self.optimizer.noise_multiplier
        elif noise_multiplier > 0:
            self.privacy_engine = PrivacyEngine(accountant="rdp")
            self.model, self.optimizer, self.dataloader = self.privacy_engine.make_private(
                module=self.model, optimizer=self.optimizer, data_loader=dataloader,
                noise_multiplier=noise_multiplier, max_grad_norm=max_grad_norm)
            self.noise_multiplier = noise_multiplier
        else:
            self.noise_multiplier = 0.0

    @property
    def module(self):
        return self.model._module if self.private else self.model

    def load_global(self, state):
        """Start the round from the server's model and clear momentum."""
        self.module.load_state_dict(state)
        opt = getattr(self.optimizer, "original_optimizer", self.optimizer)
        opt.state.clear()

    def train(self, local_epochs):
        self.model.train()
        total, batches = 0.0, 0
        for _ in range(local_epochs):
            for x, y in self.dataloader:
                if len(x) == 0:          # Poisson sampling can yield an empty batch
                    continue
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(x), y)
                loss.backward()
                self.optimizer.step()
                total += loss.item()
                batches += 1
        return total / max(batches, 1)

    def epsilon(self, delta):
        """Cumulative epsilon over every optimisation step this client has taken."""
        return self.privacy_engine.get_epsilon(delta) if self.private else float("inf")

    def get_model_update(self, global_state):
        local = self.module.state_dict()
        return {k: (local[k].detach() - global_state[k].to(local[k].device)) for k in global_state}
