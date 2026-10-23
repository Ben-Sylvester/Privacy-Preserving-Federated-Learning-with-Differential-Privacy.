import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

class DPClientTrainer:
    def __init__(self, model, dataloader, lr, noise_multiplier, max_grad_norm, device="cpu"):
        self.device = device
        self.noise_multiplier = noise_multiplier
        
        model = ModuleValidator.fix(model)
        ModuleValidator.validate(model, strict=False)
        self.model = model.to(device)
        
        # SGD, lr=0.01, momentum=0.9
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()

        if noise_multiplier > 0:
            self.privacy_engine = PrivacyEngine(accountant="rdp")
            self.model, self.optimizer, self.dataloader = self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=dataloader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm,
            )
        else:
            self.dataloader = dataloader
            self.privacy_engine = None

    def train(self, local_epochs):
        self.model.train()
        total_loss = 0.0
        for _ in range(local_epochs):
            for x, y in self.dataloader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(x)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        return total_loss / len(self.dataloader)

    def get_epsilon(self, delta):
        if self.privacy_engine:
            return self.privacy_engine.get_epsilon(delta)
        return float('inf')

    def get_model_update(self, global_model):
        local_model = self.model._module if self.privacy_engine else self.model
        local_state = local_model.state_dict()
        global_state = global_model.state_dict()
        update = {key: local_state[key] - global_state[key] for key in global_state.keys()}
        return update