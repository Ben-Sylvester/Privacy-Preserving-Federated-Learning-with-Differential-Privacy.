# import torch
# import torch.nn as nn
# import torch.optim as optim

# from opacus import PrivacyEngine
# from opacus.validators import ModuleValidator


# class DPClientTrainer:
#     def __init__(
#         self,
#         model,
#         dataloader,
#         lr,
#         noise_multiplier,
#         max_grad_norm,
#         device="cpu",
#     ):
#         self.device = device

#         # Fix model for DP compatibility
#         model = ModuleValidator.fix(model)
#         self.model = model.to(device)

#         self.optimizer = optim.SGD(
#             self.model.parameters(),
#             lr=lr,
#             momentum=0.9
#         )

#         self.criterion = nn.CrossEntropyLoss()

#         # Compatible with older & newer Opacus versions
#         self.privacy_engine = PrivacyEngine(accountant="rdp")

#         self.model, self.optimizer, self.dataloader = (
#             self.privacy_engine.make_private(
#                 module=self.model,
#                 optimizer=self.optimizer,
#                 data_loader=dataloader,
#                 noise_multiplier=noise_multiplier,
#                 max_grad_norm=max_grad_norm,
#             )
#         )

#     def train(self, local_epochs):
#         self.model.train()
#         total_loss = 0.0

#         for _ in range(local_epochs):
#             for x, y in self.dataloader:
#                 x, y = x.to(self.device), y.to(self.device)

#                 self.optimizer.zero_grad()
#                 logits = self.model(x)
#                 loss = self.criterion(logits, y)
#                 loss.backward()
#                 self.optimizer.step()

#                 total_loss += loss.item()

#         return total_loss / len(self.dataloader)

#     def get_epsilon(self, delta):
#         return self.privacy_engine.get_epsilon(delta)

#     def get_model_update(self, global_model):
#         """
#         Return model delta using unwrapped model (Opacus-safe)
#         """
#         #  Unwrap Opacus model
#         local_model = self.model._module  

#         local_state = local_model.state_dict()
#         global_state = global_model.state_dict()

#         update = {}
#         for key in global_state.keys():
#             update[key] = local_state[key] - global_state[key]

#         return update

import torch
import torch.nn as nn
import torch.optim as optim

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

class DPClientTrainer:
    def __init__(
        self,
        model,
        dataloader,
        lr,
        noise_multiplier,
        max_grad_norm,
        device="cpu",
    ):
        self.device = device

        # Fix model for DP compatibility
        model = ModuleValidator.fix(model)
        self.model = model.to(device)

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=0.9
        )

        self.criterion = nn.CrossEntropyLoss()

        self.privacy_engine = PrivacyEngine(accountant="rdp")

        # Make model private for DP
        self.model, self.optimizer, self.dataloader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=dataloader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
        )

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
        return self.privacy_engine.get_epsilon(delta)

    def get_model_update(self, global_model):
        local_model = self.model._module  
        local_state = local_model.state_dict()
        global_state = global_model.state_dict()

        update = {}
        for key in global_state.keys():
            update[key] = local_state[key] - global_state[key]

        return update
