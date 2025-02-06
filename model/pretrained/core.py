import torch
import torch.nn as nn

from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()
    @abstractmethod
    def shared_eval(self, batch, optimizer, mode):
        pass

    def configure_optimizers(self, lr=1e-3):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer
