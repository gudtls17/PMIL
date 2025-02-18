from abc import abstractmethod
import torch
import torch.nn as nn


class BaseModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self,
                connectivity: torch.tensor,
                timedelay: torch.tensor,
                timescales: torch.tensor) -> torch.tensor:
        pass
