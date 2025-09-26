import torch
from torch import nn

class ConstantNode(nn.Module):
    """
    Implements a constant terminal node. It always returns a tensor of the constant.
    """
    def __init__(self, constant: float):
        super().__init__()
        self.constant = constant
    
    def __str__(self):
        return f'CONSTANT {self.constant}'

    def forward(self, x, memoization_cache = None):
        """
        Returns a tensor of the constant with a shape of (batch_size, 1).
        
        Args:
            x (torch.Tensor): The input tensor.
        """
        batch_size = 1 if x.dim() == 1 else x.shape[0]
        return torch.full((batch_size, 1), self.constant, device=x.device, dtype=torch.float32)
