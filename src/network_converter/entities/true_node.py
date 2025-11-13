import torch
from torch import nn

class TrueNode(nn.Module):
    """
    Implements a TRUE terminal node. It always returns a tensor of ones.
    """
    def __init__(self):
        super().__init__()
        self.descendant_variables = set()

    def __str__(self):
        return 'TRUE'

    def forward(self, x, memoization_cache = None):
        """
        Returns a tensor of ones with a shape of (batch_size, 1).
        
        Args:
            x (torch.Tensor): The input tensor.
        """
        batch_size = 1 if x.dim() == 1 else x.shape[0]
        return torch.ones((batch_size, 1), device=x.device, dtype=torch.float32)
