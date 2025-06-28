import torch
from torch import nn

class FalseNode(nn.Module):
    """
    Implements a FALSE terminal node. It always returns a tensor of zeros.
    """
    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'FALSE'

    def forward(self, x, marginalized_variables = None):
        """
        Returns a tensor of zeros with a shape of (batch_size, 1).

        Args:
            x (torch.Tensor): The input tensor, used to determine batch size,
                              device, and dtype.
        """
        batch_size = 1 if x.dim() == 1 else x.shape[0]
        return torch.zeros((batch_size, 1), device=x.device, dtype=x.dtype)
