import torch
from torch import nn

class LiteralNodeModule(nn.Module):
    """Handles literal nodes by selecting the corresponding input feature."""
    def __init__(self, literal_index, negated=False):
        super().__init__()
        self.literal_index = literal_index
        self.negated = negated
    
    def __str__(self):
        return f"{'¬' if self.negated else ''}x{self.literal_index}"
    
    def forward(self, x):
        val = x[:, self.literal_index].unsqueeze(1)
        return 1 - val if self.negated else val
