import torch
from torch import nn

class ORNode(nn.Module):
    """Implements an OR gate using max operation."""
    def __init__(self, children_nodes):
        super().__init__()
        self.children_nodes = nn.ModuleList(children_nodes)

    def forward(self, x):
        child_outputs = torch.cat([child(x) for child in self.children_nodes], dim=1)
        return torch.max(child_outputs, dim=1, keepdim=True)[0]

class ANDNode(nn.Module):
    """Implements an AND gate using element-wise multiplication."""
    def __init__(self, children_nodes):
        super().__init__()
        self.children_nodes = nn.ModuleList(children_nodes)

    def forward(self, x):
        child_outputs = torch.cat([child(x) for child in self.children_nodes], dim=1)
        return torch.prod(child_outputs, dim=1, keepdim=True)

class LiteralNodeModule(nn.Module):
    """Handles literal nodes by selecting the corresponding input feature."""
    def __init__(self, literal_index, negated=False):
        super().__init__()
        self.literal_index = literal_index
        self.negated = negated
    
    def forward(self, x):
        return x[:, self.literal_index].unsqueeze(1) * (-1 if self.negated else 1)