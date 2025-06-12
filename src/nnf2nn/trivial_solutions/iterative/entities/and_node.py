import torch
from torch import nn

class ANDNode(nn.Module):
    """Implements an AND gate using element-wise multiplication."""
    def __init__(self, children_nodes):
        super().__init__()
        self.children_nodes = nn.ModuleList(children_nodes)

    def __str__(self):
        if not self.children_nodes: return '()'
        return f"({' ∧ '.join(map(str, self.children_nodes))})"

    def forward(self, child_outputs):
        return torch.prod(child_outputs, dim=1, keepdim=True)