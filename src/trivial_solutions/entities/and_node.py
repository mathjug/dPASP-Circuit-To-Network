import torch
from torch import nn

class RecursiveANDNode(nn.Module):
    """Implements an AND gate using element-wise multiplication with recursive forward pass."""
    def __init__(self, children_nodes):
        super().__init__()
        self.children_nodes = nn.ModuleList(children_nodes)

    def __str__(self):
        if len(self.children_nodes) == 0:
            return '()'
        output = '('
        for child in self.children_nodes:
            output += f"{child} ∧ "
        return output[:-3] + ')'

    def forward(self, x):
        child_outputs = torch.cat([child.forward(x) for child in self.children_nodes], dim=1)
        return torch.prod(child_outputs, dim=1, keepdim=True)

class IterativeANDNode(nn.Module):
    """Implements an AND gate using element-wise multiplication with iterative forward pass."""
    def __init__(self, children_nodes):
        super().__init__()
        self.children_nodes = nn.ModuleList(children_nodes)

    def __str__(self):
        if not self.children_nodes: return '()'
        return f"({' ∧ '.join(map(str, self.children_nodes))})"

    def forward(self, child_outputs):
        return torch.prod(child_outputs, dim=1, keepdim=True)
