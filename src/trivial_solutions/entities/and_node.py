import torch
from torch import nn

class RecursiveANDNode(nn.Module):
    """Implements an AND gate using element-wise multiplication with recursive forward pass and memoization."""
    def __init__(self, children_nodes, node_id=None):
        super().__init__()
        self.children_nodes = nn.ModuleList(children_nodes)
        self.node_id = node_id

    def __str__(self):
        if len(self.children_nodes) == 0:
            return '()'
        output = '('
        for child in self.children_nodes:
            output += f"{child} ∧ "
        return output[:-3] + ')'

    def forward(self, x, marginalized_variables = None, memoization_cache = None):
        """
        Forward pass with memoization support.
        
        Args:
            x: Input tensor
            marginalized_variables: Marginalized variables tensor
            memoization_cache: Dictionary to cache node outputs by node_id
            
        Returns:
            Output tensor
        """
        if memoization_cache is None:
            memoization_cache = {}
        
        if self.node_id is not None and self.node_id in memoization_cache:
            return memoization_cache[self.node_id]
        
        child_outputs = torch.cat([
            child.forward(x, marginalized_variables, memoization_cache) 
            for child in self.children_nodes
        ], dim=1)
        output = torch.prod(child_outputs, dim=1, keepdim=True)
        
        if self.node_id is not None:
            memoization_cache[self.node_id] = output
        
        return output

class IterativeANDNode(nn.Module):
    """Implements an AND gate using element-wise multiplication with iterative forward pass."""
    def __init__(self, children_nodes, node_id=None):
        super().__init__()
        self.children_nodes = nn.ModuleList(children_nodes)
        self.node_id = node_id

    def __str__(self):
        if not self.children_nodes: return '()'
        return f"({' ∧ '.join(map(str, self.children_nodes))})"

    def forward(self, child_outputs):
        return torch.prod(child_outputs, dim=1, keepdim=True)
