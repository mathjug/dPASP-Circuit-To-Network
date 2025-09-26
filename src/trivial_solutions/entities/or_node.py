import torch
from torch import nn

class BaseORNode(nn.Module):
    """
    Base class for OR nodes.
    
    Attributes:
        children_nodes (list): List of child nodes
        node_id (string): ID of the node
    """
    def __init__(self, children_nodes, node_id=None):
        super().__init__()
        self.children_nodes = nn.ModuleList(children_nodes)
        self.node_id = node_id
    
    def __str__(self):
        if not self.children_nodes: return '()'
        return f"({' V '.join(map(str, self.children_nodes))})"

class RecursiveORNode(BaseORNode):
    """Implements an OR gate using sum operation with recursive forward pass and memoization."""

    def __init__(self, children_nodes, node_id=None):
        super().__init__(children_nodes, node_id)

    def forward(self, x, memoization_cache = None):
        """
        Forward pass with memoization support.
        
        Args:
            x: Input tensor
            memoization_cache: Dictionary to cache node outputs by node_id
            
        Returns:
            Output tensor
        """
        if memoization_cache is None:
            memoization_cache = {}
        
        if self.node_id is not None and self.node_id in memoization_cache:
            return memoization_cache[self.node_id]
        
        child_outputs = torch.cat([child.forward(x, memoization_cache) for child in self.children_nodes], dim=1)
        output = torch.sum(child_outputs, dim=1, keepdim=True)
        
        if self.node_id is not None:
            memoization_cache[self.node_id] = output
        
        return output

class IterativeORNode(BaseORNode):
    """Implements an OR gate using sum operation with iterative forward pass."""
    
    def __init__(self, children_nodes, node_id=None):
        super().__init__(children_nodes, node_id)

    def forward(self, child_outputs):
        """
        Forward pass with iterative forward pass.
        
        Args:
            child_outputs: Output tensor from child nodes
            
        Returns:
            Output tensor
        """
        return torch.sum(child_outputs, dim=1, keepdim=True)
