import torch
from torch import nn

from src.network_converter.entities.literal_node import LiteralNodeModule

class BaseANDNode(nn.Module):
    """
    Base class for AND nodes.
    
    Attributes:
        children_nodes (list): List of child nodes
        node_id (string): ID of the node
    """
    def __init__(self, children_nodes, node_id=None):
        super().__init__()
        self.children_nodes = nn.ModuleList(children_nodes)
        self.node_id = node_id
        self.descendant_variables = self._get_descendant_variables()
    
    def __str__(self):
        if not self.children_nodes: return '()'
        return f"({' ∧ '.join(map(str, self.children_nodes))})"

    def _get_descendant_variables(self):
        """
        Gets the set of variables that are descendants of the node.
        """
        descendant_variables = set()
        for child in self.children_nodes:
            descendant_variables.update(child.descendant_variables)
        return descendant_variables

class RecursiveANDNode(BaseANDNode):
    """Implements an AND gate using element-wise multiplication with recursive forward pass and memoization."""

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
        output = torch.prod(child_outputs, dim=1, keepdim=True)
        
        if self.node_id is not None:
            memoization_cache[self.node_id] = output
        
        return output

class IterativeANDNode(BaseANDNode):
    """Implements an AND gate using element-wise multiplication with iterative forward pass."""

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
        return torch.prod(child_outputs, dim=1, keepdim=True)
