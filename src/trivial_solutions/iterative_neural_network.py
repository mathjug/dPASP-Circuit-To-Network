import torch
from torch import nn

from src.trivial_solutions.entities.or_node import IterativeORNode as ORNode
from src.trivial_solutions.entities.and_node import IterativeANDNode as ANDNode
from src.trivial_solutions.network_builder import NetworkBuilder

class IterativeNN(nn.Module):
    """
    Neural Network representation of a NNF, with a bottom-up, iterative forward pass.
    """
    def __init__(self, nnf_root):
        super().__init__()
        network_builder = NetworkBuilder(ORNode, ANDNode)
        self.root = network_builder.build_network(nnf_root)
        self.execution_order = self._topological_sort()

    def _topological_sort(self):
        """
        Performs a topological sort of the nodes in the network.
        Returns a list of nodes in the order they should be executed (leaves first).
        """
        sorted_nodes = []
        visited = set()

        def visit(node):
            if node in visited:
                return
            visited.add(node)
            # Recursively visit children first for post-order traversal
            if hasattr(node, 'children_nodes'):
                for child in node.children_nodes:
                    visit(child)
            # Add the node to the list after all its children have been added
            sorted_nodes.append(node)

        visit(self.root)
        return sorted_nodes

    def forward(self, x, marginalized_variables = None):
        """
        Executes the forward pass using the pre-computed execution order.
        
        Args:
            x (torch.Tensor): The input tensor for the network.

        Returns:
            torch.Tensor: The final output from the root node.
        """
        node_outputs = {}

        for node in self.execution_order:
            if hasattr(node, 'children_nodes'):
                # Retrieve the pre-computed outputs of all children
                # Use node IDs as dictionary keys because nn.Module is not hashable
                child_outputs_list = [node_outputs[id(child)] for child in node.children_nodes]
                child_outputs_tensor = torch.cat(child_outputs_list, dim=1)
                output = node.forward(child_outputs_tensor)
            else:
                output = node.forward(x, marginalized_variables)
            node_outputs[id(node)] = output
            
        return node_outputs[id(self.root)]
