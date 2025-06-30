from torch import nn

from src.trivial_solutions.entities.or_node import RecursiveORNode as ORNode
from src.trivial_solutions.entities.and_node import RecursiveANDNode as ANDNode
from src.trivial_solutions.network_builder import NetworkBuilder

class RecursiveNN(nn.Module):
    """
    Neural Network representation of a NNF, with a top-down, recursive forward pass.
    """
    def __init__(self, root):
        super().__init__()
        network_builder = NetworkBuilder(ORNode, ANDNode)
        self.root = network_builder.build_network(root)

    def forward(self, x, marginalized_variables = None):
        return self.root.forward(x, marginalized_variables)
