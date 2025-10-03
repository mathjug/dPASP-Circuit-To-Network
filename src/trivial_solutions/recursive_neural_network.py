from torch import nn

from src.trivial_solutions.entities.or_node import RecursiveORNode as ORNode
from src.trivial_solutions.entities.and_node import RecursiveANDNode as ANDNode
from src.trivial_solutions.network_builder import NetworkBuilder

class RecursiveNN(nn.Module):
    """
    Neural Network representation of a NNF, with a top-down, recursive forward pass.
    """
    def __init__(self, sdd_file, json_file, should_simplify=True, make_smooth=True):
        super().__init__()
        network_builder = NetworkBuilder(ORNode, ANDNode)
        build_network_response = network_builder.build_network(sdd_file, json_file, should_simplify, make_smooth)
        self.root = build_network_response.get_nn_root()
        self.num_variables = build_network_response.get_num_variables()
        self.literal_to_prob_node = build_network_response.get_literal_to_prob_node()

    def forward(self, x):
        """
        Executes the forward pass recursively from the root node, with memoization.
        
        Args:
            x (torch.Tensor): The input tensor for the network.

        Returns:
            torch.Tensor: The final output from the root node.
        """
        memoization_cache = {}
        return self.root.forward(x, memoization_cache)
    
    def get_num_variables(self):
        """
        Returns the number of variables in the neural network.
        """
        return self.num_variables
    
    def get_literal_to_prob_node(self):
        """
        Returns the mapping of literals to their probability nodes.
        """
        return self.literal_to_prob_node
