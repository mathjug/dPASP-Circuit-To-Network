from src.trivial_solutions.entities.literal_node import LiteralNodeModule
from src.trivial_solutions.entities.true_node import TrueNode
from src.trivial_solutions.entities.false_node import FalseNode

import src.parser.nnf_parser as nnf

class NetworkBuilder:
    """
    A utility class for building neural networks from NNF circuits.
    This class handles the common logic of converting NNF nodes to neural network modules,
    with proper caching to avoid duplicates when the same NNF node appears multiple times.
    """
    def __init__(self, or_node_class, and_node_class):
        """
        Initialize the network builder with the specific node classes to use.
        
        Args:
            or_node_class: The class to use for OR nodes (e.g., IterativeORNode or RecursiveORNode)
            and_node_class: The class to use for AND nodes (e.g., IterativeANDNode or RecursiveANDNode)
        """
        self.or_node_class = or_node_class
        self.and_node_class = and_node_class
    
    def build_network(self, node, node_cache=None):
        """
        Builds the neural network recursively, caching nodes by their ID to avoid duplicates.
        
        Args:
            node: The NNF node to convert
            node_cache: Dictionary mapping node IDs to already instantiated nodes
            
        Returns:
            The instantiated neural network root node
        """
        if node_cache is None:
            node_cache = {}
        
        if node.id in node_cache:
            return node_cache[node.id]
        
        if isinstance(node, nnf.LiteralNode):
            nn_node = LiteralNodeModule(node.literal - 1, node.negated)
        elif isinstance(node, nnf.TrueNode):
            nn_node = TrueNode()
        elif isinstance(node, nnf.FalseNode):
            nn_node = FalseNode()
        elif isinstance(node, nnf.AndNode):
            children_nodes = [self.build_network(child, node_cache) for child in node.children]
            nn_node = self.and_node_class(children_nodes, node_id=node.id)
        elif isinstance(node, nnf.OrNode):
            children_nodes = [self.build_network(child, node_cache) for child in node.children]
            nn_node = self.or_node_class(children_nodes, node_id=node.id)
        else:
            raise ValueError(f"Unknown node type: {type(node)}")
        
        node_cache[node.id] = nn_node
        return nn_node
