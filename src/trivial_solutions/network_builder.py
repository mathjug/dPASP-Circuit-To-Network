from src.trivial_solutions.entities.literal_node import LiteralNodeModule
from src.trivial_solutions.entities.true_node import TrueNode
from src.trivial_solutions.entities.false_node import FalseNode
from src.trivial_solutions.entities.constant_node import ConstantNode

import src.parser.nnf_parser as nnf

from src.parser.nnf_parser import NNFParser
from src.parser.probabilities_parser import ProbabilitiesParser

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
    
    def build_network(self, sdd_file, json_file):
        nnf_root = NNFParser().parse(sdd_file)
        self.literal_to_prob_node = self._build_literal_to_prob_node_mapping(json_file)
        return self._recursive_build_network(nnf_root, {})
    
    def _build_literal_to_prob_node_mapping(self, json_file):
        probabilities_parser = ProbabilitiesParser(json_file)
        literal_to_prob = probabilities_parser.variable_to_prob
        return {literal: ConstantNode(prob) for literal, prob in literal_to_prob.items()}
    
    def _recursive_build_network(self, node, node_cache):
        """
        Builds the neural network recursively, caching nodes by their ID to avoid duplicates.
        
        Args:
            node: The NNF node to convert
            node_cache: Dictionary mapping node IDs to already instantiated nodes
            
        Returns:
            The instantiated neural network node
        """
        if node.id in node_cache:
            return node_cache[node.id]
        
        if isinstance(node, nnf.LiteralNode):
            nn_node = self._create_literal_node(node)
        elif isinstance(node, nnf.TrueNode):
            nn_node = TrueNode()
        elif isinstance(node, nnf.FalseNode):
            nn_node = FalseNode()
        elif isinstance(node, nnf.AndNode):
            children_nodes = [self._recursive_build_network(child, node_cache) for child in node.children]
            nn_node = self.and_node_class(children_nodes, node_id=node.id)
        elif isinstance(node, nnf.OrNode):
            children_nodes = [self._recursive_build_network(child, node_cache) for child in node.children]
            nn_node = self.or_node_class(children_nodes, node_id=node.id)
        else:
            raise ValueError(f"Unknown node type: {type(node)}")
        
        node_cache[node.id] = nn_node
        return nn_node

    def _create_literal_node(self, node):
        input_index = 2 * (node.literal - 1) + int(node.negated)
        literal_node = LiteralNodeModule(node.literal, input_index, node.negated)
        if node.literal not in self.literal_to_prob_node:
            return literal_node
        probability_node = self.literal_to_prob_node[node.literal]
        if node.negated:
            return self._create_negated_literal_node(node.id, literal_node, probability_node)
        return self.and_node_class([literal_node, probability_node], node_id=node.id)
    
    def _create_negated_literal_node(self, node_id, literal_node, probability_node):
        and_node = self.and_node_class([probability_node, ConstantNode(-1.0)])
        or_node = self.or_node_class([and_node, ConstantNode(1.0)])
        return self.and_node_class([or_node, literal_node], node_id=node_id)