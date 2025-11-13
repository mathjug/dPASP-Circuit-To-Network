from src.network_converter.entities.literal_node import LiteralNodeModule
from src.network_converter.entities.true_node import TrueNode
from src.network_converter.entities.false_node import FalseNode
from src.network_converter.entities.constant_node import ConstantNode
from src.network_converter.entities.network_builder import BuildNetworkResponse

import src.parser.sdd_parser as nnf
from src.parser.sdd_parser import SDDParser
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
    
    def build_network(self, sdd_file, json_file, should_simplify=True, make_smooth=True):
        """
        Builds the neural network from the given SDD file and JSON file.
        
        Args:
            sdd_file (string): The path to the SDD file
            json_file (string): The path to the JSON file
            should_simplify (bool): Whether to simplify the network by removing unnecessary nodes
            make_smooth (bool): Whether to transform the network to make it represent a smooth circuit
        
        Returns:
            BuildNetworkResponse: An object containing the neural network, number of variables, and mapping
                of literals to probability nodes.
        """
        nnf_root = SDDParser().parse(sdd_file)
        self.literal_to_prob_node, num_variables = self._build_literal_to_prob_node_mapping(json_file)
        self.literal_to_literal_node = {} # maps literal to its literal node
        nn_root = self._recursive_build_network(nnf_root, {}, should_simplify)
        if make_smooth:
            self._enforce_circuit_smoothness(nn_root)
        return BuildNetworkResponse(nn_root, num_variables, self.literal_to_prob_node)
    
    def _build_literal_to_prob_node_mapping(self, json_file):
        probabilities_parser = ProbabilitiesParser(json_file)
        num_variables = len(probabilities_parser.variable_to_atom)
        literal_to_prob = probabilities_parser.variable_to_prob
        return {literal: ConstantNode(prob) for literal, prob in literal_to_prob.items()}, num_variables
    
    def _recursive_build_network(self, node, node_cache, should_simplify):
        """
        Builds the neural network recursively, caching nodes by their ID to avoid duplicates.
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
            children_nodes = [self._recursive_build_network(child, node_cache, should_simplify) for child in node.children]
            nn_node = self._create_and_node(node.id, children_nodes, should_simplify)
        elif isinstance(node, nnf.OrNode):
            children_nodes = [self._recursive_build_network(child, node_cache, should_simplify) for child in node.children]
            nn_node = self._create_or_node(node.id, children_nodes, should_simplify)
        else:
            raise ValueError(f"Unknown node type: {type(node)}")
        
        node_cache[node.id] = nn_node
        return nn_node

    def _create_literal_node(self, node):
        if (node.literal, node.negated) in self.literal_to_literal_node:
            return self.literal_to_literal_node[(node.literal, node.negated)]
        input_index = 2 * (node.literal - 1) + int(node.negated)
        literal_node = LiteralNodeModule(node.literal, input_index, node.negated)
        if node.literal not in self.literal_to_prob_node:
            self.literal_to_literal_node[(node.literal, node.negated)] = literal_node
            return literal_node
        probability_node = self.literal_to_prob_node[node.literal]
        if node.negated:
            negated_literal_node = self._create_negated_literal_node(node.id, literal_node, probability_node)
            self.literal_to_literal_node[(node.literal, True)] = negated_literal_node
            return negated_literal_node
        positive_literal_node = self.and_node_class([literal_node, probability_node], node_id=node.id)
        self.literal_to_literal_node[(node.literal, False)] = positive_literal_node
        return positive_literal_node
    
    def _create_negated_literal_node(self, node_id, literal_node, probability_node):
        and_node = self.and_node_class([probability_node, ConstantNode(-1.0)])
        or_node = self.or_node_class([and_node, ConstantNode(1.0)])
        return self.and_node_class([or_node, literal_node], node_id=node_id)
    
    def _create_and_node(self, node_id, children_nodes, should_simplify):
        if not should_simplify:
            return self.and_node_class(children_nodes, node_id=node_id)
        simplified_children = []
        for child in children_nodes:
            if isinstance(child, FalseNode):
                return child
            if not isinstance(child, TrueNode):
                simplified_children.append(child)
        if not simplified_children:
            return TrueNode()
        if len(simplified_children) == 1:
            return simplified_children[0]
        return self.and_node_class(simplified_children, node_id=node_id)
    
    def _create_or_node(self, node_id, children_nodes, should_simplify):
        if not should_simplify:
            return self.or_node_class(children_nodes, node_id=node_id)
        simplified_children = []
        for child in children_nodes:
            if isinstance(child, TrueNode):
                return child
            if not isinstance(child, FalseNode):
                simplified_children.append(child)
        if not simplified_children:
            return FalseNode()
        if len(simplified_children) == 1:
            return simplified_children[0]
        return self.or_node_class(simplified_children, node_id=node_id)
    
    def _enforce_circuit_smoothness(self, nn_root):
        """
        Enforces the smoothness of the circuit.
        """
        if isinstance(nn_root, self.or_node_class):
            self._enforce_circuit_smoothness_or_node(nn_root)
        if hasattr(nn_root, 'children_nodes'):
            for child in nn_root.children_nodes:
                self._enforce_circuit_smoothness(child)
    
    def _enforce_circuit_smoothness_or_node(self, or_node):
        """
        Enforces the smoothness of the OR node.
        """
        if len(or_node.children_nodes) < 2:
            return
        parent_variables = or_node.descendant_variables
        for i, child in enumerate(or_node.children_nodes):
            child_variables = child.descendant_variables
            missing_variables = parent_variables - child_variables
            if not missing_variables:
                continue
            new_child = child
            if not isinstance(child, self.and_node_class):
                new_child = self.and_node_class([child])
                or_node.children_nodes[i] = new_child
            for missing_var in missing_variables:
                tautology_node = self._create_tautology_node(missing_var)
                new_child.children_nodes.append(tautology_node)
    
    def _create_tautology_node(self, variable):
        """
        Creates a tautology node.
        """
        input_index = 2 * (variable - 1)
        negative_literal = self.literal_to_literal_node.setdefault((variable, True), LiteralNodeModule(variable, input_index + 1, True))
        positive_literal = self.literal_to_literal_node.setdefault((variable, False), LiteralNodeModule(variable, input_index, False))
        children_nodes = [negative_literal, positive_literal]
        return self.or_node_class(children_nodes)
