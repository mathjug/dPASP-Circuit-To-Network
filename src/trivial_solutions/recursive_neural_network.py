import torch
from torch import nn

from src.trivial_solutions.entities.or_node import RecursiveORNode as ORNode
from src.trivial_solutions.entities.and_node import RecursiveANDNode as ANDNode
from src.trivial_solutions.entities.literal_node import LiteralNodeModule
from src.trivial_solutions.entities.true_node import TrueNode
from src.trivial_solutions.entities.false_node import FalseNode

import src.parser.nnf_parser as nnf

class RecursiveNN(nn.Module):
    """
    Neural Network representation of a NNF, with a top-down, recursive forward pass.
    """
    def __init__(self, root):
        super().__init__()
        self.root = self._build_network(root)
    
    def _build_network(self, node):
        if isinstance(node, nnf.LiteralNode):
            return LiteralNodeModule(node.literal - 1, node.negated)
        elif isinstance(node, nnf.TrueNode):
            return TrueNode()
        elif isinstance(node, nnf.FalseNode):
            return FalseNode()
        elif isinstance(node, nnf.AndNode):
            children_modules = [self._build_network(child) for child in node.children]
            return ANDNode(children_modules)
        elif isinstance(node, nnf.OrNode):
            children_modules = [self._build_network(child) for child in node.children]
            return ORNode(children_modules)
    
    def forward(self, x, marginalized_variables = None):
        return self.root.forward(x, marginalized_variables)
