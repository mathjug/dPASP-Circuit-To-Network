import torch
from torch import nn

from src.nnf2nn.trivial_solutions.entities.or_node import RecursiveORNode as ORNode
from src.nnf2nn.trivial_solutions.entities.and_node import RecursiveANDNode as ANDNode
from src.nnf2nn.trivial_solutions.entities.literal_node import LiteralNodeModule

import src.nnf2nn.parser.nnf as nnf

class RecursiveNN(nn.Module):
    """
    Neural Network representation of a NNF, with a top-down, recursive forward pass.
    """
    def __init__(self, root, sym2lit: dict = None, n_vars: int = None):
        super().__init__()
        self.root = self._build_network(root)
        self.sym2lit = sym2lit
        self.n_vars = n_vars
    
    def _build_network(self, node):
        if isinstance(node, nnf.LiteralNode):
            return LiteralNodeModule(node.literal - 1, node.negated)
        elif isinstance(node, nnf.AndNode):
            children_modules = [self._build_network(child) for child in node.children]
            return ANDNode(children_modules)
        elif isinstance(node, nnf.OrNode):
            children_modules = [self._build_network(child) for child in node.children]
            return ORNode(children_modules)
    
    def forward(self, x):
        return self.root.forward(x)
    
    def build_input(self):
        probs = torch.ones(self.n_vars)
        probs[self.__get_probs_index('a(bill)')] = 0.25
        probs[self.__get_probs_index('b(carol)')] = 0.25
        probs[self.__get_probs_index('c(daniel)')] = 0.25
        probs[self.__get_probs_index('d(carol,anna)')] = 0.2
        probs[self.__get_probs_index('e(bill,anna)')] = 0.2
        probs[self.__get_probs_index('influences(bill,anna)')] = 0.3
        probs[self.__get_probs_index('influences(carol,anna)')] = 0.4
        probs[self.__get_probs_index('stress(bill)')] = 0.333
        probs[self.__get_probs_index('stress(carol)')] = 0.333
        probs[self.__get_probs_index('stress(daniel)')] = 0.334
        return probs.unsqueeze(0)
    
    def __get_probs_index(self, symbol: str):
        if 'not' in symbol:
            symbol = symbol.replace('not', '').lstrip()
            return -(self.sym2lit[symbol]) - 1
        return self.sym2lit[symbol] - 1