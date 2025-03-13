from torch import nn
from src.entities.neural_network_nodes import LiteralNodeModule, ANDNode, ORNode
import src.nnf2nn.nnf as nnf

class NNFToNN(nn.Module):
    def __init__(self, root):
        super().__init__()
        self.literals_map = self._extract_literals(root)
        self.model = self._build_network(root)

    def _extract_literals(self, node, literals=None):
        if literals is None:
            literals = {}
        
        if isinstance(node, nnf.LiteralNode):
            if node.literal not in literals:
                literals[node.literal] = len(literals)
        else:
            for child in node.children:
                self._extract_literals(child, literals)
        
        return literals
    
    def _build_network(self, node):
        if isinstance(node, nnf.LiteralNode):
            return LiteralNodeModule(self.literals_map[node.literal], node.negated)
        elif isinstance(node, nnf.AndNode):
            children_modules = [self._build_network(child) for child in node.children]
            return ANDNode(children_modules)
        elif isinstance(node, nnf.OrNode):
            children_modules = [self._build_network(child) for child in node.children]
            return ORNode(children_modules)
    
    def forward(self, x):
        return self.model(x)
