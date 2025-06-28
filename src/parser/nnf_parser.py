class NNFParser:
    """
    A parser for reading NNF/SDD files and constructing a circuit graph in memory.
    Nodes are instantiated in a bottom-up fashion: child nodes are created before their parents.
    """
    def __init__(self):
        self.nodes = {} # maps node ID to node object
        self.root = None

    def parse(self, file_path):
        """
        Parses the given NNF file and builds the circuit.

        Args:
            file_path (str): The path to the .sdd file.

        Returns:
            Node: The root node of the parsed circuit.
        """
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('c'):
                    continue

                parts = line.split()
                node_type = parts[0]
                current_node = None

                if node_type == 'L':
                    # L id-of-literal-sdd-node id-of-vtree literal
                    node_id = parts[1]
                    literal_val = int(parts[3])
                    current_node = LiteralNode(
                        id=node_id,
                        literal=abs(literal_val),
                        negated=(literal_val < 0)
                    )
                    self.nodes[node_id] = current_node
                elif node_type == 'T':
                    # T id-of-true-sdd-node
                    node_id = parts[1]
                    current_node = TrueNode(id=node_id)
                    self.nodes[node_id] = current_node
                elif node_type == 'F':
                    # F id-of-false-sdd-node
                    node_id = parts[1]
                    current_node = FalseNode(id=node_id)
                    self.nodes[node_id] = current_node
                elif node_type == 'D':
                    # D id-of-decomposition-sdd-node id-of-vtree number-of-elements {id-of-prime id-of-sub}*
                    # This represents an OR of ANDs: (p1 AND s1) OR (p2 AND s2) OR ...
                    or_node_id = parts[1]
                    num_elements = int(parts[3])
                    elements = parts[4:]
                    
                    and_children = []
                    for i in range(num_elements):
                        prime_id = elements[i*2]
                        sub_id = elements[i*2+1]
                        
                        prime_node = self.nodes[prime_id]
                        sub_node = self.nodes[sub_id]
                        
                        and_node_id = f"{or_node_id}_and_{i}"
                        and_node = AndNode(id=and_node_id, children=[prime_node, sub_node])
                        and_children.append(and_node)
                    
                    current_node = OrNode(id=or_node_id, children=and_children)
                    self.nodes[or_node_id] = current_node

                if current_node:
                    self.root = current_node

        return self.root

class Node:
    """ Abstract base node for the NNF circuit. """
    def __init__(self, id, children=None):
        super().__init__()
        self.id = str(id)
        self.children = children if children is not None else []
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Node):
            return NotImplemented
        return self.id == __value.id

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id='{self.id}')"

class AndNode(Node):
    """ AND node in an NNF circuit. """
    def __init__(self, id, children):
        super().__init__(id, children)

class OrNode(Node):
    """ OR node in an NNF circuit. """
    def __init__(self, id, children):
        super().__init__(id, children)
 
class LiteralNode(Node):
    """
    Represents a literal node in a Negation Normal Form (NNF) graph.
    A literal node is a leaf in the NNF graph.
    """
    def __init__(self, id, literal, negated=False):
        """
        Initializes a LiteralNode.
        Args:
            id (str): The unique identifier for this node.
            literal (int): The integer identifying the input variable.
            negated (bool): Flag indicating if the literal is negated.
        """
        super().__init__(id, [])
        self.literal = literal
        self.negated = negated

    def __repr__(self):
        return f"LiteralNode(id='{self.id}', literal={self.literal}, negated={self.negated})"

class TrueNode(Node):
    """ Represents a terminal node for the boolean constant TRUE. """
    def __init__(self, id):
        super().__init__(id, [])

class FalseNode(Node):
    """ Represents a terminal node for the boolean constant FALSE. """
    def __init__(self, id):
        super().__init__(id, [])

def print_circuit(node, prefix="", is_last=True):
    """
    Utility function to print the structure of the parsed circuit
    in a tree-like format for visualization.
    """
    if not node:
        return
        
    print(prefix + ("└── " if is_last else "├── ") + repr(node))
    
    children = node.children
    for i, child in enumerate(children):
        is_child_last = (i == len(children) - 1)
        new_prefix = prefix + ("    " if is_last else "│   ")
        print_circuit(child, new_prefix, is_child_last)
