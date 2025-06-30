"""
This module provides a parser for reading a circuit representation from a file
in Negation Normal Form (NNF), specifically tailored for the Sentential Decision
Diagram (SDD) format. The primary component is the `NNFParser` class, which is
responsible for parsing the file and constructing an in-memory graph
representation of the logical circuit.

The parser processes a text file line by line, interpreting different node types
that define the logical structure of the circuit. The supported node types are:

- 'L': Literal nodes, representing a boolean variable or its negation.
- 'T': A terminal node representing the logical constant True.
- 'F': A terminal node representing the logical constant False.
- 'D': Decomposition nodes, which represent a disjunction (OR) of conjunctions
       (ANDs). Each element of the decomposition is a pair of nodes (a prime
       and a sub), which are conjoined. 
"""

import os

class NNFParser:
    """
    A parser for reading NNF/SDD files and constructing a circuit graph in memory.

    Attributes:
        nodes (Dict[str, Node]): A dictionary that maps a unique node ID (from
            the file) to its corresponding Node object. This serves as a cache
            for all nodes created during parsing.
        circuit_root (Optional[Node]): The final node processed in the file, which
            represents the root of the entire logical circuit. It is `None` until
            at least one node has been successfully parsed.
        and_node_cache (Dict[Tuple[str, str], AndNode]): A dictionary that maps
            a tuple of two node IDs to its corresponding AndNode object. This
            serves as a cache for all AndNodes created during parsing.

    """
    def __init__(self):
        self.nodes = {} # maps node ID to node object
        self.and_node_cache = {}
        self.circuit_root = None
    
    def parse(self, file_path):
        """
        Parses the given NNF file and builds the circuit.

        Args:
            file_path (str): The path to the .sdd file.

        Returns:
            Optional[Node]: The root node of the parsed circuit, or None if an error occurs.
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Error: The file '{file_path}' was not found.")

            with open(file_path, 'r') as f:
                for line in f:
                    current_node = self._parse_line(line.strip())
                    if current_node:
                        self.circuit_root = current_node

        except FileNotFoundError as e:
            print(e)
            return None
        except IOError as e:
            print(f"Error: Could not read the file '{file_path}': {e}")
            return None
        except (IndexError, ValueError) as e:
            print(f"Error: Malformed content in file '{file_path}': {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None
            
        return self.circuit_root
    
    def _parse_line(self, line):
        """Parses a single line and returns the corresponding node."""
        if not line or line.startswith('c'):
            return None

        parts = line.split()
        node_type = parts[0]

        if node_type == 'L':
            return self._create_literal_node(parts)
        elif node_type == 'T':
            return self._create_true_node(parts)
        elif node_type == 'F':
            return self._create_false_node(parts)
        elif node_type == 'D':
            return self._create_decomposition_node(parts)
        
        return None
    
    def _create_literal_node(self, parts):
        """Creates a LiteralNode from parsed parts."""
        # L id-of-literal-sdd-node id-of-vtree literal
        node_id = parts[1]
        literal_val = int(parts[3])
        node = LiteralNode(
            id=node_id,
            literal=abs(literal_val),
            negated=(literal_val < 0)
        )
        self.nodes[node_id] = node
        return node
    
    def _create_true_node(self, parts):
        """Creates a TrueNode from parsed parts."""
        # T id-of-true-sdd-node
        node_id = parts[1]
        node = TrueNode(id=node_id)
        self.nodes[node_id] = node
        return node

    def _create_false_node(self, parts):
        """Creates a FalseNode from parsed parts."""
        # F id-of-false-sdd-node
        node_id = parts[1]
        node = FalseNode(id=node_id)
        self.nodes[node_id] = node
        return node
    
    def _create_decomposition_node(self, parts):
        """Creates a decomposition (Or) node from parsed parts."""
        # D id-of-decomposition-sdd-node id-of-vtree number-of-elements {id-of-prime id-of-sub}*
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
        
        node = OrNode(id=or_node_id, children=and_children)
        self.nodes[or_node_id] = node
        return node
    
    def _create_decomposition_node(self, parts):
        """
        Creates a decomposition (Or) node from parsed parts, reusing AndNodes
        to avoid duplication.
        """
        # D id-of-decomposition-sdd-node id-of-vtree number-of-elements {id-of-prime id-of-sub}*
        or_node_id = parts[1]
        num_elements = int(parts[3])
        elements = parts[4:]
        
        and_children = []
        for i in range(num_elements):
            prime_id, sub_id = elements[i*2], elements[i*2+1]
            and_node = self._create_and_node(prime_id, sub_id)
            and_children.append(and_node)
        
        node = OrNode(id=or_node_id, children=and_children)
        self.nodes[or_node_id] = node
        return node

    def _create_and_node(self, prime_id, sub_id):
        """Creates or retrieves an already existing AndNode."""
        and_node_key = (prime_id, sub_id)
        if and_node_key in self.and_node_cache:
            return self.and_node_cache[and_node_key]
        return self._create_new_and_node(prime_id, sub_id)
    
    def _create_new_and_node(self, prime_id, sub_id):
        """Creates a new AndNode."""
        prime_node = self.nodes[prime_id]
        sub_node = self.nodes[sub_id]
        and_node_id = f"and_{prime_id}_{sub_id}"
        
        and_node = AndNode(id=and_node_id, children=[prime_node, sub_node])
        self.and_node_cache[(prime_id, sub_id)] = and_node
        self.nodes[and_node_id] = and_node

        return and_node

class Node:
    """
    Abstract base node for the NNF circuit.
    
    Attributes:
        id (str): A unique string identifier for the node.
        children (list[Node]): A list containing the child nodes of this node.
    """
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

    Attributes:
        id (str): The unique identifier for this node, inherited from Node.
        children (list): An empty list, as literal nodes are terminals and have no children.
        literal (int): The integer identifying the input variable.
        negated (bool): A flag that is True if the literal is negated, otherwise False.
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
    """
    Represents a terminal node for the boolean constant TRUE.
    
    Its attributes are inherited from the base Node class, with `children` being an empty list.
    """
    def __init__(self, id):
        super().__init__(id, [])

class FalseNode(Node):
    """
    Represents a terminal node for the boolean constant FALSE.
    
    Its attributes are inherited from the base Node class, with `children` being an empty list.
    """
    def __init__(self, id):
        super().__init__(id, [])
