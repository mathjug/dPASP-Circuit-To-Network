"""
This test file is dedicated to verifying the correctness of the NetworkBuilder class.
It specifically tests that:

1. Networks with shared subexpressions are built correctly without duplicates
2. Node caching works properly using the node ID mechanism
3. The same node instances are reused when they appear multiple times in the NNF
4. The builder works correctly with different node class combinations

The tests create NNF circuits with intentionally repeated nodes and verify that
the resulting neural network contains the expected number of unique modules.
"""

import pytest
from collections import defaultdict

import src.parser.nnf_parser as nnf
from src.trivial_solutions.network_builder import NetworkBuilder
from src.trivial_solutions.entities.or_node import IterativeORNode, RecursiveORNode
from src.trivial_solutions.entities.and_node import IterativeANDNode, RecursiveANDNode

def count_unique_nodes(root_node):
    """
    Counts the number of unique nodes in a neural network.
    
    Args:
        root_node: The root node of the neural network
        
    Returns:
        dict: Dictionary with counts of each node type
    """
    node_counts = defaultdict(int)
    visited = set()
    
    def normalize_node_type(node_type_name):
        """Normalize node type names by removing implementation prefixes."""
        if node_type_name.startswith('Iterative') or node_type_name.startswith('Recursive'):
            return node_type_name[9:]
        return node_type_name
    
    def count_nodes_recursive(node):
        if id(node) in visited:
            return
        visited.add(id(node))
        
        node_type = normalize_node_type(type(node).__name__)
        node_counts[node_type] += 1
        
        if hasattr(node, 'children_nodes'):
            for child in node.children_nodes:
                count_nodes_recursive(child)
    
    count_nodes_recursive(root_node)
    return dict(node_counts)

def get_all_node_instances(root_node):
    """
    Returns all node instances in a neural network.
    
    Args:
        root_node: The root node of the neural network
        
    Returns:
        list: List of all node instances
    """
    nodes = []
    visited = set()
    
    def collect_nodes_recursive(node):
        if id(node) in visited:
            return
        visited.add(id(node))
        
        nodes.append(node)
        
        if hasattr(node, 'children_nodes'):
            for child in node.children_nodes:
                collect_nodes_recursive(child)
    
    collect_nodes_recursive(root_node)
    return nodes

network_construction_test_cases = [
    (
        "Simple shared literal: x₁ ∧ x₁",
        lambda: nnf.AndNode('A1', [
            nnf.LiteralNode('L1', 1),
            nnf.LiteralNode('L1', 1)
        ]),
        {
            'LiteralNodeModule': 1,
            'ANDNode': 1
        }
    ),
    (
        "Shared OR node: (x₁ V x₂) ∧ (x₁ V x₂)",
        lambda: nnf.AndNode('A1', [
            nnf.OrNode('O1', [nnf.LiteralNode('L1', 1), nnf.LiteralNode('L2', 2)]),
            nnf.OrNode('O1', [nnf.LiteralNode('L1', 1), nnf.LiteralNode('L2', 2)])
        ]),
        {
            'LiteralNodeModule': 2,
            'ORNode': 1,
            'ANDNode': 1
        }
    ),
    (
        "Complex shared subexpression: (x₁ ∧ x₂) V (x₁ ∧ x₂) V (x₃ ∧ x₄)",
        lambda: nnf.OrNode('O1', [
            nnf.AndNode('A1', [nnf.LiteralNode('L1', 1), nnf.LiteralNode('L2', 2)]),
            nnf.AndNode('A1', [nnf.LiteralNode('L1', 1), nnf.LiteralNode('L2', 2)]),
            nnf.AndNode('A2', [nnf.LiteralNode('L3', 3), nnf.LiteralNode('L4', 4)])
        ]),
        {
            'LiteralNodeModule': 4,
            'ANDNode': 2,
            'ORNode': 1
        }
    ),
    (
        "Deep shared structure: ((x₁ V x₂) ∧ x₃) V ((x₁ V x₂) ∧ x₄)",
        lambda: nnf.OrNode('O1', [
            nnf.AndNode('A1', [
                nnf.OrNode('O2', [nnf.LiteralNode('L1', 1), nnf.LiteralNode('L2', 2)]),
                nnf.LiteralNode('L3', 3)
            ]),
            nnf.AndNode('A2', [
                nnf.OrNode('O2', [nnf.LiteralNode('L1', 1), nnf.LiteralNode('L2', 2)]),
                nnf.LiteralNode('L4', 4)
            ])
        ]),
        {
            'LiteralNodeModule': 4,
            'ORNode': 2,
            'ANDNode': 2,
        }
    ),
    (
        "True/False nodes: (True ∧ x₁) V (False ∧ x₁)",
        lambda: nnf.OrNode('O1', [
            nnf.AndNode('A1', [nnf.TrueNode('T'), nnf.LiteralNode('L1', 1)]),
            nnf.AndNode('A2', [nnf.FalseNode('F'), nnf.LiteralNode('L1', 1)])
        ]),
        {
            'LiteralNodeModule': 1,
            'TrueNode': 1,
            'FalseNode': 1,
            'ANDNode': 2,
            'ORNode': 1
        }
    )
]

@pytest.mark.parametrize("or_node_class, and_node_class", [
    (IterativeORNode, IterativeANDNode),
    (RecursiveORNode, RecursiveANDNode)
], ids=["Iterative", "Recursive"])
@pytest.mark.parametrize("description, nnf_circuit, expected_counts", network_construction_test_cases)
def test_network_builder_no_duplicates(or_node_class, and_node_class, description, nnf_circuit, expected_counts):
    """
    Tests that NetworkBuilder constructs networks without duplicate nodes when the same NNF node
    appears multiple times in the circuit.
    """
    implementation_name = "Iterative" if or_node_class == IterativeORNode else "Recursive"
    full_description = f"{description} ({implementation_name} Implementation)"
    print(f"Testing network builder: {full_description}")

    root_node = nnf_circuit()
    builder = NetworkBuilder(or_node_class, and_node_class)
    neural_network_root = builder.build_network(root_node)

    actual_counts = count_unique_nodes(neural_network_root)
    
    assert actual_counts == expected_counts, (
        f"Module count mismatch for {full_description}\n"
        f"Expected: {expected_counts}\n"
        f"Actual: {actual_counts}"
    )

@pytest.mark.parametrize("or_node_class, and_node_class", [
    (IterativeORNode, IterativeANDNode),
    (RecursiveORNode, RecursiveANDNode)
], ids=["Iterative", "Recursive"])
def test_shared_node_instances(or_node_class, and_node_class):
    """
    Tests that the same module instance is used when the same NNF node appears multiple times.
    """
    implementation_name = "Iterative" if or_node_class == IterativeORNode else "Recursive"
    description = f"Shared node instances ({implementation_name} Implementation)"
    print(f"Testing: {description}")

    root_node = nnf.AndNode('A1', [
        nnf.LiteralNode('L1', 1),
        nnf.LiteralNode('L1', 1)
    ])
    
    builder = NetworkBuilder(or_node_class, and_node_class)
    neural_network_root = builder.build_network(root_node)

    all_nodes = get_all_node_instances(neural_network_root)
    literal_nodes = [n for n in all_nodes if type(n).__name__ == 'LiteralNodeModule']
    
    assert len(literal_nodes) == 1, (
        f"Expected 1 LiteralNodeModule instance, got {len(literal_nodes)}"
    )

@pytest.mark.parametrize("or_node_class, and_node_class", [
    (IterativeORNode, IterativeANDNode),
    (RecursiveORNode, RecursiveANDNode)
], ids=["Iterative", "Recursive"])
def test_complex_shared_structure(or_node_class, and_node_class):
    """
    Tests a more complex structure with multiple shared subexpressions.
    """
    implementation_name = "Iterative" if or_node_class == IterativeORNode else "Recursive"
    description = f"Complex shared structure ({implementation_name} Implementation)"
    print(f"Testing: {description}")

    shared_or = nnf.OrNode('O1', [nnf.LiteralNode('L1', 1), nnf.LiteralNode('L2', 2)])
    
    root_node = nnf.OrNode('O2', [
        nnf.AndNode('A1', [shared_or, nnf.LiteralNode('L3', 3)]),
        nnf.AndNode('A2', [shared_or, nnf.LiteralNode('L4', 4)]) 
    ])
    
    builder = NetworkBuilder(or_node_class, and_node_class)
    neural_network_root = builder.build_network(root_node)

    actual_counts = count_unique_nodes(neural_network_root)
    
    expected_counts = {
        'LiteralNodeModule': 4,
        'ORNode': 2,
        'ANDNode': 2,
    }
    
    assert actual_counts == expected_counts, (
        f"Module count mismatch for {description}\n"
        f"Expected: {expected_counts}\n"
        f"Actual: {actual_counts}"
    )

def test_network_builder_unknown_node_type():
    """
    Tests that NetworkBuilder raises an appropriate error for unknown node types.
    """
    class UnknownNode:
        def __init__(self, node_id):
            self.id = node_id
    
    unknown_node = UnknownNode('unknown')
    builder = NetworkBuilder(IterativeORNode, IterativeANDNode)
    
    with pytest.raises(ValueError, match="Unknown node type"):
        builder.build_network(unknown_node)

def test_network_builder_node_caching():
    """
    Tests that the node caching mechanism works correctly across multiple calls.
    """
    literal_node = nnf.LiteralNode('L1', 1)
    root_node = nnf.AndNode('A1', [literal_node, literal_node])
    
    builder = NetworkBuilder(IterativeORNode, IterativeANDNode)
    
    node_cache = {}
    root1 = builder.build_network(root_node, node_cache)
    root2 = builder.build_network(root_node, node_cache)
    
    assert root1 is root2
    assert len(node_cache) == 2
