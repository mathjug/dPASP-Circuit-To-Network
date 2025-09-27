"""
This test file is dedicated to verifying the correctness of the descendant_variables attribute
for all nodes in neural networks built by NetworkBuilder. It specifically tests that:

1. Literal nodes have the correct descendant_variables (the variable they represent)
2. AND/OR nodes correctly aggregate descendant_variables from their children
3. Constant/True/False nodes have empty descendant_variables sets
4. The descendant_variables are correctly computed for complex networks with shared subexpressions

The tests use SDD/JSON files and verify that the resulting neural network nodes have
the expected descendant_variables sets.
"""
import pytest
import os

from src.trivial_solutions.network_builder import NetworkBuilder
from src.trivial_solutions.entities.or_node import IterativeORNode
from src.trivial_solutions.entities.and_node import IterativeANDNode

test_dir = os.path.dirname(os.path.abspath(__file__))
circuit_dir = os.path.join(test_dir, "test_circuits")

def collect_all_nodes(root_node):
    """Recursively collect all nodes in the network."""
    nodes = [root_node]
    if hasattr(root_node, 'children_nodes'):
        for child in root_node.children_nodes:
            nodes.extend(collect_all_nodes(child))
    return nodes

descendant_variables_test_cases = [
    (
        "Simple AND: x₁ ∧ x₂",
        os.path.join(circuit_dir, "simple_and.sdd"),
        os.path.join(circuit_dir, "simple_and.json"),
        [{1, 2}, {1}, {2}],
    ),
    (
        "Simple OR: x₁ V x₂",
        os.path.join(circuit_dir, "simple_or.sdd"),
        os.path.join(circuit_dir, "simple_or.json"),
        [{1, 2}, {1}, {2}],
    ),
    (
        "Memoization Test: (x₁ V x₂) ∧ (x₁ V x₂) - shared subexpressions",
        os.path.join(circuit_dir, "memoization_test.sdd"),
        os.path.join(circuit_dir, "memoization_test.json"),
        [{1, 2}, {1, 2}, {1}, {2}, {1, 2}, {1}, {2}],
    ),
    (
        "Complex Memoization Test: ((x₁ ∧ x₂) V x₃) ∧ ((x₁ ∧ x₂) V x₄) - shared subexpressions",
        os.path.join(circuit_dir, "memoization_test2.sdd"),
        os.path.join(circuit_dir, "memoization_test2.json"),
        [{1, 2, 3, 4}, {1, 2, 3}, {1, 2}, {1}, {2}, {3}, {1, 2, 4}, {1, 2}, {1}, {2}, {4}],
    ),
]

@pytest.mark.parametrize("description, sdd_file, json_file, expected_descendant_variables", descendant_variables_test_cases)
def test_descendant_variables(description, sdd_file, json_file, expected_descendant_variables):
    """
    Tests that all nodes in the network have the correct descendant_variables sets.
    """
    print(f"Testing descendant variables: {description}")

    neural_network_root = NetworkBuilder(IterativeORNode, IterativeANDNode).build_network(sdd_file, json_file)
    all_nodes = collect_all_nodes(neural_network_root)
    
    assert len(all_nodes) == len(expected_descendant_variables), (
        f"Node count mismatch in {description}\n"
        f"Expected: {len(expected_descendant_variables)}, Actual: {len(all_nodes)}"
    )
    
    # Check descendant_variables for each node
    for i, node in enumerate(all_nodes):
        expected_vars = expected_descendant_variables[i]
        actual_vars = node.descendant_variables
        
        assert actual_vars == expected_vars, (
            f"Descendant variables mismatch for node {i+1} ({type(node).__name__}) in {description}\n"
            f"Expected: {expected_vars}\n"
            f"Actual: {actual_vars}\n"
            f"Node: {node}"
        )
