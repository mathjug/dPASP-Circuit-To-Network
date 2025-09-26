"""
This test file is dedicated to verifying the correctness of the NetworkBuilder class
when used with SDD/JSON files. It specifically tests that:

1. Networks with shared subexpressions are built correctly without duplicates
2. Node caching works properly using the node ID mechanism
3. The same node instances are reused when they appear multiple times in the NNF
4. The builder works correctly with different node class combinations

The tests use SDD/JSON files with intentionally repeated subexpressions and verify that
the resulting neural network contains the expected number of unique modules.
"""
import pytest
import os

from src.trivial_solutions.recursive_neural_network import RecursiveNN
from src.trivial_solutions.iterative_neural_network import IterativeNN
from tests.utils.utils import count_unique_nodes

test_dir = os.path.dirname(os.path.abspath(__file__))
circuit_dir = os.path.join(test_dir, "test_circuits")

network_construction_test_cases = [
    (
        "Simple AND: x₁ ∧ x₂",
        os.path.join(circuit_dir, "simple_and.sdd"),
        os.path.join(circuit_dir, "simple_and.json"),
        {
            'LiteralNodeModule': 2,
            'ANDNode': 1,
            'ORNode': 1
        },
    ),
    (
        "Simple OR: x₁ V x₂",
        os.path.join(circuit_dir, "simple_or.sdd"),
        os.path.join(circuit_dir, "simple_or.json"),
        {
            'LiteralNodeModule': 2,
            'ORNode': 1,
            'ANDNode': 2,
            'TrueNode': 1
        },
    ),
    (
        "Memoization Test 1: (x₁ V x₂) ∧ (x₁ V x₂) - shared subexpressions",
        os.path.join(circuit_dir, "memoization_test.sdd"),
        os.path.join(circuit_dir, "memoization_test.json"),
        {
            'LiteralNodeModule': 2, # x1 and x2 should be shared
            'ORNode': 2,            # Two OR nodes
            'ANDNode': 3,           # Three AND nodes
            'TrueNode': 1
        },
    ),
    (
        "Memoization Test 2: ((x₁ ∧ x₂) V x₃) ∧ ((x₁ ∧ x₂) V x₄) - shared subexpressions",
        os.path.join(circuit_dir, "memoization_test2.sdd"),
        os.path.join(circuit_dir, "memoization_test2.json"),
        {
            'LiteralNodeModule': 4, # x1, x2, x3, x4
            'ANDNode': 5,           # Five AND nodes
            'ORNode': 4,            # Four OR nodes
            'TrueNode': 1
        },
    ),
    (
        "Simple AND NOT: x₁ ∧ ¬x₂",
        os.path.join(circuit_dir, "simple_and_not.sdd"),
        os.path.join(circuit_dir, "simple_and_not.json"),
        {
            'LiteralNodeModule': 2,
            'ANDNode': 1 + 3,
            'ORNode': 1 + 1,
            'ConstantNode': 4
        },
    ),
    (
        "Complex Shared AND: (x₁ ∧ x₂) V (x₁ ∧ x₂) V (x₃ ∧ x₄) - shared subexpressions",
        os.path.join(circuit_dir, "complex_shared_and.sdd"),
        os.path.join(circuit_dir, "complex_shared_and.json"),
        {
            'LiteralNodeModule': 4,  # x1, x2, x3, x4
            'ANDNode': 4 + 4,        # Eight AND nodes
            'ORNode': 3,             # Three OR nodes
            'ConstantNode': 4,       # Four constant nodes
            'TrueNode': 1
        },
    ),
    (
        "Deep Nested Shared: ((x₁ V x₂) ∧ x₃) V ((x₁ V x₂) ∧ x₄) V ((x₁ V x₂) ∧ x₅) - shared subexpressions",
        os.path.join(circuit_dir, "deep_nested_shared.sdd"),
        os.path.join(circuit_dir, "deep_nested_shared.json"),
        {
            'LiteralNodeModule': 5,  # x1, x2, x3, x4, x5
            'ANDNode': 8 + 5,        # Thirteen AND nodes
            'ORNode': 5,             # Five OR nodes
            'ConstantNode': 5,       # Five constant nodes
            'TrueNode': 1
        },
    )
]

@pytest.mark.parametrize("implementation_class", [RecursiveNN, IterativeNN], ids=["Recursive", "Iterative"])
@pytest.mark.parametrize("description, sdd_file, json_file, expected_counts", network_construction_test_cases)
def test_network_builder_no_duplicates(implementation_class, description, sdd_file, json_file, expected_counts):
    """
    Tests that NetworkBuilder constructs networks without duplicate nodes when the same NNF node
    appears multiple times in the circuit.
    """
    implementation_name = "Recursive" if implementation_class == RecursiveNN else "Iterative"
    full_description = f"{description} ({implementation_name} Implementation)"
    print(f"Testing network builder: {full_description}")

    neural_network = implementation_class(sdd_file, json_file)
    actual_counts = count_unique_nodes(neural_network.root)
    
    assert actual_counts == expected_counts, (
        f"Module count mismatch for {full_description}\n"
        f"Expected: {expected_counts}\n"
        f"Actual: {actual_counts}"
    )

def test_network_builder_node_caching():
    """
    Tests that the node caching mechanism works correctly across multiple calls.
    """
    sdd_file = os.path.join(circuit_dir, "memoization_test.sdd")
    json_file = os.path.join(circuit_dir, "memoization_test.json")
    
    nn1 = RecursiveNN(sdd_file, json_file)
    nn2 = RecursiveNN(sdd_file, json_file)
    counts1 = count_unique_nodes(nn1.root)
    counts2 = count_unique_nodes(nn2.root)
    
    assert counts1 == counts2, "Different network instances should have the same structure"
