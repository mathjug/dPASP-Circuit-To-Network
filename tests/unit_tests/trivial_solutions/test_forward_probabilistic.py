"""
This test file is dedicated to verifying the correctness of the forward pass
for both RecursiveNN and IterativeNN implementations, using probabilistic inputs.
It checks if the computed output of various circuits matches the analytically
calculated expected output.

Tests are divided into two main scenarios:
1. Standard Forward Pass: Evaluates circuits with probabilistic inputs.
2. Marginalized Forward Pass: Evaluates circuits where the behavior of negated
   literals is altered for specific "marginalized" variables.
"""

import torch
import pytest

import src.parser.nnf_parser as nnf
from tests.utils.utils import implementations

# --- Test Cases for Standard Forward Pass ---
# Each tuple: (description, nnf_circuit, input_data, marginalized_vars, expected_output)

standard_test_cases = [
    (
        "Complex AND of ORs: (x₁ V ¬x₂) ∧ (¬x₃ V x₄)",
        # Circuit function: f = (p(x₁) + (1 - p(x₂))) * ((1 - p(x₃)) + p(x₄))
        lambda: nnf.AndNode('A1', [
            nnf.OrNode('O1', [nnf.LiteralNode('L1', 1), nnf.LiteralNode('L2', 2, negated=True)]),
            nnf.OrNode('O2', [nnf.LiteralNode('L3', 3, negated=True), nnf.LiteralNode('L4', 4)])
        ]),
        torch.tensor([
            [0.2, 0.8, 0.1, 0.9],  # (0.2 + 0.2) * (0.9 + 0.9) = 0.4 * 1.8 = 0.72
            [0.5, 0.5, 0.5, 0.5],  # (0.5 + 0.5) * (0.5 + 0.5) = 1.0 * 1.0 = 1.0
            [1.0, 0.0, 0.0, 1.0],  # (1.0 + 1.0) * (1.0 + 1.0) = 2.0 * 2.0 = 4.0
        ]),
        torch.tensor([0, 0, 0, 0]),
        torch.tensor([[0.72], [1.0], [4.0]])
    ),
    (
        "Complex OR of ANDs with a Literal: (x₁ ∧ ¬x₂) V (x₃ ∧ ¬x₄) V x₅",
        # Circuit function: f = (p(x₁) * (1-p(x₂))) + (p(x₃) * (1-p(x₄))) + p(x₅)
        lambda: nnf.OrNode('O1', [
            nnf.AndNode('A1', [nnf.LiteralNode('L1', 1), nnf.LiteralNode('L2', 2, negated=True)]),
            nnf.AndNode('A2', [nnf.LiteralNode('L3', 3), nnf.LiteralNode('L4', 4, negated=True)]),
            nnf.LiteralNode('L5', 5)
        ]),
        torch.tensor([
            [0.9, 0.1, 0.8, 0.2, 0.3], # (0.9*0.9) + (0.8*0.8) + 0.3 = 0.81 + 0.64 + 0.3 = 1.75
            [0.0, 1.0, 1.0, 0.0, 0.5], # (0.0*0.0) + (1.0*1.0) + 0.5 = 0.0 + 1.0 + 0.5 = 1.5
        ]),
        torch.tensor([0, 0, 0, 0, 0]),
        torch.tensor([[1.75], [1.5]])
    ),
    (
        "Circuit with True/False nodes: ((x₁ V True) ∧ x₂) V (x₃ ∧ False)",
        # Circuit function: f = ((p(x₁) + 1) * p(x₂)) + (p(x₃) * 0)
        lambda: nnf.OrNode('O1', [
            nnf.AndNode('A1', [
                nnf.OrNode('O2', [nnf.LiteralNode('L1', 1), nnf.TrueNode('T')]),
                nnf.LiteralNode('L2', 2)
            ]),
            nnf.AndNode('A2', [nnf.LiteralNode('L3', 3), nnf.FalseNode('F')])
        ]),
        torch.tensor([
            [0.7, 0.5, 0.9], # ((0.7 + 1) * 0.5) + (0.9 * 0) = 1.7 * 0.5 = 0.85
            [0.1, 0.2, 0.3], # ((0.1 + 1) * 0.2) + (0.3 * 0) = 1.1 * 0.2 = 0.22
        ]),
        torch.tensor([0, 0, 0]),
        torch.tensor([[0.85], [0.22]])
    )
]

# --- Test Cases for Marginalized Forward Pass ---
# Each tuple: (description, nnf_circuit, input_data, marginalized_vars, expected_output)

marginalized_test_cases = [
    (
        "Complex AND of ORs with Marginalization: (x₁ V ¬x₂) ∧ (¬x₃ V x₄)",
        # Marginalize x₂ and x₃. ¬x₂ -> 1, ¬x₃ -> 1
        # Function: f = (p(x₁) + 1) * (1 + p(x₄))
        lambda: nnf.AndNode('A1', [
            nnf.OrNode('O1', [nnf.LiteralNode('L1', 1), nnf.LiteralNode('L2', 2, negated=True)]),
            nnf.OrNode('O2', [nnf.LiteralNode('L3', 3, negated=True), nnf.LiteralNode('L4', 4)])
        ]),
        torch.tensor([
            [0.2, 0.8, 0.1, 0.9], # (0.2 + 1) * (1 + 0.9) = 1.2 * 1.9 = 2.28
            [0.5, 0.5, 0.5, 0.5], # (0.5 + 1) * (1 + 0.5) = 1.5 * 1.5 = 2.25
        ]),
        torch.tensor([0, 1, 1, 0]),
        torch.tensor([[2.28], [2.25]])
    ),
    (
        "Complex OR of ANDs with Partial Marginalization: (x₁ ∧ ¬x₂) V (x₃ ∧ ¬x₄)",
        # Marginalize only x₄. ¬x₄ -> 1
        # Function: f = (p(x₁) * (1 - p(x₂))) + (p(x₃) * 1)
        lambda: nnf.OrNode('O1', [
            nnf.AndNode('A1', [nnf.LiteralNode('L1', 1), nnf.LiteralNode('L2', 2, negated=True)]),
            nnf.AndNode('A2', [nnf.LiteralNode('L3', 3), nnf.LiteralNode('L4', 4, negated=True)]),
        ]),
        torch.tensor([
            [0.9, 0.1, 0.8, 0.2], # (0.9 * 0.9) + (0.8 * 1) = 0.81 + 0.8 = 1.61
            [0.5, 0.2, 0.3, 0.4], # (0.5 * 0.8) + (0.3 * 1) = 0.4 + 0.3 = 0.7
        ]),
        torch.tensor([0, 0, 0, 1]),
        torch.tensor([[1.61], [0.7]])
    )
]

# --- Test Cases for Memoization Cache Verification ---
# Each tuple: (description, nnf_circuit, input_data, marginalized_vars, expected_output)

memoization_test_cases = [
    (
        "Probabilistic Circuit with Shared Subexpressions: (x₁ V x₂) ∧ (x₁ V x₂)",
        lambda: nnf.AndNode('A1', [
            nnf.OrNode('O1', [nnf.LiteralNode('L1', 1), nnf.LiteralNode('L2', 2)]),
            nnf.OrNode('O2', [nnf.LiteralNode('L3', 1), nnf.LiteralNode('L4', 2)])
        ]),
        torch.tensor([
            [0.8, 0.2],  # (0.8 + 0.2) * (0.8 + 0.2) = 1.0 * 1.0 = 1.0
            [0.5, 0.5],  # (0.5 + 0.5) * (0.5 + 0.5) = 1.0 * 1.0 = 1.0
            [0.9, 0.1],  # (0.9 + 0.1) * (0.9 + 0.1) = 1.0 * 1.0 = 1.0
        ]),
        torch.tensor([0, 0]),
        torch.tensor([[1.0], [1.0], [1.0]])
    ),
    (
        "Probabilistic Complex Circuit with Multiple Shared Nodes: ((x₁ V x₂) ∧ x₃) V ((x₁ V x₂) ∧ x₄)",
        lambda: nnf.OrNode('O1', [
            nnf.AndNode('A1', [
                nnf.OrNode('O2', [nnf.LiteralNode('L1', 1), nnf.LiteralNode('L2', 2)]),
                nnf.LiteralNode('L3', 3)
            ]),
            nnf.AndNode('A2', [
                nnf.OrNode('O3', [nnf.LiteralNode('L4', 1), nnf.LiteralNode('L5', 2)]),
                nnf.LiteralNode('L6', 4)
            ])
        ]),
        torch.tensor([
            [0.7, 0.3, 0.8, 0.2],  # ((0.7+0.3)*0.8) + ((0.7+0.3)*0.2) = 0.8 + 0.2 = 1.0
            [0.4, 0.6, 0.5, 0.5],  # ((0.4+0.6)*0.5) + ((0.4+0.6)*0.5) = 0.5 + 0.5 = 1.0
            [0.9, 0.1, 0.3, 0.7],  # ((0.9+0.1)*0.3) + ((0.9+0.1)*0.7) = 0.3 + 0.7 = 1.0
        ]),
        torch.tensor([0, 0, 0, 0]),
        torch.tensor([[1.0], [1.0], [1.0]])
    ),
    (
        "Probabilistic Deep Circuit with Shared Paths: ((x₁ ∧ x₂) V x₃) ∧ ((x₁ ∧ x₂) V x₄)",
        lambda: nnf.AndNode('A1', [
            nnf.OrNode('O1', [
                nnf.AndNode('A2', [nnf.LiteralNode('L1', 1), nnf.LiteralNode('L2', 2)]),
                nnf.LiteralNode('L3', 3)
            ]),
            nnf.OrNode('O2', [
                nnf.AndNode('A3', [nnf.LiteralNode('L4', 1), nnf.LiteralNode('L5', 2)]),
                nnf.LiteralNode('L6', 4)
            ])
        ]),
        torch.tensor([
            [0.8, 0.7, 0.2, 0.3],  # ((0.8*0.7)+0.2) * ((0.8*0.7)+0.3) = 0.76 * 0.86 = 0.6536
            [0.5, 0.5, 0.5, 0.5],  # ((0.5*0.5)+0.5) * ((0.5*0.5)+0.5) = 0.75 * 0.75 = 0.5625
            [0.9, 0.1, 0.8, 0.2],  # ((0.9*0.1)+0.8) * ((0.9*0.1)+0.2) = 0.89 * 0.29 = 0.2581
        ]),
        torch.tensor([0, 0, 0, 0]),
        torch.tensor([[0.6536], [0.5625], [0.2581]])
    )
]

test_cases = standard_test_cases + marginalized_test_cases + memoization_test_cases

@pytest.mark.parametrize("implementation", implementations, ids=[i['name'] for i in implementations])
@pytest.mark.parametrize("description, nnf_circuit, input_data, marginalized_vars, expected_output", test_cases)
def test_forward_pass_marginalized(implementation, description, nnf_circuit, input_data, marginalized_vars, expected_output):
    """
    Tests the forward pass output against analytical results using probabilistic inputs.
    """
    full_description = f"{description} ({implementation['name']} Implementation)"
    print(f"Testing marginalized forward pass: {full_description}")

    root_node = nnf_circuit()
    implementation_class = implementation["implementation_class"]
    neural_network = implementation_class(root_node)

    computed_output = neural_network.forward(input_data, marginalized_variables=marginalized_vars)

    torch.testing.assert_close(
        computed_output,
        expected_output,
        msg=f"Output mismatch for marginalized circuit: {full_description}\n Expected: {expected_output}\n Actual: {computed_output}\n"
    )
