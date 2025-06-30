"""
This test file is dedicated to verifying the correctness of the backward pass
for both RecursiveNN and IterativeNN implementations, using probabilistic inputs.
It checks if the computed gradients of various circuits matches the analytically
calculated expected gradients.

Tests are divided into two main scenarios:
1. Standard Forward Pass: Evaluates circuits with probabilistic inputs.
2. Marginalized Forward Pass: Evaluates circuits where the behavior of negated
   literals is altered for specific "marginalized" variables.
"""

import torch
import pytest

import src.parser.nnf_parser as nnf
from tests.utils.utils import calculate_individual_gradients, implementations

# --- Test Cases for Standard Backward Pass ---
# Each tuple: (description, nnf_circuit, input_data, marginalized_variables, expected_gradients)

standard_test_cases = [
    (
        "Probabilistic AND: x₁ ∧ x₂",
        # f = x₁ * x₂
        # ∂f/∂x₁ = x₂, ∂f/∂x₂ = x₁
        lambda: nnf.AndNode('A1', [nnf.LiteralNode('L1', 1), nnf.LiteralNode('L2', 2)]),
        torch.tensor([[0.2, 0.7], [0.5, 0.5], [0.9, 0.1]]),
        torch.tensor([0, 0]),
        torch.tensor([[0.7, 0.2], [0.5, 0.5], [0.1, 0.9]])
    ),
    (
        "Probabilistic OR: x₁ V x₂",
        # f = x₁ + x₂
        # ∂f/∂x₁ = 1, ∂f/∂x₂ = 1
        lambda: nnf.OrNode('O1', [nnf.LiteralNode('L1', 1), nnf.LiteralNode('L2', 2)]),
        torch.tensor([[0.1, 0.9], [0.8, 0.3]]),
        torch.tensor([0, 0]),
        torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    ),
    (
        "Probabilistic AND with Negation: ¬x₁ ∧ x₂",
        # f = (1 - x₁) * x₂
        # ∂f/∂x₁ = -x₂, ∂f/∂x₂ = 1 - x₁
        lambda: nnf.AndNode('A1', [nnf.LiteralNode('L1', 1, negated=True), nnf.LiteralNode('L2', 2)]),
        torch.tensor([[0.3, 0.6], [0.8, 0.9]]),
        torch.tensor([0, 0]),
        torch.tensor([[-0.6, 0.7], [-0.9, 0.2]])
    ),
    (
        "Probabilistic Complex Circuit: (¬x₁ ∧ x₂) V x₃",
        # f = (1 - x₁) * x₂ + x₃
        # ∂f/∂x₁ = -x₂, ∂f/∂x₂ = 1 - x₁, ∂f/∂x₃ = 1
        lambda: nnf.OrNode('O1', [
            nnf.AndNode('A1', [nnf.LiteralNode('L1', 1, negated=True), nnf.LiteralNode('L2', 2)]), 
            nnf.LiteralNode('L3', 3)
        ]),
        torch.tensor([[0.2, 0.8, 0.5], [0.9, 0.1, 0.3]]),
        torch.tensor([0, 0, 0]),
        torch.tensor([[-0.8, 0.8, 1.0], [-0.1, 0.1, 1.0]])
    ),
    (
        "AND with TrueNode: x₁ ∧ True",
        # f = x₁ * 1 = x₁
        # ∂f/∂x₁ = 1
        lambda: nnf.AndNode('A1', [nnf.LiteralNode('L1', 1), nnf.TrueNode('T')]),
        torch.tensor([[0.25], [0.75]]),
        torch.tensor([0, 0]),
        torch.tensor([[1.0], [1.0]])
    ),
    (
        "OR with TrueNode: x₁ V True",
        # f = x₁ + 1
        # ∂f/∂x₁ = 1
        lambda: nnf.OrNode('O1', [nnf.LiteralNode('L1', 1), nnf.TrueNode('T')]),
        torch.tensor([[0.25], [0.75]]),
        torch.tensor([0, 0]),
        torch.tensor([[1.0], [1.0]])
    ),
    (
        "AND with FalseNode: x₁ ∧ False",
        # f = x₁ * 0 = 0
        # ∂f/∂x₁ = 0
        lambda: nnf.AndNode('A1', [nnf.LiteralNode('L1', 1), nnf.FalseNode('F')]),
        torch.tensor([[0.4], [0.6]]),
        torch.tensor([0, 0]),
        torch.tensor([[0.0], [0.0]])
    ),
    (
        "OR with FalseNode: x₁ V False",
        # f = x₁ + 0 = x₁
        # ∂f/∂x₁ = 1
        lambda: nnf.OrNode('O1', [nnf.LiteralNode('L1', 1), nnf.FalseNode('F')]),
        torch.tensor([[0.4], [0.6]]),
        torch.tensor([0, 0]),
        torch.tensor([[1.0], [1.0]])
    ),
    (
        "Complex with True/False: (x₁ ∧ ¬x₂) V (x₃ ∧ True) V False",
        # f = x₁ * (1 - x₂) + x₃ * 1 + 0 = x₁ - x₁*x₂ + x₃
        # ∂f/∂x₁ = 1 - x₂, ∂f/∂x₂ = -x₁, ∂f/∂x₃ = 1
        lambda: nnf.OrNode('O1', [
            nnf.OrNode('O2', [
                nnf.AndNode('A1', [nnf.LiteralNode('L1', 1), nnf.LiteralNode('L2', 2, negated=True)]),
                nnf.AndNode('A2', [nnf.LiteralNode('L3', 3), nnf.TrueNode('T')])
            ]),
            nnf.FalseNode('F')
        ]),
        torch.tensor([[0.5, 0.2, 0.9], [0.1, 0.8, 0.4]]),
        torch.tensor([0, 0, 0]),
        torch.tensor([[0.8, -0.5, 1.0], [0.2, -0.1, 1.0]])
    ),
]

# --- Test Cases for Marginalized Backward Pass ---
# Each tuple: (description, nnf_circuit, input_data, marginalized_variables, expected_gradients)

marginalized_test_cases = [
    (
        "Simple Marginalized Negation: ¬x₁",
        # x₁ is marginalized. f = x₁.
        # ∂f/∂x₁ = 1
        lambda: nnf.LiteralNode('L1', 1, negated=True),
        torch.tensor([[0.3], [0.8]]),
        torch.tensor([1]),
        torch.tensor([[1.0], [1.0]])
    ),
    (
        "AND with Marginalized Negation: ¬x₁ ∧ x₂",
        # x₁ is marginalized. f = x₁ * x₂.
        # ∂f/∂x₁ = x₂, ∂f/∂x₂ = x₁
        lambda: nnf.AndNode('A1', [nnf.LiteralNode('L1', 1, negated=True), nnf.LiteralNode('L2', 2)]),
        torch.tensor([[0.4, 0.5], [0.9, 0.2]]),
        torch.tensor([1, 0]),
        torch.tensor([[0.5, 0.4], [0.2, 0.9]])
    ),
    (
        "OR with Marginalized Negation: x₁ V ¬x₂",
        # x₂ is marginalized. f = x₁ + x₂.
        # ∂f/∂x₁ = 1, ∂f/∂x₂ = 1
        lambda: nnf.OrNode('O1', [nnf.LiteralNode('L1', 1), nnf.LiteralNode('L2', 2, negated=True)]),
        torch.tensor([[0.1, 0.6], [0.7, 0.3]]),
        torch.tensor([0, 1]),
        torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    ),
    (
        "Complex Marginalized: (¬x₁ ∧ x₂) V ¬x₃",
        # x₁ and x₃ are marginalized. f = (x₁ * x₂) + x₃.
        # ∂f/∂x₁ = x₂, ∂f/∂x₂ = x₁, ∂f/∂x₃ = 1
        lambda: nnf.OrNode('O1', [
            nnf.AndNode('A1', [nnf.LiteralNode('L1', 1, negated=True), nnf.LiteralNode('L2', 2)]),
            nnf.LiteralNode('L3', 3, negated=True)
        ]),
        torch.tensor([[0.5, 0.8, 0.2], [0.1, 0.9, 0.7]]),
        torch.tensor([1, 0, 1]),
        torch.tensor([[0.8, 0.5, 1.0], [0.9, 0.1, 1.0]])
    ),
    (
        "Mixed Negation: (¬x₁ ∧ x₂) V ¬x₃",
        # Only x₁ is marginalized. f = (x₁ * x₂) + (1 - x₃).
        # ∂f/∂x₁ = x₂, ∂f/∂x₂ = x₁, ∂f/∂x₃ = -1
        lambda: nnf.OrNode('O1', [
            nnf.AndNode('A1', [nnf.LiteralNode('L1', 1, negated=True), nnf.LiteralNode('L2', 2)]),
            nnf.LiteralNode('L3', 3, negated=True)
        ]),
        torch.tensor([[0.5, 0.8, 0.2], [0.1, 0.9, 0.7]]),
        torch.tensor([1, 0, 0]),
        torch.tensor([[0.8, 0.5, -1.0], [0.9, 0.1, -1.0]])
    )
]

# --- Test Cases for Probabilistic Backward Pass with Shared Subexpressions ---
# Each tuple: (description, nnf_circuit, input_data, marginalized_variables, expected_gradients)

shared_subexpression_test_cases = [
    (
        "Probabilistic Circuit with Shared OR: (x₁ V x₂) ∧ (x₁ V x₂)",
        # f = (x₁ + x₂) * (x₁ + x₂) = (x₁ + x₂)²
        # ∂f/∂x₁ = 2(x₁ + x₂), ∂f/∂x₂ = 2(x₁ + x₂)
        lambda: nnf.AndNode('A1', [
            nnf.OrNode('O1', [nnf.LiteralNode('L1', 1), nnf.LiteralNode('L2', 2)]),
            nnf.OrNode('O1', [nnf.LiteralNode('L1', 1), nnf.LiteralNode('L2', 2)])
        ]),
        torch.tensor([[0.3, 0.7], [0.5, 0.5], [0.8, 0.2]]),
        torch.tensor([0, 0]),
        torch.tensor([[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]])
    ),
    (
        "Probabilistic Circuit with Shared AND: ((x₁ ∧ x₂) V x₃) ∧ ((x₁ ∧ x₂) V x₄)",
        # f = ((x₁ * x₂) + x₃) * ((x₁ * x₂) + x₄)
        # ∂f/∂x₁ = x₂ * ((x₁ * x₂) + x₄) + x₂ * ((x₁ * x₂) + x₃)
        # ∂f/∂x₂ = x₁ * ((x₁ * x₂) + x₄) + x₁ * ((x₁ * x₂) + x₃)
        # ∂f/∂x₃ = (x₁ * x₂) + x₄
        # ∂f/∂x₄ = (x₁ * x₂) + x₃
        lambda: nnf.AndNode('A1', [
            nnf.OrNode('O1', [
                nnf.AndNode('A2', [nnf.LiteralNode('L1', 1), nnf.LiteralNode('L2', 2)]),
                nnf.LiteralNode('L3', 3)
            ]),
            nnf.OrNode('O2', [
                nnf.AndNode('A2', [nnf.LiteralNode('L1', 1), nnf.LiteralNode('L2', 2)]),
                nnf.LiteralNode('L6', 4)
            ])
        ]),
        torch.tensor([[0.6, 0.4, 0.3, 0.7], [0.2, 0.8, 0.5, 0.5], [0.9, 0.1, 0.4, 0.6], [0.3, 0.7, 0.2, 0.8]]),
        torch.tensor([0, 0, 0, 0]),
        torch.tensor([[0.592, 0.888, 0.94, 0.54], [1.056, 0.264, 0.66, 0.66], [0.118, 1.062, 0.69, 0.49], [0.994, 0.426, 1.01, 0.41]])
    ),
    (
        "Probabilistic Circuit with Shared Negation: (¬x₁ ∧ x₂) V (¬x₁ ∧ x₃)",
        # f = ((1 - x₁) * x₂) + ((1 - x₁) * x₃) = (1 - x₁) * (x₂ + x₃)
        # ∂f/∂x₁ = -x₂ - x₃
        # ∂f/∂x₂ = 1 - x₁
        # ∂f/∂x₃ = 1 - x₁
        lambda: nnf.OrNode('O1', [
            nnf.AndNode('A1', [nnf.LiteralNode('L1', 1, negated=True), nnf.LiteralNode('L2', 2)]),
            nnf.AndNode('A2', [nnf.LiteralNode('L1', 1, negated=True), nnf.LiteralNode('L4', 3)])
        ]),
        torch.tensor([[0.4, 0.6, 0.8], [0.7, 0.3, 0.5]]),
        torch.tensor([0, 0, 0]),
        torch.tensor([[-1.4, 0.6, 0.6], [-0.8, 0.3, 0.3]])
    )
]

test_cases = standard_test_cases + marginalized_test_cases + shared_subexpression_test_cases

@pytest.mark.parametrize("implementation", implementations, ids=[i['name'] for i in implementations])
@pytest.mark.parametrize("description, nnf_circuit, input_data, marginalized_vars, expected_gradients", test_cases)
def test_circuit_derivatives_probabilistic(implementation, description, nnf_circuit, input_data, marginalized_vars, expected_gradients):
    """
    Tests that computed gradients match analytical derivatives for various circuits 
    and implementations using probabilistic inputs.
    """
    full_description = f"{description} ({implementation['name']} Implementation)"
    print(f"Testing circuit: {full_description}")
    
    root_node = nnf_circuit()

    implementation_class = implementation["implementation_class"]
    computed_gradients = calculate_individual_gradients(root_node, input_data.clone(), implementation_class, marginalized_vars)

    torch.testing.assert_close(
        computed_gradients, 
        expected_gradients, 
        msg=f"Gradient mismatch for circuit: {full_description}\n Expected: {expected_gradients}\n Actual: {computed_gradients}\n"
    )
