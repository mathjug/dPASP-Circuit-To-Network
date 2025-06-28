import torch
import pytest

from src.trivial_solutions.iterative_neural_network import IterativeNN
from src.trivial_solutions.recursive_neural_network import RecursiveNN
import src.parser.nnf_parser as nnf

from tests.utils.gradient_calculator import calculate_individual_gradients

# --- Test Cases Definition ---

implementations = [
    {
        "name": "Recursive",
        "implementation_class": RecursiveNN,
    },
    {
        "name": "Iterative",
        "implementation_class": IterativeNN,
    }
]

# Each test case is defined as a tuple:
# (description, nnf_circuit, input_data, expected_gradients)
test_cases = [
    (
        "Probabilistic AND: x₁ ∧ x₂",
        # f = x₁ * x₂
        # ∂f/∂x₁ = x₂, ∂f/∂x₂ = x₁
        lambda: nnf.AndNode('A1', [nnf.LiteralNode('L1', 1), nnf.LiteralNode('L2', 2)]),
        torch.tensor([[0.2, 0.7], [0.5, 0.5], [0.9, 0.1]]),
        torch.tensor([[0.7, 0.2], [0.5, 0.5], [0.1, 0.9]])
    ),
    (
        "Probabilistic OR: x₁ V x₂",
        # f = x₁ + x₂
        # ∂f/∂x₁ = 1, ∂f/∂x₂ = 1
        lambda: nnf.OrNode('O1', [nnf.LiteralNode('L1', 1), nnf.LiteralNode('L2', 2)]),
        torch.tensor([[0.1, 0.9], [0.8, 0.3]]),
        torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    ),
    (
        "Probabilistic AND with Negation: ¬x₁ ∧ x₂",
        # f = (1 - x₁) * x₂
        # ∂f/∂x₁ = -x₂, ∂f/∂x₂ = 1 - x₁
        lambda: nnf.AndNode('A1', [nnf.LiteralNode('L1', 1, negated=True), nnf.LiteralNode('L2', 2)]),
        torch.tensor([[0.3, 0.6], [0.8, 0.9]]),
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
        torch.tensor([[-0.8, 0.8, 1.0], [-0.1, 0.1, 1.0]])
    ),
    (
        "AND with TrueNode: x₁ ∧ True",
        # f = x₁ * 1 = x₁
        # ∂f/∂x₁ = 1
        lambda: nnf.AndNode('A1', [nnf.LiteralNode('L1', 1), nnf.TrueNode('T')]),
        torch.tensor([[0.25], [0.75]]),
        torch.tensor([[1.0], [1.0]])
    ),
    (
        "OR with TrueNode: x₁ V True",
        # f = x₁ + 1
        # ∂f/∂x₁ = 1
        lambda: nnf.OrNode('O1', [nnf.LiteralNode('L1', 1), nnf.TrueNode('T')]),
        torch.tensor([[0.25], [0.75]]),
        torch.tensor([[1.0], [1.0]])
    ),
    (
        "AND with FalseNode: x₁ ∧ False",
        # f = x₁ * 0 = 0
        # ∂f/∂x₁ = 0
        lambda: nnf.AndNode('A1', [nnf.LiteralNode('L1', 1), nnf.FalseNode('F')]),
        torch.tensor([[0.4], [0.6]]),
        torch.tensor([[0.0], [0.0]])
    ),
    (
        "OR with FalseNode: x₁ V False",
        # f = x₁ + 0 = x₁
        # ∂f/∂x₁ = 1
        lambda: nnf.OrNode('O1', [nnf.LiteralNode('L1', 1), nnf.FalseNode('F')]),
        torch.tensor([[0.4], [0.6]]),
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
        torch.tensor([[0.8, -0.5, 1.0], [0.2, -0.1, 1.0]])
    ),
]

@pytest.mark.parametrize("implementation", implementations, ids=[i['name'] for i in implementations])
@pytest.mark.parametrize("description, nnf_circuit, input_data, expected_gradients", test_cases)
def test_circuit_partial_derivatives_probabilistic(implementation, description, nnf_circuit, input_data, expected_gradients):
    """
    Tests that computed gradients match analytical derivatives for various circuits 
    and implementations using probabilistic inputs.
    """
    full_description = f"{description} ({implementation['name']} Implementation)"
    print(f"Testing circuit: {full_description}")
    
    root_node = nnf_circuit()

    implementation_class = implementation["implementation_class"]
    computed_gradients = calculate_individual_gradients(root_node, input_data.clone(), implementation_class)

    torch.testing.assert_close(
        computed_gradients, 
        expected_gradients, 
        msg=f"Gradient mismatch for circuit: {full_description}\n Expected: {expected_gradients}\n Actual: {computed_gradients}\n"
    )
