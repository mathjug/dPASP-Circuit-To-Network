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
        "Simple AND: x₁ ∧ x₂",
        lambda: nnf.AndNode('A1', [nnf.LiteralNode('L1', 1), nnf.LiteralNode('L2', 2)]),
        torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]]),
        torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    ),
    (
        "Simple OR: x₁ V x₂",
        lambda: nnf.OrNode('O1', [nnf.LiteralNode('L1', 1), nnf.LiteralNode('L2', 2)]),
        torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]]),
        torch.tensor([[1., 1.], [1., 1.], [1., 1.], [1., 1.]])
    ),
    (
        "Simple Negation: ¬x₁",
        lambda: nnf.LiteralNode('L1', 1, negated=True),
        torch.tensor([[0.], [1.]]),
        torch.tensor([[-1.], [-1.]])
    ),
    (
        "Simple AND with Negation: ¬x₁ ∧ x₂",
        lambda: nnf.AndNode('A1', [nnf.LiteralNode('L1', 1, negated=True), nnf.LiteralNode('L2', 2)]),
        torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]]),
        torch.tensor([[0., 1.], [-1., 1.], [0., 0.], [-1., 0.]])
    ),
    (
        "Complex Circuit 1: (¬x₁ ∧ x₂) V x₃",
        lambda: nnf.OrNode('O1', [
            nnf.AndNode('A1', [nnf.LiteralNode('L1', 1, negated=True), nnf.LiteralNode('L2', 2)]), 
            nnf.LiteralNode('L3', 3)
        ]),
        torch.tensor([[0., 1., 0.], [1., 1., 0.], [0., 0., 1.], [1., 0., 1.]]),
        torch.tensor([[-1., 1., 1.], [-1., 0., 1.], [ 0., 1., 1.], [ 0., 0., 1.]])
    ),
    (
        "Complex Circuit 2: (x₀ ∧ x₁) V (x₂ ∧ ¬x₃)",
        lambda: nnf.OrNode("6", [
            nnf.AndNode("4", [nnf.LiteralNode("0", 1), nnf.LiteralNode("1", 2)]),
            nnf.AndNode("5", [nnf.LiteralNode("2", 3), nnf.LiteralNode("3", 4, negated=True)])
        ]),
        torch.tensor([[1., 1., 0., 0.], [0., 1., 1., 1.], [1., 0., 1., 0.]]),
        torch.tensor([[1., 1., 1., -0.], [1., 0., 0., -1.], [0., 1., 1., -1.]])
    )
]

@pytest.mark.parametrize("implementation", implementations, ids=[i['name'] for i in implementations])
@pytest.mark.parametrize("description, nnf_circuit, input_data, expected_gradients", test_cases)
def test_circuit_partial_derivatives_boolean(implementation, description, nnf_circuit, input_data, expected_gradients):
    """
    Tests that computed gradients match analytical derivatives for various circuits and implementations
    using boolean inputs.
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
