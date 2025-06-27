import torch
import pytest

from src.trivial_solutions.iterative_neural_network import IterativeNN
from src.trivial_solutions.recursive_neural_network import RecursiveNN
import src.parser.nnf_parser as nnf

def calculate_individual_gradients(root_node, input_tensor, executor_class):
    """
    Calculates the gradient of the output w.r.t each input for each sample in a batch.
    
    Returns:
        torch.Tensor: A tensor of the same shape as input_tensor, containing the gradients.
    """
    input_tensor.requires_grad_(True)
    neural_network = executor_class(root_node)
    output = neural_network.forward(input_tensor)
    
    batch_size = input_tensor.shape[0]
    num_features = input_tensor.shape[1]
    all_grads = torch.zeros_like(input_tensor)

    for i in range(batch_size):
        # Clear previous gradients before calculating the new one
        if input_tensor.grad is not None:
            input_tensor.grad.zero_()
        output[i].backward(retain_graph=True)        
        all_grads[i, :] = input_tensor.grad[i, :]
        
    input_tensor.grad.zero_()
    input_tensor.requires_grad_(False)    
    return all_grads

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
def test_circuit_partial_derivatives(implementation, description, nnf_circuit, input_data, expected_gradients):
    """
    Tests that computed gradients match analytical derivatives for various circuits and implementations.
    """
    full_description = f"{description} ({implementation['name']} Implementation)"
    print(f"Testing circuit: {full_description}")
    
    root_node = nnf_circuit()

    implementation_class = implementation["implementation_class"]
    computed_gradients = calculate_individual_gradients(root_node, input_data.clone(), implementation_class)

    torch.testing.assert_close(computed_gradients, expected_gradients, msg=f"Gradient mismatch for circuit: {full_description}\
    \n Expected: {expected_gradients}\n Actual: {computed_gradients}\n")
