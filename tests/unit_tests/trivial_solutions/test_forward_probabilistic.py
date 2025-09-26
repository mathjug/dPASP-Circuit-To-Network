"""
This test file is dedicated to verifying the correctness of the forward pass
for both RecursiveNN and IterativeNN implementations, using probabilistic inputs.
It checks if the computed output of various circuits matches the analytically
calculated expected output.
"""

import torch
import pytest
import tempfile
import os

import src.parser.nnf_parser as nnf
from tests.utils.utils import implementations

# Get the directory where this test file is located
test_dir = os.path.dirname(os.path.abspath(__file__))
circuit_dir = os.path.join(test_dir, "test_circuits")

# --- Test Cases for Standard Forward Pass ---
# Each tuple: (description, sdd_file, json_file, input_data, expected_output)

standard_test_cases = [
    (
        "Simple AND: x₁ ∧ x₂",
        os.path.join(circuit_dir, "simple_and.sdd"),
        os.path.join(circuit_dir, "simple_and.json"),
        torch.tensor([
            [0.2, 0.8, 0.8, 0.2],  # x1=0.2, ¬x1=0.8, x2=0.8, ¬x2=0.2 -> 0.2 * 0.8 = 0.16
            [0.5, 0.5, 0.5, 0.5],  # x1=0.5, ¬x1=0.5, x2=0.5, ¬x2=0.5 -> 0.5 * 0.5 = 0.25
            [1.0, 0.0, 0.0, 1.0],  # x1=1.0, ¬x1=0.0, x2=0.0, ¬x2=1.0 -> 1.0 * 0.0 = 0.0
        ]),
        torch.tensor([[0.16], [0.25], [0.0]])
    ),
    (
        "Simple OR: x₁ V x₂",
        os.path.join(circuit_dir, "simple_or.sdd"),
        os.path.join(circuit_dir, "simple_or.json"),
        torch.tensor([
            [0.2, 0.8, 0.8, 0.2],  # x1=0.2, ¬x1=0.8, x2=0.8, ¬x2=0.2 -> 0.2 + 0.8 = 1.0
            [0.5, 0.5, 0.5, 0.5],  # x1=0.5, ¬x1=0.5, x2=0.5, ¬x2=0.5 -> 0.5 + 0.5 = 1.0
            [1.0, 0.0, 0.0, 1.0],  # x1=1.0, ¬x1=0.0, x2=0.0, ¬x2=1.0 -> 1.0 + 0.0 = 1.0
        ]),
        torch.tensor([[1.0], [1.0], [1.0]])
    ),
    (
        "Simple AND NOT: x₁ ∧ ¬x₂",
        os.path.join(circuit_dir, "simple_and_not.sdd"),
        os.path.join(circuit_dir, "simple_and_not.json"),
        torch.tensor([
            [1.0, 0.0, 1.0, 0.0],  # x1=1.0, ¬x1=0.0, x2=1.0, ¬x2=0.0 -> 0.3 * 0.0 = 0.00
            [0.0, 1.0, 1.0, 0.0],  # x1=0.0, ¬x1=1.0, x2=1.0, ¬x2=0.0 -> 0.0 * 0.4 = 0.00
            [1.0, 0.0, 0.0, 1.0],  # x1=1.0, ¬x1=0.0, x2=0.0, ¬x2=1.0 -> 0.3 * 0.4 = 0.12
        ]),
        torch.tensor([[0.00], [0.00], [0.12]])
    )
]

# --- Test Cases for Memoized and Marginalized Forward Pass ---
# Each tuple: (description, sdd_file, json_file, input_data, expected_output)

marginalized_test_cases = [
    (
        "Memoization Test 1: (x₁ V x₂) ∧ (x₁ V x₂) - shared subexpressions",
        os.path.join(circuit_dir, "memoization_test.sdd"),
        os.path.join(circuit_dir, "memoization_test.json"),
        torch.tensor([
            [0.2, 0.8, 0.8, 0.2],  # x1=0.2, ¬x1=0.8, x2=0.8, ¬x2=0.2 -> (0.2 + 0.8) * (0.2 + 0.8) = 1.0 * 1.0 = 1.0
            [0.5, 0.5, 0.5, 0.5],  # x1=0.5, ¬x1=0.5, x2=0.5, ¬x2=0.5 -> (0.5 + 0.5) * (0.5 + 0.5) = 1.0 * 1.0 = 1.0
            [0.9, 0.1, 0.4, 0.6],  # x1=0.9, ¬x1=0.1, x2=0.4, ¬x2=0.6 -> (0.9 + 0.4) * (0.9 + 0.4) = 1.3 * 1.3 = 1.69
        ]),
        torch.tensor([[1.0], [1.0], [1.69]])
    ),
    (
        "Memoization Test 2: ((x₁ ∧ x₂) V x₃) ∧ ((x₁ ∧ x₂) V x₄) - shared subexpressions",
        os.path.join(circuit_dir, "memoization_test2.sdd"),
        os.path.join(circuit_dir, "memoization_test2.json"),
        torch.tensor([
            [0.8, 0.2, 0.7, 0.3, 0.2, 0.8, 0.3, 0.7],  # x1=0.8, ¬x1=0.2, x2=0.7, ¬x2=0.3, x3=0.2, ¬x3=0.8, x4=0.3, ¬x4=0.7 -> ((0.8*0.7)+0.2) * ((0.8*0.7)+0.3) = 0.76 * 0.86 = 0.6536
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # x1=0.5, ¬x1=0.5, x2=0.5, ¬x2=0.5, x3=0.5, ¬x3=0.5, x4=0.5, ¬x4=0.5 -> ((0.5*0.5)+0.5) * ((0.5*0.5)+0.5) = 0.75 * 0.75 = 0.5625
            [0.9, 0.1, 0.1, 0.9, 0.8, 0.2, 0.2, 0.8],  # x1=0.9, ¬x1=0.1, x2=0.1, ¬x2=0.9, x3=0.8, ¬x3=0.2, x4=0.2, ¬x4=0.8 -> ((0.9*0.1)+0.8) * ((0.9*0.1)+0.2) = 0.89 * 0.29 = 0.2581
        ]),
        torch.tensor([[0.6536], [0.5625], [0.2581]])
    ),
    (
        "Marginalized AND: x₁ ∧ x₂ with x₂ marginalized",
        os.path.join(circuit_dir, "simple_and.sdd"),
        os.path.join(circuit_dir, "simple_and.json"),
        torch.tensor([
            [0.2, 0.8, 1.0, 1.0],  # x1=0.2, ¬x1=0.8, x2=1.0, ¬x2=1.0 (marginalized) -> 0.2 * 1.0 = 0.2
            [0.5, 0.5, 1.0, 1.0],  # x1=0.5, ¬x1=0.5, x2=1.0, ¬x2=1.0 (marginalized) -> 0.5 * 1.0 = 0.5
            [1.0, 0.0, 1.0, 1.0],  # x1=1.0, ¬x1=0.0, x2=1.0, ¬x2=1.0 (marginalized) -> 1.0 * 1.0 = 1.0
        ]),
        torch.tensor([[0.2], [0.5], [1.0]])
    ),
    (
        "Marginalized OR: x₁ V x₂ with x₁ marginalized",
        os.path.join(circuit_dir, "simple_or.sdd"),
        os.path.join(circuit_dir, "simple_or.json"),
        torch.tensor([
            [1.0, 1.0, 0.8, 0.2],  # x1=1.0, ¬x1=1.0 (marginalized), x2=0.8, ¬x2=0.2 -> 1.0 + 0.8 = 1.8
            [1.0, 1.0, 0.5, 0.5],  # x1=1.0, ¬x1=1.0 (marginalized), x2=0.5, ¬x2=0.5 -> 1.0 + 0.5 = 1.5
            [1.0, 1.0, 0.0, 1.0],  # x1=1.0, ¬x1=1.0 (marginalized), x2=0.0, ¬x2=1.0 -> 1.0 + 0.0 = 1.0
        ]),
        torch.tensor([[1.8], [1.5], [1.0]])
    )
]

test_cases = standard_test_cases + marginalized_test_cases

@pytest.mark.parametrize("implementation", implementations, ids=[i['name'] for i in implementations])
@pytest.mark.parametrize("description, sdd_file, json_file, input_data, expected_output", test_cases)
def test_forward_pass_probabilistic(implementation, description, sdd_file, json_file, input_data, expected_output):
    """
    Tests the forward pass output against analytical results using probabilistic inputs.
    """
    full_description = f"{description} ({implementation['name']} Implementation)"
    print(f"Testing probabilistic forward pass: {full_description}")

    implementation_class = implementation["implementation_class"]
    neural_network = implementation_class(sdd_file, json_file)

    computed_output = neural_network.forward(input_data)

    torch.testing.assert_close(
        computed_output,
        expected_output,
        msg=f"Output mismatch for probabilistic circuit: {full_description}\n Expected: {expected_output}\n Actual: {computed_output}\n"
    )
