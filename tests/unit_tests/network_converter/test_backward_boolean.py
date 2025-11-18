"""
This test file is dedicated to verifying the correctness of the backward pass
for both RecursiveNN and IterativeNN implementations, using logic variables.
It checks if the computed gradients of various circuits matches the analytically
calculated expected gradients.
"""

import torch
import pytest
import os

from tests.utils.utils import calculate_individual_gradients, implementations

test_dir = os.path.dirname(os.path.abspath(__file__))
circuit_dir = os.path.join(test_dir, "test_circuits")

# --- Test Cases for Boolean Backward Pass ---
# Each tuple: (description, sdd_file, json_file, input_data, expected_gradients)

test_cases = [
    (
        "Simple AND: x₁ ∧ x₂",
        os.path.join(circuit_dir, "simple_and.sdd"),
        os.path.join(circuit_dir, "simple_and.json"),
        torch.tensor([
            [1., 0., 1., 0.],  # x1=1, ¬x1=0, x2=1, ¬x2=0 -> 1 AND 1 = 1
            [1., 0., 0., 1.],  # x1=1, ¬x1=0, x2=0, ¬x2=1 -> 1 AND 0 = 0
            [0., 1., 1., 0.],  # x1=0, ¬x1=1, x2=1, ¬x2=0 -> 0 AND 1 = 0
            [0., 1., 0., 1.]   # x1=0, ¬x1=1, x2=0, ¬x2=1 -> 0 AND 0 = 0
        ]),
        torch.tensor([
            [1., 0., 1., 0.],  # ∂f/∂x1=1, ∂f/∂¬x1=0, ∂f/∂x2=1, ∂f/∂¬x2=0
            [0., 0., 1., 0.],  # ∂f/∂x1=0, ∂f/∂¬x1=0, ∂f/∂x2=1, ∂f/∂¬x2=0
            [1., 0., 0., 0.],  # ∂f/∂x1=1, ∂f/∂¬x1=0, ∂f/∂x2=0, ∂f/∂¬x2=0
            [0., 0., 0., 0.]   # ∂f/∂x1=0, ∂f/∂¬x1=0, ∂f/∂x2=0, ∂f/∂¬x2=0
        ])
    ),
    (
        "Simple OR: x₁ V x₂",
        os.path.join(circuit_dir, "simple_or.sdd"),
        os.path.join(circuit_dir, "simple_or.json"),
        torch.tensor([
            [1., 0., 1., 0.],  # x1=1, ¬x1=0, x2=1, ¬x2=0 -> 1 OR 1 = 2
            [1., 0., 0., 1.],  # x1=1, ¬x1=0, x2=0, ¬x2=1 -> 1 OR 0 = 1
            [0., 1., 1., 0.],  # x1=0, ¬x1=1, x2=1, ¬x2=0 -> 0 OR 1 = 1
            [0., 1., 0., 1.]   # x1=0, ¬x1=1, x2=0, ¬x2=1 -> 0 OR 0 = 0
        ]),
        torch.tensor([
            [1., 0., 1., 0.],  # ∂f/∂x1=1, ∂f/∂¬x1=0, ∂f/∂x2=1, ∂f/∂¬x2=0
            [1., 0., 1., 0.],  # ∂f/∂x1=1, ∂f/∂¬x1=0, ∂f/∂x2=1, ∂f/∂¬x2=0
            [1., 0., 1., 0.],  # ∂f/∂x1=1, ∂f/∂¬x1=0, ∂f/∂x2=1, ∂f/∂¬x2=0
            [1., 0., 1., 0.]   # ∂f/∂x1=1, ∂f/∂¬x1=0, ∂f/∂x2=1, ∂f/∂¬x2=0
        ])
    ),
    (
        "Memoization Test 1: (x₁ V x₂) ∧ (x₁ V x₂) - shared subexpressions",
        os.path.join(circuit_dir, "memoization_test.sdd"),
        os.path.join(circuit_dir, "memoization_test.json"),
        torch.tensor([
            [1., 0., 1., 0.],  # (1 OR 1) AND (1 OR 1) = 2 AND 2 = 4
            [1., 0., 0., 1.],  # (1 OR 0) AND (1 OR 0) = 1 AND 1 = 1
            [0., 1., 1., 0.],  # (0 OR 1) AND (0 OR 1) = 1 AND 1 = 1
            [0., 1., 0., 1.]   # (0 OR 0) AND (0 OR 0) = 0 AND 0 = 0
        ]),
        torch.tensor([
            [4., 0., 4., 0.],  # ∂f/∂x1=4, ∂f/∂¬x1=0, ∂f/∂x2=4, ∂f/∂¬x2=0
            [2., 0., 2., 0.],  # ∂f/∂x1=2, ∂f/∂¬x1=0, ∂f/∂x2=2, ∂f/∂¬x2=0
            [2., 0., 2., 0.],  # ∂f/∂x1=2, ∂f/∂¬x1=0, ∂f/∂x2=2, ∂f/∂¬x2=0
            [0., 0., 0., 0.]   # ∂f/∂x1=0, ∂f/∂¬x1=0, ∂f/∂x2=0, ∂f/∂¬x2=0
        ])
    ),
    (
        "Memoization Test 2: ((x₁ ∧ x₂) V x₃) ∧ ((x₁ ∧ x₂) V x₄) - shared subexpressions",
        os.path.join(circuit_dir, "memoization_test2.sdd"),
        os.path.join(circuit_dir, "memoization_test2.json"),
        torch.tensor([
            [1., 0., 1., 0., 0., 1., 0., 1.],  # ((1 AND 1) OR 0) AND ((1 AND 1) OR 0) = 1 AND 1 = 1
            [1., 0., 0., 1., 1., 0., 1., 0.],  # ((1 AND 0) OR 1) AND ((1 AND 0) OR 1) = 1 AND 1 = 1
            [0., 1., 1., 0., 1., 0., 1., 0.],  # ((0 AND 1) OR 1) AND ((0 AND 1) OR 1) = 1 AND 1 = 1
        ]),
        torch.tensor([
            [2., 0., 2., 0., 1., 0., 1., 0.],  # ∂f/∂x1=2, ∂f/∂¬x1=0, ∂f/∂x2=2, ∂f/∂¬x2=0, ∂f/∂x3=1, ∂f/∂¬x3=0, ∂f/∂x4=1, ∂f/∂¬x4=0
            [0., 0., 2., 0., 1., 0., 1., 0.],  # ∂f/∂x1=0, ∂f/∂¬x1=0, ∂f/∂x2=2, ∂f/∂¬x2=0, ∂f/∂x3=1, ∂f/∂¬x3=0, ∂f/∂x4=1, ∂f/∂¬x4=0
            [2., 0., 0., 0., 1., 0., 1., 0.],  # ∂f/∂x1=2, ∂f/∂¬x1=0, ∂f/∂x2=0, ∂f/∂¬x2=0, ∂f/∂x3=1, ∂f/∂¬x3=0, ∂f/∂x4=1, ∂f/∂¬x4=0
        ])
    ),
]

@pytest.mark.parametrize("implementation", implementations, ids=[i['name'] for i in implementations])
@pytest.mark.parametrize("description, sdd_file, json_file, input_data, expected_gradients", test_cases)
def test_circuit_derivatives_boolean(implementation, description, sdd_file, json_file, input_data, expected_gradients):
    """
    Tests that computed gradients match analytical derivatives for various circuits and implementations
    using logic variables.
    """
    full_description = f"{description} ({implementation['name']} Implementation)"
    print(f"Testing circuit: {full_description}")

    implementation_class = implementation["implementation_class"]
    computed_gradients = calculate_individual_gradients(sdd_file, json_file, input_data.clone(), implementation_class)

    torch.testing.assert_close(
        computed_gradients,
        expected_gradients,
        msg=f"Gradient mismatch for circuit: {full_description}\n Expected: {expected_gradients}\n Actual: {computed_gradients}\n"
    )
