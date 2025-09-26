"""
This test file is dedicated to verifying the correctness of the backward pass
for both RecursiveNN and IterativeNN implementations, using probabilistic variables.
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
        "Simple AND with Negation: x₁ ∧ ¬x₂",
        os.path.join(circuit_dir, "simple_and_not.sdd"),
        os.path.join(circuit_dir, "simple_and_not.json"),
        torch.tensor([
            [1., 0., 1., 0.],  # x1=1, ¬x1=0, x2=1, ¬x2=0 -> 1 AND ¬1 = 0
            [1., 0., 0., 1.],  # x1=1, ¬x1=0, x2=0, ¬x2=1 -> 1 AND ¬0 = 1
            [0., 1., 1., 0.],  # x1=0, ¬x1=1, x2=1, ¬x2=0 -> 0 AND ¬1 = 0
            [0., 1., 0., 1.]   # x1=0, ¬x1=1, x2=0, ¬x2=1 -> 0 AND ¬0 = 0
        ]),
        torch.tensor([
            [0.0000, 0.0000, 0.0000, 0.1200],  # ∂f/∂x1=0.00, ∂f/∂¬x1=0, ∂f/∂x2=0, ∂f/∂¬x2=0.12
            [0.1200, 0.0000, 0.0000, 0.1200],  # ∂f/∂x1=0.12, ∂f/∂¬x1=0, ∂f/∂x2=0, ∂f/∂¬x2=0.12
            [0.0000, 0.0000, 0.0000, 0.0000],  # ∂f/∂x1=0.00, ∂f/∂¬x1=0, ∂f/∂x2=0, ∂f/∂¬x2=0
            [0.1200, 0.0000, 0.0000, 0.0000]   # ∂f/∂x1=0.12, ∂f/∂¬x1=0, ∂f/∂x2=0, ∂f/∂¬x2=0
        ])
    ),
    (
        "Complex Shared AND: (x₁ ∧ x₂) V (x₁ ∧ x₂) V (x₃ ∧ x₄)",
        os.path.join(circuit_dir, "complex_shared_and.sdd"),
        os.path.join(circuit_dir, "complex_shared_and.json"),
        torch.tensor([
            [1., 0., 1., 0., 1., 0., 1., 0.],  # (1 AND 1) OR (1 AND 1) OR (1 AND 1) = 1 OR 1 OR 1 = 3
            [1., 0., 1., 0., 0., 1., 0., 1.],  # (1 AND 1) OR (1 AND 1) OR (0 AND 0) = 1 OR 1 OR 0 = 2
            [0., 1., 0., 1., 1., 0., 1., 0.],  # (0 AND 0) OR (0 AND 0) OR (1 AND 1) = 0 OR 0 OR 1 = 1
        ]),
        torch.tensor([
            [0.5000, 0.0000, 0.5000, 0.0000, 0.2500, 0.0000, 0.2500, 0.0000],  # ∂f/∂x1=0.5, ∂f/∂¬x1=0, ∂f/∂x2=0.5, ∂f/∂¬x2=0, ∂f/∂x3=0.25, ∂f/∂¬x3=0, ∂f/∂x4=0.25, ∂f/∂¬x4=0
            [0.5000, 0.0000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],  # ∂f/∂x1=0.5, ∂f/∂¬x1=0, ∂f/∂x2=0.5, ∂f/∂¬x2=0, ∂f/∂x3=0, ∂f/∂¬x3=0, ∂f/∂x4=0, ∂f/∂¬x4=0
            [0.0000, 0.0000, 0.0000, 0.0000, 0.2500, 0.0000, 0.2500, 0.0000],  # ∂f/∂x1=0, ∂f/∂¬x1=0, ∂f/∂x2=0, ∂f/∂¬x2=0, ∂f/∂x3=0.25, ∂f/∂¬x3=0, ∂f/∂x4=0.25, ∂f/∂¬x4=0
        ])
    ),
    (
        "Deep Nested Shared: ((x₁ V x₂) ∧ x₃) V ((x₁ V x₂) ∧ x₄) V ((x₁ V x₂) ∧ x₅)",
        os.path.join(circuit_dir, "deep_nested_shared.sdd"),
        os.path.join(circuit_dir, "deep_nested_shared.json"),
        torch.tensor([
            [1., 0., 1., 0., 1., 0., 1., 0., 1., 0.],  # ((1 OR 1) AND 1) OR ((1 OR 1) AND 1) OR ((1 OR 1) AND 1) = 2 OR 2 OR 2 = 6
            [1., 0., 1., 0., 0., 1., 1., 0., 0., 1.],  # ((1 OR 1) AND 0) OR ((1 OR 1) AND 1) OR ((1 OR 1) AND 0) = 0 OR 2 OR 0 = 2
            [0., 1., 0., 1., 1., 0., 1., 0., 1., 0.],  # ((0 OR 0) AND 1) OR ((0 OR 0) AND 1) OR ((0 OR 0) AND 1) = 0 OR 0 OR 0 = 0
        ]),
        torch.tensor([
            [0.7500, 0.0000, 0.7500, 0.0000, 0.5000, 0.0000, 0.5000, 0.0000, 0.5000, 0.0000],  # ∂f/∂x1=0.75, ∂f/∂¬x1=0, ∂f/∂x2=0.75, ∂f/∂¬x2=0, ∂f/∂x3=0.5, ∂f/∂¬x3=0, ∂f/∂x4=0.5, ∂f/∂¬x4=0, ∂f/∂x5=0.5, ∂f/∂¬x5=0
            [0.2500, 0.0000, 0.2500, 0.0000, 0.5000, 0.0000, 0.5000, 0.0000, 0.5000, 0.0000],  # ∂f/∂x1=0.25, ∂f/∂¬x1=0, ∂f/∂x2=0.25, ∂f/∂¬x2=0, ∂f/∂x3=0.5, ∂f/∂¬x3=0, ∂f/∂x4=0.5, ∂f/∂¬x4=0, ∂f/∂x5=0.5, ∂f/∂¬x5=0
            [0.7500, 0.0000, 0.7500, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],  # ∂f/∂x1=0.75, ∂f/∂¬x1=0, ∂f/∂x2=0.75, ∂f/∂¬x2=0, ∂f/∂x3=0, ∂f/∂¬x3=0, ∂f/∂x4=0, ∂f/∂¬x4=0, ∂f/∂x5=0, ∂f/∂¬x5=0
        ])
    )
]

@pytest.mark.parametrize("implementation", implementations, ids=[i['name'] for i in implementations])
@pytest.mark.parametrize("description, sdd_file, json_file, input_data, expected_gradients", test_cases)
def test_circuit_derivatives_probabilistic(implementation, description, sdd_file, json_file, input_data, expected_gradients):
    """
    Tests that computed gradients match analytical derivatives for various circuits and implementations
    using probabilistic variables.
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
