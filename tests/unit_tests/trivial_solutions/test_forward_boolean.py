"""
This test file is dedicated to verifying the correctness of the forward pass
for both RecursiveNN and IterativeNN implementations using BOOLEAN inputs (0.0 or 1.0).
It checks if the computed output of various complex circuits matches the
analytically calculated expected output.

Note that the OR operation is implemented as summation, so `1 V 1` will result
in `2.0`, not `1.0`.
"""

import torch
import pytest
import os

from tests.utils.utils import implementations

# --- Test Cases for Standard Forward Pass ---
# Each tuple: (description, sdd_file, json_file, input_data, expected_output)

# Get the directory where this test file is located
test_dir = os.path.dirname(os.path.abspath(__file__))
circuit_dir = os.path.join(test_dir, "test_circuits")

standard_test_cases = [
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
        torch.tensor([[1.], [0.], [0.], [0.]])
    ),
    (
        "Simple OR: x₁ V x₂",
        os.path.join(circuit_dir, "simple_or.sdd"),
        os.path.join(circuit_dir, "simple_or.json"),
        torch.tensor([
            [1., 0., 1., 0.],  # x1=1, ¬x1=0, x2=1, ¬x2=0 -> 1 OR 1 = 2 (1+1)
            [1., 0., 0., 1.],  # x1=1, ¬x1=0, x2=0, ¬x2=1 -> 1 OR 0 = 1 (1+0)
            [0., 1., 1., 0.],  # x1=0, ¬x1=1, x2=1, ¬x2=0 -> 0 OR 1 = 1 (0+1)
            [0., 1., 0., 1.]   # x1=0, ¬x1=1, x2=0, ¬x2=1 -> 0 OR 0 = 0 (0+0)
        ]),
        torch.tensor([[2.], [1.], [1.], [0.]])
    ),
]

# --- Test Cases for Memoization Cache Verification ---
# Each tuple: (description, sdd_file, json_file, input_data, expected_output)

memoization_test_cases = [
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
        torch.tensor([[4.], [1.], [1.], [0.]])
    ),
    (
        "Memoization Test 2: ((x₁ ∧ x₂) V x₃) ∧ ((x₁ ∧ x₂) V x₄) - shared subexpressions",
        os.path.join(circuit_dir, "memoization_test2.sdd"),
        os.path.join(circuit_dir, "memoization_test2.json"),
        torch.tensor([
            [1., 0., 1., 0., 0., 1., 0., 1.],  # ((1 AND 1) OR 0) AND ((1 AND 1) OR 0) = (1 OR 0) AND (1 OR 0) = 1 AND 1 = 1
            [1., 0., 0., 1., 1., 0., 1., 0.],  # ((1 AND 0) OR 1) AND ((1 AND 0) OR 1) = (0 OR 1) AND (0 OR 1) = 1 AND 1 = 1
            [0., 1., 1., 0., 1., 0., 1., 0.],  # ((0 AND 1) OR 1) AND ((0 AND 1) OR 1) = (0 OR 1) AND (0 OR 1) = 1 AND 1 = 1
        ]),
        torch.tensor([[1.], [1.], [1.]])
    ),
]

test_cases = standard_test_cases + memoization_test_cases

@pytest.mark.parametrize("implementation", implementations, ids=[i['name'] for i in implementations])
@pytest.mark.parametrize("description, sdd_file, json_file, input_data, expected_output", test_cases)
def test_forward_pass_boolean(implementation, description, sdd_file, json_file, input_data, expected_output):
    """
    Tests the forward pass output against analytical results using boolean inputs.
    """
    full_description = f"{description} ({implementation['name']} Implementation)"
    print(f"Testing boolean forward pass: {full_description}")

    implementation_class = implementation["implementation_class"]
    neural_network = implementation_class(sdd_file, json_file, make_smooth=False)

    computed_output = neural_network.forward(input_data)

    torch.testing.assert_close(
        computed_output,
        expected_output,
        msg=f"Output mismatch for boolean circuit: {full_description}\n Expected: {expected_output}\n Actual: {computed_output}\n"
    )
