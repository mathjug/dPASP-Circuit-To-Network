"""
This file contains unit tests for the QueryExecutor class.

The primary goal of these tests is to verify the internal logic of the
QueryExecutor in isolation from its dependencies, specifically the actual
neural network computation.

To achieve this, the tests use a mock neural network implementation that
substitutes the real neural network classes (IterativeNN or RecursiveNN).
Instead of performing real neural network computation, the mocked neural
network returns predictable, predefined values.

This allows the tests to confirm that the QueryExecutor correctly:
1.  Orchestrates the two required calls to the neural network forward method (one for the
    numerator and one for the denominator).
2.  Prepares the correct input tensors for each call.
3.  Performs the final division correctly based on the mocked return values.
4.  Handles input validation and edge cases, such as encountering a zero-probability
    denominator.
"""

import pytest
import torch
import json
from array import array

from src.queries.query_executor import QueryExecutor

# --- Mock Dependencies ---

class MockNeuralNetwork:
    """
    A mock neural network class to substitute the real neural network implementations.
    It allows us to control the output of the forward method for predictable testing.
    """
    def __init__(self, sdd_file, json_file):
        self.call_count = 0
        self.call_args_list = []
        self.return_values = [0.0, 0.0]  # [numerator, denominator]
        self.num_variables = 3  # Default value, will be overridden in tests

    def forward(self, x):
        """
        Mock method that returns predefined values based on call count.
        """
        self.call_args_list.append({'input': x})
        value_to_return = self.return_values[self.call_count]
        self.call_count += 1
        return torch.tensor(value_to_return, dtype=torch.float32)

@pytest.fixture
def create_test_files(tmp_path):
    """
    A pytest fixture to create temporary sdd and json files for testing.
    The content of the SDD file doesn't matter since we are mocking the neural network,
    but the file needs to exist and be parsable.
    """
    def _create_files(json_data):
        sdd_file = tmp_path / "test.sdd"
        json_file = tmp_path / "test.json"
        sdd_content = "sdd 1\nT 0"
        sdd_file.write_text(sdd_content)
        with open(json_file, 'w') as f:
            json.dump(json_data, f)
        return str(sdd_file), str(json_file)
    return _create_files

# --- Test Cases for Successful Queries ---
# Each tuple: (description, json_data, query_variables, evidence_variables, numerator_result, denominator_result, 
#              expected_numerator_input, expected_denominator_input)

successful_query_test_cases = [
    (
        "Standard query without evidence",
        {
            "atom_mapping": {"1": "a", "2": "b", "3": "c"},
            "prob": {
                "pfacts": [["1", "0.5"], ["2", "0.3"], ["3", "0.7"]],
                "pvars": []
            }
        },
        [2, 3], [], 0.4, 0.8,
        [1.0, 1.0, 1.0, 0.0, 1.0, 0.0],  # x2=1, x3=1
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]   # all variables marginalized
    ),
    (
        "Query with evidence variables",
        {
            "atom_mapping": {"1": "a", "2": "b", "3": "c", "4": "d"},
            "prob": {
                "pfacts": [["1", "0.6"], ["2", "0.4"], ["3", "0.8"], ["4", "0.2"]],
                "pvars": []
            }
        },
        [2], [1, 4], 0.3, 0.5,
        [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0],  # x1=1, x2=1, x4=1
        [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]   # x1=1, x4=1
    ),
    (
        "Query with non-probabilistic evidence variable",
        {
            "atom_mapping": {"1": "a", "2": "b", "3": "c", "4": "d"},
            "prob": {
                "pfacts": [["1", "0.6"], ["2", "0.4"], ["4", "0.2"]],
                "pvars": []
            }
        },
        [2, 4], [1, 3], 0.3, 0.5,
        [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],  # x1=1, x2=1, x3=1, x4=1
        [1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]   # x1=1, x3=1
    ),
    (
        "Query with empty evidence variables list",
        {
            "atom_mapping": {"1": "a", "2": "b", "3": "c"},
            "prob": {
                "pfacts": [["1", "0.5"], ["2", "0.3"], ["3", "0.7"]],
                "pvars": []
            }
        },
        [2], [], 0.4, 0.8,
        [1.0, 1.0, 1.0, 0.0, 1.0, 1.0],  # x2=1
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]   # all variables marginalized
    ),
]

@pytest.mark.parametrize("description, json_data, query_variables, evidence_variables, numerator_result, denominator_result, expected_numerator_input, expected_denominator_input", successful_query_test_cases)
def test_successful_queries(monkeypatch, create_test_files, description, json_data, query_variables, evidence_variables, numerator_result, denominator_result, expected_numerator_input, expected_denominator_input):
    """
    Tests successful query executions with various configurations.
    """
    sdd_file, json_file = create_test_files(json_data)
    mock_nn = MockNeuralNetwork(sdd_file, json_file)
    mock_nn.return_values = [numerator_result, denominator_result]    
    mock_nn.num_variables = len(json_data["atom_mapping"])

    def mock_nn_implementation(sdd_file, json_file):
        return mock_nn

    executor = QueryExecutor(sdd_file, json_file, mock_nn_implementation)
    result = executor.execute_query(query_variables=query_variables, evidence_variables=evidence_variables)

    assert result == pytest.approx(numerator_result / denominator_result)
    assert mock_nn.call_count == 2
    forward_calls = [call for call in mock_nn.call_args_list]
    assert len(forward_calls) == 2
    numerator_input = forward_calls[0]['input']
    denominator_input = forward_calls[1]['input']
    assert list(numerator_input) == pytest.approx(list(expected_numerator_input))
    assert list(denominator_input) == pytest.approx(list(expected_denominator_input))

# --- Test Cases for Error Handling ---
# Each tuple: (description, json_data, query_variables, evidence_variables, expected_error_type, expected_error_message)

error_test_cases = [
    (
        "Query variable out of bounds (too high)",
        {"atom_mapping": {"1": "a", "2": "b"}, "prob": {"pfacts": [["1", "0.5"], ["2", "0.3"]], "pvars": []}},
        [3], [], ValueError, "Variable \\(X_3\\) out of bounds for tensor of size 2\\."
    ),
    (
        "Query variable out of bounds (zero)",
        {"atom_mapping": {"1": "a", "2": "b"}, "prob": {"pfacts": [["1", "0.5"], ["2", "0.3"]], "pvars": []}},
        [0], [], ValueError, "Variable \\(X_0\\) out of bounds for tensor of size 2\\."
    ),
    (
        "Evidence variable out of bounds (too high)",
        {"atom_mapping": {"1": "a", "2": "b"}, "prob": {"pfacts": [["1", "0.5"], ["2", "0.3"]], "pvars": []}},
        [1], [3], ValueError, "Variable \\(X_3\\) out of bounds for tensor of size 2\\."
    ),
    (
        "Evidence variable out of bounds (zero)",
        {"atom_mapping": {"1": "a", "2": "b"}, "prob": {"pfacts": [["1", "0.5"], ["2", "0.3"]], "pvars": []}},
        [1], [0], ValueError, "Variable \\(X_0\\) out of bounds for tensor of size 2\\."
    ),
]

@pytest.mark.parametrize("description, json_data, query_variables, evidence_variables, expected_error_type, expected_error_message", error_test_cases)
def test_error_handling(monkeypatch, create_test_files, description, json_data, query_variables, evidence_variables, expected_error_type, expected_error_message):
    """
    Tests error handling for various invalid inputs.
    """
    sdd_file, json_file = create_test_files(json_data)    
    mock_nn = MockNeuralNetwork(sdd_file, json_file)    
    mock_nn.num_variables = len(json_data["atom_mapping"])

    def mock_nn_implementation(sdd_file, json_file):
        return mock_nn
    
    executor = QueryExecutor(sdd_file, json_file, mock_nn_implementation)
    with pytest.raises(expected_error_type, match=expected_error_message):
        executor.execute_query(query_variables=query_variables, evidence_variables=evidence_variables)

# --- Test Cases for Zero Denominator ---
# Each tuple: (description, json_data, query_variables, evidence_variables, mock_logic)

zero_denominator_test_cases = [
    (
        "Zero denominator without evidence",
        {"atom_mapping": {"1": "a", "2": "b"}, "prob": {"pfacts": [["1", "0.5"], ["2", "0.3"]], "pvars": []}},
        [1], [],
        lambda input_tensor: 0.0 if len(input_tensor) == 4 and input_tensor[1] == 1.0 else 0.1
    ),
    (
        "Zero denominator with evidence",
        {"atom_mapping": {"1": "a", "2": "b", "3": "c"}, "prob": {"pfacts": [["1", "0.5"], ["2", "0.3"], ["3", "0.7"]], "pvars": []}},
        [2], [1],
        lambda input_tensor: 0.0 if len(input_tensor) == 6 and input_tensor[3] == 1.0 else 0.1
    ),
]

@pytest.mark.parametrize("description, json_data, query_variables, evidence_variables, mock_logic", zero_denominator_test_cases)
def test_zero_denominator_handling(monkeypatch, create_test_files, description, json_data, query_variables, evidence_variables, mock_logic):
    """
    Tests that queries with zero denominator return 0.0 without division-by-zero errors.
    """
    sdd_file, json_file = create_test_files(json_data)    
    mock_nn = MockNeuralNetwork(sdd_file, json_file)
    mock_nn.num_variables = len(json_data["atom_mapping"])
    
    def mock_forward_logic(self, x):
        self.call_args_list.append({'input': x})
        value_to_return = mock_logic(x)
        self.call_count += 1
        return torch.tensor(value_to_return, dtype=torch.float32)
    
    mock_nn.forward = mock_forward_logic.__get__(mock_nn, MockNeuralNetwork)
        
    def mock_nn_implementation(sdd_file, json_file):
        return mock_nn
    
    executor = QueryExecutor(sdd_file, json_file, mock_nn_implementation)
    result = executor.execute_query(query_variables=query_variables, evidence_variables=evidence_variables)
    assert result == 0.0
