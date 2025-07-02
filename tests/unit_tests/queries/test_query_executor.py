"""
This file contains unit tests for the QueryExecutor class.

The primary goal of these tests is to verify the internal logic of the
QueryExecutor in isolation from its dependencies, specifically the actual
neural network computation.

To achieve this, the tests use the 'monkeypatch' fixture to mock the `forward`
method of the neural network. Instead of performing a real, complex neural
network pass, the mocked method returns predictable, predefined values.

This allows the tests to confirm that the QueryExecutor correctly:
1.  Orchestrates the two required calls to the `forward` method (one for the
    numerator and one for the denominator).
2.  Prepares the correct `input_tensor` and `marginalized_variables` for each call.
3.  Performs the final division correctly based on the mocked return values.
4.  Handles input validation and edge cases, such as querying an evidence
    variable or encountering a zero-probability denominator.
"""

import pytest
import torch
import json

from src.queries.query_executor import QueryExecutor

# --- Mock Dependencies ---

class MockExecutorNN:
    """
    A mock neural network class to substitute RecursiveNN or IterativeNN.
    It allows us to control the output of the `forward` method for predictable testing.
    """
    def __init__(self, root_node):
        pass

    def forward(self, input_tensor, marginalized_variables=None):
        """
        This default forward method will be replaced by the tests using monkeypatch.
        """
        return torch.tensor([0.0])

@pytest.fixture
def create_test_files(tmp_path):
    """
    A pytest fixture to create temporary sdd and json files for testing.
    This ensures tests are isolated and don't depend on external files.

    The content of the SDD file doesn't matter since we are mocking the NN,
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
# Each tuple: (description, json_data, query_variable, evidence_variables, numerator_result, denominator_result, 
#              expected_numerator_input, expected_numerator_marginalized, expected_denominator_input, expected_denominator_marginalized)

successful_query_test_cases = [
    (
        "Standard query without evidence",
        {
            "atom_mapping": {"1": "a", "2": "b", "3": "c"},
            "prob": {"pfacts": [["1", "0.8"]]}
        },
        2, [], 0.4, 0.8,
        torch.tensor([0.8, 1.0, 1.0]),
        torch.tensor([0, 0, 1], dtype=torch.int64),
        torch.tensor([0.8, 1.0, 1.0]),
        torch.tensor([0, 1, 1], dtype=torch.int64)
    ),
    (
        "Query with evidence variables",
        {
            "atom_mapping": {"1": "a", "2": "b", "3": "c", "4": "d"},
            "prob": {"pfacts": [["1", "0.8"], ["3", "0.6"]]}
        },
        2, [1, 4], 0.3, 0.5,
        torch.tensor([1.0, 1.0, 0.6, 1.0]),
        torch.tensor([0, 0, 0, 0], dtype=torch.int64),
        torch.tensor([1.0, 1.0, 0.6, 1.0]),
        torch.tensor([0, 1, 0, 0], dtype=torch.int64)
    ),
    (
        "Query with non-probabilistic evidence variable",
        {
            "atom_mapping": {"1": "a", "2": "b", "3": "c", "4": "d"},
            "prob": {"pfacts": [["1", "0.8"], ["3", "0.6"]]}
        },
        2, [1, 3], 0.3, 0.5,
        torch.tensor([1.0, 1.0, 1.0, 1.0]),
        torch.tensor([0, 0, 0, 1], dtype=torch.int64),
        torch.tensor([1.0, 1.0, 1.0, 1.0]),
        torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    ),
    (
        "Query with empty evidence variables list",
        {
            "atom_mapping": {"1": "a", "2": "b", "3": "c"},
            "prob": {"pfacts": [["1", "0.8"]]}
        },
        2, [], 0.4, 0.8,
        torch.tensor([0.8, 1.0, 1.0]),
        torch.tensor([0, 0, 1], dtype=torch.int64),
        torch.tensor([0.8, 1.0, 1.0]),
        torch.tensor([0, 1, 1], dtype=torch.int64)
    ),
]

# --- Test Cases for Error Handling ---
# Each tuple: (description, json_data, query_variable, evidence_variables, expected_error_type, expected_error_message)

error_test_cases = [
    (
        "Query on evidence variable raises error",
        {
            "atom_mapping": {"1": "a", "2": "b"},
            "prob": {"pfacts": [["1", "0.9"]]}
        },
        1, [], ValueError, "Probability is already known for X_1"
    ),
    (
        "Query variable out of bounds (too high)",
        {"atom_mapping": {"1": "a", "2": "b"}},
        3, [], ValueError, "Query variable .* out of bounds"
    ),
    (
        "Query variable out of bounds (zero)",
        {"atom_mapping": {"1": "a", "2": "b"}},
        0, [], ValueError, "Query variable .* out of bounds"
    ),
    (
        "Evidence variable out of bounds (too high)",
        {"atom_mapping": {"1": "a", "2": "b"}},
        1, [3], ValueError, "Evidence variable .* out of bounds"
    ),
    (
        "Evidence variable out of bounds (zero)",
        {"atom_mapping": {"1": "a", "2": "b"}},
        1, [0], ValueError, "Evidence variable .* out of bounds"
    ),
]

# --- Test Cases for Zero Denominator ---
# Each tuple: (description, json_data, query_variable, evidence_variables, mock_logic)

zero_denominator_test_cases = [
    (
        "Zero denominator without evidence",
        {"atom_mapping": {"1": "a", "2": "b"}},
        1, [],
        lambda marginalized_variables: torch.tensor([0.0]) if torch.equal(marginalized_variables, torch.tensor([1, 1], dtype=torch.int64)) else torch.tensor([0.1])
    ),
    (
        "Zero denominator with evidence",
        {"atom_mapping": {"1": "a", "2": "b", "3": "c"}},
        2, [1],
        lambda marginalized_variables: torch.tensor([0.0]) if torch.equal(marginalized_variables, torch.tensor([0, 1, 1], dtype=torch.int64)) else torch.tensor([0.1])
    ),
]

@pytest.mark.parametrize("description, json_data, query_variable, evidence_variables, numerator_result, denominator_result, expected_numerator_input, expected_numerator_marginalized, expected_denominator_input, expected_denominator_marginalized", successful_query_test_cases)
def test_successful_queries(monkeypatch, create_test_files, description, json_data, query_variable, evidence_variables, numerator_result, denominator_result, expected_numerator_input, expected_numerator_marginalized, expected_denominator_input, expected_denominator_marginalized):
    """
    Tests successful query executions with various configurations.
    """
    class MockForwarder:
        """
        Helper class to replicate the behavior of returning different values on each call.
        """
        def __init__(self, return_values):
            self.return_values = return_values
            self.call_count = 0
            self.call_args_list = []

        def __call__(self, input_tensor, marginalized_variables=None):
            self.call_args_list.append(
                {'input': input_tensor, 'marginalized': marginalized_variables}
            )
            value_to_return = self.return_values[self.call_count]
            self.call_count += 1
            return value_to_return

    sdd_file, json_file = create_test_files(json_data)
    
    mock_forwarder = MockForwarder(return_values=[
        torch.tensor([numerator_result]),   # Mocked result for the numerator call
        torch.tensor([denominator_result])  # Mocked result for the denominator call
    ])
    
    # Replace 'forward' method with the mock forwarder
    monkeypatch.setattr(MockExecutorNN, 'forward', mock_forwarder)

    executor = QueryExecutor(MockExecutorNN, sdd_file, json_file)
    result = executor.execute_query(query_variable=query_variable, evidence_variables=evidence_variables)

    assert result == pytest.approx(numerator_result / denominator_result)
    assert mock_forwarder.call_count == 2
    
    numerator_call_args = mock_forwarder.call_args_list[0]
    assert torch.equal(numerator_call_args['input'], expected_numerator_input)
    assert torch.equal(numerator_call_args['marginalized'], expected_numerator_marginalized)

    denominator_call_args = mock_forwarder.call_args_list[1]
    assert torch.equal(denominator_call_args['input'], expected_denominator_input)
    assert torch.equal(denominator_call_args['marginalized'], expected_denominator_marginalized)

@pytest.mark.parametrize("description, json_data, query_variable, evidence_variables, expected_error_type, expected_error_message", error_test_cases)
def test_error_handling(create_test_files, description, json_data, query_variable, evidence_variables, expected_error_type, expected_error_message):
    """
    Tests error handling for various invalid inputs.
    """
    sdd_file, json_file = create_test_files(json_data)
    executor = QueryExecutor(MockExecutorNN, sdd_file, json_file)

    with pytest.raises(expected_error_type, match=expected_error_message):
        executor.execute_query(query_variable=query_variable, evidence_variables=evidence_variables)

@pytest.mark.parametrize("description, json_data, query_variable, evidence_variables, mock_logic", zero_denominator_test_cases)
def test_zero_denominator_handling(monkeypatch, create_test_files, description, json_data, query_variable, evidence_variables, mock_logic):
    """
    Tests that queries with zero denominator return 0.0 without division-by-zero errors.
    """
    sdd_file, json_file = create_test_files(json_data)
    
    # Mock the forward method to return zero for denominator
    def mock_forward_logic(self, input_tensor, marginalized_variables=None):
        return mock_logic(marginalized_variables)
        
    monkeypatch.setattr(MockExecutorNN, 'forward', mock_forward_logic)
    
    executor = QueryExecutor(MockExecutorNN, sdd_file, json_file)
    result = executor.execute_query(query_variable=query_variable, evidence_variables=evidence_variables)
    
    assert result == 0.0
