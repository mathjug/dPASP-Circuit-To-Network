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

# --- Test Cases ---

def test_successful_query(monkeypatch, create_test_files):
    """
    Tests a standard, successful query execution using the built-in monkeypatch fixture.
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

    # 1. Setup: Create test files and prepare the mock forwarder
    json_data = {
        "atom_mapping": {"1": "a", "2": "b", "3": "c"},
        "prob": {"pfacts": [["1", "0.8"]]}
    }
    sdd_file, json_file = create_test_files(json_data)
    
    mock_forwarder = MockForwarder(return_values=[
        torch.tensor([0.4]), # Mocked result for the numerator call
        torch.tensor([0.8])  # Mocked result for the denominator call
    ])
    
    # Replace 'forward' method with the mock forwarder
    monkeypatch.setattr(MockExecutorNN, 'forward', mock_forwarder)

    # 2. Execution: Initialize and run the query
    executor = QueryExecutor(MockExecutorNN, sdd_file, json_file)
    result = executor.execute_query(query_variable=2)

    # 3. Assertions
    assert result == pytest.approx(0.4 / 0.8)
    assert mock_forwarder.call_count == 2
    
    numerator_call_args = mock_forwarder.call_args_list[0]
    assert torch.equal(numerator_call_args['input'], torch.tensor([0.8, 1.0, 1.0]))
    assert torch.equal(numerator_call_args['marginalized'], torch.tensor([0, 0, 1], dtype=torch.int64))

    denominator_call_args = mock_forwarder.call_args_list[1]
    assert torch.equal(denominator_call_args['input'], torch.tensor([0.8, 1.0, 1.0]))
    assert torch.equal(denominator_call_args['marginalized'], torch.tensor([0, 1, 1], dtype=torch.int64))

def test_query_on_evidence_variable_raises_error(create_test_files):
    """
    Tests that querying for a variable that is already part of the evidence
    (i.e., not marginalized) raises a ValueError.
    """
    json_data = {
        "atom_mapping": {"1": "a", "2": "b"},
        "prob": {"pfacts": [["1", "0.9"]]}
    }
    sdd_file, json_file = create_test_files(json_data)
    
    executor = QueryExecutor(MockExecutorNN, sdd_file, json_file)

    with pytest.raises(ValueError, match="Probability is already known for X_1"):
        executor.execute_query(query_variable=1)

def test_query_out_of_bounds_raises_error(create_test_files):
    """
    Tests that querying for a variable outside the valid range raises a ValueError.
    """
    json_data = {"atom_mapping": {"1": "a", "2": "b"}}
    sdd_file, json_file = create_test_files(json_data)
    
    executor = QueryExecutor(MockExecutorNN, sdd_file, json_file)

    with pytest.raises(ValueError, match="Query variable .* out of bounds"):
        executor.execute_query(query_variable=3)
        
    with pytest.raises(ValueError, match="Query variable .* out of bounds"):
        executor.execute_query(query_variable=0)

def test_query_with_zero_denominator(monkeypatch, create_test_files):
    """
    Tests that if the probability of the evidence (denominator) is zero,
    the query returns 0.0 without a division-by-zero error.
    """
    json_data = {"atom_mapping": {"1": "a", "2": "b"}}
    sdd_file, json_file = create_test_files(json_data)
    
    # Mock the forward method to return a zero for the denominator
    def mock_forward_logic(self, input_tensor, marginalized_variables=None):
        # If this is the denominator call (original marginalized vars), return 0
        if torch.equal(marginalized_variables, torch.tensor([1, 1], dtype=torch.int64)):
            return torch.tensor([0.0])
        # Otherwise, return a non-zero numerator
        return torch.tensor([0.1])
        
    monkeypatch.setattr(MockExecutorNN, 'forward', mock_forward_logic)
    
    executor = QueryExecutor(MockExecutorNN, sdd_file, json_file)
    result = executor.execute_query(query_variable=1)
    
    assert result == 0.0
