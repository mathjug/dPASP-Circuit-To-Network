"""
This file contains unit tests for the QueryExecutor class.

The primary goal of these tests is to verify the internal logic of the
QueryExecutor in isolation from its dependencies, specifically the actual
PySDD circuit computation.

To achieve this, the tests use the 'monkeypatch' fixture to mock the WMC
(Weighted Model Counting) methods. Instead of performing real circuit
computation, the mocked methods return predictable, predefined values.

This allows the tests to confirm that the QueryExecutor correctly:
1.  Orchestrates the two required calls to the WMC propagate method (one for the
    numerator and one for the denominator).
2.  Prepares the correct input arrays for each call.
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

class MockWMC:
    """
    A mock WMC (Weighted Model Counting) class to substitute the real PySDD WMC.
    It allows us to control the output of the propagate method for predictable testing.
    """
    def __init__(self):
        self.call_count = 0
        self.call_args_list = []
        self.return_values = [0.0, 0.0]  # [numerator, denominator]

    def set_literal_weights_from_array(self, weights):
        """
        Mock method to track the weights being set.
        """
        self.call_args_list.append({'weights': weights, 'call_type': 'set_weights'})

    def propagate(self):
        """
        Mock method that returns predefined values based on call count.
        """
        self.call_args_list.append({'call_type': 'propagate'})
        value_to_return = self.return_values[self.call_count]
        self.call_count += 1
        return value_to_return

class MockCircuitRoot:
    """
    A mock circuit root that returns a mock WMC instance.
    """
    def __init__(self):
        self._wmc_instance = MockWMC()

    def wmc(self, log_mode=False):
        return self._wmc_instance

@pytest.fixture
def create_test_files(tmp_path):
    """
    A pytest fixture to create temporary sdd, json, and vtree files for testing.
    This ensures tests are isolated and don't depend on external files.

    The content of the SDD and vtree files don't matter since we are mocking the circuit,
    but the files need to exist and be parsable.
    """
    def _create_files(json_data):
        sdd_file = tmp_path / "test.sdd"
        json_file = tmp_path / "test.json"
        vtree_file = tmp_path / "test.vtree"

        sdd_content = "sdd 1\nT 0"
        sdd_file.write_text(sdd_content)
        
        vtree_content = "vtree 1\nL 1"
        vtree_file.write_text(vtree_content)

        with open(json_file, 'w') as f:
            json.dump(json_data, f)
            
        return str(sdd_file), str(json_file), str(vtree_file)

    return _create_files

# --- Test Cases for Successful Queries ---
# Each tuple: (description, json_data, query_variable, evidence_variables, numerator_result, denominator_result, 
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
        2, [], 0.4, 0.8,
        [0.3, 0.0, 0.5, 0.5, 0.3, 0.7],  # [-3, -2, -1, 1, 2, 3] with x2=1, ¬x2=0
        [0.3, 0.7, 0.5, 0.5, 0.3, 0.7]   # [-3, -2, -1, 1, 2, 3] all variables free
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
        2, [1, 4], 0.3, 0.5,
        [0.0, 0.2, 0.0, 0.0, 0.6, 0.4, 0.8, 0.2],  # [-4, -3, -2, -1, 1, 2, 3, 4] with x1=1, x2=1, x4=1
        [0.0, 0.2, 0.6, 0.0, 0.6, 0.4, 0.8, 0.2]   # [-4, -3, -2, -1, 1, 2, 3, 4] with x1=1, x4=1
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
        2, [1, 3], 0.3, 0.5,
        [0.8, 0.0, 0.0, 0.0, 0.6, 0.4, 1.0, 0.2],  # [-4, -3, -2, -1, 1, 2, 3, 4] with x1=1, x2=1, x3=1
        [0.8, 0.0, 0.6, 0.0, 0.6, 0.4, 1.0, 0.2]   # [-4, -3, -2, -1, 1, 2, 3, 4] with x1=1, x3=1
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
        2, [], 0.4, 0.8,
        [0.3, 0.0, 0.5, 0.5, 0.3, 0.7],  # [-3, -2, -1, 1, 2, 3] with x2=1, ¬x2=0
        [0.3, 0.7, 0.5, 0.5, 0.3, 0.7]   # [-3, -2, -1, 1, 2, 3] all variables free
    ),
]

@pytest.mark.parametrize("description, json_data, query_variable, evidence_variables, numerator_result, denominator_result, expected_numerator_input, expected_denominator_input", successful_query_test_cases)
def test_successful_queries(monkeypatch, create_test_files, description, json_data, query_variable, evidence_variables, numerator_result, denominator_result, expected_numerator_input, expected_denominator_input):
    """
    Tests successful query executions with various configurations.
    """
    sdd_file, json_file, vtree_file = create_test_files(json_data)
    mock_circuit_root = MockCircuitRoot()
    mock_circuit_root._wmc_instance.return_values = [numerator_result, denominator_result]
    
    # Mock the _get_circuit_root method
    def mock_get_circuit_root(self, sdd_file, vtree_file):
        return mock_circuit_root
    monkeypatch.setattr(QueryExecutor, '_get_circuit_root', mock_get_circuit_root)

    executor = QueryExecutor(sdd_file, json_file, vtree_file)
    result = executor.execute_query(query_variable=query_variable, evidence_variables=evidence_variables)

    assert result == pytest.approx(numerator_result / denominator_result)
    assert mock_circuit_root._wmc_instance.call_count == 2
    set_weights_calls = [call for call in mock_circuit_root._wmc_instance.call_args_list if call['call_type'] == 'set_weights']
    assert len(set_weights_calls) == 2
    numerator_weights = set_weights_calls[0]['weights']
    denominator_weights = set_weights_calls[1]['weights']
    assert list(numerator_weights) == pytest.approx(list(expected_numerator_input))
    assert list(denominator_weights) == pytest.approx(list(expected_denominator_input))

# --- Test Cases for Error Handling ---
# Each tuple: (description, json_data, query_variable, evidence_variables, expected_error_type, expected_error_message)

error_test_cases = [
    (
        "Query variable out of bounds (too high)",
        {"atom_mapping": {"1": "a", "2": "b"}, "prob": {"pfacts": [["1", "0.5"], ["2", "0.3"]], "pvars": []}},
        3, [], ValueError, "Variable \\(X_3\\) out of bounds for tensor of size 2\\."
    ),
    (
        "Query variable out of bounds (zero)",
        {"atom_mapping": {"1": "a", "2": "b"}, "prob": {"pfacts": [["1", "0.5"], ["2", "0.3"]], "pvars": []}},
        0, [], ValueError, "Variable \\(X_0\\) out of bounds for tensor of size 2\\."
    ),
    (
        "Evidence variable out of bounds (too high)",
        {"atom_mapping": {"1": "a", "2": "b"}, "prob": {"pfacts": [["1", "0.5"], ["2", "0.3"]], "pvars": []}},
        1, [3], ValueError, "Variable \\(X_3\\) out of bounds for tensor of size 2\\."
    ),
    (
        "Evidence variable out of bounds (zero)",
        {"atom_mapping": {"1": "a", "2": "b"}, "prob": {"pfacts": [["1", "0.5"], ["2", "0.3"]], "pvars": []}},
        1, [0], ValueError, "Variable \\(X_0\\) out of bounds for tensor of size 2\\."
    ),
]

@pytest.mark.parametrize("description, json_data, query_variable, evidence_variables, expected_error_type, expected_error_message", error_test_cases)
def test_error_handling(monkeypatch, create_test_files, description, json_data, query_variable, evidence_variables, expected_error_type, expected_error_message):
    """
    Tests error handling for various invalid inputs.
    """
    sdd_file, json_file, vtree_file = create_test_files(json_data)    
    mock_circuit_root = MockCircuitRoot()
    
    # Mock the _get_circuit_root method
    def mock_get_circuit_root(self, sdd_file, vtree_file):
        return mock_circuit_root
    
    monkeypatch.setattr(QueryExecutor, '_get_circuit_root', mock_get_circuit_root)
    executor = QueryExecutor(sdd_file, json_file, vtree_file)
    with pytest.raises(expected_error_type, match=expected_error_message):
        executor.execute_query(query_variable=query_variable, evidence_variables=evidence_variables)

# --- Test Cases for Zero Denominator ---
# Each tuple: (description, json_data, query_variable, evidence_variables, mock_logic)

zero_denominator_test_cases = [
    (
        "Zero denominator without evidence",
        {"atom_mapping": {"1": "a", "2": "b"}, "prob": {"pfacts": [["1", "0.5"], ["2", "0.3"]], "pvars": []}},
        1, [],
        lambda weights: 0.0 if len(weights) == 4 and weights[1] == 0.0 else 0.1
    ),
    (
        "Zero denominator with evidence",
        {"atom_mapping": {"1": "a", "2": "b", "3": "c"}, "prob": {"pfacts": [["1", "0.5"], ["2", "0.3"], ["3", "0.7"]], "pvars": []}},
        2, [1],
        lambda weights: 0.0 if len(weights) == 6 and weights[2] == 0.0 else 0.1
    ),
]

@pytest.mark.parametrize("description, json_data, query_variable, evidence_variables, mock_logic", zero_denominator_test_cases)
def test_zero_denominator_handling(monkeypatch, create_test_files, description, json_data, query_variable, evidence_variables, mock_logic):
    """
    Tests that queries with zero denominator return 0.0 without division-by-zero errors.
    """
    sdd_file, json_file, vtree_file = create_test_files(json_data)    
    mock_circuit_root = MockCircuitRoot()
    
    # Mock the propagate method to return zero for denominator
    def mock_propagate_logic(self):
        # Get the weights from the last set_literal_weights_from_array call
        set_weights_calls = [call for call in self.call_args_list if call['call_type'] == 'set_weights']
        if set_weights_calls:
            weights = set_weights_calls[-1]['weights']
            return mock_logic(weights)
        return 0.1
    
    mock_circuit_root._wmc_instance.propagate = mock_propagate_logic.__get__(mock_circuit_root._wmc_instance, MockWMC)
    
    # Mock the _get_circuit_root method
    def mock_get_circuit_root(self, sdd_file, vtree_file):
        return mock_circuit_root
    
    monkeypatch.setattr(QueryExecutor, '_get_circuit_root', mock_get_circuit_root)
    executor = QueryExecutor(sdd_file, json_file, vtree_file)
    result = executor.execute_query(query_variable=query_variable, evidence_variables=evidence_variables)
    assert result == 0.0
