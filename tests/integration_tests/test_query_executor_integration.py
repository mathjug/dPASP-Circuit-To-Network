"""
This file contains integration tests for computing conditional probabilities using
the QueryExecutor class.

Specifically, this test ensures the end-to-end pipeline is functional:
1.  The `NNFParser` correctly parses a `.sdd` file.
2.  The `ProbabilitiesParser` correctly parses a corresponding `.json` file.
3.  The real `RecursiveNN` and `IterativeNN` classes are instantiated with the circuit.
4.  The `QueryExecutor` correctly uses these components to perform two `forward` passes
    and compute a final, accurate conditional probability.
5.  The system correctly handles multiple evidence variables and complex queries.
"""

import torch
import pytest
import os
from src.queries.query_executor import QueryExecutor
from tests.utils.utils import implementations

@pytest.fixture
def alarm_files():
    """A pytest fixture to provide paths to the alarm example files."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    examples_dir = os.path.join(current_dir, "..", "..", "examples", "alarm")
    sdd_file = os.path.join(examples_dir, "alarm_balanced.sdd")
    json_file = os.path.join(examples_dir, "alarm.json")
    return sdd_file, json_file

# Test cases for alarm queries
# Each tuple: (description, query_variables, evidence_variables, expected_output)
alarm_test_cases = [
    ("P(alarm)", [4], None, 0.28),
    ("P(hears_alarm)", [3], [], 0.7),
    ("P(alarm | burglary=1)", [4], [1], 1.0),
    ("P(alarm | hears_alarm=1)", [4], [3], 0.28),
    ("P(alarm | calls=1)", [4], [5], 1.0),
    ("P(alarm | alarm=1)", [4], [4], 1.0),
    ("P(alarm | earthquake=1, burglary=1)", [4], [1, 2], 1.0),
    ("P(alarm | hears_alarm=1, earthquake=1)", [4], [3, 2], 1.0),
]

# Test cases for calls queries
# Each tuple: (description, query_variables, evidence_variables, expected_output)
calls_test_cases = [
    ("P(calls)", [5], None, 0.196),
    ("P(calls | earthquake=1)", [5], [2], 0.7),
    ("P(calls | hears_alarm=1)", [5], [3], 0.28),
    ("P(calls | alarm=1)", [5], [4], 0.7),
    ("P(calls | calls=1)", [5], [5], 1.0),
    ("P(calls | earthquake=1, hears_alarm=1)", [5], [2, 3], 1.0),
    ("P(calls | earthquake=1, alarm=1)", [5], [2, 4], 0.7),
    ("P(calls | alarm=1, hears_alarm=1)", [5], [3, 4], 1.0),
    ("P(calls | earthquake=1, burglary=1, alarm=1)", [5], [1, 2, 4], 0.7),
    ("P(calls | earthquake=1, burglary=1, alarm=1, hears_alarm=1)", [5], [1, 2, 3, 4], 1.0),
]

# Test cases with multiple query variables
# Each tuple: (description, query_variables, evidence_variables, expected_output)
multiple_query_variables_test_cases = [
    ("P(burglary, earthquake)", [1, 2], [], 0.02),
    ("P(burglary, earthquake | hears_alarm=1)", [1, 2], [3], 0.02),
    ("P(burglary | alarm=1)", [1], [4], 0.35714286),
    ("P(earthquake | calls=1)", [2], [5], 0.71428571),
    ("P(burglary, earthquake | alarm=1)", [1, 2], [4], 0.071428571),
    ("P(burglary, calls)", [1, 5], [], 0.07),
    ("P(hears_alarm, alarm)", [3, 4], [], 0.196),
    ("P(alarm, calls)", [4, 5], [], 0.196),
    ("P(alarm, calls | burglary=1)", [4, 5], [1], 0.7),
    ("P(burglary, hears_alarm, alarm, calls)", [1, 3, 4, 5], [], 0.07),
    ("P(earthquake, hears_alarm, alarm, calls)", [2, 3, 4, 5], [], 0.14),
    ("P(burglary, earthquake, hears_alarm, alarm, calls)", [1, 2, 3, 4, 5], [], 0.014),
]

all_test_cases = alarm_test_cases + calls_test_cases + multiple_query_variables_test_cases

@pytest.mark.parametrize("implementation", implementations, ids=[i['name'] for i in implementations])
@pytest.mark.parametrize("description, query_variables, evidence_variables, expected_output", all_test_cases)
def test_query_executor_integration(alarm_files, implementation, description, query_variables, evidence_variables, expected_output):
    """
    Tests the QueryExecutor integration with various query and evidence combinations.
    """
    sdd_file, json_file = alarm_files
    executor = QueryExecutor(sdd_file, json_file, implementation["implementation_class"])
    result = executor.execute_query(query_variables=query_variables, evidence_variables=evidence_variables)
    
    torch.testing.assert_close(
        result, 
        expected_output,
        msg=f"Query mismatch for {description}\n Expected: {expected_output}\n Actual: {result}\n"
    )
