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
# Each tuple: (description, query_variable, evidence_variables, expected_output)
alarm_test_cases = [
    ("P(alarm)", 4, None, 0.28),
    ("P(alarm | burglary=1)", 4, [1], 1.0),
    ("P(alarm | hears_alarm=1)", 4, [3], 0.28),
    ("P(alarm | calls=1)", 4, [5], 1.0),
    ("P(alarm | earthquake=1, burglary=1)", 4, [1, 2], 1.0),
    ("P(alarm | hears_alarm=1, earthquake=1)", 4, [3, 2], 1.0),
    ("P(alarm | [])", 4, [], 0.28),
]

# Test cases for calls queries
# Each tuple: (description, query_variable, evidence_variables, expected_output)
calls_test_cases = [
    ("P(calls)", 5, None, 0.196),
    ("P(calls | earthquake=1)", 5, [2], 0.7),
    ("P(calls | hears_alarm=1)", 5, [3], 0.28),
    ("P(calls | alarm=1)", 5, [4], 0.7),
    ("P(calls | earthquake=1, hears_alarm=1)", 5, [2, 3], 1.0),
    ("P(calls | earthquake=1, alarm=1)", 5, [2, 4], 0.7),
    ("P(calls | alarm=1, hears_alarm=1)", 5, [3, 4], 1.0),
    ("P(calls | earthquake=1, burglary=1, alarm=1)", 5, [1, 2, 4], 0.7),
    ("P(calls | earthquake=1, burglary=1, alarm=1, hears_alarm=1)", 5, [1, 2, 3, 4], 1.0),
]

all_test_cases = alarm_test_cases + calls_test_cases

@pytest.mark.parametrize("implementation", implementations, ids=[i['name'] for i in implementations])
@pytest.mark.parametrize("description, query_variable, evidence_variables, expected_output", all_test_cases)
def test_query_executor_integration(implementation, alarm_files, description, query_variable, evidence_variables, expected_output):
    """
    Tests the QueryExecutor integration with various query and evidence combinations.
    """
    sdd_file, json_file = alarm_files
    
    executor = QueryExecutor(implementation["implementation_class"], sdd_file, json_file)
    result = executor.execute_query(query_variable=query_variable, evidence_variables=evidence_variables)
    
    torch.testing.assert_close(
        result, 
        expected_output,
        msg=f"Query mismatch for {description} ({implementation['name']} Implementation)\n Expected: {expected_output}\n Actual: {result}\n"
    )
