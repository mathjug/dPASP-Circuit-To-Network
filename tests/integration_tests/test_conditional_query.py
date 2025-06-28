"""
This file contains integration tests for computing conditional probabilities using
the QueryExecutor class.

Specifically, this test ensures the end-to-end pipeline is functional:
1.  The `NNFParser` correctly parses a minimal `.sdd` file.
2.  The `ProbabilitiesParser` correctly parses a corresponding `.json` file.
3.  The real `RecursiveNN` and `IterativeNN` classes are instantiated with the circuit.
4.  The `QueryExecutor` correctly uses these components to perform two `forward` passes
    and compute a final, accurate conditional probability.
"""

import pytest
import json
from src.queries.query_executor import QueryExecutor
from tests.utils.utils import implementations

# A minimal, valid SDD file for the circuit: f = x1 AND x2
SDD_CONTENT = """c ids of sdd nodes start at 0
c circuit for: (x1 AND x2)
sdd 6
F 0
T 1
L 2 0 1
L 3 0 -1
L 4 1 2
D 5 2 2 2 4 3 0
"""

JSON_DATA = {
    "atom_mapping": {
        "1": "x1",
        "2": "x2"
    },
    "prob": {
        "pfacts": [
            ["2", "0.8"]
        ]
    }
}

@pytest.fixture
def create_test_files(tmp_path):
    """A pytest fixture to create the sdd and json files needed for the test."""
    sdd_file = tmp_path / "test.sdd"
    json_file = tmp_path / "test.json"

    sdd_file.write_text(SDD_CONTENT)
    with open(json_file, 'w') as f:
        json.dump(JSON_DATA, f)
        
    return str(sdd_file), str(json_file)

@pytest.mark.parametrize("implementation", implementations, ids=[i['name'] for i in implementations])
def test_integration_conditional_query(implementation, create_test_files):
    """
    Performs an integration test calling the QueryExecutor.

    This test uses the real neural network implementations and file parsers
    to verify the end-to-end query logic for the circuit (x1 AND x2).
    """
    sdd_file, json_file = create_test_files
    
    executor = QueryExecutor(implementation["implementation_class"], sdd_file, json_file)
    result = executor.execute_query(query_variable=1) # We want to find P(x1 | x2 = 0.8)

    assert result == pytest.approx(1.0)
