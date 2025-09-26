import pytest
import json
import torch

from src.parser.probabilities_parser import ProbabilitiesParser

@pytest.fixture
def valid_json_file(tmp_path):
    """Pytest fixture to create a temporary, valid JSON file for testing."""
    file_path = tmp_path / "test_alarm.json"
    valid_data = {
        "atom_mapping": {
            "1": "burglary",
            "2": "earthquake",
            "3": "alarm"
        },
        "prob": {
            "pfacts": [
                [1, 0.1],
                [2, 0.2]
            ]
        },
        "metadata": {
            "num_atoms": 3
        }
    }
    with open(file_path, 'w') as f:
        json.dump(valid_data, f)
    yield file_path

def test_successful_parsing(valid_json_file):
    """Tests if the parser correctly populates attributes with valid data."""
    parser = ProbabilitiesParser(valid_json_file)

    expected_mapping = {"1": "burglary", "2": "earthquake", "3": "alarm"}
    assert parser.variable_to_atom == expected_mapping

    expected_probs = {1: 0.1, 2: 0.2}
    assert parser.variable_to_prob == expected_probs

def test_file_not_found():
    """Tests the parser's behavior when the file does not exist."""
    parser = ProbabilitiesParser("non_existent_file.json")
    assert parser.variable_to_atom == {}
    assert parser.variable_to_prob == {}

def test_invalid_json(tmp_path):
    """Tests the parser's behavior with a malformed JSON file."""
    invalid_json_path = tmp_path / "invalid.json"
    with open(invalid_json_path, 'w') as f:
        f.write("{'bad': 'json',}")

    parser = ProbabilitiesParser(invalid_json_path)
    assert parser.variable_to_atom == {}
    assert parser.variable_to_prob == {}

def test_missing_prob_field(tmp_path):
    """Tests parsing when the 'prob' field is missing from the JSON."""
    file_path = tmp_path / "missing_prob.json"
    data = {"atom_mapping": {"1": "a", "2": "b"}}
    with open(file_path, 'w') as f:
        json.dump(data, f)
    
    parser = ProbabilitiesParser(file_path)
    assert parser.variable_to_atom == {"1": "a", "2": "b"}
    assert parser.variable_to_prob == {}

def test_missing_atom_mapping(tmp_path):
    """Tests parsing when the 'atom_mapping' field is missing."""
    file_path = tmp_path / "missing_mapping.json"
    data = {"prob": {"pfacts": [[1, 0.5]]}}
    with open(file_path, 'w') as f:
        json.dump(data, f)
            
    parser = ProbabilitiesParser(file_path)
    assert parser.variable_to_atom == {}
