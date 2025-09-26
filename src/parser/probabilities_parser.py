"""
This module provides a parser for processing JSON files that describe the
properties of a boolean circuit, specifically focusing on its probabilistic aspects.
The primary component is the `ProbabilitiesParser` class, which extracts key
information and transforms it into a format suitable for probabilistic inference
or machine learning applications.

The parser's main responsibilities include:
- Mapping internal variable numbers to human-readable string representations,
  referred to as 'atoms'.
- Extracting the assigned probabilities for the probabilistic variables.
"""

import json

class ProbabilitiesParser:
    """
    Parses a JSON file describing a boolean circuit to extract atom mappings,
    probabilities. The results are stored as instance attributes.

    Attributes:
        variable_to_atom (dict): A dictionary mapping variable numbers (str) to their literal names (str).
                                 Populated after parsing.
        variable_to_prob (dict): A dictionary mapping variable numbers (int) to their probabilities (float).
                                 Populated after parsing.
    """
    def __init__(self, file_path):
        """
        Initializes the parser and immediately processes the file.

        Args:
            file_path: The path to the JSON circuit file.
        """
        self.variable_to_atom = {}
        self.variable_to_prob = {}
        self._parse_json_file(file_path)

    def _parse_json_file(self, file_path):
        """
        Private method to handle the file reading and data extraction.

        This method reads the JSON file specified by `self.file_path` and populates
        the instance's attributes based on its content.
        
        After execution, it will have set the following attributes:
        - self.variable_to_atom: A dictionary where keys are the variable numbers from the
          boolean circuit and values are the represented atoms (e.g., '1': 'hears_alarm(john)').
        - self.variable_to_prob: A dictionary mapping the integer variable number to their corresponding
          float probabilities (e.g., 1: 0.1).
        """
        data = load_json_file(file_path)
        if data is None:
            return
        
        self.variable_to_atom = data.get("atom_mapping", {})
        self.variable_to_prob = self._map_variables_to_probabilities(data)

    def _map_variables_to_probabilities(self, data):
        """Maps variables to their probabilities."""
        pfacts = data.get("prob", {}).get("pfacts", [])
        variable_to_prob = {int(pfact[0]): float(pfact[1]) for pfact in pfacts}
        pvars = data.get("prob", {}).get("pvars", [])
        for pvar in pvars:
            if pvar not in variable_to_prob:
                variable_to_prob[pvar] = 1/2
        return variable_to_prob

def load_json_file(file_path):
    """Loads and parses a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file at {file_path} is not a valid JSON file.")
        return
    return data
