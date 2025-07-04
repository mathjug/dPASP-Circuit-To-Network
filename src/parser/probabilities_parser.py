"""
This module provides a parser for processing JSON files that describe the
properties of a boolean circuit, specifically focusing on its probabilistic aspects.
The primary component is the `ProbabilitiesParser` class, which extracts key
information and transforms it into a format suitable for probabilistic inference
or machine learning applications.

The parser's main responsibilities include:
- Mapping internal variable numbers to human-readable string representations,
  referred to as 'atoms'.
- Extracting the assigned probabilities for a subset of these variables.
- Identifying variables that are "marginalized", meaning they have not been
  assigned a specific probability in the input file.
- Constructing a `torch.tensor` of input probabilities, which can be directly
  used in computational frameworks that leverage tensor operations.
"""

import json
import torch

class ProbabilitiesParser:
    """
    Parses a JSON file describing a boolean circuit to extract atom mappings,
    probabilities, and construct an input tensor. The results are stored
    as instance attributes.

    Attributes:
        file_path (str): The path to the JSON circuit file provided on initialization.
        variable_to_atom (dict): A dictionary mapping variable numbers (str) to their literal names (str).
                                 Populated after parsing.
        variable_to_prob (dict): A dictionary mapping variable numbers (int) to their probabilities (float).
                                 Populated after parsing.
        marginalized_variables (torch.tensor): An array with 1 for marginalized variables and 0 otherwise.
        input_tensor (torch.tensor): An array holding the input probabilities for each atom.
                                     The index corresponds to (variable_number - 1). Populated after parsing.
    """
    def __init__(self, file_path):
        """
        Initializes the parser and immediately processes the file.

        Args:
            file_path: The path to the JSON circuit file.
        """
        self.file_path = file_path
        self.variable_to_atom = {}
        self.variable_to_prob = {}
        self.marginalized_variables = torch.tensor([])  
        self.input_tensor = torch.tensor([])
        self._parse()

    def _parse(self):
        """
        Private method to handle the file reading and data extraction.

        This method reads the JSON file specified by `self.file_path` and populates
        the instance's attributes based on its content.
        
        After execution, it will have set the following attributes:
        - self.variable_to_atom: A dictionary where keys are the variable numbers from the
          boolean circuit and values are the represented atoms (e.g., '1': 'hears_alarm(john)').
        - self.variable_to_prob: A dictionary mapping the integer variable number to their corresponding
          float probabilities (e.g., 1: 0.1).
        - self.marginalized_variables: An array of integers, where 0 represents the non-marginalized
          variables, 1 are the marginalized variables, and 2 is the variable whose probability will be queried
        - self.input_tensor: An array of probabilities, ordered by variable number. The
          size of the tensor is determined by 'num_atoms' in the metadata.
          The value at index `i` is the probability of atom `i+1`.
        """
        data = load_json_file(self.file_path)
        if data is None:
            return
        num_atoms = self._build_dictionaries(data)
        self._build_marginalized_variables(num_atoms)
        self._build_input_tensor(num_atoms)
    
    def _build_dictionaries(self, data):
        """Extracts mappings and probabilities from the data to populate the instance's dictionaries."""
        self.variable_to_atom = data.get("atom_mapping", {})
        num_atoms = len(self.variable_to_atom)

        pfacts = data.get("prob", {}).get("pfacts", [])
        self.variable_to_prob = {int(pfact[0]): float(pfact[1]) for pfact in pfacts}
        return num_atoms
    
    def _build_marginalized_variables(self, num_atoms):
        """
        Constructs a vector that indicates which variables are marginalized.

        A variable is considered marginalized if it has not been assigned a
        specific probability. This vector uses a 1 for marginalized variables
        and a 0 otherwise.
        """
        self.marginalized_variables = torch.zeros(num_atoms, dtype=torch.int64)
        for variable_index in range(num_atoms):
            if variable_index + 1 not in self.variable_to_prob:
                self.marginalized_variables[variable_index] = 1
    
    def _build_input_tensor(self, num_atoms):
        """Constructs the input tensor based on the extracted probabilities."""
        self.input_tensor = torch.ones(num_atoms)
        for variable_number, probability in self.variable_to_prob.items():
            if 0 < variable_number <= num_atoms:
                self.input_tensor[variable_number - 1] = probability

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
