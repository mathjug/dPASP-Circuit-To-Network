import json
import numpy as np

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
        input_tensor (np.ndarray): A NumPy array holding the input probabilities for each atom.
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
        self.input_tensor = np.array([])      
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
        - self.input_tensor: A NumPy array of probabilities, ordered by variable number. The
          size of the tensor is determined by 'num_atoms' in the metadata.
          The value at index `i` is the probability of atom `i+1`.
        """
        data = load_json_file(self.file_path)
        if data is None:
            return
        num_atoms = self._build_dictionaries(data)
        self._build_input_tensor(num_atoms)
    
    def _build_dictionaries(self, data):
        """Extracts mappings and probabilities from the data to populate the instance's dictionaries."""
        self.variable_to_atom = data.get("atom_mapping", {})
        num_atoms = len(self.variable_to_atom)

        pfacts = data.get("prob", {}).get("pfacts", [])
        self.variable_to_prob = {int(pfact[0]): float(pfact[1]) for pfact in pfacts}
        return num_atoms
    
    def _build_input_tensor(self, num_atoms):
        """Constructs the input tensor based on the extracted probabilities."""
        self.input_tensor = np.ones(num_atoms)        
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

def main():
    file_path = '/Users/mjurgensen/Desktop/Personal/TCC/PASDD/plp/programs/alarm/alarm.json'
    
    parser = ProbabilitiesParser(file_path)

    if parser.input_tensor.size > 0:
        print("--- Parsing Successful ---\n")
        
        print("1. Variable to Atom Mapping:")
        for variable_number, atom in sorted(parser.variable_to_atom.items(), key=lambda item: int(item[0])):
            print(f"  Variable Number {variable_number}: {atom}")
            
        print("\n2. Probabilities Found (Variable Number -> Probability):")
        for variable_number, prob in sorted(parser.variable_to_prob.items()):
            atom_name = parser.variable_to_atom.get(str(variable_number), "Unknown")
            print(f"  {atom_name} ({variable_number}): {prob}")

        print("\n3. Constructed Input Tensor:")
        print(parser.input_tensor)
        
        print("\n--- Tensor Breakdown ---")
        for i, prob in enumerate(parser.input_tensor):
            variable_number = i + 1
            atom_name = parser.variable_to_atom.get(str(variable_number), "Unknown")
            if variable_number in parser.variable_to_prob:
                print(f"  Index {i} (Variable {variable_number}: {atom_name}): {prob:.2f}")
            else:
                print(f"  Index {i} (Variable {variable_number}: {atom_name}): {prob:.2f} (Default value)")

if __name__ == "__main__":
    main()
