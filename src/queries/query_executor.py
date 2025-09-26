import torch
from array import array
from pathlib import Path
from pysdd.sdd import SddManager, Vtree

from src.parser.probabilities_parser import ProbabilitiesParser

class QueryExecutor:
    """
    Executes conditional probability queries on a compiled NNF circuit.
    
    The executor calculates P(query_variable | evidence) using the formula:
    P(Q | E) = P(Q, E) / P(E)

    Attributes:
        neural_network: An instance of the neural network model that represents the NNF circuit.
        num_variables (int): The number of variables in the NNF circuit.
    """
    
    def __init__(self, sdd_file, json_file, vtree_file):
        """
        Initializes the QueryExecutor.

        Args:
            sdd_file (string): The `.sdd` file with the definition of the NNF circuit.
            json_file (string): The `.json` file with the probabilities of each variable.
            vtree_file (string): The `.vtree` file with the structure of the Vtree.
        """
        self.circuit_root = self._get_circuit_root(sdd_file, vtree_file)
        prob_parser = ProbabilitiesParser(json_file)
        self.variable_to_prob = prob_parser.variable_to_prob
        self.num_variables = len(prob_parser.variable_to_atom)
    
    def execute_query(self, query_variable, evidence_variables = []):
        """
        Executes a conditional probability query.

        Args:
            query_variable (int): The ID of the variable to query (e.g., P(X_i=1 | ...)).
            evidence_variables (list): The IDs of the variables to use as evidence (e.g., [X_j=1, X_k=1]).
        
        Returns:
            float: The calculated conditional probability P(query_variable | evidence).
        """
        numerator_input, denominator_input = self._build_input_tensors(
            query_variable,
            evidence_variables if evidence_variables is not None else []
        )

        # Weighted Model Counting
        wmc = self.circuit_root.wmc(log_mode=False)

        # 1. Calculate the numerator: P(query=1, evidence=1)
        wmc.set_literal_weights_from_array(numerator_input)
        numerator_prob = wmc.propagate()

        # 2. Calculate the denominator: P(evidence=1)
        wmc.set_literal_weights_from_array(denominator_input)
        denominator_prob = wmc.propagate()

        # 3. Compute conditional probability
        if denominator_prob == 0:
            return 0.0        
        return numerator_prob / denominator_prob
    
    def _get_circuit_root(self, sdd_file, vtree_file):
        """
        Gets the root node of the circuit.
        """
        current_dir = Path(__file__).parent.parent.parent
        vtree = Vtree.from_file(bytes(current_dir / vtree_file))
        sdd = SddManager.from_vtree(vtree)
        root = sdd.read_sdd_file(bytes(current_dir / sdd_file))
        return root
    
    def _build_input_tensors(self, query_variable, evidence_variables):
        """
        Builds the input tensors for the numerator and denominator of the conditional probability.
        The input tensor is indexed as [-n, -(n-1), ..., -1, 1, 2, ..., n] for n variables.
        """
        numerator_input = self._initialize_input_tensor()
        denominator_input = self._initialize_input_tensor()

        neg_index, pos_index = self._get_indexes_of_literals(query_variable)
        numerator_input[neg_index] = 0.
        numerator_input[pos_index] = self._get_positive_literal_input_value(query_variable)

        for evidence_variable in evidence_variables:
            neg_index, pos_index = self._get_indexes_of_literals(evidence_variable)
            numerator_input[neg_index] = 0.
            numerator_input[pos_index] = self._get_positive_literal_input_value(evidence_variable)
            denominator_input[neg_index] = 0.
            denominator_input[pos_index] = self._get_positive_literal_input_value(evidence_variable)
        
        return array('d', numerator_input), array('d', denominator_input)
    
    def _initialize_input_tensor(self):
        """
        Initializes the input tensor for the numerator and denominator of the conditional probability.
        The array is indexed as [-n, -(n-1), ..., -1, 1, 2, ..., n] for n variables.
        """
        input_tensor = torch.ones(self.num_variables * 2, dtype=torch.float32)
        for variable, prob in self.variable_to_prob.items():
            neg_index, pos_index = self._get_indexes_of_literals(variable)
            input_tensor[neg_index] = 1 - prob
            input_tensor[pos_index] = prob
        return input_tensor
    
    def _get_indexes_of_literals(self, variable):
        """
        Gets the indexes of the positive and negative literals for a variable.
        """
        if variable < 1 or variable > self.num_variables:
            raise ValueError(f"Variable (X_{variable}) out of bounds for tensor of size {self.num_variables}.")
        neg_index = self.num_variables - variable
        pos_index = self.num_variables + variable - 1
        return neg_index, pos_index
    
    def _get_positive_literal_input_value(self, variable):
        """
        Gets the input value for the positive literal of a variable.
        """
        if variable not in self.variable_to_prob:
            return 1.
        return self.variable_to_prob[variable]
