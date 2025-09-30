import torch

class QueryExecutor:
    """
    Executes conditional probability queries on a compiled NNF circuit.
    
    The executor calculates P(query_variable | evidence) using the formula:
    P(Q | E) = P(Q, E) / P(E)

    Attributes:
        neural_network: An instance of the neural network model that represents the NNF circuit.
        num_variables (int): The number of variables in the NNF circuit.
    """
    
    def __init__(self, sdd_file, json_file, nn_implementation):
        """
        Initializes the QueryExecutor.

        Args:
            sdd_file (string): The `.sdd` file with the definition of the NNF circuit.
            json_file (string): The `.json` file with the probabilities of each variable.
            nn_implementation (nn.Module): The implementation (IterativeNN or RecursiveNN) of the neural network.
        """
        self.neural_network = nn_implementation(sdd_file, json_file)
        self.num_variables = self.neural_network.num_variables
    
    def execute_query(self, query_variables, evidence_variables = []):
        """
        Executes a conditional probability query.

        Args:
            query_variables (list): The IDs of the variables to query (e.g., P(X_i=1, X_j=1 | ...)).
            evidence_variables (list): The IDs of the variables to use as evidence (e.g., [X_j=1, X_k=1]).
        
        Returns:
            float: The calculated conditional probability P(query_variables | evidence).
        """
        numerator_input, denominator_input = self._build_input_tensors(
            query_variables,
            evidence_variables if evidence_variables is not None else []
        )

        # 1. Calculate the numerator: P(query=1, evidence=1)
        numerator_prob = self.neural_network.forward(numerator_input)

        # 2. Calculate the denominator: P(evidence=1)
        denominator_prob = self.neural_network.forward(denominator_input)

        # 3. Compute conditional probability
        if float(denominator_prob) == 0.:
            return 0.
        return float(numerator_prob / denominator_prob)
    
    def _build_input_tensors(self, query_variables, evidence_variables):
        """
        Builds the input tensors for the numerator and denominator of the conditional probability.
        The input tensor is indexed as [1, -1, 2, -2, ..., n, -n] for n variables.
        """
        numerator_input = torch.ones(self.num_variables * 2, dtype=torch.float32)
        denominator_input = torch.ones(self.num_variables * 2, dtype=torch.float32)

        for query_variable in query_variables:
            neg_index, pos_index = self._get_indexes_of_literals(query_variable)
            numerator_input[neg_index] = 0.
            numerator_input[pos_index] = 1.

        for evidence_variable in evidence_variables:
            neg_index, pos_index = self._get_indexes_of_literals(evidence_variable)
            numerator_input[neg_index] = 0.
            numerator_input[pos_index] = 1.
            denominator_input[neg_index] = 0.
            denominator_input[pos_index] = 1.
        
        return numerator_input, denominator_input
    
    def _get_indexes_of_literals(self, variable):
        """
        Gets the indexes of the positive and negative literals for a variable.
        """
        if variable < 1 or variable > self.num_variables:
            raise ValueError(f"Variable (X_{variable}) out of bounds for tensor of size {self.num_variables}.")
        neg_index = 2 * (variable - 1) + 1
        pos_index = 2 * (variable - 1)
        return neg_index, pos_index
