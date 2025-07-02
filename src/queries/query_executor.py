from src.parser.nnf_parser import NNFParser
from src.parser.probabilities_parser import ProbabilitiesParser

class QueryExecutor:
    """
    Executes conditional probability queries on a compiled NNF circuit.
    
    The executor calculates P(query_variable | evidence) using the formula:
    P(Q | E) = P(Q, E) / P(E)

    Attributes:
        neural_network: An instance of the neural network model that represents the NNF circuit.
        input_tensor (torch.Tensor): A tensor holding the initial probabilities for all variables,
                                     where evidence variables are fixed and others are marginalized.
        marginalized_variables (torch.Tensor): A tensor indicating which variables are marginalized (1)
                                               and which are part of the evidence (0).
    """
    
    def __init__(self, executor_class, sdd_file, json_file):
        """
        Initializes the QueryExecutor.

        Args:
            executor_class: The neural network implementation class (e.g., RecursiveNN, IterativeNN).
            sdd_file (string): The `.sdd` file with the definition of the NNF circuit.
            json_file (string): The `.json` file with the probabilities of each variable.
        """
        self._build_neural_network(executor_class, sdd_file)
        self._build_inputs(json_file)
    
    def execute_query(self, query_variable, evidence_variables = []):
        """
        Executes a conditional probability query.

        Args:
            query_variable (int): The ID of the variable to query (e.g., P(X_i=1 | ...)).
            evidence_variables (list): The IDs of the variables to use as evidence (e.g., [X_j=1, X_k=1]).
        
        Returns:
            float: The calculated conditional probability P(query_variable | evidence).
        """
        query_variable_index = self._get_query_variable_index(query_variable)
        evidence_variables_indices = self._get_evidence_variables_indices(evidence_variables) if evidence_variables is not None else []

        numerator_input, denominator_input = self._build_input_tensors(
            query_variable_index,
            evidence_variables_indices
        )
        numerator_marginalized_vars, denominator_marginalized_vars = self._build_marginalized_variables(
            query_variable_index,
            evidence_variables_indices
        )

        # 1. Calculate the numerator: P(query=1, evidence=1)
        numerator_prob = self.neural_network.forward(
            numerator_input,
            marginalized_variables=numerator_marginalized_vars
        ).item()

        # 2. Calculate the denominator: P(evidence=1)
        denominator_prob = self.neural_network.forward(
            denominator_input,
            marginalized_variables=denominator_marginalized_vars
        ).item()

        # 3. Compute conditional probability
        if denominator_prob == 0:
            return 0.0        
        return numerator_prob / denominator_prob
    
    def _get_query_variable_index(self, query_variable):
        """
        Validates the query variable and converts it to a 0-based tensor index.

        Args:
            query_variable (int): The 1-indexed ID of the variable to query.

        Returns:
            int: The 0-based index corresponding to the query variable.

        Raises:
            ValueError: If the query variable is out of bounds or is already part of the evidence.
        """
        if query_variable < 1 or query_variable > self.input_tensor.numel():
            raise ValueError(f"Query variable (X_{query_variable}) out of bounds \
                               for tensor of size {self.input_tensor.numel()}.")

        variable_index = query_variable - 1

        if self.marginalized_variables[variable_index] == 0:
            raise ValueError(f"Probability is already known for X_{query_variable}: \
                               {self.input_tensor[variable_index]}")

        return variable_index
    
    def _get_evidence_variables_indices(self, evidence_variables):
        """
        Validates the evidence variables and converts them to 0-based tensor indices.
        """
        evidence_variables_indices = []
        for evidence_variable in evidence_variables:
            if evidence_variable < 1 or evidence_variable > self.input_tensor.numel():
                raise ValueError(f"Evidence variable (X_{evidence_variable}) out of bounds \
                                   for tensor of size {self.input_tensor.numel()}.")
            evidence_variable_index = evidence_variable - 1
            evidence_variables_indices.append(evidence_variable_index)
        return evidence_variables_indices

    def _build_input_tensors(self, query_variable_index, evidence_variables_indices):
        """
        Builds the input tensors for the numerator and denominator of the conditional probability.
        """
        numerator_input = self.input_tensor.clone()
        denominator_input = self.input_tensor.clone()

        numerator_input[query_variable_index] = 1

        for evidence_variable in evidence_variables_indices:
            numerator_input[evidence_variable] = 1
            denominator_input[evidence_variable] = 1
        
        return numerator_input, denominator_input
    
    def _build_marginalized_variables(self, query_variable_index, evidence_variables_indices):
        """
        Builds the marginalized variables tensor for the denominator of the conditional probability.
        """
        numerator_marginalized_variables = self.marginalized_variables.clone()
        denominator_marginalized_variables = self.marginalized_variables.clone()

        numerator_marginalized_variables[query_variable_index] = 0
        denominator_marginalized_variables[query_variable_index] = 1

        for evidence_variable in evidence_variables_indices:
            numerator_marginalized_variables[evidence_variable] = 0
            denominator_marginalized_variables[evidence_variable] = 0
        
        return numerator_marginalized_variables, denominator_marginalized_variables
    
    def _build_neural_network(self, executor_class, sdd_file):
        """
        Parses the SDD file to construct and store the neural network model.
        """
        nnf_parser = NNFParser()
        nnf_root = nnf_parser.parse(sdd_file)
        if nnf_root is None:
            raise ValueError("NNF root node should not be none.")
        self.neural_network = executor_class(nnf_root)
    
    def _build_inputs(self, json_file):
        """
        Parses the JSON file to create and store the input and marginalized tensors.
        """
        probabilities_parser = ProbabilitiesParser(json_file)
        self.input_tensor = probabilities_parser.input_tensor
        self.marginalized_variables = probabilities_parser.marginalized_variables
