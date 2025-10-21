import torch
import random

class AlarmDatasetGenerator:
    """
    Generates a dataset for the neural network of the Alarm problem.
    """
    def __init__(self, neural_network):
        """
        Initializes the AlarmDatasetGenerator.

        Args:
            neural_network (nn.Module): The neural network to generate a dataset for (IterativeNN or RecursiveNN).
        """
        self.neural_network = neural_network
    
    def generate_dataset(self, num_samples=500):
        """
        Generates a dataset for the neural network.

        Args:
            num_samples (int): The number of samples to generate.
        
        Returns:
            torch.Tensor: The input tensor for the network (X_train).
        """
        X_train = []
        for i in range(num_samples):
            literal_to_instantiated_value_map = self._get_literal_to_instantiated_value_map()
            self._add_logical_variables_to_map(literal_to_instantiated_value_map)
            input_sample = self._build_input_sample(literal_to_instantiated_value_map)
            X_train.append(input_sample)
        return torch.stack(X_train)
    
    def _get_literal_to_instantiated_value_map(self):
        """
        Returns the mapping of literals to their instantiated values.
        """
        literal_to_instantiated_value_map = {}
        for literal, prob_node in self.neural_network.get_literal_to_prob_node().items():
            literal_probability = prob_node.get_constant()
            literal_to_instantiated_value_map[literal] = 1 if random.random() < literal_probability else 0
        return literal_to_instantiated_value_map
    
    def _add_logical_variables_to_map(self, literal_to_instantiated_value_map):
        """
        Adds the logical variables to the mapping.
        """
        burglary = literal_to_instantiated_value_map[1]
        earthquake = literal_to_instantiated_value_map[2]
        hears_alarm = literal_to_instantiated_value_map[3]
        alarm = 0
        calls = 0

        if burglary or earthquake:
            alarm = 1
        if alarm and hears_alarm:
            calls = 1
        literal_to_instantiated_value_map[4] = alarm
        literal_to_instantiated_value_map[5] = calls
    
    def _build_input_sample(self, literal_to_instantiated_value_map):
        """
        Builds an input sample for the neural network.
        """
        input_sample = torch.ones(self.neural_network.get_num_variables() * 2, dtype=torch.float32)
        rate_of_hidden_prob_variables = 0.8
        lower = 0
        higher = self.neural_network.get_num_variables()
        if random.random() < rate_of_hidden_prob_variables:
            lower = 3
        for i in range(lower, higher):
            input_sample[2 * i] = literal_to_instantiated_value_map[i + 1]
            input_sample[2 * i + 1] = 1 - literal_to_instantiated_value_map[i + 1]
        return input_sample
