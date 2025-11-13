class BuildNetworkResponse:
    """
    Response object for the build_network method of the NetworkBuilder class.
    """
    def __init__(self, nn_root, num_variables, literal_to_prob_node):
        self.nn_root = nn_root
        self.num_variables = num_variables
        self.literal_to_prob_node = literal_to_prob_node
    
    def get_nn_root(self):
        """
        Returns the root node of the neural network.
        """
        return self.nn_root
    
    def get_num_variables(self):
        """
        Returns the number of variables in the neural network.
        """
        return self.num_variables
    
    def get_literal_to_prob_node(self):
        """
        Returns the mapping of literals to their probability nodes.
        """
        return self.literal_to_prob_node
