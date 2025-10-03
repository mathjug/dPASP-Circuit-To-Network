import torch

class DatasetGenerator:
    """
    Generates a dataset for the neural network.
    """
    def __init__(self, neural_network):
        """
        Initializes the DatasetGenerator.

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
            torch.Tensor: The labels for the network (y_train).
        """
        num_variables = self.neural_network.get_num_variables()
        X_train = torch.randint(0, 2, (num_samples, 2 * num_variables)).float()
        y_train = torch.tensor([self.neural_network.forward(x) for x in X_train]).unsqueeze(-1)
        return X_train, y_train
