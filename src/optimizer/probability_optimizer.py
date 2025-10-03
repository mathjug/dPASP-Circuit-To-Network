import torch
from torch import nn, optim

class ProbabilityOptimizer:
    """
    Optimizes the probabilities of the neural network.
    """
    def __init__(self, neural_network):
        """
        Initializes the ProbabilityOptimizer.

        Args:
            neural_network (nn.Module): The neural network to optimize (IterativeNN or RecursiveNN).
        """
        self.neural_network = neural_network
        self.literal_to_prob_node = neural_network.get_literal_to_prob_node()

    def learn_probability(self, literal_to_learn, X_train, y_train, learning_rate=0.01, num_epochs=100):
        """
        Learns the probability of a literal in the network.

        Args:
            literal_to_learn (int): The literal to learn the probability of.
            X_train (torch.Tensor): The input tensor for the network.
            y_train (torch.Tensor): The output tensor for the network.
            learning_rate (float): The learning rate for the optimizer.
            num_epochs (int): The number of epochs to train.
        
        Returns:
            dict: Dictionary of literals to their optimized probabilities
            float: The final loss of the training loop.
        """
        self._add_learnable_parameter_to_network(literal_to_learn)
        self._validate_hyperparameters(learning_rate, num_epochs)
        final_loss = self._run_training_loop(X_train, y_train, learning_rate, num_epochs)
        final_learned_probs = self._get_learned_probabilities()
        return final_learned_probs, final_loss
    
    def _add_learnable_parameter_to_network(self, literal):
        """
        Adds a learnable parameter to the network for a given literal.
        """
        if literal not in self.literal_to_prob_node:
            raise ValueError(f"Literal {literal} not found in the network as a probabilistic node")
        prob_node = self.literal_to_prob_node[literal]
        learnable_parameter = nn.Parameter(torch.tensor(0.0))
        prob_node.set_constant(learnable_parameter)
    
    def _validate_hyperparameters(self, learning_rate, num_epochs):
        """
        Validates the hyperparameters for the training loop.
        """
        if learning_rate <= 0:
            raise ValueError(f"Learning rate must be positive, got {learning_rate}")
        if num_epochs <= 0:
            raise ValueError(f"Number of epochs must be positive, got {num_epochs}")

    def _run_training_loop(self, X_train, y_train, learning_rate, num_epochs):
        """
        Runs the training loop for the network.
        """
        loss_function = nn.MSELoss() # Mean Squared Error Loss
        optimizer = optim.SGD(self._get_learnable_parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            outputs = self.neural_network.forward(X_train)
            loss = loss_function(outputs, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self._print_progress(epoch, num_epochs, loss.item())
        return loss.item()
    
    def _get_learnable_parameters(self):
        """
        Collects all learnable parameters from probability nodes in the network.
        
        Returns:
            list: List of torch.nn.Parameter objects that are learnable
        """
        learnable_params = []
        for prob_node in self.literal_to_prob_node.values():
            if prob_node.is_learnable():
                learnable_params.append(prob_node.get_constant())
        return learnable_params
    
    def _get_learned_probabilities(self):
        """
        Collects all learned probabilities from probability nodes in the network.
        """
        learned_probs = {}
        for literal, prob_node in self.literal_to_prob_node.items():
            if prob_node.is_learnable():
                learned_probs[literal] = torch.sigmoid(prob_node.get_constant())
        return learned_probs
    
    def _print_progress(self, epoch, num_epochs, loss):
        """
        Prints the progress of the training loop every 10% of epochs.
        """
        progress_interval = max(1, num_epochs // 10)
        if (epoch + 1) % progress_interval == 0:
            percentage = int((epoch + 1) / num_epochs * 100)
            print(f'[{percentage}%] Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.6f}')
