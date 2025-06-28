import torch
from src.trivial_solutions.iterative_neural_network import IterativeNN
from src.trivial_solutions.recursive_neural_network import RecursiveNN

implementations = [
    {
        "name": "Recursive",
        "implementation_class": RecursiveNN,
    },
    {
        "name": "Iterative",
        "implementation_class": IterativeNN,
    }
]

def calculate_individual_gradients(root_node, input_tensor, executor_class, marginalized_variables = None):
    """
    Calculates the gradient of the output w.r.t each input for each sample in a batch.
    
    Returns:
        torch.Tensor: A tensor of the same shape as input_tensor, containing the gradients.
    """
    input_tensor.requires_grad_(True)
    neural_network = executor_class(root_node)
    output = neural_network.forward(input_tensor, marginalized_variables)
    
    batch_size = input_tensor.shape[0]
    num_features = input_tensor.shape[1]
    all_grads = torch.zeros_like(input_tensor)

    for i in range(batch_size):
        # Clear previous gradients before calculating the new one
        if input_tensor.grad is not None:
            input_tensor.grad.zero_()
        output[i].backward(retain_graph=True)        
        all_grads[i, :] = input_tensor.grad[i, :]
        
    input_tensor.grad.zero_()
    input_tensor.requires_grad_(False)    
    return all_grads
