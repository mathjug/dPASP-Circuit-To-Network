import torch
from collections import defaultdict
from src.network_converter.iterative_neural_network import IterativeNN
from src.network_converter.recursive_neural_network import RecursiveNN

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

def calculate_individual_gradients(sdd_file, json_file, input_tensor, nn_implementation):
    """
    Calculates the gradient of the output w.r.t each input for each sample in a batch.
    
    Returns:
        torch.Tensor: A tensor of the same shape as input_tensor, containing the gradients.
    """
    input_tensor.requires_grad_(True)
    neural_network = nn_implementation(sdd_file, json_file, make_smooth=False)
    output = neural_network.forward(input_tensor)
    
    batch_size = input_tensor.shape[0]
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

def count_unique_nodes(root_node):
    """
    Counts the number of unique nodes in a neural network.
    
    Args:
        root_node: The root node of the neural network
        
    Returns:
        dict: Dictionary with counts of each node type
    """
    node_counts = defaultdict(int)
    visited = set()
    
    def normalize_node_type(node_type_name):
        """Normalize node type names by removing implementation prefixes."""
        if node_type_name.startswith('Iterative') or node_type_name.startswith('Recursive'):
            return node_type_name[9:]
        return node_type_name
    
    def count_nodes_recursive(node):
        if id(node) in visited:
            return
        visited.add(id(node))
        
        node_type = normalize_node_type(type(node).__name__)
        node_counts[node_type] += 1
        
        if hasattr(node, 'children_nodes'):
            for child in node.children_nodes:
                count_nodes_recursive(child)
    
    count_nodes_recursive(root_node)
    return dict(node_counts)
