import torch
from torch import nn

class ConstantNode(nn.Module):
    """
    Implements a constant terminal node. It always returns a tensor of the constant.
    Can handle both fixed constants and learnable parameters with sigmoid transformation.
    """
    def __init__(self, constant: float):
        super().__init__()
        self.constant = constant
        self.descendant_variables = set()
        self._is_learnable = False
    
    def __str__(self):
        return f'CONSTANT {self.get_constant()}'

    def forward(self, x, memoization_cache = None):
        """
        Returns a tensor of the constant with a shape of (batch_size, 1).
        If the constant is learnable, applies sigmoid transformation: 1/(1+e^(-theta))
        
        Args:
            x (torch.Tensor): The input tensor.
        """
        batch_size = 1 if x.dim() == 1 else x.shape[0]

        const_value = self.get_constant()
        if self.is_learnable():
            const_value = torch.sigmoid(const_value)
        else:
            const_value = torch.tensor(const_value, dtype=x.dtype, device=x.device)

        return const_value.expand(batch_size, 1)
    
    def get_constant(self):
        """
        Returns the constant value.
        """
        return self.constant
    
    def set_constant(self, constant):
        """
        Sets the constant value. If constant is a torch.nn.Parameter, 
        marks this node as learnable and enables sigmoid transformation.
        
        Args:
            constant: Either a float value or a torch.nn.Parameter for learnable parameters
        """
        self.constant = constant
        self._is_learnable = isinstance(constant, nn.Parameter)
    
    def is_learnable(self):
        """
        Returns True if the constant is learnable, False otherwise.
        """
        return self._is_learnable
