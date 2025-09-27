from torch import nn

class LiteralNodeModule(nn.Module):
    """Handles literal nodes by selecting the corresponding input feature."""
    def __init__(self, literal_index, input_index, negated=False):
        super().__init__()
        self.literal_index = literal_index # 1-indexed
        self.input_index = input_index # 0-indexed
        self.negated = negated
        self.descendant_variables = set([literal_index])
    
    def __str__(self):
        return f"{'¬' if self.negated else ''}x{self.literal_index}"

    def forward(self, x, memoization_cache = None):
        """
        Forward pass for literal node. Memoization cache is passed only for compatibility with non-literal nodes.
        """
        x = x.unsqueeze(0) if x.dim() == 1 else x
        result = x[:, self.input_index].unsqueeze(1)
        return result.float()
