from torch import nn

class LiteralNodeModule(nn.Module):
    """Handles literal nodes by selecting the corresponding input feature."""
    def __init__(self, literal_index, negated=False):
        super().__init__()
        self.literal_index = literal_index
        self.negated = negated
    
    def __str__(self):
        return f"{'¬' if self.negated else ''}x{self.literal_index}"

    def forward(self, x, marginalized_variables = None, memoization_cache = None):
        x = x.unsqueeze(0) if x.dim() == 1 else x
        val = x[:, self.literal_index].unsqueeze(1)
        if self._is_marginalized_variable(marginalized_variables):
            # Use val * 0 + 1 to maintain gradient connectivity
            return val * 0 + 1
        return 1 - val if self.negated else val
    
    def _is_marginalized_variable(self, marginalized_variables):
        return (marginalized_variables is not None and
            0 <= self.literal_index < len(marginalized_variables) and
            marginalized_variables[self.literal_index] == 1)
