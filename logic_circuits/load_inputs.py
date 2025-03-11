import pandas as pd
import torch

def load_inputs(input_file, expected_columns):
    """Loads input values from a CSV file and converts to a PyTorch tensor."""
    df = pd.read_csv(input_file)

    for col in expected_columns:
        if col not in df.columns:
            raise ValueError(f"Missing input column: {col} in {input_file}")

    return torch.tensor(df[expected_columns].values, dtype=torch.float)
