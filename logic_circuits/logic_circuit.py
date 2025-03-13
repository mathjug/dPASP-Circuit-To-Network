import torch
import torch.nn as nn
import json
from load_inputs import load_inputs

class ORNode(nn.Module):
    """Implements an OR gate using max operation."""
    def forward(self, x):
        return torch.max(x, dim=1, keepdim=True)[0]

class ANDNode(nn.Module):
    """Implements an AND gate using element-wise multiplication."""
    def forward(self, x):
        return torch.prod(x, dim=1, keepdim=True)

class LogicCircuit(nn.Module):
    """Dynamically builds and evaluates a logic circuit from a JSON definition."""
    def __init__(self, circuit_file):
        super(LogicCircuit, self).__init__()

        with open(circuit_file, 'r') as f:
            self.circuit = json.load(f)

        self.nodes = nn.ModuleDict()

        for gate in self.circuit["gates"]:
            if gate["type"] == "OR":
                self.nodes[gate["id"]] = ORNode()
            elif gate["type"] == "AND":
                self.nodes[gate["id"]] = ANDNode()

    def forward(self, inputs):
        """Computes the output of the logic circuit dynamically."""
        values = {name: inputs[:, i].unsqueeze(1) for i, name in enumerate(self.circuit["inputs"])}

        for gate in self.circuit["gates"]:
            gate_inputs = torch.cat([values[inp] for inp in gate["inputs"]], dim=1)
            values[gate["id"]] = self.nodes[gate["id"]](gate_inputs)

        return values[self.circuit["output"]]

def main():
    circuit_file = "examples/example_1/example_circuit_1.json"
    circuit = LogicCircuit(circuit_file)
    input_file = "examples/example_1/input_circuit_1.csv"
    input_circuit = load_inputs(input_file, circuit.circuit["inputs"])
    output = circuit(input_circuit)
    print(output)

if __name__ == "__main__":
    main()
