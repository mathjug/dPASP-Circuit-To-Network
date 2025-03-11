from logic_circuits.logic_circuit import LogicCircuit
from logic_circuits.load_inputs import load_inputs

def main():
    circuit_file = "logic_circuits/examples/example_1/example_circuit_1.json" # (A ∨ B) ∧ ((C ∨ D) ∨ E)
    input_file = "logic_circuits/examples/example_1/input_circuit_1.csv"

    network = LogicCircuit(circuit_file)
    X = load_inputs(input_file, network.circuit["inputs"])

    output = (network(X) > 0.5).bool()
    print("Output:\n", output)

if __name__ == "__main__":
    main()
