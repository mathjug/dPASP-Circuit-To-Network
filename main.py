import torch

import src.parser.nnf_parser as nnf

from src.trivial_solutions.iterative_neural_network import IterativeNN
from src.trivial_solutions.recursive_neural_network import RecursiveNN
from src.queries.query_executor import QueryExecutor

def small_example(nn_implementation):
    ''' (x1 AND x2) OR (x3 OR x4) '''

    # create the NNF
    nnf_root = create_small_nnf()

    # converting the NNF to a neural network
    nn_model = nn_implementation(nnf_root)
    print(nn_model)
    print("\nFormula: " + str(nn_model.root))

    # querying the neural network
    test_input = torch.tensor([ [1, 0, 0, 0],
                                [1, 1, 0, 0],
                                [0, 0, 0, 1],
                                [0, 0, 1, 0],
                                [0, 1, 0, 0],])
    output = nn_model(test_input)
    print("\nNeural Network Output:\n", output)

def create_small_nnf():
    x1 = nnf.LiteralNode("1", literal=1)
    x2 = nnf.LiteralNode("2", literal=2)
    x3 = nnf.LiteralNode("3", literal=3)
    x4 = nnf.LiteralNode("4", literal=4)
    and_node = nnf.AndNode("5", [x1, x2])
    or_node = nnf.OrNode("6", [x3, x4])
    root = nnf.OrNode("7", [and_node, or_node])
    return root

def alarm_example(nn_implementation):
    sdd_file = "examples/alarm/alarm_balanced.sdd"
    json_file = "examples/alarm/alarm.json"
    query_executor = QueryExecutor(nn_implementation, sdd_file, json_file)
    query_variable = 5
    conditional_prob = query_executor.execute_query(query_variable)
    print(f"\nP(calls(john) | evidence): {conditional_prob}\n")

def main():
    print("\n=================== recursive small example ===================\n")
    small_example(RecursiveNN)
    print("\n=================== iterative small example ===================\n")
    small_example(IterativeNN)
    print("\n======================== alarm example ========================\n")
    alarm_example(IterativeNN)

if __name__ == "__main__":
    main()
