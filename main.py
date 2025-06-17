import torch

from src.pasp2cnf.pasp2cnf import pasp2cnf
from src.cnf2nnf.cnf2nnf import cnf2nnf
import src.nnf2nn.parser.nnf as nnf

from src.nnf2nn.trivial_solutions.iterative_neural_network import IterativeNN
from src.nnf2nn.trivial_solutions.recursive_neural_network import RecursiveNN

def small_example_recursive():
    ''' (x1 AND x2) OR (x3 OR x4) '''

    # create the NNF
    nnf_root = create_small_nnf()

    # converting the NNF to a neural network
    nn_model = RecursiveNN(nnf_root, None, None)
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

def small_example_iterative():
    ''' (x1 AND x2) OR (x3 OR x4) '''

    # create the NNF
    nnf_root = create_small_nnf()

    # converting the NNF to a neural network
    nn_model = IterativeNN(nnf_root)
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

def smoke_example_recursive():
    # converting PASP to CNF
    filename, sym2lit = pasp2cnf('examples/smoke.pasp')
    
    # converting CNF to NNF
    # filename = cnf2nnf(filename, 'src/cnf2nnf/c2d_linux')
    filename = 'examples/smoke.pasp.cnf.nnf'

    # storing NNF in memory
    rootId, _, nodeDict, n_vars = nnf.parse(filename)

    # converting NNF to a neural network
    nn_model = RecursiveNN(nodeDict[rootId], sym2lit, n_vars)
    print(nn_model.root)

    nn_input = nn_model.build_input()
    print("\n", nn_input)

def main():
    print("\n=================== recursive small example ===================\n")
    small_example_recursive()
    print("\n=================== iterative small example ===================\n")
    small_example_iterative()
    print()

if __name__ == "__main__":
    main()
