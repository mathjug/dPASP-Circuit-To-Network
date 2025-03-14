from src.pasp2cnf.pasp2cnf import pasp2cnf
import src.nnf2nn.nnf as nnf
from src.nnf2nn.nnf_to_neural_network import NNFToNN
from src.cnf2nnf.cnf2nnf import cnf2nnf
import torch

def small_example():
    ''' (x1 AND x2) OR (x3 OR x4) '''

    # create the NNF
    x1 = nnf.LiteralNode("1", literal=1)
    x2 = nnf.LiteralNode("2", literal=2)
    x3 = nnf.LiteralNode("3", literal=3)
    x4 = nnf.LiteralNode("4", literal=4)
    and_node = nnf.AndNode("5", [x1, x2])
    or_node = nnf.OrNode("6", [x3, x4])
    root = nnf.OrNode("7", [and_node, or_node])

    # converting the NNF to a neural network
    nn_model = NNFToNN(root, None, None)
    print(nn_model)
    print("\nFormula: " + str(nn_model.model))

    # querying the neural network
    test_input = torch.tensor([ [1, 0, 0, 0],
                                [1, 1, 0, 0],
                                [0, 0, 0, 1],
                                [0, 0, 1, 0],
                                [0, 1, 0, 0],])
    output = nn_model(test_input)
    print("\nNeural Network Output:\n", output)

def smoke_example():
    # converting PASP to CNF
    filename, sym2lit = pasp2cnf('examples/smoke.pasp')
    
    # converting CNF to NNF
    # filename = cnf2nnf(filename, 'src/cnf2nnf/c2d_linux')
    filename = 'examples/smoke.pasp.cnf.nnf'

    # storing NNF in memory
    rootId, _, nodeDict, n_vars = nnf.parse(filename)

    # converting NNF to a neural network
    nn_model = NNFToNN(nodeDict[rootId], sym2lit, n_vars)
    print(nn_model.model)

    nn_input = nn_model.build_input()
    print("\n", nn_input)

def main():
    print("\n=================== small example ===================\n")
    small_example()
    print("\n=================== smoke.pasp example ===================\n")
    smoke_example()
    print()

if __name__ == "__main__":
    main()
