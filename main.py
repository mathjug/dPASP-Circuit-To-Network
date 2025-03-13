from src.pasp2cnf.pasp2cnf import pasp2cnf
from src.nnf2nn.nnf import parse
from src.nnf2nn.nnf_to_neural_network import NNFToNN
from src.cnf2nnf.cnf2nnf import cnf2nnf

def main():
    # PASP --> CNF
    filename, symbols = pasp2cnf('examples/smoke.pasp')
    
    # CNF --> NNF
    # filename = cnf2nnf(filename, 'src/cnf2nnf/c2d_linux')
    filename = 'examples/smoke.pasp.cnf.nnf'

    # NNF --> NNF (in memory)
    rootId, _, nodeDict, nvars = parse(filename)

    # NNF --> NN
    nn_model = NNFToNN(nodeDict[rootId])
    print(nn_model)

if __name__ == "__main__":
    main()
