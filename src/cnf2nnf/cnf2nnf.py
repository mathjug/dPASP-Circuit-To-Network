from subprocess import Popen, PIPE

def cnf2nnf(filename, c2d_executable):
    """ Given a CNF as an input file, writes out a NNF version of it. """
    process = Popen([c2d_executable, "-in", filename, "-dt_method", "4"], stdout=PIPE)
    (poutput, perr) = process.communicate()
    exit_code = process.wait()

    if exit_code != 0:
        if poutput:
            print(poutput.decode("utf-8"))
        if perr:
            print(perr.decode("utf-8"))
        exit(exit_code)
    
    return filename + '.nnf'
