from src.pasp2cnf.program import Program

def pasp2cnf(filename):
    """ Given an ASP (or annotated ASP) program, outputs a CNF form of it with a dict of symbols (clauses in the program) """
    program_str = ''
    with open(filename) as infile:
        program_str = infile.read()
    database_str = ''
    program = Program(program_str, database_str)
    if program.grounded_program.check_tightness():
        cnf = program.clark_completion()
        str_cnf = str(cnf).replace("w", "c")
        filename += ".cnf"
        with open(filename, "w") as outfile:
            outfile.write(str_cnf)

    return filename, program.grounded_program.symbol2literal