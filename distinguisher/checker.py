import os
import time
import subprocess



class ModelChecker:

    def generate_symbolic_representation(self, programs, input_constraints):
        pass


class CBMC(ModelChecker):
    """ The CBMC requires a template to generate C.
        The template must implement the is_equiv and not_equiv functions.
        It must also contain a variable named input."""

    var = "VAR"
    positive_assertion = "assert(is_equiv({}));".format(var)
    negative_assertion = "assert(not_equiv({}));".format(var)
    cbmc_positive_assertion = "return_value_is_equiv"
    cbmc_negative_assertion = "return_value_not_equiv"
    cbmc_input_name = "c main::1::input!0@1#1 "
    template_assertions = "ASSERTIONS"

    n_soft_clauses = 1024

    def __init__(self, template):
        self.template = template

    def generate_symbolic_representation(self, programs, input_constraints):
        c_program, equiv_vars = self.template.generate_c(programs, input_constraints)
        c_program = self.add_assertions(c_program, equiv_vars)

        file_n = round(time.time() * 100)
        with open("/tmp/cbmc_{}.c".format(file_n), "a+") as f:
            f.write(c_program)

        cmd = "cbmc /tmp/cbmc_{}.c --dimacs --object-bits 10".format(file_n)
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        lns = str(out, encoding='utf-8')

        n_vars, n_clauses, inc_dimacs = self.get_dimacs(lns.splitlines())
        eq_vars = self.get_eq_vars(lns.splitlines())
        neq_vars = self.get_neq_vars(lns.splitlines())
        input_vars = self.get_input_vars(lns.splitlines())

        return SymbolicRepresentation(n_vars, n_clauses, self.n_soft_clauses, inc_dimacs, eq_vars, neq_vars, input_vars)

    def add_assertions(self, c_program, equiv_vars):
        assertions = ""
        for i in range(len(equiv_vars)):
            assertions += self.positive_assertion.replace(self.var, equiv_vars[i]) + os.linesep
        for i in range(len(equiv_vars)):
            assertions += self.negative_assertion.replace(self.var, equiv_vars[i]) + os.linesep
        return c_program.replace(self.template_assertions, assertions)

    def get_dimacs(self, lines):
        while lines[0].find("p cnf ") == -1:
            lines = lines[1:]
        header = lines[0][len("p cnf "):].split(" ")
        inc_dimacs = list(filter(lambda line: not line.startswith("c "), lines[1:]))
        inc_dimacs = ["{} {}".format(self.n_soft_clauses + 1, el) for el in inc_dimacs]
        inc_dimacs = "{}".format(os.linesep).join(inc_dimacs)
        return int(header[0]), int(header[1]), inc_dimacs

    def get_eq_vars(self, lines):
        eq_vars = []
        for line in lines:
            if line.find(self.cbmc_positive_assertion) != -1:
                line = line[line.find(self.cbmc_positive_assertion):].split(" ")
                if line[2] == "FALSE" or line[2] == "TRUE":
                    eq_vars += [line[1]]
        return sorted(eq_vars, key=int)

    def get_neq_vars(self, lines):
        neq_vars = []
        for line in lines:
            if line.find(self.cbmc_negative_assertion) != -1:
                line = line[line.find(self.cbmc_negative_assertion):].split(" ")
                if line[2] == "FALSE" or line[2] == "TRUE":
                    neq_vars += [line[1]]
        return sorted(neq_vars, key=int)

    def get_input_vars(self, lines):
        inpt_vars = []
        for line in lines:
            if line.find(self.cbmc_input_name) != -1:
                line = line[len(self.cbmc_input_name):].split(" ")
                inpt_vars += line
        return inpt_vars




class SymbolicRepresentation:

    def __init__(self, n_vars, n_clauses, n_soft_clauses, inc_dimacs, eq_vars, neq_vars, input_vars):
        self.n_vars = n_vars
        self.n_clauses = n_clauses
        self.n_soft_clauses = n_soft_clauses
        self.inc_dimacs = inc_dimacs + os.linesep
        self.eq_vars = eq_vars
        self.neq_vars = neq_vars
        self.input_vars = input_vars

    def add_variable(self):
        self.n_vars += 1
        return str(self.n_vars)

    def add_hard_clause(self, variables):
        clause = "{} ".format(self.n_soft_clauses + 1)
        for var in variables:
            clause += "{} ".format(var)
        clause += "0{}".format(os.linesep)

        self.n_clauses += 1
        self.inc_dimacs += clause

    def add_soft_clause(self, weight, variables):
        clause = "{} ".format(weight)
        for var in variables:
            clause += "{} ".format(var)
        clause += "0{}".format(os.linesep)

        self.n_clauses += 1
        self.inc_dimacs += clause

    def get_dimacs(self):
        header = "p wcnf {} {} {}".format(self.n_vars, self.n_clauses, self.n_soft_clauses + 1) + os.linesep
        return header + self.inc_dimacs

    def create_totalizer(self, l, u, p):
        ret = [self.add_variable()]
        ending = [self.add_variable()]
        self.add_hard_clause([ret[0]])
        self.add_hard_clause(['-' + ending[0]])
        if l >= u:
            return ret + [p[l]] + ending

        left_vars = self.create_totalizer(l, l + (u-l)//2, p)  #[q]
        right_vars = self.create_totalizer(l + (u-l)//2 + 1, u, p) #[r]

        for i in range(l, u+1):
            ret += [self.add_variable()]

        ret += ending

        for i in range(len(left_vars)-1):
            for j in range(len(right_vars)-1):
                self.add_hard_clause(['-' + left_vars[i], '-' + right_vars[j], ret[i+j]]) #[p]
                self.add_hard_clause([left_vars[i+1], right_vars[j+1], '-' + ret[i+j+1]]) #[p]

        return ret


