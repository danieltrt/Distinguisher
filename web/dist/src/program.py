import os
import re
import threading

class CProgram:
    idx = 1
    lock = threading.Lock()

    def __init__(self, raw_string, string, output_type, input_type):
        self.raw_string = raw_string
        self.string = string
        self.output_type = output_type
        self.input_type = input_type
        self.idx = CProgram.idx
        CProgram.idx += 1

    def get_input_type(self):
        return self.input_type

    def call(self, var):
        first_line = self.string.split(os.linesep)[0]
        begin = len(self.output_type) + 1
        end = first_line.find("(")
        call_string = first_line[begin:end]
        call_string += "({});".format(var)
        return call_string

    def __lt__(self, other):
        return self.idx < other.idx

    def __str__(self):
        first_line = self.string.split(os.linesep)[0]
        begin = len(self.output_type) + 1
        end = first_line.find("(")
        return first_line[begin:end]


class UnchartItProgram(CProgram):

    def __init__(self, path=None, f=None, n_cols=10, vars=None):
        CProgram.lock.acquire()
        self.vars = vars
        self.mappings = {"LIS": 3, "OPO": 2, "GVA": 1}
        self.mappings_c = 4
        self.count = 0
        self.n_cols = n_cols
        rx = re.compile(r'\.R$')
        if path is not None:
            with open(path, "r+") as f:
                raw_string = f.read()
                if rx.match(str(f)):
                    string = self.r_to_c(raw_string)
                else:
                    string = raw_string
                super().__init__(raw_string, string, "void", "dataframe")
        elif f is not None:
            raw_string = f.read().decode(encoding='UTF-8')
            if rx.search(str(f)):
                string = self.r_to_c(raw_string)
            else:
                string = raw_string
            super().__init__(raw_string, string, "void", "dataframe")
        CProgram.lock.release()

    def r_to_c(self, r_program):
        idx = CProgram.idx
        header = f'void unchartit_gen{idx}(dataframe *df)' + '{' + os.linesep
        lines = ''
        for line in r_program.split(os.linesep):
            if line != os.linesep:
                lines += ' ' * 4 + self.map_line(line) + os.linesep
        return header + lines + '}'

    def map_line(self, line):

        if line.find('filter') != -1:
            rx = re.compile(r'filter\((.*)\)')
            if line.find('>=') != -1:
                col, val = rx.search(line).group(1).replace(' ', '').split('>=')
                self.check_col(col)
                return f'filter(df, gte, {self.vars[col]}, {val});'
            elif line.find('<=') != -1:
                col, val = rx.search(line).group(1).replace(' ', '').split('<=')
                self.check_col(col)
                return f'filter(df, lte, {self.vars[col]}, {val});'
            elif line.find('==') != -1:
                col, val = rx.search(line).group(1).replace(' ', '').split('==')
                if '"' in val:
                    val = val[1:-1]
                    arg = self.mappings.get(val, None)
                    if None:
                        self.mappings[val] = self.mappings_c
                        self.mappings_c += 1
                    val = self.mappings[val]
                self.check_col(col)
                return f'filter(df, eq, {self.vars[col]}, {val});'
            elif line.find('!=') != -1:
                col, val = rx.search(line).group(1).replace(' ', '').split('!=')
                if '"' in val:
                    val = val[1:-1]
                    arg = self.mappings.get(val, None)
                    if None:
                        self.mappings[val] = self.mappings_c
                        self.mappings_c += 1
                    val = self.mappings[val]
                self.check_col(col)
                return f'filter(df, ne, {self.vars[col]}, {val});'
        elif line.find('mutate_date') != -1:
            rx = re.compile(r'mutate_date\((.*)\)')
            _, expr = rx.search(line).group(1).replace(' ', '').split('=')
            rx = re.compile(r'(.*)\((.*),(.*)\)')
            aggr, col1, col2 = rx.search(expr).groups()
            self.check_col(col1)
            self.check_col(col2)
            return f'mutate_date(df, {aggr}, {self.vars[col1]}, {self.vars[col2]});'
        elif line.find('mutate') != -1:
            rx = re.compile(r'mutate\((.*)\)')
            _, expr = rx.search(line).group(1).replace(' ', '').split('=')
            rx = re.compile(r'(.*)\((.*)\)')
            aggr, col = rx.search(expr).groups()
            self.check_col(col)
            return f'mutate(df, {aggr}, {self.vars[col]});'
        elif line.find('arrange') != -1:
            rx = re.compile(r'arrange\((.*)\)')
            expr = rx.search(line).group(1)
            rx = re.compile(r'desc\((.*)\)')
            if rx.search(expr) is not None:
                arg = rx.search(expr).group(1)
                self.check_col(arg)
                return f'arrange(df, descending, {self.vars[arg]});'
            else:
                self.check_col(expr)
                return f'arrange(df, ascending, {self.vars[expr]});'
        elif line.find('bottom_n') != -1:
            rx = re.compile(r'bottom_n\((.*),(.*)\)')
            n, arg = rx.search(line).groups()
            arg = arg.replace(' ', '')
            n = n.replace(' ', '')
            self.check_col(arg)
            return f'bottom_n(df, {self.vars[arg]}, {n});'
        elif line.find('top_n') != -1:
            rx = re.compile(r'top_n\((.*),(.*)\)')
            n, arg = rx.search(line).groups()
            arg = arg.replace(' ', '')
            n = n.replace(' ', '')
            self.check_col(arg)
            return f'top_n(df, {self.vars[arg]}, {n});'
        elif line.find('count') != -1:
            self.vars['n'] = self.n_cols
            return f'count(df);'
        elif line.find('group_by') != -1:
            rx = re.compile(r'group_by\((.*)\)')
            arg = rx.search(line).group(1).replace(' ', '')
            self.check_col(arg)
            return f'group_by(df, {self.vars[arg]});'
        elif line.find('summarize') != -1:
            rx = re.compile(r'summarize\((.*)\)')
            new_col, expr = rx.search(line).group(1).replace(' ', '').split('=')
            self.vars[new_col] = self.n_cols
            rx = re.compile(r'(.*)\((.*)\)')
            aggr, col = rx.search(expr).groups()
            self.check_col(col)
            return f'summarize(df, {aggr}, {self.vars[col]});'

    def check_col(self, col):
        arg = self.vars.get(col, None)
        if arg is None:
            self.vars[col] = self.count
            self.count += 1

