from distinguisher.model import *
from distinguisher.solver import *
from distinguisher.distinguisher import *
from distinguisher.interpreter import *
from distinguisher.program import *
from distinguisher.template import *
from argparse import *
from os import listdir
import json


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-d", "--details", type=str)
    arg_parser.add_argument("-s", "--solver", type=str)
    cmd_args = arg_parser.parse_args()

    with open(cmd_args.details) as f:
        data = json.load(f)

    # UnchartItSpecific
    programs = []
    programs_paths = [data['programs'] + f for f in listdir(data['programs'])]
    for program_path in programs_paths:
        programs += [UnchartItProgram(program_path)]
    template = UnchartItTemplate(data["template"])
    interpreter = UnchartItInterpreter()

    # Generic, but must comply with the specification
    model_checker = CBMC(template)
    solver = Solver(cmd_args.solver, template)
    interaction_model = OptionsInteractionModel(model_checker, solver, interpreter)

    dst = Distinguisher(interaction_model, programs, data['input_constraints'])
    dst.distinguish()