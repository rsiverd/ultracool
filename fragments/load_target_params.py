
import ast

param_file = '2m0415.par'
with open(param_file, 'r') as ff:
    pardict = ast.literal_eval(ff.read())

