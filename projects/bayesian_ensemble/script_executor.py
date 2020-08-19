import os
import sys


script = sys.argv[1]
args = sys.argv[2:]
# script = 'projects/bayesian_ensemble/fit_bayesian_ensemble.py'
# args = [f'projects/bayesian_ensemble/experiments/{arg}' for arg in ['exp_test']]

for arg in args:
    os.system(f'python {script} {arg}')