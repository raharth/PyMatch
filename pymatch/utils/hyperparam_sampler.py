import json
from shutil import copyfile
import numpy as np
import os


class HyperparamSampler:
    def __init__(self, source_root, target_root, n_rand_experiments, prefix='exp_'):
        self.source_root = source_root
        self.target_root = f'{target_root}/{prefix}'
        self.n_rand_experiments = n_rand_experiments

    def sample_values(self, source_dict):
        target_dict = {}
        for k, v in source_dict.items():
            if type(v) == dict:
                target_dict[k] = self.sample_values(v)
            elif type(v) == list:
                target_dict[k] = v[np.random.randint(len(v))]
            else:
                target_dict[k] = v
        return target_dict

    def sample_experiments(self):
        for i in range(self.n_rand_experiments):
            exp_path = f'{self.target_root}_{i}'
            os.mkdir(exp_path)
            for file_name in os.listdir(self.source_root):
                if file_name == 'params.json':
                    with open(f'{self.source_root}/{file_name}', 'r') as f:
                        sample_params = json.load(f)
                    param_sample = self.sample_values(sample_params)
                    with open(f'{exp_path}/params.json', 'w') as f:
                        json.dump(param_sample, f, indent=4)
                else:
                    copyfile(src=f'{self.source_root}/{file_name}', dst=f'{exp_path}/{file_name}')