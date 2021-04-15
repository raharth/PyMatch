import os
import time
from pydoc import locate
from multiprocessing import Process
import numpy as np


class ExperimentWorkerHandler:
    def __init__(self, experiment_root, source_file, source_function, num_workers, folders=None):
        self.experiment_root = experiment_root
        self.source_file = source_file
        self.source_function = source_function
        self.num_workers = num_workers
        self.processes = self.define_processes(folders)

    def count_active_processes(self):
        return np.array([process.is_alive() for process in self.processes]).sum()

    def get_func(self):
        return locate(f'{self.source_file.replace("/", ".")}.{self.source_function}')

    def define_processes(self, folders=None):
        if folders is None:
            folders = []
            for dir in os.listdir(self.experiment_root):
                sub_dir = f'{self.experiment_root}/{dir}'
                if os.listdir(sub_dir):
                    folders += [sub_dir]
        return [Process(target=self.get_func(), args=(folder, f'{self.source_file}.py')) for folder in folders]

    def start(self):
        print(f'processes: {self.processes}')
        for process in self.processes:
            while self.count_active_processes() >= self.num_workers:
                time.sleep(5)
            process.start()

    def join(self):
        for process in self.processes:
            process.join()
