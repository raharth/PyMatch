import json
from pydoc import locate
from shutil import copyfile
import os
import datetime
import sys
import wandb
from pymatch.DeepLearning.learner import Learner
from pymatch.utils.exception import OverwriteException


class Experiment:
    def __init__(self, root):
        """
        Experiment class, used for documenting experiments in a standardized manner.

        Args:
            root:
        """
        self.root = root
        self.info = {"mode": "interactive" if sys.argv[0] == '' or sys.argv[0].split('\\')[-1] == 'pydevconsole.py'
                                           else "script"}

    def get_params(self, param_source='params.json'):
        """
        Loads parameters from a json file

        Args:
            param_source:   source file

        Returns:
            dictionary of parameters
        """
        with open(f'{self.root}/{param_source}', 'r') as f:
            params = json.load(f)
        return params

    def get_model_class(self, source_file='model', source_class='Model'):
        """
        Loads a model from a python file.

        Args:
            source_file:    source file
            source_class:   model class name

        Returns:
            callable model object
        """
        import_path = self.root.replace('/', '.')
        return locate(f'{import_path}.{source_file}.{source_class}')

    def get_factory(self, source_file='factory', source_function='factory'):
        """
        Loads factory method from python file

        Args:
            source_file:        python source file
            source_function:    factory function name

        Returns:
            callable function
        """
        import_path = self.root.replace('/', '.')
        return locate(f'{import_path}.{source_file}.{source_function}')

    def document_script(self, script_path):
        """
        Saves the training script.

        Args:
            script_path:    path to training script

        Returns:
            None
        """
        if os.path.isfile(f'{self.root}/train_script.py'):
            raise OverwriteException('There is already a stored script. Please remove the script before re-running')
        copyfile(script_path, f'{self.root}/train_script.py')

    def start(self, overwrite=False):
        """
        Starts a new training process, writing basic information.

        Returns:
            None

        """
        if os.path.isfile(f'{self.root}/meta_data.json') and not overwrite:
            raise OverwriteException('This experiment has been already run. Please set `overwrite` to True if you are '
                                     'sure to do so.')
        self.start_time = datetime.datetime.now()
        self.info['start time'] = str(self.start_time)
        self.write_json(self.info)

    def finish(self):
        """
        Finishes a training process, writing basic information.

        Returns:
            None

        """
        self.info["time_taken"] = str(datetime.datetime.now() - self.start_time)
        self.write_json(self.info)

    def write_json(self, data, path='meta_data.json'):
        """
        Writes a dictionary to a json file.

        Args:
            data:   dictionary to dumpy
            path:   path and file name to write it to

        Returns:
            None

        """
        if not os.path.exists(self.root):
            print(f'Creating missing directory: {self.root}')
            os.makedirs(self.root)
        with open(f'{self.root}/{path}', 'w') as json_file:
            json_file.write(json.dumps(data, indent=2))


class WandbExperiment(Experiment):

    def __init__(self, root, param_source):
        """
        Experiment linking to Wandb.

        Args:
            root:               experiment root as for the regular experiment
            wandb_init_args:    wandb.init() arguments

        Returns:

        """
        super(WandbExperiment, self).__init__(root)
        self.params = self.get_params(param_source=param_source)
        wandb.init(**self.params)

    def watch(self, learner: Learner):
        wandb.watch(learner.model)

    def log(self, info):
        wandb.log(info)
