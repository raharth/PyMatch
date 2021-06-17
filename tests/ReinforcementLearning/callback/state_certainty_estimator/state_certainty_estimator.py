from pymatch.ReinforcementLearning.callback import StateCertaintyEstimator, MemoryUpdater
import pymatch.ReinforcementLearning.callback as rcb
from pymatch.utils.experiment import Experiment
from pymatch.DeepLearning.ensemble import Ensemble
from pymatch.ReinforcementLearning.memory import Memory, PriorityMemory

root = 'tests/callback/state_certainty_estimator/experiment'

experiment = Experiment(root=root)
factory = experiment.get_factory()
params = experiment.get_params()
params['factory_args']['learner_args']['dump_path'] = root
Model = experiment.get_model_class()
memory = PriorityMemory(**params['memory_args'])
params['factory_args']['learner_args']['memory'] = memory

learner = Ensemble(model_class=Model,
                   trainer_factory=factory,
                   trainer_args=params['factory_args'],
                   n_model=params['n_learner'],
                   callbacks=[StateCertaintyEstimator()])

# learner.start_callbacks()
filler = rcb.MemoryUpdater(1.)
filler(learner.learners[0])

len(memory)
learner.train_loader = memory
len(learner.learners[0].train_loader)

learner.callbacks[0](learner)

print(memory.probability.sum())

import numpy as np
import matplotlib.pyplot as plt
test = np.concatenate([memory.sample_indices(100) for _ in range(10000)])

v, c = np.unique(test, return_counts=True)
c = c / c.sum()

plt.plot(memory.probability)
plt.plot(c)
plt.show()
