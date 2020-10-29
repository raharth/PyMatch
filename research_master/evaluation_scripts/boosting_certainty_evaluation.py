import sys
import pymatch.DeepLearning.hat as hat
from pymatch.ReinforcementLearning.selection_policy import GreedyValueSelection
from pymatch.ReinforcementLearning.torch_gym import TorchGym, CartPole
from pymatch.utils.experiment import Experiment
from pymatch.utils.functional import interactive_python_mode, eval_mode
from pymatch.DeepLearning.ensemble import Ensemble
from pymatch.DeepLearning.pipeline import Pipeline
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

if interactive_python_mode():
    root = 'research_master/DQN/CartPole/boosting/exp_58'
else:
    root = sys.argv[1]

experiment = Experiment(root=root)
factory = experiment.get_factory()
params = experiment.get_params()
params['factory_args']['learner_args']['dump_path'] = root
Model = experiment.get_model_class()

Core = experiment.get_model_class(source_file='core', source_class='Core')
params['factory_args']['core'] = Core(**params['core_args'])

learner = Ensemble(model_class=Model,
                   trainer_factory=factory,
                   trainer_args=params['factory_args'],
                   n_model=params['n_learner']
                   )

learner.load_checkpoint(path=f'{root}/checkpoint', tag='checkpoint')
pipeline = Pipeline(pipes=[learner, hat.EnsembleHatStd()])

env = CartPole(**params['factory_args']['env_args'])
# action_selector = GreedyValueSelection()

episodes_stds = []
episodes_values = []

for i in tqdm(range(100)):
    stds = []
    value = []
    with eval_mode(learner):
        terminate = False
        episode_reward = 0
        observation = env.reset().detach()
        while not terminate:
            # action = action_selector(pipeline, observation)
            actions, std = pipeline(observation)
            action = actions.argmax().item()
            value += [actions.squeeze(0)[action]]
            stds += [std.squeeze(0)[action].item()]
            new_observation, reward, done, _ = env.step(action)
            episode_reward += reward
            observation = new_observation
            terminate = done
            # env.render()
    episodes_values += [value]
    episodes_stds += [stds]

count = 0
for p in episodes_stds:
    if len(p) < 499:
        count += 1
        plt.plot(p[::-1], c='grey', alpha=.1)
# x_ticks = np.arange(0, 501, 100)
# plt.xticks(x_ticks, np.flip(x_ticks))
plt.title('Std on best action on failing attempts')
plt.xlabel('steps before failing')
plt.ylabel('std')
plt.show()
print(count)

for p in episodes_values:
    if len(p) < 499:
        count += 1
        plt.plot(p[::-1], c='grey', alpha=.1)
    else:
        plt.plot(p[::-1], c='blue', alpha=.03)
plt.title('Value estimation')
plt.xlabel('steps before failing')
plt.show()