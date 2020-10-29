import sys

from pymatch.DeepLearning.hat import EnsembleHat
import pymatch.ReinforcementLearning.callback as cb
from pymatch.ReinforcementLearning.selection_policy import GreedyValueSelection
from pymatch.ReinforcementLearning.torch_gym import TorchGym
from pymatch.utils.experiment import Experiment
from pymatch.utils.functional import interactive_python_mode, eval_mode
from pymatch.DeepLearning.ensemble import Ensemble

if interactive_python_mode():
    root = 'research_master/DQN/CartPole/ensemble/exp_39'
else:
    root = sys.argv[1]

experiment = Experiment(root=root)
factory = experiment.get_factory()
params = experiment.get_params()
params['factory_args']['learner_args']['dump_path'] = root
Model = experiment.get_model_class()

learner = Ensemble(model_class=Model,
                   trainer_factory=factory,
                   trainer_args=params['factory_args'],
                   n_model=params['n_learner'],
                   callbacks=[
                       # cb.EnvironmentEvaluator(
                       #     env=TorchGym(**params['factory_args']['env_args']),
                       #     n_evaluations=10,
                       #     action_selector=GreedyValueSelection(
                       #         post_pipeline=[EnsembleHat()]
                       #     )),
                       # cb.EnsembleRewardPlotter()
                   ])
learner.load_checkpoint(path=f'{root}/checkpoint', tag='checkpoint')

# viz = cb.AgentVisualizer(env=TorchGym(**params['factory_args']['env_args']),
#                          frequency=1,
#                          action_selector=GreedyValueSelection(
#                              post_pipeline=[EnsembleHat()]
#                          ))
# viz(learner)
env = TorchGym(**params['factory_args']['env_args'])
action_selector = GreedyValueSelection(post_pipeline=[EnsembleHat()])

with eval_mode(learner):
    terminate = False
    episode_reward = 0
    observation = env.reset().detach()
    while not terminate:
        action = action_selector(learner, observation)
        new_observation, reward, done, _ = env.step(action)
        episode_reward += reward
        observation = new_observation
        terminate = done
        env.render()
