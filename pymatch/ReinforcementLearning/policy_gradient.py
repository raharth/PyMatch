import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.distributions import Categorical

from tqdm import tqdm
import numpy as np
from pymatch.ReinforcementLearning.memory import Memory
from pymatch.ReinforcementLearning.loss import REINFORCELoss
from pymatch.DeepLearning.learner import Learner, ClassificationLearner


class PolicyGradient(Learner):
    def __init__(self,
                 env,
                 model,
                 optimizer,
                 memory_updater,
                 n_samples,
                 batch_size,
                 crit=REINFORCELoss(),
                 gamma=.9,
                 memory_size=1000,
                 grad_clip=None,
                 load_checkpoint=False,
                 name='',
                 callbacks=None,
                 dump_path='./tmp',
                 device='cpu'):
        """
        Policy Gradient algorithm

        Args:
            model:              neural network
            optimizer:          optimizer to optimize the network with
            crit:               loss function
            memory:             memory
            grad_clip:
            load_checkpoint:
            name:
            callbacks:
            dump_path:
            device:
        """
        super().__init__(model=model,
                         optimizer=optimizer,
                         crit=crit,
                         train_loader=None,
                         grad_clip=grad_clip,
                         load_checkpoint=load_checkpoint,
                         name=name,
                         callbacks=callbacks,
                         dump_path=dump_path,
                         device=device)
        self.memory = Memory(['log_prob', 'reward'], buffer_size=memory_size)
        self.env = env
        self.memory_updater = memory_updater
        self.train_dict['rewards'] = []
        self.gamma = gamma
        self.batch_size = batch_size
        self.n_samples = n_samples

    def fit_epoch(self, device, verbose=1):
        """
        Train a single epoch.

        Args:
            device: device t-o run it on 'cpu' or 'cuda'
            verbose: verbosity of the learning

        Returns:
            current loss
        """
        self.memory_updater(self)
        self.model.train()
        self.model.to(device)

        losses = []

        memory_loader = torch.utils.data.DataLoader(    # @todo this can be moved to the outside with the indices being manipulated during training
            self.memory,
            batch_size=self.batch_size,
            sampler=SubsetRandomSampler(indices=self.memory.sample_indices(n_samples=self.n_samples))
        )

        for batch, (log_prob, reward) in tqdm(enumerate(memory_loader)):
            log_prob, reward = log_prob.to(device), reward.to(device)
            loss = self.crit(log_prob, reward)
            self._backward(loss)
            losses += [loss.item()]
        loss = np.mean(losses)
        self.train_dict['train_losses'] += [loss]
        if verbose == 1:
            print(f'train loss: {loss:.4f} - average reward: {np.mean(self.train_dict["rewards"]):.4f}')
        return loss

    def chose_action(self, observation):
        self.model.to(self.device)
        probs = self.model(observation.to(self.device))
        dist = Categorical(probs.squeeze())
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    # def play_episode(self, episode_length=None, render=False):
    #     """
    #     Plays a single episode.
    #     This might need to be changed when using a non openAI gym environment.
    #
    #     Args:
    #         episode_length (int): max length of an episode
    #         render (bool): render environment
    #
    #     Returns:
    #         episode reward
    #     """
    #     observation = self.env.reset()
    #     episode_reward = 0
    #     step_counter = 0
    #     terminate = False
    #
    #     while not terminate:
    #         step_counter += 1
    #         action, log_prob = self.chose_action(observation)
    #         new_observation, reward, done, _ = self.env.step(action)
    #
    #         episode_reward += reward
    #         self.memory.memorize((log_prob, torch.tensor(reward)), ['log_prob', 'reward'])
    #         observation = new_observation
    #         terminate = done or (episode_length is not None and step_counter >= episode_length)
    #
    #         if render:
    #             self.env.render()
    #         if done:
    #             break
    #
    #     self.memory.cumul_reward(gamma=self.gamma)
    #     self.memory.memorize(self.memory, self.memory.memory_cell_names)
    #     self.rewards += [episode_reward]
    #
    #     if episode_reward > self.best_performance:
    #         self.best_performance = episode_reward
    #         self.dump_checkpoint(self.episodes_run, self.early_stopping_path)
    #
    #     return episode_reward
