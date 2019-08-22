# general imports
import numpy as np
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
from torch.distributions import Categorical

# own imports
from ReinforcementLearning.ReinforcementLearner import ReinforcementLearner
from ReinforcementLearning.Loss import REINFORCELoss
from ReinforcementLearning.Memory import Memory


class PolicyGradient(ReinforcementLearner):

    def __init__(self, agent, optimizer, env, crit, grad_clip=0., load_checkpoint=False):
        """

        Args:
            agent (nn.Module): neural network
            optimizer (torch.optim): Optimizer
            env(any): environment to interact with
            crit (any): loss function
        """
        super(PolicyGradient, self).__init__(agent, optimizer, env, crit, grad_clip=grad_clip, load_checkpoint=load_checkpoint)
        self.memory = Memory(['log_prob', 'reward'], buffer_size=None)

    def play_episode(self, episode_length=None, render=False):
        """
        Plays a single episode.
        This might need to be changed when using a non openAI gym environment.

        Args:
            episode_length (int): max length of an episode
            render (bool): render environment

        Returns:
            episode reward
        """
        observation = self.env.reset().detach()
        episode_reward = 0
        step_counter = 0
        terminate = False
        episode_memory = Memory(['log_prob', 'reward'])

        while not terminate:
            step_counter += 1
            action, log_prob = self.chose_action(observation)
            new_observation, reward, done, _ = self.env.step(action)

            episode_reward += reward
            episode_memory.memorize((log_prob, torch.tensor(reward)), ['log_prob', 'reward'])
            observation = new_observation
            terminate = done or (episode_length is not None and step_counter >= episode_length)

            if render:
                self.env.render()
            if done:
                break

        episode_memory.cumul_reward(gamma=self.gamma)
        self.memory.memorize(episode_memory, episode_memory.memory_cell_names)
        self.rewards += [episode_reward]

        if episode_reward > self.best_performance:
            self.best_performance = episode_reward
            self.dump_checkpoint(self.episodes_run, self.early_stopping_path)

        return episode_reward

    def replay_memory(self, device, verbose=1):
        log_prob, reward = self.memory.sample(None)
        log_prob, reward = log_prob.to(device), reward.to(device)
        loss = self.crit(log_prob, reward)
        self.losses += [loss]
        self.backward(loss)
        self.memory.memory_reset()

    def chose_action(self, observation):
        probs = self.agent(observation)
        dist = Categorical(probs.squeeze())
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob
