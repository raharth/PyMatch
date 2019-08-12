import torch
import torch.nn as nn

import os
import numpy as np

class ReinforcementLearner:

    def __init__(self, agent, optimizer, env, crit, grad_clip=0., buffer_size=None, checkpoint_root=None, load_checkpoint=False):
        self.agent = agent
        self.optimizer = optimizer
        self.crit = crit
        self.env = env

        self.memory = Memory(env.action_space, buffer_size)

        self.losses = []
        self.rewards = []
        self.episodes_run = 0
        self.best_performance = -torch.tensor(float('inf'))

        self.grad_clip = grad_clip

        if checkpoint_root is None:
            checkpoint_root = './tmp'
        if not os.path.isdir(checkpoint_root):
            os.mkdir(checkpoint_root)

        self.checkpoint_path = checkpoint_root + '/checkpoint.pth'
        self.early_stopping_path = checkpoint_root + '/early_stopping.pth'

        if load_checkpoint:
            self.load_checkpoint(self.checkpoint_path)

    def train(self, episodes, device, checkpoint_int=10, validation_int=10, restore_early_stopping=False):
        self.agent.train()

        for episode in range(episodes):
            print('episode: {}'.format(episode))
            
            self.play_episode()
            self.replay_memory(device)
            self.episodes_run += 1
            
            if episode % checkpoint_int == 0:
                self.dump_checkpoint(self.epochs_run + episode)
            if episode % validation_int == 0:
                performance = self.validate(device=device)
                self.val_losses += [[performance, self.episodes_run]]
                if performance < self.best_performance:
                    self.best_performance = performance
                    self.dump_checkpoint(epoch=self.episodes_run, path=self.early_stopping_path)

        if restore_early_stopping:
            self.load_checkpoint(self.early_stopping_path)
        self.dump_checkpoint(self.epochs_run, self.checkpoint_path)
        return

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
        observation = torch.tensor(self.env.reset())
        episode_reward = 0
        step_counter = 0
        terminate = False

        while not terminate:
            step_counter += 1
            action = self.chose_action(observation)
            new_observation, reward, done, _ = self.env.step(action)

            # if self.reward_transform is not None:
            #     reward = self.reward_transform(reward, new_observation, done)

            episode_reward += reward
            self.memory.memorize(observation, action, reward)
            observation = new_observation
            terminate = done or (episode_length is not None and step_counter >= episode_length)

            if render:
                self.env.render()

        self.rewards += [episode_reward]

        if episode_reward > self.best_performance:
            self.best_performance = episode_reward
            self.dump_checkpoint(self.episodes_run, self.early_stopping_path)

        return episode_reward

    def backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0.:
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.grad_clip)
        self.optimizer.step()
        return

    def chose_action(self, observation):
        raise NotImplementedError

    def dump_checkpoint(self, epoch, path=None):
        if path is None:
            path = self.checkpoint_path
        torch.save({'epoch': epoch,
                    'model_state_dict': self.agent.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': self.losses,
                    'best_performance': self.best_performance,
                    'memory': self.memory
                    },
                   path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.agent.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epochs_run = checkpoint['epoch']
        self.losses = checkpoint['loss']
        self.best_performance = checkpoint['best_performance']
        self.memory = checkpoint['memory']


class Memory:

    def __init__(self, action_space, buffer_size=None):
        """
        Memory class for RL algorithm.

        Args:
            action_space (int):
            buffer_size (int): max buffer size
        """
        self.memory_observations = []
        self.memory_actions = []
        self.memory_rewards = []
        self.action_space = action_space
        self.buffer_size = buffer_size

    def memorize(self, observation, action, reward):
        # @todo data types
        self.memory_observations += [list(observation)]
        a = torch.zeros(self.action_space.n, dtype=torch.int)
        a[action] = 1
        self.memory_actions += [list(a)]
        self.memory_rewards += [reward]

        self._reduce_buffer()
        return

    def memory_reset(self):
        # @todo data types
        self.memory_observations = []
        self.memory_actions = []
        self.memory_rewards = []
        return

    def _reduce_buffer(self):
        if self.buffer_size is not None:
            self.memory_observations = self.memory_observations[-self.buffer_size:]
            self.memory_actions = self.memory_actions[-self.buffer_size:]
            self.memory_rewards = self.memory_rewards[-self.buffer_size:]
        return

    def sample(self, n):
        mask = np.random.choice(range(len(self.memory_actions)), n)
        observation_sample = torch.tensor(self.memory_observations)[mask]
        action_sample = torch.tensor(self.memory_actions)[mask]
        reward_sample = torch.tensor(self.memory_rewards)[mask]
        return observation_sample, action_sample, reward_sample
