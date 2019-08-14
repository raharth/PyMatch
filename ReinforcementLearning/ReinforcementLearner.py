import torch
import torch.nn as nn

import os
import numpy as np
from tqdm import tqdm

from ReinforcementLearning.Memory import Memory

class ReinforcementLearner:

    def __init__(self, agent, optimizer, env, crit, grad_clip=0., buffer_size=None, checkpoint_root=None, load_checkpoint=False):
        self.agent = agent
        self.optimizer = optimizer
        self.crit = crit
        self.env = env

        self.memory = Memory(['log_prob', 'reward'], buffer_size)

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

    def train(self, episodes, device, checkpoint_int=10, restore_early_stopping=False, render=False):
        self.agent.train()

        for episode in tqdm(range(episodes)):
            # print('episode: {}'.format(episode))
            
            reward = self.play_episode(render=render)
            self.replay_memory(device)
            self.episodes_run += 1
            
            if episode % checkpoint_int == 0:
                self.dump_checkpoint(self.episodes_run + episode)
            if reward > self.best_performance:
                self.best_performance = reward
                self.dump_checkpoint(epoch=self.episodes_run, path=self.early_stopping_path)

        if restore_early_stopping:
            self.load_checkpoint(self.early_stopping_path)
        self.dump_checkpoint(self.episodes_run, self.checkpoint_path)
        return

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

