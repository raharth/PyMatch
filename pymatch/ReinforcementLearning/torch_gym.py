import torch
import gym

class TorchGym:

    def __init__(self, env_name, max_episode_length=None):
        self.env = gym.make(env_name)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.max_episode_length = max_episode_length

    def reset(self):
        return torch.tensor(self.env.reset()).float().unsqueeze(0)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = torch.tensor(observation).float().unsqueeze(0)
        return observation, reward, done, info

    def render(self, mode='human', **kwargs):
        self.env.render(mode=mode, **kwargs)
