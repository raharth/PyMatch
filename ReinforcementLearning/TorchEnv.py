import torch
import gym

class TorchEnv:

    def __init__(self, env_name):
        self.env = gym.make(env_name)

    def reset(self):
        return torch.tensor(self.env.reset())

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = torch.tensor(observation)
        return observation, reward, done, info

    def render(self, mode='human', **kwargs):
        self.env.render(mode=mode, **kwargs)

    def action_space(self):
        return self.env.action_space

    def observation_space(self):
        return self.env.observation_space