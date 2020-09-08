import torch
import torch.nn as nn
import torch.tensor as tt

from pymatch.ReinforcementLearning.reinforcement_learner import ReinforcementLearner
from pymatch.ReinforcementLearning.memory import Memory


class Q_Learner(ReinforcementLearner):

    def __init__(self, agent, optimizer, env, selection_policy, grad_clip=0., load_checkpoint=False):
        crit = nn.L1Loss()
        super(Q_Learner, self).__init__(agent, optimizer, env, crit, grad_clip=grad_clip, load_checkpoint=load_checkpoint)
        self.memory = Memory(['state', 'action', 'reward', 'next_state'], [4, 1, 1, 4], buffer_size=10000)
        self.selection_policy = selection_policy

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

        while not terminate:
            step_counter += 1
            action = self.chose_action(observation)
            new_observation, reward, done, _ = self.env.step(action)
            self.memory.memorize((observation.squeeze(), tt(action), tt(reward), new_observation.squeeze()), ['state', 'action', 'reward', 'next_state'])

            episode_reward += reward
            observation = new_observation
            terminate = done or (episode_length is not None and step_counter >= episode_length)

            if render:
                self.env.render()
            if done:
                break

        self.rewards += [episode_reward]

        if episode_reward > self.best_performance:
            self.best_performance = episode_reward
            self.dump_checkpoint(self.episodes_run, self.early_stopping_path)

        return episode_reward

    def replay_memory(self, device, batch_size=128, verbose=1):
        if batch_size > self.memory.get_size():
            return
        state, action, reward, next_state = self.memory.sample(batch_size)
        state, reward, next_state = state.to(device), reward.to(device), next_state.to(device)
        prediction = self.agent(state)
        with torch.no_grad():
            self.agent.eval()
            max_next = self.agent(next_state).max(dim=1)[0]
        target = prediction.clone().detach()

        # batch TD
        for t, a, r, m in zip(target, action, reward, max_next):
            t[a] += 1. * (r + self.gamma * m - t[a])

        loss = self.crit(prediction, target)
        self.losses += [loss.item()]
        self.backward(loss)

    def chose_action(self, observation):
        q_values = self.agent(observation)
        action = self.selection_policy.choose(q_values)
        return action.item()
