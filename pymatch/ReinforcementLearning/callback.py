import torch
from pymatch.DeepLearning.callback import Callback
from pymatch.ReinforcementLearning.memory import Memory


class MemoryUpdater(Callback):
    def __init__(self, memory_refresh_rate, update_frequ=1):
        super().__init__()
        if not 0. <= memory_refresh_rate <= 1.:
            raise ValueError(f'memory_refresh_rate was set to {memory_refresh_rate} but has to be in ]0., 1.]')
        self.memory_refresh_rate = memory_refresh_rate
        self.update_frequ = update_frequ

    def __call__(self, agent):
        if agent.train_dict['epochs_run'] % self.update_frequ == 0:
            reduce_to = int(len(agent.memory) * (1 - self.memory_refresh_rate))
            self.memory._reduce_buffer(reduce_to)
            self.fill_memory()

    def fill_memory(self, agent):
        while len(agent.memory) < agent.memory.buffer_size:
            game = self.play_episode(agent=agent)
            agent.memory.memorize(game)
        agent.memory._reduce_buffer()

    def play_episode(self, agent):
        observation = agent.env.reset().detach()
        episode_reward = 0
        step_counter = 0
        terminate = False
        episode_memory = Memory(['log_prob', 'reward'])

        while not terminate:
            step_counter += 1
            action, log_prob = agent.chose_action(observation)
            new_observation, reward, done, _ = self.env.step(action)

            episode_reward += reward
            episode_memory.memorize((log_prob, torch.tensor(reward)), ['log_prob', 'reward'])
            observation = new_observation
            terminate = done or (self.max_episode_length is not None and step_counter >= self.max_episode_length)

            # self.env.render()
            if done:
                break

        episode_memory.cumul_reward(gamma=self.gamma)
        agent.memory.memorize(episode_memory, episode_memory.memory_cell_names)
        agent.train_dict['rewards'] = agent.train_dict.get('rewards', []) + [episode_reward]
        return episode_reward
