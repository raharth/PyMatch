import torch
import gym


class TorchGym:

    def __init__(self, env_name, post_pipeline=[]):
        """

        Args:
            env_name:       Gym environment name to make
            post_pipeline:  List of pipeline elements that can be used to alter the output of the environment.
                            This can be used e.g. to down-sample states that are images to save memory.
        """
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        # `n_instances` is not used in the default version but necessary to properly run the MultiInstanceEnvironment
        self.n_instances = 1
        self.post_pipeline = post_pipeline

        # self.max_episode_length = max_episode_length
        # @todo this is an unused variably there is a way of registering new env with the gym framework and reduced
        #   number of max episodes, this should be used instead

    def reset(self):
        return torch.tensor(self.env.reset()).float().unsqueeze(0)

    def step(self, action):
        observation, reward, done, info = self.env.step(action.item())

        for pipe in self.post_pipeline:
            observation, reward, done, info = pipe(observation, reward, done, info)

        return torch.tensor(observation).float().unsqueeze(0), \
               torch.tensor(reward).float().unsqueeze(0), \
               torch.tensor(done).unsqueeze(0), \
               info

    def render(self, mode='human', **kwargs):
        self.env.render(mode=mode, **kwargs)

    def close(self):
        self.env.close()


# @todo depricated should not be used anymore
# class CartPole(TorchGym):
#
#     def __init__(self):
#         super().__init__('CartPole-v1')
#         self.steps = 0
#
#     def reset(self):
#         self.steps = 0
#         return torch.tensor(self.env.reset()).float().unsqueeze(0)
#
#     def step(self, action):
#         self.steps += 1
#         observation, reward, done, info = super().step(action.item())
#         if done:  # and self.steps < 500:
#             reward = -10
#         if self.steps == 500:
#             reward = 0
#         return observation, reward, done, info


class MultiInstanceGym:
    def __init__(self, env_name, n_instances):
        """

        Args:
            env_name:       environment name as known by gym
            n_instances:    number of simultaneously run instances
        """
        self.env_name = env_name
        self.n_instances = n_instances
        self.envs = [gym.make(env_name) for _ in range(n_instances)]
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space
        self.done = None

    def reset(self):
        self.done = [False for i in range(self.n_instances)]
        return torch.stack([torch.tensor(env.reset()).float() for env in self.envs])

    def step(self, actions):
        observations, rewards, infos = [], [], []
        done_mask = list(torch.where(torch.tensor(self.done) == False)[0].numpy())
        try:
            for action, i in zip(actions, done_mask):
                if not self.done[i]:
                    observation, reward, done, info = self.envs[i].step(action.item())
                    observation = torch.tensor(observation).float()
                    observations += [observation]
                    rewards += [reward]
                    self.done[i] = done
                    infos += [info]
        except Exception as e:
            raise e
        return torch.stack(observations), \
               torch.tensor(rewards, dtype=torch.double).view(-1, 1), \
               torch.tensor(self.done)[done_mask].view(-1, 1), \
               infos

    def render(self, mode='human', **kwargs):
        for env in self.envs:
            env.render(mode=mode, **kwargs)

    def close(self):
        for env in self.envs:
            env.close()


if __name__ == '__main__':

    import time

    m_env = MultiInstanceGym('CartPole-v1', 2)
    m_env.reset()
    actions = torch.tensor([[0], [1]], dtype=torch.int8)
    obs, r, done, _ = m_env.step(actions)

    m_env.envs[0].step(0)

    action = actions[0]
    env = m_env.envs[0]

    m_env.done[1] = True
    actions = torch.tensor([[0]], dtype=torch.int8)
    obs, r, done, _ = m_env.step(actions)

    m_env = MultiInstanceGym('CartPole-v1', 100)
    actions = torch.zeros((100, 1), dtype=torch.int8)
    t_from = time.time()
    for k in range(1000):
        states = m_env.reset()
        m_env.step(actions)
    print(f'Multi env time: {(time.time() - t_from)/1000}')



    s_env = TorchGym('CartPole-v1')
    t_from = time.time()
    act_ops = [0, 1]
    for k in range(1000):
        state = s_env.reset()
        for i in range(100):
            s_env.step(act_ops[i % 2])
    print(f'Single env time: {(time.time() - t_from) / 1000}')
