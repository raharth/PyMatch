import torch
import torch.nn as nn
from gym.envs.classic_control import rendering


class Environment:
    def __init__(self, action_space, observation_space, max_length):
        self.action_space = action_space
        self.observation_space = observation_space
        self.max_length = max_length
        self.steps = 0

        # viewer parameters
        self.viewer = None
        self.field_size = 20
        self.agent_geometry = None

    def step(self, action):
        raise NotImplementedError
        return observation, reward, done, info

    def reset(self):
        raise NotImplementedError
        return observation

    def render(self):
        raise NotImplementedError


class LavaWorld(Environment):
    def __init__(self, size, horizon, p=.15, max_length=100, randomized=False, wall=False, rgb=False):
        super().__init__(4, observation_space=(horizon, horizon), max_length=max_length)
        self.size = size
        self.horizon = horizon
        self.p = p
        self.position = None
        self.map = None
        self.goal = None
        self.screen_height, self.screen_width = None, None
        self.randomized = randomized
        self.wall = wall
        self.rgb = rgb
        self.action_map = {0: torch.tensor([-1, 0]),
                           1: torch.tensor([0, 1]),
                           2: torch.tensor([1, 0]),
                           3: torch.tensor([0, -1])}
        self.color_map = {-1: (.5, 0, 0), 0: (1, 1, 1), 1: (1, 1, 0)}

    def step(self, action):
        """
        Take a step in the environment

        Args:
            action: 0: up, 1: right, 2: down, 3: left

        Returns:
            observation, reward, done, info

        """
        self.steps += 1
        self.position = self.update_position(action)
        return self.observe(), self.reward(), self.done(), ''

    def done(self):
        return (self.steps >= self.max_length) or (self.map[tuple(self.position)].item() in [-1, 1])

    def observe(self):
        """
        Make the observation.

        Returns:
            observation made by the agent

        """
        if self.rgb:
            observation = self.world[:,
                          self.position[0] - self.horizon: self.position[0] + self.horizon + 1,
                          self.position[1] - self.horizon: self.position[1] + self.horizon + 1]
        else:
            observation = self.world[
                          self.position[0] - self.horizon: self.position[0] + self.horizon + 1,
                          self.position[1] - self.horizon: self.position[1] + self.horizon + 1]
        return observation

    def update_position(self, action):
        """
        Update the position by an action.

        Args:
            action:

        Returns:

        """
        position = self.position + self.action_map[action]
        if self.wall:
            position = torch.clamp(position, min=self.horizon)
            position[0] = min(position[0], self.size[0] + self.horizon)
            position[1] = min(position[1], self.size[1] + self.horizon)
        return position

    def reset(self):
        """
        Resets the env to the initial state

        Returns:

        """
        self.steps = 0
        if self.map is None or self.randomized:
            self.init_world(self.size)

        if self.viewer is not None:
            if self.randomized:
                self.viewer.close()
                self.viewer = None
            else:
                self.agent_geometry.set_color(.0, .5, .0)
        self.init_position()
        # self.init_goal()

        return self.observe()

    def reward(self):
        """
        Computes the reward for the current state

        Returns:

        """
        # if self.world[tuple(self.position)] == -1:
        if self.done():
            return -1.
        return (1. - torch.norm((self.position - self.goal).type(torch.float), 1)
                / torch.norm(torch.tensor([20, 20.]), 1)).item()

    def init_goal(self):
        """
        Initialize goal state
        Returns:

        """
        self.goal = torch.tensor(self.size) + self.horizon - 1
        self.map[tuple(self.goal)] = 1.  # self.color_map[1.] if self.rgb else 1.

    def init_position(self):
        """
        Initialize position.

        Returns:
            None
        """
        self.position = torch.tensor([self.horizon, self.horizon])

    def init_world(self, size):
        """
        Initialize world.

        Args:
            size:

        Returns:
            None
        """
        world = self.create_map(size)
        self.map = nn.functional.pad(world, [self.horizon] * 4, value=-1.)
        self.init_goal()
        if self.rgb:
            self.world = self.colorize_map()
        else:
            self.world = self.map

    def create_map(self, size):
        world = torch.bernoulli(torch.ones(size) * self.p) * (-1)
        world[:3, :3] = 0.
        return world

    def render(self, mode='human'):
        if self.viewer is None:
            self.screen_height, self.screen_width = torch.tensor(self.map.shape) * self.field_size
            self.viewer = rendering.Viewer(self.screen_width, self.screen_height)

            for i, row in enumerate(self.map):
                for j, v in enumerate(row):
                    # color = v if self.rgb else self.color_map[v.item()]
                    self.add_element2viewer((i, j), color=self.color_map[v.item()])

            agent = self.add_element2viewer(self.position, color=(0, .5, 0))
            self.agent_geometry = agent

        if self.map is None:
            return None

        l, b, t, r = self.map_index2position(self.position)
        self.agent_geometry.v = [(l, b), (l, t), (r, t), (r, b)]
        if self.reward() == -1.:
            self.agent_geometry.set_color(0, 0, 0)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def map_index2position(self, index):
        t = self.screen_height - index[0] * self.field_size
        l = index[1] * self.field_size
        b = t - self.field_size
        r = l + self.field_size
        return l, b, t, r

    def add_element2viewer(self, index, color):
        l, b, t, r = self.map_index2position(index)
        element = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        element.set_color(*color)
        self.viewer.add_geom(element)
        return element

    def close(self):
        self.viewer.close()

    def colorize_map(self):
        c_world = torch.stack([torch.zeros(size=self.map.shape)] * 3, 0)
        for i, row in enumerate(self.map):
            for j, v in enumerate(row):
                c_world[:, i, j] = torch.tensor(self.color_map[v.item()])
        return c_world


class LavaWorld1(LavaWorld):
    def __init__(self, horizon):
        super().__init__(size=(15, 15), horizon=horizon)

    def create_map(self, size):
        """
        Initialize world.

        Args:
            size:

        Returns:
            None
        """
        world = torch.tensor([[0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                              [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0.],
                              [0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0.],
                              [1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0.],
                              [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0.],
                              [0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 1., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 1.],
                              [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                              [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]) * -1.
        return world

#
# import time
#
# #
# env = LavaWorld(horizon=2, size=(7, 8), randomized=False, rgb=True)
# env.reset()
# env.colorize_map()
# import matplotlib.pyplot as plt
# #
# # env.colorize_map()
# #
# plt.imshow(env.colorize_map().permute(1, 2, 0))
# plt.show()
#
# # obs = env.reset()
# #
# for i in range(5):
#     print(i)
#     done = False
#     env.reset()
#     env.render()
#     time.sleep(.5)
#     while not done:
#         obs, r, done, info = env.step(torch.randint(4, size=(1,)).item())
#         env.render()
#         time.sleep(.1)
#     time.sleep(1.)

# torch.stack([torch.zeros(size=(5, 5))] * 3).view(-1)
#
# torch.zeros(size=(5, 5)).unsqueeze(0).repeat(1)
# torch.zeros(size=(5, 5)).unsqueeze(0).shape
