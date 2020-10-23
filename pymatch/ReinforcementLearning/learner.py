import torch
from tqdm import tqdm
import numpy as np
from pymatch.ReinforcementLearning.memory import Memory
from pymatch.ReinforcementLearning.loss import REINFORCELoss
from pymatch.DeepLearning.learner import Learner
from pymatch.utils.functional import one_hot_encoding, eval_mode
import pymatch.ReinforcementLearning.selection_policy as sp
import copy

import pymatch.DeepLearning.hat as hat
from pymatch.DeepLearning.pipeline import Pipeline
from torch.utils.data.sampler import SubsetRandomSampler
from torch.distributions import Categorical, MultivariateNormal
import torch.nn.functional as F


class ReinforcementLearner(Learner):
    def __init__(self,
                 model,
                 optimizer,
                 crit,
                 memory,
                 env,
                 memory_updater,
                 action_selector,
                 gamma,
                 grad_clip=None,
                 load_checkpoint=False,
                 name='',
                 callbacks=None,
                 dump_path='./tmp',
                 device='cpu'
                 ):
        """
        Abstract class for Reinforcement Learners

        Args:
            model:              neural network
            optimizer:          optimizer
            crit:               loss function
            memory:             memory to store and load the memory
            env:                environment to interact with
            memory_updater:     memory updater, also implementing the update policy
            action_selector:    action selection strategy
            gamma:              discount factor for reward over time
            grad_clip:          gradient clipping
            load_checkpoint:    bool, if a checkpoint should be loaded
            name:               name of the model
            callbacks:          list of callbacks to use
            dump_path:          dump path for the model and callbacks
            device:             device to run the model on
        """
        super().__init__(model,
                         optimizer,
                         crit,
                         train_loader=memory,
                         grad_clip=grad_clip,
                         load_checkpoint=load_checkpoint,
                         name=name,
                         callbacks=callbacks,
                         dump_path=dump_path,
                         device=device
                         )

        self.env = env
        self.memory_updater = memory_updater
        self.train_dict['rewards'] = []
        self.gamma = gamma
        self.chose_action = action_selector

    def fit_epoch(self, device, verbose=1):
        raise NotImplementedError

    def play_episode(self):
        raise NotImplementedError


class PolicyGradient(ReinforcementLearner):
    def __init__(self,
                 env,
                 model,
                 optimizer,
                 memory_updater,
                 n_samples,
                 batch_size,
                 crit=REINFORCELoss(),
                 action_selector=sp.PolicyGradientActionSelection(),
                 memory_size=1000,
                 gamma=.95,
                 grad_clip=None,
                 load_checkpoint=False,
                 name='',
                 callbacks=None,
                 dump_path='./tmp',
                 device='cpu'):
        """
        Policy Gradient learner.

        Args:
            env:                environment to interact with
            model:              neural network
            optimizer:          optimizer
            memory_updater:     memory updater, also implementing the update policy
            n_samples:          number samples to sample for each update
            batch_size:         batch size for updates
            crit:               loss function
            action_selector:    action selection strategy
            memory_size:        memory size, storing passed memories
            gamma:              discount factor for rewards over time
            grad_clip:          gradient clipping
            load_checkpoint:    bool, if checkpoint should be loaded
            name:               name of the agent
            callbacks:          list of callbacks to use during training
            dump_path:          dump path for the model and the callbacks
            device:             device to run the model on
        """
        super().__init__(model=model,
                         optimizer=optimizer,
                         crit=crit,
                         env=env,
                         gamma=gamma,
                         memory=Memory(['log_prob', 'reward'],
                                       buffer_size=memory_size,
                                       n_samples=n_samples,
                                       gamma=gamma,
                                       batch_size=batch_size),
                         memory_updater=memory_updater,
                         action_selector=action_selector,
                         grad_clip=grad_clip,
                         load_checkpoint=load_checkpoint,
                         name=name,
                         callbacks=callbacks,
                         dump_path=dump_path,
                         device=device)

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

        for batch, (log_prob, reward) in tqdm(enumerate(self.train_loader)):
            log_prob, reward = log_prob.to(device), reward.to(device)
            loss = self.crit(log_prob, reward)
            self._backward(loss)
            losses += [loss.item()]
        loss = np.mean(losses)
        self.train_dict['train_losses'] += [loss]
        # self.train_dict['epochs_run'] += 1
        if verbose == 1:
            print(f'epoch: {self.train_dict["epochs_run"]}\t'
                  f'average reward: {np.mean(self.train_dict["rewards"]):.2f}\t'
                  f'latest average reward: {self.train_dict["avg_reward"][-1]:.2f}')
        return loss

    def play_episode(self, render=False):
        """
        Plays a single episode.
        This might need to be changed when using a non openAI gym environment.

        Args:
            render (bool): render environment

        Returns:
            episode reward
        """
        observation = self.env.reset().detach()
        episode_reward = 0
        step_counter = 0
        terminate = False
        episode_memory = Memory(['log_prob', 'reward'], gamma=self.gamma)

        while not terminate:
            step_counter += 1
            action, log_prob = self.chose_action(self, observation)
            new_observation, reward, done, _ = self.env.step(action)

            episode_reward += reward
            episode_memory.memorize((log_prob, torch.tensor(reward).float()), ['log_prob', 'reward'])
            observation = new_observation
            terminate = done or (self.env.max_episode_length is not None
                                 and step_counter >= self.env.max_episode_length)
            if render:
                self.env.render()
            # if done:
            #     break

        episode_memory.cumul_reward()
        self.train_loader.memorize(episode_memory, episode_memory.memory_cell_names)
        self.train_dict['rewards'] = self.train_dict.get('rewards', []) + [episode_reward]

        if episode_reward > self.train_dict.get('best_performance', -np.inf):
            self.train_dict['best_performance'] = episode_reward

        return episode_reward

    def _backward(self, loss):
        """
        Backward pass for the model, also performing a grad clip if defined for the learner.

        Args:
            loss: loss the backward pass is based on

        Returns:
            None

        """
        self.optimizer.zero_grad()
        loss.clone().backward(retain_graph=True)
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()


class QLearner(ReinforcementLearner):
    def __init__(self,
                 model,
                 optimizer,
                 crit,
                 env,
                 memory_updater,
                 action_selector,
                 alpha,
                 gamma,
                 memory_size=None,
                 n_samples=None,
                 batch_size=None,
                 memory=None,
                 grad_clip=None,
                 load_checkpoint=False,
                 name='q_learner',
                 callbacks=[],
                 dump_path='./tmp',
                 device='cpu'):
        """
        Deep Q-Learning algorithm, as introduced by http://arxiv.org/abs/1312.5602
        Args:
            model:              pytorch graph derived from torch.nn.Module
            optimizer:          optimizer
            crit:               loss function
            env:                environment to interact with
            memory_updater:     object that iteratively updates the memory
            action_selector:    policy after which actions are selected, it has to be a stochastic one to be used in
                                learning
            alpha:              TD-learning rate @todo this might be dropped in a future implementation
            gamma:              disount factor for future rewards
            memory_size:        size of the replay memory (number of memories to be hold)
            n_samples:          number of samples to be drawn from the memory each update
            batch_size:         batch size when updating
            memory:             alternatively the memory can be explicitly specified, instead by (memory_size,
                                n_samples, batch_size)
            grad_clip:          gradient_clipping
            load_checkpoint:    loading previous checkpoint @todo might be dropped in the future
            name:               name for the learner
            callbacks:          list of callbacks to be called during training
            dump_path:          path to root folder, where the model and its callbacks is dumping stuff to
            device:             device on which the learning has to be performed
        """
        if memory is None and (memory_size is None or n_samples is None or batch_size is None):
            raise ValueError('Learner lacks the memory, it has to be explicitly given, or defined by the params:'
                             '`memory_size`, `n_samples`, `batch_size`')
        if memory is not None and (memory_size is not None or
                                   n_samples is not None or
                                   batch_size is not None):
            raise ValueError('Ambiguous memory specification, either `memory` or `memory_size`, `n_samples`, '
                             '`batch_size` have to be provided')
        if memory is None:
            memory = Memory(['action', 'state', 'reward', 'new_state', 'terminal'],
                            buffer_size=memory_size,
                            n_samples=n_samples,
                            gamma=gamma,
                            batch_size=batch_size)
        super().__init__(model=model,
                         optimizer=optimizer,
                         crit=crit,
                         env=env,
                         gamma=gamma,
                         memory=memory,
                         memory_updater=memory_updater,
                         action_selector=action_selector,
                         grad_clip=grad_clip,
                         load_checkpoint=load_checkpoint,
                         name=name,
                         callbacks=callbacks,
                         dump_path=dump_path,
                         device=device)
        self.train_dict['train_losses'] = []
        self.alpha = alpha

    def fit_epoch(self, device, verbose=1):
        self.memory_updater(self)
        self.model.train()
        self.model.to(device)

        for batch, (action, state, reward, new_state, terminal) in tqdm(enumerate(self.train_loader)):
            action, state, reward, new_state = action.to(self.device), state.to(self.device), reward.to(
                self.device), new_state.to(self.device)
            prediction = self.model(state.squeeze(1))
            target = prediction.clone().detach()
            max_next = self.get_max_Q_for_states(new_state)

            mask = one_hot_encoding(action).type(torch.BoolTensor)
            target[mask] = (1 - self.alpha) * target[mask] + self.alpha * (
                        reward + self.gamma * max_next * (1 - terminal.type(torch.FloatTensor)))

            loss = self.crit(prediction, target)
            self.train_dict['train_losses'] += [loss.item()]
            self._backward(loss)

        if verbose == 1:
            print(f'epoch: {self.train_dict["epochs_run"]}\t'
                  f'average reward: {np.mean(self.train_dict["rewards"]):.2f}\t'
                  f'latest average reward: {self.train_dict["avg_reward"][-1]:.2f}')
        return loss

    def get_max_Q_for_states(self, states):
        with eval_mode(self):  # @todo we might have trouble with the MC Dropout here
            max_Q = self.model(states.squeeze(1)).max(dim=1)[0]
        return max_Q

    def play_episode(self):
        observation = self.env.reset().detach()
        episode_reward = 0
        step_counter = 0
        terminate = False
        episode_memory = Memory(['action', 'state', 'reward', 'new_state', 'terminal'], gamma=self.gamma)
        # self.eval()
        with eval_mode(self):
            while not terminate:
                step_counter += 1
                with torch.no_grad():
                    action = self.chose_action(self, observation)
                new_observation, reward, done, _ = self.env.step(action)

                episode_reward += reward
                episode_memory.memorize((action,
                                         observation,
                                         torch.tensor(reward).float(),
                                         new_observation,
                                         done),
                                        ['action', 'state', 'reward', 'new_state', 'terminal'])
                observation = new_observation
                terminate = done or (self.env.max_episode_length is not None
                                     and step_counter >= self.env.max_episode_length)

        # self.train()
        self.train_loader.memorize(episode_memory, episode_memory.memory_cell_names)
        self.train_dict['rewards'] = self.train_dict.get('rewards', []) + [episode_reward]

        if episode_reward > self.train_dict.get('best_performance', -np.inf):
            self.train_dict['best_performance'] = episode_reward

        return episode_reward


class DoubleQLearner(QLearner):
    def __init__(self,
                 model,
                 optimizer,
                 crit,
                 env,
                 memory_updater,
                 action_selector,
                 alpha,
                 gamma,
                 tau,
                 memory_size=None,
                 n_samples=None,
                 batch_size=None,
                 memory=None,
                 grad_clip=None,
                 load_checkpoint=False,
                 name='q_learner',
                 callbacks=[],
                 dump_path='./tmp',
                 device='cpu'):
        """
        Double Deep Q-Learning algorithm, as introduced in `Deep Reinforcement Learning with Double Q-Learning` by
        Hasselt et al.

        Args:
            model:              pytorch graph derived from torch.nn.Module
            optimizer:          optimizer
            crit:               loss function
            env:                environment to interact with
            memory_updater:     object that iteratively updates the memory
            action_selector:    policy after which actions are selected, it has to be a stochastic one to be used in
                                learning
            alpha:              TD-learning rate @todo this might be dropped in a future implementation
            gamma:              discount factor for future rewards
            tau:                update for target network, if provided as ]0,1[ it constantly updates the weights of the
                                target network by the fraction defined by tau. If provides as an integer it keeps the
                                weights fixed and sets the target weights every tau epochs to the online weights.
            memory_size:        size of the replay memory (number of memories to be hold)
            n_samples:          number of samples to be drawn from the memory each update
            batch_size:         batch size when updating
            memory:             alternatively the memory can be explicitly specified, instead by (memory_size,
                                n_samples, batch_size)
            grad_clip:          gradient_clipping
            load_checkpoint:    loading previous checkpoint @todo might be dropped in the future
            name:               name for the learner
            callbacks:          list of callbacks to be called during training
            dump_path:          path to root folder, where the model and its callbacks is dumping stuff to
            device:             device on which the learning has to be performed
        """
        super().__init__(
            model=model,
            optimizer=optimizer,
            crit=crit,
            env=env,
            memory_updater=memory_updater,
            action_selector=action_selector,
            alpha=alpha,
            gamma=gamma,
            memory_size=memory_size,
            n_samples=n_samples,
            batch_size=batch_size,
            memory=memory,
            grad_clip=grad_clip,
            load_checkpoint=load_checkpoint,
            name=name,
            callbacks=callbacks,
            dump_path=dump_path,
            device=device)
        self.target_model = copy.deepcopy(model)
        self.tau = tau

    def get_max_Q_for_states(self, states):
        with eval_mode(self):  # @todo we might have trouble with the MC Dropout here
            max_Q = self.target_model(states.squeeze(1)).max(dim=1)[0]
        return max_Q

    def fit_epoch(self, device, verbose=1):
        loss = super(DoubleQLearner, self).fit_epoch(device=device, verbose=verbose)
        self.update_target_network()
        return loss

    def update_target_network(self):
        if self.tau < 1.:
            for params_target, params_online in zip(self.target_model.parameters(), self.model.parameters()):
                params_target.data.copy_(self.tau * params_online.data + params_target.data * (1.0 - self.tau))
        else:
            if self.train_dict['epochs_run'] % self.tau == 0:
                for params_target, params_online in zip(self.target_model.parameters(), self.model.parameters()):
                    params_target.data.copy_(params_online.data)


class SARSA(QLearner):
    def __init__(self,
                 model,
                 optimizer,
                 crit,
                 env,
                 memory_updater,
                 action_selector,
                 gamma,
                 alpha,
                 memory_size,
                 n_samples,
                 batch_size,
                 grad_clip=None,
                 load_checkpoint=False,
                 name='q_learner',
                 callbacks=[],
                 dump_path='./tmp',
                 device='cpu'):
        memory = Memory(['action', 'state', 'reward', 'new_state', 'new_action', 'terminal'],
                        buffer_size=memory_size,
                        n_samples=n_samples,
                        gamma=gamma,
                        batch_size=batch_size)
        super().__init__(model=model,
                         optimizer=optimizer,
                         crit=crit,
                         env=env,
                         memory_updater=memory_updater,
                         action_selector=action_selector,
                         gamma=gamma,
                         alpha=alpha,
                         memory_size=memory_size,
                         n_samples=n_samples,
                         batch_size=batch_size,
                         grad_clip=grad_clip,
                         load_checkpoint=load_checkpoint,
                         name=name,
                         callbacks=callbacks,
                         dump_path=dump_path,
                         device=device,
                         memory=memory)

    def fit_epoch(self, device, verbose=1):
        self.memory_updater(self)
        self.model.train()
        self.model.to(device)

        for batch, (action, state, reward, new_state, new_action, terminal) in tqdm(enumerate(self.train_loader)):
            prediction = self.model(state.squeeze(1))
            new_action = one_hot_encoding(new_action)
            with eval_mode(self):
                next_Q = (self.model(new_state.squeeze(1)) * new_action).sum(1)
            target = prediction.clone().detach()

            for t, a, r, nq, term in zip(target, action, reward, next_Q, terminal):
                if term:
                    nq = torch.tensor(-0.)
                t[a.item()] = (1 - self.alpha) * t[a.item()] + self.alpha * (r + self.gamma * nq)
            loss = self.crit(prediction, target)
            self.train_dict['train_losses'] += [loss.item()]
            self._backward(loss)

        if verbose == 1:
            print(f'epoch: {self.train_dict["epochs_run"]}\t'
                  f'average reward: {np.mean(self.train_dict["rewards"]):.2f}\t'
                  f'latest average reward: {self.train_dict["avg_reward"][-1]:.2f}')
        return loss

    def play_episode(self):
        state_old = self.env.reset().detach()
        episode_reward = 0
        step_counter = 0
        terminate = False
        episode_memory = Memory(['action', 'state', 'reward', 'new_state', 'new_action', 'terminal'], gamma=self.gamma)

        action_old = None
        rewards_old = None
        while not terminate:
            with eval_mode(self):
                action = self.chose_action(self, state_old)
            state, reward, done, _ = self.env.step(action)

            episode_reward += reward
            if step_counter > 0:
                episode_memory.memorize((action_old,
                                         state_old,
                                         torch.tensor(rewards_old).float(),
                                         state,
                                         action,
                                         done),
                                        ['action', 'state', 'reward', 'new_state', 'new_action', 'terminal'])
            state_old = state
            rewards_old = reward
            action_old = action
            step_counter += 1
            terminate = done or (self.env.max_episode_length is not None  # @todo could be done better
                                 and step_counter >= self.env.max_episode_length)

        # memorize final step
        episode_memory.memorize((action_old,
                                 state_old,
                                 torch.tensor(rewards_old).float(),
                                 state,
                                 action,
                                 done),
                                ['action', 'state', 'reward', 'new_state', 'new_action', 'terminal'])

        self.train_loader.memorize(episode_memory, episode_memory.memory_cell_names)
        self.train_dict['rewards'] = self.train_dict.get('rewards', []) + [episode_reward]

        if episode_reward > self.train_dict.get('best_performance', -np.inf):
            self.train_dict['best_performance'] = episode_reward

        return episode_reward
