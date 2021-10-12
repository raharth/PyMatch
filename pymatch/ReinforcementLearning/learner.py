import torch
from tqdm import tqdm
import numpy as np

from pymatch.ReinforcementLearning.memory import Memory
from pymatch.ReinforcementLearning.loss import REINFORCELoss
from pymatch.DeepLearning.learner import Learner
from pymatch.utils.functional import one_hot_encoding, eval_mode, train_mode
import pymatch.ReinforcementLearning.selection_policy as sp
import copy


class ReinforcementLearner(Learner):
    def __init__(self,
                 model,
                 optimizer,
                 crit,
                 memory,
                 env,
                 selection_strategy,
                 gamma,
                 fitter=None,
                 grad_clip=None,
                 name='',
                 callbacks=None,
                 dump_path='./tmp',
                 device='cpu',
                 store_memory=False,
                 *args,
                 **kwargs
                 ):
        """
        Abstract class for Reinforcement Learners

        Args:
            model:              neural network
            optimizer:          optimizer
            crit:               loss function
            memory:             memory to store and load the memory
            env:                environment to interact with
            action_selector:    action selection strategy
            gamma:              discount factor for reward over time
            fitter:             module that updates the weights of the agent
            grad_clip:          gradient clipping
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
                         name=name,
                         callbacks=callbacks,
                         dump_path=dump_path,
                         device=device,
                         *args,
                         **kwargs
                         )
        self.memory = memory
        self.env = env
        self.fitter = fitter
        self.train_dict['rewards'] = []
        self.gamma = gamma
        self.selection_strategy = selection_strategy
        self.store_memory = store_memory

    def fit_epoch(self, device, verbose=1):
        return self.fitter(self, device)

    def play_episode(self):
        raise NotImplementedError

    def create_state_dict(self):
        """
        Creates the state dictionary of a learner.
        This should be redefined by each derived learner that introduces own members. Always call the parents method.
        This dictionary can then be extended by
        the derived learner's members

        Returns:
            state dictionary of the learner

        """
        state_dict = super().create_state_dict()
        if self.store_memory:
            state_dict['memory'] = self.train_loader.create_state_dict()
        return state_dict

    def restore_checkpoint(self, checkpoint):
        super(ReinforcementLearner, self).restore_checkpoint(checkpoint=checkpoint)
        if self.store_memory:
            self.train_loader.restore_checkpoint(checkpoint['memory'])


class PolicyGradient(ReinforcementLearner):
    def __init__(self,
                 env,
                 model,
                 optimizer,
                 n_samples=None,
                 batch_size=None,
                 crit=REINFORCELoss(),
                 action_selector=sp.PolicyGradientActionSelection(),
                 memory=None,
                 memory_size=None,
                 gamma=.95,
                 grad_clip=None,
                 name='',
                 callbacks=None,
                 dump_path='./tmp',
                 device='cpu',
                 *args,
                 **kwargs):
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
            name:               name of the agent
            callbacks:          list of callbacks to use during training
            dump_path:          dump path for the model and the callbacks
            device:             device to run the model on
        """
        if memory is None and (memory_size is None or batch_size is None):
            raise ValueError('Learner lacks the memory, it has to be explicitly given, or defined by the params:'
                             '`memory_size`, `batch_size`')
        if memory is not None and (memory_size is not None or
                                   batch_size is not None):
            raise ValueError('Ambiguous memory specification, either `memory` or `memory_size`, `batch_size` have to '
                             'be provided')
        if memory is None:
            memory = Memory(['log_prob', 'reward'],
                            memory_size=memory_size,
                            n_samples=n_samples,
                            gamma=gamma,
                            batch_size=batch_size)
        super().__init__(model=model,
                         optimizer=optimizer,
                         crit=crit,
                         env=env,
                         gamma=gamma,
                         memory=memory,
                         action_selector=action_selector,
                         grad_clip=grad_clip,
                         name=name,
                         callbacks=callbacks,
                         dump_path=dump_path,
                         device=device,
                         *args,
                         **kwargs)

    def fit_epoch(self, device, verbose=1):
        """
        Train a single epoch.

        Args:
            device: device t-o run it on 'cpu' or 'cuda'
            verbose: verbosity of the learning

        Returns:
            current loss
        """
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
            action, log_prob = self.selection_strategy(self, observation)
            new_observation, reward, terminate, _ = self.env.step(action)

            episode_reward += reward
            episode_memory.memorize((log_prob, torch.tensor(reward).float()), ['log_prob', 'reward'])
            observation = new_observation

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
    def __init__(self, model, optimizer, crit, env, selection_strategy, alpha, gamma, player, memory_size=None,
                 n_samples=None, batch_size=None, memory=None, grad_clip=None, name='q_learner', callbacks=[],
                 dump_path='./tmp', device='cpu', **kwargs):
        """
        Deep Q-Learning algorithm, as introduced by http://arxiv.org/abs/1312.5602
        Args:
            model:              pytorch graph derived from torch.nn.Module
            optimizer:          optimizer
            crit:               loss function
            env:                environment to interact with
            memory_updater:     object that iteratively updates the memory
            selection_strategy:    policy after which actions are selected, it has to be a stochastic one to be used in
                                learning
            alpha:              TD-learning rate @todo this might be dropped in a future implementation
            gamma:              disount factor for future rewards
            memory_size:        size of the replay memory (number of memories to be hold)
            n_samples:          number of samples to be drawn from the memory each update
            batch_size:         batch size when updating
            memory:             alternatively the memory can be explicitly specified, instead by (memory_size,
                                n_samples, batch_size)
            grad_clip:          gradient_clipping
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
                            memory_size=memory_size,
                            n_samples=n_samples,
                            gamma=gamma,
                            batch_size=batch_size)
        super().__init__(model=model,
                         optimizer=optimizer,
                         crit=crit,
                         env=env,
                         gamma=gamma,
                         memory=memory,
                         selection_strategy=selection_strategy,
                         grad_clip=grad_clip,
                         name=name,
                         callbacks=callbacks,
                         dump_path=dump_path,
                         device=device,
                         **kwargs)
        self.train_dict['train_losses'] = []
        self.player = player
        self.alpha = alpha

    def get_max_Q_for_states(self, states):
        """
        Computes the max Q value estimates for states.

        Args:
            states: to get the max Q-value for

        Returns:

        """
        with eval_mode(self):
            max_Q = self.model(states.squeeze(1)).max(dim=1)[0]
        return max_Q

    def play_episode(self):
        """
        Plays an episode, memorizing all state transitions.

        Returns:

        """
        return self.player(self, self.selection_strategy, self.train_loader)


class DQNFitter:  # @todo this could become a callback
    def __call__(self, agent, device, verbose=True):
        agent.model.train()
        agent.model.to(device)

        losses = []

        for batch, (action, state, reward, new_state, terminal) in tqdm(enumerate(agent.train_loader)):
            action, state, reward, new_state = action.to(agent.device), state.to(agent.device), reward.to(
                agent.device), new_state.to(agent.device)
            prediction = agent.model(state.squeeze(1))
            target = prediction.clone().detach()
            max_next = agent.get_max_Q_for_states(new_state)

            mask = one_hot_encoding(action, n_categories=agent.env.action_space.n).type(torch.BoolTensor).to(
                agent.device)
            target[mask] = (1 - agent.alpha) * target[mask] + agent.alpha * (
                    reward.view(-1) + agent.gamma * max_next * (1 - terminal.view(-1).type(torch.FloatTensor)).to(
                agent.device))

            loss = agent.crit(prediction, target)
            losses += [loss.item()]
            agent._backward(loss)

        loss = np.mean(losses).item()
        agent.train_dict['train_losses'] += [loss]

        if verbose == 1:
            print(f'epoch: {agent.train_dict["epochs_run"]}\t'
                  f'average reward: {np.mean(agent.train_dict["rewards"]):.2f}\t',
                  f'last loss: {agent.train_dict["train_losses"][-1]:.2f}',
                  f'latest average reward: {agent.train_dict.get("avg_reward", [np.nan])[-1]:.2f}')
        return loss


class DQNIntegratedFitter:
    def __init__(self, sample_size=32, skip_steps=4):
        self.skip_steps = skip_steps
        self.step_counter = 0
        self.sample_size = sample_size

    def __call__(self, agent, device, verbose=True):
        observation = agent.env.reset().detach()
        episode_reward = 0
        terminate = False
        # episode_memory = Memory(['action', 'state', 'reward', 'new_state', 'terminal'],
        #                         gamma=agent.memory.gamma)
        # with eval_mode(agent):
        while not terminate:
            self.step_counter += 1
            agent.to(agent.device)
            action = agent.selection_strategy(agent, observation.to(agent.device))
            new_observation, reward, terminate, _ = agent.env.step(action)

            episode_reward += torch.sum(reward).item() / agent.env.n_instances
            agent.memory.memorize((action,
                                   observation,
                                   torch.tensor(reward).float(),
                                   new_observation,
                                   terminate),
                                  ['action', 'state', 'reward', 'new_state', 'terminal'])
            observation = new_observation[~terminate.view(-1)]
            terminate = terminate.min().item()

            if self.step_counter % self.skip_steps == 0:
                self.update(agent, verbose)

        agent.train_dict['rewards'] = agent.train_dict.get('rewards', []) + [episode_reward]
        agent.train_dict['env_steps'] = agent.train_dict.get('env_steps', 0) + 1

        if episode_reward > agent.train_dict.get('best_performance', -np.inf):
            agent.train_dict['best_performance'] = episode_reward

        return episode_reward

    def update(self, agent, verbose):
        with train_mode(agent):
            agent.model.to(agent.device)
            losses = []

            for batch, (action, state, reward, new_state, terminal) in tqdm(
                    enumerate(agent.memory.sample_loader(n_samples=self.sample_size))):
                action, state, reward, new_state = action.to(agent.device), state.to(agent.device), reward.to(
                    agent.device), new_state.to(agent.device)
                prediction = agent.model(state.squeeze(1))
                target = prediction.clone().detach()
                max_next = agent.get_max_Q_for_states(new_state)

                mask = one_hot_encoding(action, n_categories=agent.env.action_space.n).type(torch.BoolTensor).to(
                    agent.device)
                target[mask] = (1 - agent.alpha) * target[mask] + agent.alpha * (
                        reward.view(-1) + agent.gamma * max_next * (1 - terminal.view(-1).type(torch.FloatTensor)).to(
                    agent.device))

                loss = agent.crit(prediction, target)
                losses += [loss.item()]
                agent._backward(loss)

            agent.train_dict['train_losses'] += [np.mean(losses).item()]

            if verbose == 1:
                print(f'epoch: {agent.train_dict["epochs_run"]}\t'
                      f'average reward: {np.mean(agent.train_dict["rewards"]):.2f}\t',
                      f'last loss: {agent.train_dict["train_losses"][-1]:.2f}',
                      f'latest average reward: {agent.train_dict.get("avg_reward", [np.nan])[-1]:.2f}')
            return loss


class DoubleQLearner(QLearner):
    def __init__(self,
                 model,
                 optimizer,
                 crit,
                 env,
                 action_selector,
                 alpha,
                 gamma,
                 tau,
                 memory_size=None,
                 n_samples=None,
                 batch_size=None,
                 memory=None,
                 grad_clip=None,
                 name='q_learner',
                 callbacks=[],
                 dump_path='./tmp',
                 device='cpu',
                 **kwargs):
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
        super().__init__(model=model, optimizer=optimizer, crit=crit, env=env, selection_strategy=action_selector,
                         alpha=alpha, gamma=gamma, memory_size=memory_size, n_samples=n_samples, batch_size=batch_size,
                         memory=memory, grad_clip=grad_clip, name=name, callbacks=callbacks, dump_path=dump_path,
                         device=device, **kwargs)
        self.target_model = copy.deepcopy(model)
        self.tau = tau

    def get_max_Q_for_states(self, states):
        with eval_mode(self):  # @todo we might have trouble with the MC Dropout here
            max_Q = self.target_model(states.squeeze(1)).max(dim=1)[0]
        return max_Q

    def fit_epoch(self, device, verbose=1):
        self.target_model.to(device)
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

    def to(self, device):
        self.target_model.to(device)
        self.model.to(device)
        # super(DoubleQLearner).to(device)


class SARSA(DoubleQLearner):
    def __init__(self, *args, **kwargs):
        # memory = Memory(['action', 'state', 'reward', 'new_state', 'new_action', 'terminal'],
        #                 memory_size=memory_size,
        #                 n_samples=n_samples,
        #                 gamma=gamma,
        #                 batch_size=batch_size)
        super().__init__(*args, **kwargs)

    def fit_epoch(self, device, verbose=1):
        self.model.train()
        self.model.to(device)
        self.target_model.to(device)

        for batch, (action, state, reward, next_state, next_action, terminal) in tqdm(enumerate(self.train_loader)):
            action, state, reward, next_state = action.to(self.device), state.to(self.device), reward.to(
                self.device), next_state.to(self.device)
            prediction = self.model(state.squeeze(1))
            next_action = one_hot_encoding(next_action).to(self.device)
            with eval_mode(self):  # @todo this is not working with DDQN so far
                next_Q = (self.target_model(next_state.squeeze(1)) * next_action).sum(1)
            target = prediction.clone().detach()

            mask = one_hot_encoding(action).type(torch.BoolTensor)
            target[mask] = (1 - self.alpha) * target[mask] + self.alpha * (
                    reward + self.gamma * next_Q * (1. - terminal.type(torch.FloatTensor)).to(self.device))

            loss = self.crit(prediction, target)
            self.train_dict['train_losses'] += [loss.item()]
            self._backward(loss)

        self.update_target_network()

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

        while not terminate:
            with eval_mode(self):
                action = self.selection_strategy(self, state_old)
            state, reward, terminate, _ = self.env.step(action)

            episode_reward += reward
            if step_counter > 0:
                episode_memory.memorize((action_old,
                                         state_old,
                                         torch.tensor(reward_old).float(),
                                         state,
                                         action,
                                         False),
                                        ['action', 'state', 'reward', 'new_state', 'new_action', 'terminal'])
            state_old = state
            reward_old = reward
            action_old = action
            step_counter += 1

        # memorize final step
        episode_memory.memorize((action_old,
                                 state_old,
                                 torch.tensor(reward_old).float(),
                                 state,
                                 action,
                                 True),
                                ['action', 'state', 'reward', 'new_state', 'new_action', 'terminal'])

        self.train_loader.memorize(episode_memory, episode_memory.memory_cell_names)
        self.train_dict['rewards'] = self.train_dict.get('rewards', []) + [episode_reward]

        if episode_reward > self.train_dict.get('best_performance', -np.inf):
            self.train_dict['best_performance'] = episode_reward

        return episode_reward


class A3C(PolicyGradient):
    def __init__(self,
                 critics,
                 env,
                 model,
                 optimizer,
                 n_samples,
                 batch_size,
                 crit=REINFORCELoss(),
                 action_selector=sp.PolicyGradientActionSelection(),
                 memory=None,
                 memory_size=1000,
                 gamma=.95,
                 grad_clip=None,
                 name='',
                 callbacks=None,
                 dump_path='./tmp',
                 device='cpu',
                 **kwargs):
        if memory is None and (memory_size is None or n_samples is None or batch_size is None):
            raise ValueError('Learner lacks the memory, it has to be explicitly given, or defined by the params:'
                             '`memory_size`, `n_samples`, `batch_size`')
        if memory is not None and (memory_size is not None or
                                   n_samples is not None or
                                   batch_size is not None):
            raise ValueError('Ambiguous memory specification, either `memory` or `memory_size`, `n_samples`, '
                             '`batch_size` have to be provided')
        if memory is None:
            memory = Memory(['log_prob', 'reward', 'state'],
                            memory_size=memory_size,
                            n_samples=n_samples,
                            gamma=gamma,
                            batch_size=batch_size)
        super().__init__(env=env,
                         model=model,
                         optimizer=optimizer,
                         n_samples=None,
                         batch_size=None,
                         crit=crit,
                         action_selector=action_selector,
                         memory=memory,
                         memory_size=None,
                         gamma=gamma,
                         grad_clip=grad_clip,
                         name=name,
                         callbacks=callbacks,
                         dump_path=dump_path,
                         device=device,
                         **kwargs)
        self.critics = critics  # @todo probably create it in here

    def fit_epoch(self, device, verbose=1):
        if verbose:
            print('fitting actor')
        actor_loss = self.fit_epoch_actor(device=device, verbose=verbose)
        if verbose:
            print('fitting critics')
        critics_loss = self.critics.fit_epoch(device=device, verbose=verbose)
        return (actor_loss + critics_loss) / 2

    def fit_epoch_actor(self, device, verbose=1):
        """
        Train a single epoch.

        Args:
            device: device t-o run it on 'cpu' or 'cuda'
            verbose: verbosity of the learning

        Returns:
            current loss
        """
        # self.memory_updater(self)
        self.model.train()
        self.model.to(device)

        losses = []
        with train_mode(self):
            for batch, (log_prob, reward, state) in tqdm(enumerate(self.train_loader)):
                log_prob, reward = log_prob.to(device), reward.to(device)
                loss = self.crit(log_prob, reward, baseline=self.critics.get_max_Q_for_states(state))
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
        episode_memory_pg = Memory(['log_prob', 'reward', 'state'], gamma=self.gamma)
        episode_memory_q = Memory(['action', 'state', 'reward', 'new_state', 'terminal'], gamma=self.gamma)

        while not terminate:
            step_counter += 1
            action, log_prob = self.selection_strategy(self, observation)
            new_observation, reward, terminate, _ = self.env.step(action)

            episode_reward += reward
            episode_memory_pg.memorize((log_prob, torch.tensor(reward).float(), observation),
                                       ['log_prob', 'reward', 'state'])
            episode_memory_q.memorize((action,
                                       observation,
                                       torch.tensor(reward).float(),
                                       new_observation,
                                       terminate),
                                      ['action', 'state', 'reward', 'new_state', 'terminal'])
            observation = new_observation

        episode_memory_pg.cumul_reward()
        self.train_loader.memorize(episode_memory_pg, episode_memory_pg.memory_cell_names)
        self.critics.train_loader.memorize(episode_memory_q, episode_memory_q.memory_cell_names)
        self.train_dict['rewards'] = self.train_dict.get('rewards', []) + [episode_reward]

        if episode_reward > self.train_dict.get('best_performance', -np.inf):
            self.train_dict['best_performance'] = episode_reward

        return episode_reward

    def create_state_dict(self):
        state_dict = super(A3C, self).create_state_dict()
        state_dict['critics_state_dict'] = self.critics.create_state_dict()
        return state_dict

    def restore_checkpoint(self, checkpoint):
        super(A3C, self).restore_checkpoint(checkpoint=checkpoint)
        self.critics.restore_checkpoint(checkpoint['critics_state_dict'])
