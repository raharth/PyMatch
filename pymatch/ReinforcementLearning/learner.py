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
                 action_selector,
                 gamma,
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

        self.env = env
        self.train_dict['rewards'] = []
        self.gamma = gamma
        self.chose_action = action_selector
        self.store_memory = store_memory

    def fit_epoch(self, device, verbose=1):
        raise NotImplementedError

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
            action, log_prob = self.chose_action(self, observation)
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
    def __init__(self,
                 model,
                 optimizer,
                 crit,
                 env,
                 action_selector,
                 alpha,
                 gamma,
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
                         action_selector=action_selector,
                         grad_clip=grad_clip,
                         name=name,
                         callbacks=callbacks,
                         dump_path=dump_path,
                         device=device,
                         **kwargs)
        self.train_dict['train_losses'] = []
        self.alpha = alpha

    def fit_epoch(self, device, verbose=1):
        self.model.train()
        self.model.to(device)

        losses = []

        for batch, (action, state, reward, new_state, terminal) in tqdm(enumerate(self.train_loader)):
            action, state, reward, new_state = action.to(self.device), state.to(self.device), reward.to(
                self.device), new_state.to(self.device)
            prediction = self.model(state.squeeze(1))
            target = prediction.clone().detach()
            max_next = self.get_max_Q_for_states(new_state)

            mask = one_hot_encoding(action, n_categories=self.env.action_space.n).type(torch.BoolTensor).to(self.device)
            target[mask] = (1 - self.alpha) * target[mask] + self.alpha * (
                    reward.view(-1) + self.gamma * max_next * (1 - terminal.view(-1).type(torch.FloatTensor)).to(self.device))

            loss = self.crit(prediction, target)
            losses += [loss.item()]
            self._backward(loss)

        self.train_dict['train_losses'] += [np.mean(losses).item()]

        # This is using the DQL loss defined in 'Asynchronous Methods for Deep Reinforcement Learning' by Mnih et al.,
        # though this is not working
        # for batch, (action, state, reward, new_state, terminal) in tqdm(enumerate(self.train_loader)):
        #     action, state, reward, new_state = action.to(self.device), state.to(self.device), reward.to(
        #         self.device), new_state.to(self.device)
        #     prediction = self.model(state.squeeze(1))
        #     max_next = self.get_max_Q_for_states(new_state)
        #     mask = one_hot_encoding(action, n_categories=self.env.action_space.n).type(torch.BoolTensor)
        #
        #     loss = self.crit(gamma=self.gamma, pred=prediction[mask], max_next=max_next, reward=reward)
        #     self.train_dict['train_losses'] += [loss.item()]
        #     self._backward(loss)

        if verbose == 1:
            print(f'epoch: {self.train_dict["epochs_run"]}\t'
                  f'average reward: {np.mean(self.train_dict["rewards"]):.2f}\t',
                  f'last loss: {self.train_dict["train_losses"][-1]:.2f}',
                  f'latest average reward: {self.train_dict.get("avg_reward", [np.nan])[-1]:.2f}')
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
        with eval_mode(self):
            while not terminate:
                step_counter += 1
                with torch.no_grad():
                    action = self.chose_action(self, observation)
                new_observation, reward, terminate, _ = self.env.step(action)

                episode_reward += reward
                episode_memory.memorize((action,
                                         observation,
                                         torch.tensor(reward).float(),
                                         new_observation,
                                         terminate),
                                        ['action', 'state', 'reward', 'new_state', 'terminal'])
                observation = new_observation

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
        super().__init__(
            model=model,
            optimizer=optimizer,
            crit=crit,
            env=env,
            action_selector=action_selector,
            alpha=alpha,
            gamma=gamma,
            memory_size=memory_size,
            n_samples=n_samples,
            batch_size=batch_size,
            memory=memory,
            grad_clip=grad_clip,
            name=name,
            callbacks=callbacks,
            dump_path=dump_path,
            device=device,
            **kwargs)
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
                action = self.chose_action(self, state_old)
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
            action, log_prob = self.chose_action(self, observation)
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
