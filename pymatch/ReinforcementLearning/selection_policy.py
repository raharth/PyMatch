import torch
from torch.distributions import Categorical, MultivariateNormal
import torch.nn.functional as F
import numpy as np
import pymatch.DeepLearning.hat as hat
from pymatch.utils.functional import eval_mode
from pymatch.DeepLearning.pipeline import Pipeline


def get_selection_policy(key, params):
    if key == 'AdaptiveQSelection':
        return AdaptiveQActionSelectionEntropy(post_pipeline=[hat.EntropyHat()], **params)
    if key == 'QSelectionCertainty':
        return QActionSelectionCertainty(post_pipeline=[hat.EntropyHat()], **params)
    if key == 'QSelection':
        return QActionSelection(post_pipeline=[hat.EnsembleHat()], **params)
    if key == 'DuelingQSelection':
        return DuelingQActionSelection(post_pipeline=[hat.DuelingQHat()], **params)
    if key == 'EpsilonGreedy':
        return EpsilonGreedyActionSelection(post_pipeline=[hat.EnsembleHat()], **params)
    if key == 'AdaptiveEpsilonGreedy':
        return AdaptiveEpsilonGreedyActionSelection(post_pipeline=[hat.EntropyHat()], **params)
    if key == 'Greedy':
        return GreedyValueSelection(post_pipeline=[hat.EnsembleHat()], **params)
    if key == 'ThompsonGreedy':
        return EpsilonGreedyActionSelection(post_pipeline=[hat.ThompsonAggregation()], **params)
    raise ValueError('Unknown selection strategy')


class SelectionPolicy:
    def __init__(self, pre_pipeline=[], post_pipeline=[]):
        """
        Abstraction of Selection policy.
        This is primarily used to have the pre and post pipeline available to all Selection Policies

        Args:
            pre_pipeline:   List of functions/objects sequentially applied to the input before predicted by the agent.
                            This can be of use when using e.g. MC Dropout, where the input has to be stacked multiple
                            times before it is predicted.
            post_pipeline:  List of functions/objects sequentially applied to the output of the agent.
                            This is useful if used with ensembles, where the output of the networks have to be
                            aggregated or if certain measures have to be estimated.

        Info:
            Why is the selection policy a wrapper around a model in the first place instead of the policy a part of the
            agent?

            Using this approach it is possible to have a greedy evaluation policy, while still having a non-greedy
            policy for the training. Using this approach you can also simply alter the Selection Policy after training.
        """
        self.pre_pipeline = pre_pipeline
        self.post_pipeline = post_pipeline

    # def forward(self, agent, observation):
    #     raise NotImplementedError

    def pre_pipe(self, x):
        for pipe in self.pre_pipeline:
            x = pipe(x)
        return x

    def post_pipe(self, x):
        for pipe in self.post_pipeline:
            x = pipe(x)
        return x


class PolicyGradientActionSelection(SelectionPolicy):
    """
    Probability based selection strategy, used for Policy Gradient
    """

    def __call__(self, agent, observation):
        agent.to(agent.device)
        observation = self.pre_pipe(observation)
        probs = agent(observation.to(agent.device))
        probs = self.post_pipe(probs)
        dist = Categorical(probs.squeeze())
        action = dist.sample()
        log_prob = dist.log_prob(action)
        action = action.view(-1, 1) if len(action.shape) > 0 else action.view(-1)
        return action, log_prob


class BayesianDropoutPGActionSelection(SelectionPolicy):
    def __init__(self, predictions: int, reduce_hat=hat.EnsembleHatStd(), *args, **kwargs):
        """
        Probability based selection strategy, used for Policy Gradient, using multiple drop out forward passes
        to estimate the reliability of the prediction.

        Args:
            predictions:    number of iterations used for the bayesian ensemble
            reduce_hat:     ensemble hat reducing the ouput of the ensemble to a single probability distribution
        """
        super().__init__(*args, **kwargs)
        self.predictions = predictions
        self.reduce_hat = reduce_hat

    def __call__(self, agent, observation):
        observation = observation.to(agent.device)
        agent.to(agent.device)
        action_probs = agent.model(torch.cat(self.predictions * [observation]))
        prob_mean, prob_std = self.reduce_hat(action_probs)
        dist = Categorical(prob_mean.squeeze())
        action = dist.sample()
        log_prob = dist.log_prob(action)
        action = action.view(-1, 1) if len(action.shape) > 0 else action.view(-1)
        return action, log_prob


class QActionSelection(SelectionPolicy):
    def __init__(self, temperature=1., *args, **kwargs):
        """
        Temperature based exponential selection strategy

        Args:
            temperature:    Temperature, which controls the degree of randomness.
        """
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    def __call__(self, agent, observation):
        agent.to(agent.device)
        observation = self.pre_pipe(observation)
        qs = agent(observation.to(agent.device))
        qs = self.post_pipe(qs)
        probs = F.softmax(qs / self.temperature, dim=-1)
        dist = Categorical(probs.squeeze())
        action = dist.sample()
        action = action.view(-1, 1) if len(action.shape) > 0 else action.view(-1)
        return action


class QActionSelectionCertainty(SelectionPolicy):
    def __init__(self, temperature=1., *args, **kwargs):
        """
        Temperature based exponential selection strategy

        Args:
            temperature:    Temperature, which controls the degree of randomness.
        """
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    def __call__(self, agent, observation):
        agent.to(agent.device)
        observation = self.pre_pipe(observation)
        qs = agent(observation.to(agent.device))
        qs, stds = self.post_pipe(qs)
        probs = F.softmax(qs / self.temperature, dim=-1)
        dist = Categorical(probs.squeeze())
        action = dist.sample()
        action = action.view(-1, 1) if len(action.shape) > 0 else action.view(-1)
        return action, stds


class DuelingQActionSelection(SelectionPolicy):
    def __init__(self, temperature=1., *args, **kwargs):
        """
        Temperature based exponential selection strategy

        Args:
            temperature:    Temperature, which controls the degree of randomness.
        """
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    def __call__(self, agent, observation):
        agent.to(agent.device)
        observation = self.pre_pipe(observation)
        qs, val_q, adv_q = agent(observation.to(agent.device))
        qs, val_q, adv_q = self.post_pipe(qs, val_q, adv_q)
        probs = F.softmax(qs / self.temperature, dim=-1)
        dist = Categorical(probs.squeeze())
        action = dist.sample()
        action = action.view(-1, 1) if len(action.shape) > 0 else action.view(-1)
        return action, val_q, adv_q


class AdaptiveQActionSelectionStd(SelectionPolicy):
    def __init__(self, temperature=1., history=1000, hist_scaling=1., warm_up=100, return_uncertainty=True,
                 *args, **kwargs):
        """
        Adaptive value selection based on the uncertainty of the model

        Args:
            temperature:    Temperature, which controls the degree of randomness. This is the standard parameter of q-selection
            history:        Number of previous states on which the normalization of certainty is based.
            hist_scaling:   This determins how much impact uncertainty has. Basically it is used as a temperature for
                            the sigmoid when adapting the temperature.
            min_length:     Minimal number of samples to base the scaling of uncertainty on. Before that the temperature
                            is not altered in any way.

        """
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.values = []
        self.history = history
        self.hist_scaling = hist_scaling
        self.warm_up = warm_up
        self.return_uncertainty = return_uncertainty

    def __call__(self, agent, observation):
        agent.to(agent.device)
        observation = self.pre_pipe(observation)
        qs = agent(observation.to(agent.device))
        qs, uncertainties = self.post_pipe(qs)

        probs = F.softmax(qs / self.adjust_temp(uncertainties=uncertainties), dim=-1)
        dist = Categorical(probs.squeeze())
        action = dist.sample()
        action = action.view(-1, 1) if len(action.shape) > 0 else action.view(-1)
        if self.return_uncertainty:
            return action.item(), uncertainties
        else:
            return action.item()

    def adjust_temp(self, uncertainties):
        uncertainty = uncertainties.max()
        self.values += [uncertainty.item()]
        self.values = self.values[-self.history:]
        if self.warm_up > len(self.values):
            return self.temperature
        uncertainty = (uncertainty - np.mean(self.values)) / np.std(self.values)
        return torch.sigmoid(uncertainty / self.hist_scaling) * self.temperature


class AdaptiveQActionSelectionEntropy(SelectionPolicy):
    def __init__(self, sensitivity=5., selection_temp=1., min_temp=.1, return_uncertainty=True, warm_up=1000, *args,
                 **kwargs):
        """
        Adaptive value selection based on the entropy of the model.

        Args:
            sensitivity:            Determines how sensible it is to uncertainty. A larger value leads to higher
                                    exploration even at low entropy. A smaller values means less random actions even at
                                    high entropy
            selection_temp:         Max temperature used for action sampling
            min_temp:               Min temperature used for action sampling
            warm_up:                Initial steps for which the unscaled selection temperature is used
            return_uncertainty:     If set to `True` entropy is returned in addition to the action
        """
        if 'post_pipeline' not in kwargs.keys():
            kwargs['post_pipeline'] = [hat.EntropyHat()]

        super().__init__(*args, **kwargs)
        self.selection_temp = selection_temp
        self.min_temp = min_temp
        self.sensitivity = sensitivity
        self.return_uncertainty = return_uncertainty
        self.warm_up = warm_up

    def __call__(self, agent, observation):
        agent.to(agent.device)
        observation = self.pre_pipe(observation)
        qs = agent(observation.to(agent.device))
        qs, uncertainties = self.post_pipe(qs)

        probs = F.softmax(qs / self.adjust_temp(uncertainties=uncertainties), dim=-1)
        dist = Categorical(probs.squeeze())
        action = dist.sample()
        action = action.view(-1, 1) if len(action.shape) > 0 else action.view(-1)
        if self.return_uncertainty:
            return action, uncertainties
        else:
            return action

    def adjust_temp(self, uncertainties):
        self.warm_up -= 1
        if self.warm_up < 0:
            return torch.clamp(
                (self.selection_temp * (1 - torch.exp(-uncertainties * self.sensitivity))).max(-1)[0],
                min=self.min_temp).view(-1, 1)
            # return max(self.selection_temp * (1 - torch.exp(-uncertainties * self.sensitivity)), self.min_temp)
        return self.selection_temp


class EpsilonGreedyActionSelection(SelectionPolicy):
    def __init__(self, action_space: list, epsilon=.1, **kwargs):
        """
        Epsilon greedy selection strategy, choosing the best or with p=epsilon choosing a random action

        Args:
            action_space:   list of possible actions
            epsilon:        probability for random action
        """
        super().__init__(**kwargs)
        self.action_space = action_space
        self.epsilon = epsilon

    def __call__(self, agent, observation):
        agent.to(agent.device)
        observation = self.pre_pipe(observation)
        qs = agent(observation.to(agent.device))
        qs = self.post_pipe(qs)
        greedy_actions = qs.max(-1)[1].type(torch.int64)
        random_actions = torch.tensor(np.random.choice(self.action_space, size=len(observation)), dtype=torch.int64)
        epsilon_mask = torch.rand(len(observation)).le(self.epsilon)
        if epsilon_mask.sum() > 0:
            greedy_actions[epsilon_mask] = random_actions[epsilon_mask]
        return greedy_actions.view(-1, 1)
        # if np.random.uniform() > self.epsilon:
        #     return qs.max(-1)[1]
        # return np.random.choice(self.action_space)


class AdaptiveEpsilonGreedyActionSelection(SelectionPolicy):
    def __init__(self, action_space, epsilon=1.0, min_epsilon=0.0, warm_up=1000, sensitivity=.1, **kwargs):
        """
        Adaptive Epsilon greedy selection strategy, choosing the best or with p=epsilon choosing a random action

        Args:
            action_space:   list of possible actions
            epsilon:        probability for random action
        """
        super().__init__(**kwargs)
        self.action_space = action_space
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.warm_up = warm_up
        self.sensitivity = sensitivity

    def __call__(self, agent, observation):
        agent.to(agent.device)
        observation = self.pre_pipe(observation)
        qs = agent(observation.to(agent.device))
        qs, uncertainties = self.post_pipe(qs)
        greedy_actions = qs.max(-1)[1].type(torch.int64)
        random_actions = torch.tensor(np.random.choice(self.action_space, size=len(observation)), dtype=torch.int64)
        epsilon_mask = torch.rand(len(observation)).le(self.adjust_temp(uncertainties=uncertainties))
        if epsilon_mask.sum() > 0:
            greedy_actions[epsilon_mask] = random_actions[epsilon_mask]
        return greedy_actions.view(-1, 1)

        # if np.random.uniform() > self.adjust_temp(uncertainties=uncertainties):
        #     return qs.argmax()
        # return np.random.choice(self.action_space)

    def adjust_temp(self, uncertainties):
        self.warm_up -= 1
        if self.warm_up < 0:
            return max(self.epsilon * (1 - torch.exp(-uncertainties * self.sensitivity)), self.min_epsilon)
        return self.epsilon


class GreedyValueSelection(SelectionPolicy):
    """
    Choosing the best possible option, necessary for evaluation
    """

    def __call__(self, agent, observation):
        observation = self.pre_pipe(observation)
        qs = agent(observation.to(agent.device))
        qs = self.post_pipe(qs)
        return qs.max(-1)[1].type(torch.int64).view(-1, 1)
        # return qs.argmax()


class NormalThompsonSampling(SelectionPolicy):
    """
    Implementation of Thompson sampling based in the Normal distribution.
    Estimates the distribution over a model, sampling from it.
    @ todo  a better abstraction would be using a general distribution, defined in the constructor. Then the last
            pipe element has to provide the final inputs to that distribution, right now it can only use the Normal
            distribution but no other.
            Why is this part of the sampling in the first place?
    """

    def __call__(self, agent, observation):
        pipeline = Pipeline(pipes=self.pre_pipes + [agent] + self.post_pipes)
        with torch.no_grad():  # @todo why am I using a no grad here? This policy can then not be used for training?
            with eval_mode(agent):
                y_mean, y_std = pipeline(observation)
        shape = y_mean.shape
        dist = MultivariateNormal(loc=y_mean.view(-1),
                                  covariance_matrix=y_std.view(-1) ** 2 * torch.eye(len(y_std.view(-1))))
        return dist.sample().view(shape)
