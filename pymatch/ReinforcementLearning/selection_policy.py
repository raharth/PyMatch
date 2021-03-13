import torch
from torch.distributions import Categorical, MultivariateNormal
import torch.nn.functional as F
import numpy as np
import pymatch.DeepLearning.hat as hat
from pymatch.utils.functional import eval_mode
from pymatch.DeepLearning.pipeline import Pipeline


class SelectionPolicy:
    # @todo really necessary?
    def __init__(self, pre_pipeline=[], post_pipeline=[]):
        self.pre_pipeline = pre_pipeline
        self.post_pipeline = post_pipeline

    def forward(self, agent, observation):
        raise NotImplementedError

    def preparation(self, x):
        for pipe in self.pre_pipeline:
            x = pipe(x)
        return x

    def aggregation(self, x):
        for pipe in self.post_pipeline:
            x = pipe(x)
        return x

# class Softmax_Selection(SelectionPolicy):
#
#     def __init__(self, temperature=1.):
#         super(Softmax_Selection, self).__init__()
#         self.temperature = temperature
#
#     def choose(self, q_values):
#         p = F.softmax(q_values / self.temperature, dim=1)
#         dist = Categorical(p.squeeze())
#         return dist.sample()
#
#
# class EpsilonGreedy(SelectionPolicy):
#
#     def __init__(self, epsilon):
#         super(EpsilonGreedy, self).__init__()
#         self.epsilon = epsilon
#
#     def choose(self, q_values):
#         if torch.rand(1) < self.epsilon:
#             return torch.LongTensor(q_values.shape[0]).random_(0, q_values.shape[1])
#         else:
#             return q_values.argmax(dim=1)

class PolicyGradientActionSelection:
    """
    Probability based selection strategy, used for Policy Gradient
    """

    def __call__(self, agent, observation):
        agent.model.to(agent.device)
        probs = agent.model(observation.to(agent.device))
        dist = Categorical(probs.squeeze())
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob


class BayesianDropoutPGActionSelection:
    def __init__(self, predictions: int, reduce_hat=hat.EnsembleHatStd()):
        """
        Probability based selection strategy, used for Policy Gradient, using multiple drop out forward passes
        to estimate the reliability of the prediction.

        Args:
            predictions:    number of iterations used for the bayesian ensemble
            reduce_hat:     ensemble hat reducing the ouput of the ensemble to a single probability distribution
        """
        self.predictions = predictions
        self.reduce_hat = reduce_hat

    def __call__(self, agent, observation):
        observation = observation.to(agent.device)
        agent.model.to(agent.device)
        action_probs = agent.model(torch.cat(self.predictions * [observation]))
        prob_mean, prob_std = self.reduce_hat(action_probs)
        dist = Categorical(prob_mean.squeeze())
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob


class QActionSelection(SelectionPolicy):
    def __init__(self, temperature=1., **kwargs):
        """
        Temperature based exponential selection strategy

        Args:
            temperature:
        """
        super().__init__(**kwargs)
        self.temperature = temperature

    def __call__(self, agent, observation):
        agent.model.to(agent.device)
        observation = self.preparation(observation)
        qs = agent.model(observation.to(agent.device))
        qs = self.aggregation(qs)
        probs = F.softmax(qs / self.temperature, dim=1)
        dist = Categorical(probs.squeeze())
        action = dist.sample()
        return action.item()


class EpsilonGreedyActionSelection(SelectionPolicy):
    def __init__(self, action_space, epsilon=.9, **kwargs):
        """
        Epsilon greedy selection strategy, choosing the best or with p=1-epsilon choosing a random action

        Args:
            action_space:   list of possible actions
            epsilon:        probability for max
        """
        super().__init__(**kwargs)
        self.action_space = action_space
        self.epsilon = epsilon

    def __call__(self, agent, observation):
        agent.model.to(agent.device)
        observation = self.preparation(observation)
        qs = agent.model(observation.to(agent.device))
        qs = self.aggregation(qs)
        if np.random.uniform() < self.epsilon:
            return qs.argmax().item()
        return np.random.choice(self.action_space)


class GreedyValueSelection(SelectionPolicy):
    """
    Choosing the best possible option, necessary for evaluation
    @todo abstract pre and post-pipeline
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, agent, observation):
        observation = self.preparation(observation)
        qs = agent(observation.to(agent.device))
        qs = self.aggregation(qs)
        return qs.argmax().item()


class NormalThompsonSampling(SelectionPolicy):
    def __init__(self, **kwargs):
        """
        Implementation of Thompson sampling based in the Normal distribution.
        Estimates the distribution over a model, sampling from it.
        Args:
            pre_pipes:  pipeline elements before the agent
            post_pipes: pipeline elements after the agent the last element of this list has to provide the
                        parameterization of the distribution
        @ todo  a better abstraction would be using a general distribution, defined in the constructor. Then the last
                pipe element has to provide the final inputs to that distribution, right now it can only use the Normal
                distribution but no other.
                Why is this part of the sampling in the first place?
        """
        # self.repeater = hat.InputRepeater(n_iter)
        super().__init__(**kwargs)

    def __call__(self, agent, observation):
        pipeline = Pipeline(pipes=self.pre_pipes + [agent] + self.post_pipes)
        with torch.no_grad():
            with eval_mode(agent):
                y_mean, y_std = pipeline(observation)
        shape = y_mean.shape
        dist = MultivariateNormal(loc=y_mean.view(-1),
                                  covariance_matrix=y_std.view(-1)**2 * torch.eye(len(y_std.view(-1))))
        return dist.sample().view(shape)