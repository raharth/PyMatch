import torch
from torch.distributions import Categorical, MultivariateNormal
import torch.nn.functional as F
import numpy as np
import pymatch.DeepLearning.hat as hat
from pymatch.utils.functional import eval_mode
from pymatch.DeepLearning.pipeline import Pipeline


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

    def forward(self, agent, observation):
        raise NotImplementedError

    def pre_pipe(self, x):
        for pipe in self.pre_pipeline:
            x = pipe(x)
        return x

    def post_pipe(self, x):
        for pipe in self.post_pipeline:
            x = pipe(x)
        return x

    # def __call__(self, agent, observation):
    #     observation = self.preparation(observation)
    #     x = self.forward(agent, observation)
    #     return self.aggregation(x)

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
        return action.item(), log_prob



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
        return action.item(), log_prob


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
        return action.item()


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
        return action.item(), stds


class EpsilonGreedyActionSelection(SelectionPolicy):
    def __init__(self, action_space, epsilon=.1, **kwargs):
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
        agent.to(agent.device)
        observation = self.pre_pipe(observation)
        qs = agent(observation.to(agent.device))
        qs = self.post_pipe(qs)
        if np.random.uniform() > self.epsilon:
            return qs.argmax().item()
        return np.random.choice(self.action_space)


class GreedyValueSelection(SelectionPolicy):
    """
    Choosing the best possible option, necessary for evaluation
    """
    def __call__(self, agent, observation):
        observation = self.pre_pipe(observation)
        qs = agent(observation.to(agent.device))
        qs = self.post_pipe(qs)
        return qs.argmax().item()


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
        with torch.no_grad():   # @todo why am I using a no grad here? This policy can then not be used for training?
            with eval_mode(agent):
                y_mean, y_std = pipeline(observation)
        shape = y_mean.shape
        dist = MultivariateNormal(loc=y_mean.view(-1),
                                  covariance_matrix=y_std.view(-1)**2 * torch.eye(len(y_std.view(-1))))
        return dist.sample().view(shape)
