import itertools


class HyperparameterSearch:

    def __init__(self, performance_measure, factory, factory_args, training_args):
        """

        Args:
            performance_measure: object that implements a 'evaluate' function returning some performance measure
            factory: factory function returning a trainable model
            factory_args: arguments used for factory other than hyperparams
            training_args: arguments used for training(epochs, device, checkpoint_int, validation_int, restore_early_stopping)
        """
        self.performance_measure = performance_measure
        self.factory_args = factory_args
        self.factory = factory
        self.training_args = training_args

        self.models = []
        self.performances = []
        self.best_performance = 0
        self.best_model = None
        self.best_hyperparams = None

    def optimize(self):
        raise NotImplementedError

    def _optimize(self, learner_args):
        model = self.factory(**self.factory_args, **learner_args)
        self.models += [model]
        model.train(**self.training_args)
        # model.train(epochs=run_epochs, device=device, checkpoint_int=checkpoint_int, validation_int=validation_int,
        #             restore_early_stopping=restore_early_stopping)
        performance = self.performance_measure.evaluate(model)
        self.performances += [performance]
        if performance > self.best_performance:
            self.best_model = model
            self.best_performance = performance
            self.best_hyperparams = learner_args


class GridSearch(HyperparameterSearch):

    def __init__(self, performance_measure, factory, factory_args, training_args, search_space):
        """
        Args:
            factory_args:

        """
        super(GridSearch, self).__init__(performance_measure, factory, factory_args, training_args)
        self.search_space = search_space
        self.hyperparameter_names = search_space.keys()
        self.values = [self.search_space[k] for k in self.search_space]

    def optimize(self):
        for hyperparams in itertools.product(*self.values):
            learner_args = {key: hyp for (key, hyp) in zip(self.hyperparameter_names, hyperparams)}
            self._optimize(learner_args)

        return self.best_model


class RandomSearch(HyperparameterSearch):

    def __init__(self, performance_measure, factory, factory_args, training_args, n_samples, hyperparameter_samplers:  dict):
        super(RandomSearch, self).__init__(performance_measure, factory, factory_args, training_args)
        self.n_samples = n_samples
        self.hyperparameter_names = hyperparameter_samplers.keys()
        self.hyperparameter_samplers = hyperparameter_samplers

    def optimize(self):
        for i in range(self.n_samples):
            hyperparams = self.sample_hyperparams()
            self._optimize(hyperparams)

        return self.best_model

    def sample_hyperparams(self):
        hyperparams = [self.hyperparameter_samplers[hyperparam].sample() for hyperparam in self.hyperparameter_samplers]
        return {key: hyp for (key, hyp) in zip(self.hyperparameter_names, hyperparams)}


class IterativeRandomSearch(HyperparameterSearch):

    def __int__(self, performance_measure, factory, factory_args, training_args, n_samples, n_iterations, hyperparameter_samplers:  dict, shrinking):
        super(IterativeRandomSearch, self).__init__(performance_measure, factory, factory_args, training_args)
        self.n_iterations = n_iterations
        self.n_samples = n_samples
        self.hyperparameter_samplers = hyperparameter_samplers
        self.shrinking = shrinking

    def optimize(self):
        for iteration in range(self.n_iterations):
            search = RandomSearch(self.performance_measure, self.factory_args, self.n_samples, self.hyperparameter_samplers)
            search.optimize()
            for center, sampler in zip(search.best_hyperparams, self.hyperparameter_samplers):
                sampler.shrink(center, self.shrinking)
        return search
