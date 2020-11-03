# PyMatch

PyMatch is a PyTorch wrapper library, a little like keras is for tensorflow. Though, in difference to keras it is not trying to simplify process of building a network, but aims to abstract the learning frameworks, especially when it comes to Reinforcement Learning or ensemble learning, and data handling as well as the infrastructure around the model. Therefore it  leaves the architectural design untouched, building a hull around the actual network, giving the user full flexibility. The big advantage is that the you will not have to deal with the overhead of loading and storing things, creating plots or debugging issues with the backprop or devices, but that you can focus on building the network architecture and hyper-parameters of your models - though the latter could also be automized.

## DeepLearning

### The Learner
The most fundamental abstraction of PyMatch is the so called `leaner`. A learner is simply a hull around a network (`torch.nn.Module`), a data loader, a loss function and an optimizer. As with keras a learner provides a `fit()` method, that trains the model on the provided data with the the given loss and optimizer. It also provides an easy way of storing and loading your model, including the optimizer and the loss, restoring their states. It also holds a member `train_dict` which can be used to store all kinds of metrics and values over the training process. This `train_dict` is also stored and loaded automatically, which frees the user of annoying and unnecessary data-loss or juggling saving and loading operations all over the learning script. In addition learners can be moved and trained on devices, so the tiresome swapping of variables from and to devices is handled by the learner itself. 

#### Callbacks
A can leaner also have a list of `callbacks`. Those callbacks are checked to be executed after each epoch and can be used to either compute and store values (e.g. to the `train_dict`) or to create plots during training. Out of the box there are callbacks like a model evaluation, checkpointing, early stopping/early termination or a metric plotter.

#### ClassificationLearner and RegressionLearner
Basically the only dofference between those two are the metrics stored during training, besides that the model remains the same

#### Inheriting from the Learner class
The most important method to redefine when inheriting from the `learner` is `fit_epoch()`. This method specifies how data is loaded and fed forward and how the output is then used to update the weights. As long as additional variables are stored in the `train_dict` of the learner there is no further adjustment necessary, but in the case, that the learner has additional member variables, `create_state_dict()` and `restore_checkpoint()` have to be redefined as well (don't forget to call their super() equivalents as well). The basic idea is that they create and pass a dictionary containing all necessary variables through the stack of inheritance. Redefining these methods is sufficient to properly store and load your model.

#### Examples
##### Auto-Encoder
An interesting example is the is the auto-encoder, when implementing this using PyMatch there are just few adjustments to make. Basically all you have to do is to build a `nn.Module` with the two sub-modules use them with a RegressionLearner and a dataloader that provides X as X and Y (which can also be done using a pipeline element). After those adjustments you can use your standard training script. All in all depending on the size of your auto-encoder, those changes can be done in a handful lines of code.

### Experiments
PyMatch provides a very simple experiment documentation class, which exist in two versions: a homebrew one, storing everything on your local machine and one that uses the [weights and bias (wandb)](https://www.wandb.com/) interface, monitoring even the hardware of your machine. Though to use wandb you need to create your own account. PyMatch Experiments are designed to be used with three files defining the experiment:
1. a model-file: This file contains the nn.Module only requiring a defined `foward()` method
2. a factory-file: This file contains a factory method for the learner, i.e. it defines how the learner is created which loss and crit are used etc.
3. a parameter-file: This file defines all additional hyper-parameter that are necessary to create and train the model

All three files can be easily loaded by a simple call of the `Experiment` class, provided by PyMatch. Using this structure enables the user to use a single training script for all experiment runs and it ensures that an experiment always stays reproducible. In addition the `Experiment` class can store the training script in the experiment folder, by a single method call.

This structure is particularly useful when running multiple experiments with different architectures and hyper-parameters and is required when using the `Ensemble` class provided by PyMatch

### Ensemble
PyMatch provides a simple way of constructing ensembles. In principal the `Ensemble` is just a wrapper around a list of `learners`, but it also has its own loading and storing method, it's own `train_dict` and `callbacks`. When using the experiment structure as suggested above there is nearly no adjustment to make to go from a single learner to an ensemble of learners. Though, using an ensemble disables you to work with wandb, since wandb is only able to store weights and metrics for a single model, but not multiple models wrapped in a single one (though you could store all of them individually).

The `Ensemble` comes with a `KFold` data folder, that can be used in the factory. The Ensemble can also be used to easily implement Neural Network Boosting and automated hyper-parameter searches, if the factory of the ensemble is randomized or used with a predefined set of different hyper-parameters.

Ensembles provide the same `fit()` method as learners do, but they can either train the learners one by one or partition the training of the learners. After each of those partitions the `callbacks` of the ensemble are called, e.g. this can be the evaluation of the ensemble as a whole, or to plot metrics from the learners or the ensemble.

### Hats
Hats are meant to be placed on the head of a network or ensemble. The idea is to keep the aggregation and handling of the models output as flexible as possible. Hats can also be stacked on the head of a network only requiring that the output and input of consecutive hats match. This is particularly useful when dealing with ensembles, for either the aggregation of the different models or certainty estimations on the predicted values (e.g. estimating the standard deviation or the entropy). Hats are usually not part of the training process but could be incorporated in the learner as well.

### Pipelines 
The `Pipeline` provided by PyMatch is a very rudimentary approach. It basically only hold a list of elements (learner, ensemble, hats, etc.) which all have to provide a `__call__()` method. 

#### MC Dropout Ensemble
Implementing MC Dropout using Pipelines and hats is a straightforward thing to do. All there is to do is to use the `InputRepeater` as the first element of the Pipeline, with a model that has a redefined `train()` method, not setting the Dropout layers to eval-mode. The later pipeline elements are then the same as for a regular ensemble, dealing with the multiple predictions of the same datapoint.  

## Reinforcement Learning
PyMatch also implements a number of Reinforcement Learning (RL) algorithms. They are build upon the OpenAI gym interface and can be used with any environment that fulfills that interface.

### Learners (also called agents)
As of by now there are 3 main learning frameworks implemented:
1. Policy Gradient (off- and on-policy depending on the memory chosen) `PolicyGradient` 
2. Deep Q-Learning `DQN`
3. Double Deep Q-Learning `DoubleDQN`
(4. SARSA - though it is still buggy) `SARSA`
(5. Actor-Critics - coming soon) `AC3`

The learners themselves are derived from the `learner` and follow the same concept, though the main difference is that they implement a `play_episode()` method, that can be used to fill a memory. The memory itself also fulfills the interface of a dataloader and replaces the regular dataloader of the usual learner. In addition to the modules of a standard learner, they also have a `selection_policy` and a `memory_updater`. The selection policy determines in which way the leaner/agent selects the next action (during training), the memory updater is used to replace older memories with fresh ones and discount the reward in the case of a PolicyGradient 

### Memory
The memory is one of the core elements of a RL algorithm and replaces the dataloader in comparison to a regular learner. Hence, it implements the same interface as a dataloader, but in difference to that it is not based on a static dataset, but an updatable one. It gives the option of memorizing single steps or merging a memory into the current one. It also randomizes the data by bootstrapping it. When using a `PolicyGradient` the memory can also be used to discount future rewards.

A Memory comes with a memory updater, which is basically the strategy of how to replace old memories with new ones.

### Callbacks
RL learners have their own set of callbacks, especially when it comes to evaluation of the model. In this case the callback also has a selection policy, usually a greedy strategy.

### Selection Policy
`selection_policy` are strategies to select actions, this can be greedy, epsilon-greedy, softmax, Thompson Sampling, or any other arbitrary selection strategy. 

## Install
Necessary to monitor the hardware
https://openhardwaremonitor.org/downloads/