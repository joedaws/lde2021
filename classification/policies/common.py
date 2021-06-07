"""
Defines the Policy interface
"""
from abc import ABC, abstractmethod
from random import shuffle

import numpy as np
from pandas import DataFrame


class Policy(ABC):
    """Generic policy interface"""
    @abstractmethod
    def predict(self, feature_vector: np.ndarray) -> str:
        """Make prediction given feature vector.

        returns the label with the highest score.
        """
        pass

    @abstractmethod
    def to_matrix(self, feature_list: list, loss_index_map: dict) -> object:
        """Creates a matrix which represents the policy."""
        pass


class BehaviorPolicyHelper:
    """Policy used to generate the historical data.

    distribution type is either 'uniform' or 'gaussian'

    """
    def __init__(self, labels: list, test_data: list, distribution_type: str = 'gaussian'):
        self.labels = labels
        self.num_labels = len(self.labels)
        self.distribution_type = distribution_type
        self._p = None
        self.reset_p(num_data=len(test_data))

    def reset_p(self, num_data):
        if self.distribution_type == 'uniform':
            self.p = [self.get_uniform_p() for _ in range(num_data)]
        elif self.distribution_type == 'gaussian':
            self.p = [self.get_gaussian_p() for _ in range(num_data)]
        else:
            raise NotImplementedError(f'{self.distribution_type} unknown')

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, new_p):
        self._p = new_p

    def choose_label(self, index):
        # return RNG.choice(self.labels, p=list(self.p[index].values()))
        return np.random.choice(self.labels, p=list(self.p[index].values()))

    def get_probability_of_label(self, label) -> float:
        """Returns probability of choosing label given feature vector"""
        return self.p[label]

    def get_gaussian_p(self):
        # p = RNG.normal(size=self.num_labels)
        p = np.random.randn(self.num_labels)
        # normalize p
        p = np.exp(p) / np.sum(np.exp(p))
        return {label: p[i] for i, label in zip(list(range(self.num_labels)), self.labels)}

    def get_uniform_p(self):
        return {label: 1/self.num_labels for i, label in zip(list(range(self.num_labels)), self.labels)}


class ClassifierPolicyHelper:
    """Creates a policy on the test_data with a specified value.

    test data has the form
    (feature_vector, loss_vector, label)
    """
    def __init__(self, data: list, labels: list, value: float):
        if value > 1.0 or value < 0.0:
            raise ValueError(f'{value} passed. Number should be between 0 and 1 inclusive.')

        self.data = data
        self.labels = labels
        self.value = value
        self._policy = None
        self.set_policy(data, value)

    @property
    def policy(self):
        return self._policy

    @policy.setter
    def policy(self, new_policy):
        self._policy = new_policy

    def set_policy(self, data, value):
        """Choose which elements of the test set will be correct and which will not be correct so
        that the resultant policy has the prescribed value."""
        n = len(data)

        # number of correct labels to attain value
        num_correct = int(value*n)

        # choose data which are true
        corrects = RNG.choice(list(range(n)), replace=False, size=num_correct)

        p = []
        for i, t in enumerate(data):
            if i in corrects:
                p.append(t[2])

            else:
                wrong_labels = [label for label in self.labels if label != t[2]]
                p.append(RNG.choice(wrong_labels))

        self._policy = p

    def predict(self, feature_vector: np.ndarray) -> str:
        # look up index of feature vector
        for i, t in enumerate(self.data):
            if np.all(t[0] == feature_vector):
                return self.policy[i]

        else:
            raise ValueError(f'WHOOPS made it to the end')


class PolicySetupStrategy(ABC):
    """
    Sets up the policy
    """
    @abstractmethod
    def make_policy(self, data: DataFrame) -> Policy:
        pass


class TrainStrategy(ABC):
    """
    Strategy for training a policy.
    """
    @abstractmethod
    def train(self, policy: Policy, training_data: list, loss_index_map: dict, num_steps: int) -> Policy:
        pass


class TrainStepStrategy(ABC):
    @abstractmethod
    def step(self, step_num: int, policy: Policy, training_example: tuple, **kwargs) -> Policy:
        pass


class Trainer(TrainStrategy):
    """
    This is the context part of a strategy design pattern.
    """
    def __init__(self, train_step_strategy: TrainStepStrategy):
        self._train_step_strategy = train_step_strategy

    def train(self, policy: Policy, training_data: list, num_steps: int, **kwargs) -> Policy:
        """Use the training strategy to train a policy"""
        old_policy = policy
        for t, example in enumerate(self.training_example_generator(training_data, num_steps)):
            new_policy = self._train_step_strategy.step(t, old_policy, example, **kwargs)
            old_policy = new_policy

        return old_policy

    @staticmethod
    def training_example_generator(training_data: list, num_steps: int):
        """Creates a generator for use in training.

        Since we don't want to modify the training data itself we
        shuffle a list of indices instead of the training data list.
        """
        num_train = len(training_data)
        idx_list = list(range(num_train))
        for epoch in range(num_steps // num_train):
            for idx in idx_list:
                yield training_data[idx]
            shuffle(idx_list)
        for idx in idx_list[0:(num_steps % num_train)]:
            yield training_data[idx]