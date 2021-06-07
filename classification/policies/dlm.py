"""
Direct loss minimization policies and functions
"""
import operator
from typing import List

import numpy as np
from pandas import DataFrame

from classification.policies.common import Policy, PolicySetupStrategy, TrainStepStrategy

EPSILON = 0.1


class DLMPolicy(Policy):
    """Deterministic policy for direct loss minimization for classification."""
    def __init__(self, num_features: int, labels: List[str]) -> None:
        self.num_features = num_features
        self.labels = labels
        self.theta = self._init_theta()

    def _init_theta(self) -> dict:
        # return {label: np.ones(self.num_features) + np.random.randn(self.num_features) for label in self.labels}
        return {label: np.random.randn(self.num_features) for label in self.labels}

    def predict(self, feature_vector: np.ndarray) -> str:
        """Make prediction given feature vector.

        returns the label with the highest score.
        """
        # dot product of feature vector and all thetas
        products = {label: np.dot(feature_vector, self.theta[label])
                    for label in self.labels}
        return max(products.items(), key=operator.itemgetter(1))[0]

    def get_a1(self, feature_vector: np.ndarray, loss_vector: np.ndarray, loss_index_map: dict) -> str:
        products = {label: np.dot(feature_vector, self.theta[label]) - EPSILON*loss_vector[loss_index_map[label]]
                    for label in self.labels}
        return max(products.items(), key=operator.itemgetter(1))[0]

    def to_matrix(self, feature_list: list, loss_index_map: dict) -> object:
        """Creates a matrix which represents the policy."""
        probs = np.eye(len(loss_index_map))
        return np.array([probs[loss_index_map[self.predict(feature_vector)]] for feature_vector in feature_list])


def make_policy(data: DataFrame) -> Policy:
    """Make a policy based on the provided dataframe.

    This is kind of like a factory method.
    """
    num_features = len([name for name in data.columns if 'feature_' in name])
    labels = list(data['label'].unique())
    return DLMPolicy(num_features=num_features, labels=labels)


class DLMPolicySetup(PolicySetupStrategy):
    def make_policy(self, data: DataFrame) -> Policy:
        return make_policy(data)


class DLMTrainStepStrategy(TrainStepStrategy):
    @classmethod
    def step(cls, step_num: int, policy: DLMPolicy, training_example: tuple, **kwargs) -> DLMPolicy:
        eta = (step_num + 1) ** (-0.3) / 2  # since t starts at 0 need to + 1
        feature_vector = training_example[0]
        loss_vector = training_example[1]

        # get a1
        a1 = policy.get_a1(feature_vector, loss_vector, kwargs.get('loss_index_map'))

        # get a2
        a2 = policy.predict(feature_vector)

        # make updates
        policy.theta[a1] = policy.theta[a1] + eta * feature_vector
        policy.theta[a2] = policy.theta[a2] - eta * feature_vector

        return policy