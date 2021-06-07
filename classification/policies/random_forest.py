"""
Random forest classifier policy
"""
from typing import List

import numpy as np
import sklearn
from numpy.random import default_rng
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from classification.policies.common import Policy, PolicySetupStrategy, TrainStrategy

rng = default_rng()
default_params = {'max_depth': 5, 'n_estimators': 30}


class RFPolicy(Policy):
    """Random forest classifier implemented as a policy."""
    def __init__(self, num_features: int, labels: List[str], rf_params: dict = default_params) -> None:
        if rf_params is None:
            rf_params = default_params
        self.num_features = num_features
        self.labels = labels
        self.clf = RandomForestClassifier(**rf_params)
        # do a meaningless fit to setup the classifier
        self.clf.fit(np.random.rand(len(labels), num_features), labels)

    def predict(self, feature_vector: np.ndarray) -> str:
        """Make prediction given feature vector.

        When it is not trained then just return a sample from uniform.

        returns the label with the highest score.
        """
        try:
            return self.clf.predict(feature_vector.reshape(1, -1))[0]
        except sklearn.exceptions.NotFittedError:
            return rng.choice(self.clf.classes_)

    def to_matrix(self, feature_list: list, loss_index_map: dict) -> object:
        """Creates a matrix which represents the policy."""
        probs = np.eye(len(loss_index_map))
        return np.array([probs[loss_index_map[self.predict(feature_vector)]]
                         for feature_vector in feature_list])


def make_policy(data: DataFrame, rf_params: dict = default_params) -> RFPolicy:
    """Makes a random forest classifier policy"""
    num_features = len([name for name in data.columns if 'feature_' in name])
    labels = list(data['label'].unique())
    return RFPolicy(num_features=num_features, labels=labels, rf_params=rf_params)


def train_rf(policy: RFPolicy, training_data: list) -> RFPolicy:
    # convert training data to X, y
    X, y = get_sklearn_x_y(training_data)

    # train the rf_classifier
    policy.clf.fit(X, y)

    return policy


def get_sklearn_x_y(training_data: list):
    X, _, y = zip(*training_data)
    return X, y


class RFPolicySetup(PolicySetupStrategy):
    def make_policy(self, data: DataFrame) -> Policy:
        return make_policy(data)


class RFTrain(TrainStrategy):
    def train(self, policy: RFPolicy, training_data: list, loss_index_map: dict, num_steps: int) -> Policy:
        # convert training data lists into numpy array
        X = np.array([t[0] for t in training_data])
        y = np.array([t[2] for t in training_data])
        # call policy.clf.fit()
        policy.clf.fit(X, y)
        # print score
        score = policy.clf.score(X, y)
        print(f'train score: {score}')
        return policy
