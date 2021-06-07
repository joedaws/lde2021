"""
Transforms data into forms suitable for training
"""
from typing import Tuple

import numpy as np
import pandas as pd

from classification.policies.common import BehaviorPolicyHelper


def make_loss_index_map(data: pd.DataFrame):
    """makes a mapping between a label and it's index in the loss vector"""
    labels = sorted(list(data['label'].unique()))

    # mapping between label name and corresponding index in loss vector
    loss_index_map = {name: i for i, name in enumerate(labels)}

    return loss_index_map


def split_df_train_test(data: pd.DataFrame, perc_train: float) -> Tuple[list, list]:
    """Create train test split"""
    # get percentages
    assert perc_train >= 0.0
    assert perc_train <= 1.0

    # get number of examples to use for training
    num_train = int(perc_train * len(data))

    # get number of examples to use for testing
    num_test = len(data) - num_train

    # get all labels
    labels = sorted(list(data['label'].unique()))
    label_to_index_map = {label: i for i, label in enumerate(labels)}

    # make sure to get some examples from each class
    num_per_label = {label: len(data[data['label'] == label]) for label in labels}

    # initialize dictionaries
    train = {label: None for label in labels}
    test = {label: None for label in labels}

    # TODO replace all data_one_hot references with np.eye rows.
    reward_vecs = np.eye(len(labels))

    # get feature columns
    feature_cols = [col for col in data.columns if 'feature_' in col]

    # perform test train split for each label
    for label in labels:
        # exact dataframe associated with current label
        label_df = data[data['label'] == label]

        # randomly select some rows to train with
        label_df_train = label_df.sample(
            n=int(perc_train * num_per_label[label]),
            replace=False
        )

        # store the remaining rows as test rows
        complement = [idx for idx in label_df.index if idx not in label_df_train.index]
        label_df_test = label_df[label_df.index.isin(complement)]

        # define processor function
        def get_feature_list(df):
            """
            Obtain a tuple for each row in the current label dataframe:
            (feature vector, loss or reward vector, label)
            """
            return [(feature[1].values, 1-reward_vecs[label_to_index_map[label]], label)
                    for feature in df[feature_cols].iterrows()]

        # set up training list examples
        # Each element of this list is a tuple (feature vec, loss vec, label)
        train[label] = get_feature_list(label_df_train)

        # set up test list examples
        # Each element of this list is a tuple (feature vec, loss vec, label)
        test[label] = get_feature_list(label_df_test)

    # flatten dictionary
    train_list = [item for sublist in train.values() for item in sublist]

    # flatten dictionary
    test_list = [item for sublist in test.values() for item in sublist]

    return train_list, test_list


def make_feature_list(data: pd.DataFrame) -> list:
    return [row[1].values for row in data[[f'feature_{i}' for i in range(1, 8)]].iterrows()]


def make_single_interaction_dataset(data_list: list, loss_index_map: map, policy: BehaviorPolicyHelper) -> list:
    """Make the transformed data set as described in section 5.1.1

    The result is a list with 3-tuples of the form
    (feature vector, loss_component, choosen label)
    """

    def choose_loss(vec, index):
        """Choose a random component of the vector"""
        # idx = RNG.choice(p=behavior_probabilities, axis=0, shuffle=True)
        label = policy.choose_label(index)
        return vec[loss_index_map[label]], label

    # create transformed list
    # It's not pretty but * is here to unpack tuple returned by choose_loss
    transformed_list = [(example[0], *choose_loss(example[1], index))
                        for index, example in enumerate(data_list)]

    return transformed_list


def make_reward_matrix(data: pd.DataFrame, loss_index_map: dict) -> np.ndarray:
    """Make a reward matrix for the given data"""
    label_series = data['label'].values
    num_labels = len(data['label'].unique())
    loss_vectors = np.eye(num_labels)
    reward_matrix = np.array([loss_vectors[loss_index_map[label]] for label in label_series])
    return reward_matrix


def reward_to_loss_matrix(reward: np.ndarray) -> np.ndarray:
    """Convert the reward matrix into a loss matrix.

    I had another more convoluted solution before:
    return np.array(list(map(lambda val: 1 - val, reward))).reshape(reward.shape)
    """
    return 1-reward


def make_loss_matrix(data: pd. DataFrame, loss_index_map: dict) -> np.ndarray:
    """Make a loss matrix for the given data"""
    reward = make_reward_matrix(data, loss_index_map)
    return reward_to_loss_matrix(reward)
