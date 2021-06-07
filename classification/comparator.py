"""
Implements a comparison class
"""

import os
import pickle
import random
from typing import Callable

import numpy as np

from classification.data_utils.transformer import make_loss_index_map
from classification.data_utils.transformer import make_single_interaction_dataset
from classification.data_utils.transformer import split_df_train_test
from classification.plots.make_plots import make_plots
from classification.policies.common import BehaviorPolicyHelper, PolicySetupStrategy, TrainStrategy
from classification.util import compute_classification_error
from classification.util import compute_direct_method
from classification.util import compute_doubly_robust_estimator
from classification.util import compute_inverse_propensity_score
from classification.util import compute_limited_data_estimator

PERCENT_TRAIN = .5
NUM_POLICIES = 10
EPOCHS = 1000
NUM_TRAINING_ROUNDS = 10
NUM_DATA_ROUNDS = 100
DATA_PERC = .8


class ComparatorData:
    """Holds data generated during a comparison run. Methods for saving and loading"""
    METHOD_KEYS = ['true', 'dim', 'ips', 'dre', 'lde']

    def __init__(self):
        self.ranks = {key: [] for key in self.METHOD_KEYS}
        self.metrics = {key: [] for key in self.METHOD_KEYS}
        self.scores = {key: [] for key in self.METHOD_KEYS}
        self.values = {key: [] for key in self.METHOD_KEYS}
        self._average_scores = None
        self._score_stds = None

    def average_score(self, method: str):
        """average score of a certain method"""
        return np.mean(self.scores[method])

    def score_std(self, method: str):
        """Standard deviation of observed scores"""
        return np.std(self.scores[method])

    def average_error(self, method: str):
        return np.mean(np.array(self.metrics[method]) - np.array(self.metrics['true']))

    def error_std(self, method: str):
        return np.std(np.array(self.metrics[method]) - np.array(self.metrics['true']))

    def push_rank(self, method: str, new_rank):
        self.ranks[method].append(new_rank)

    def push_metric(self, method: str, new_metric):
        self.metrics[method].append(new_metric)

    def push_score(self, method: str, new_score):
        self.scores[method].append(new_score)

    def push_value(self, method: str, new_value):
        self.values[method].append(new_value)


class Comparator:
    """Uses metric computer to compare a list of policies.
    """

    def __init__(self,
                 policy_setup_strategy: PolicySetupStrategy,
                 train_strategy: TrainStrategy):
        self._policy_setup_strategy = policy_setup_strategy
        self._train_strategy = train_strategy
        self.comp_data = {}

    def compare(self, data, data_name, seed=2021):
        """Evaluates a policy using the provided strategies"""

        # fix random seed for reproducibility
        np.random.seed(seed)
        random.seed(seed)

        train_data, test_data = split_df_train_test(data=data, perc_train=PERCENT_TRAIN)

        # make loss index map
        loss_index_map = make_loss_index_map(data)
        loss_index_map_keys = list(loss_index_map.keys())

        behavior_policy = BehaviorPolicyHelper(labels=loss_index_map_keys,
                                               test_data=test_data,
                                               distribution_type='gaussian')

        # placeholder matrix which is filled in below
        r_hat_matrix = self.get_r_hat_matrix(test_data, loss_index_map)

        # create new data object for storing data for this experiment
        exp_data = ComparatorData()

        print('----------------------------')
        print(data_name)
        print('----------------------------')
        for training_round in range(NUM_TRAINING_ROUNDS):
            print(f'running test {training_round+1}/{NUM_TRAINING_ROUNDS}...')
            trained_policies = []
            observed_values = []
            unique_policy_count = 0

            # to avoid comparing the same two policies we search for NUM_POLICIES distinct policies
            while unique_policy_count < NUM_POLICIES:
                # prepare policy to be evaluated.
                policy = self._policy_setup_strategy.make_policy(data)
                policy = self._train_strategy.train(policy=policy,
                                                    training_data=train_data[:int(DATA_PERC*len(train_data))],
                                                    loss_index_map=loss_index_map,
                                                    num_steps=len(train_data) * EPOCHS)

                # add policy to list if it has a different value
                this_value = compute_classification_error(policy, test_list=test_data)
                if this_value not in observed_values:
                    observed_values.append(this_value)
                    trained_policies.append(policy)
                    unique_policy_count += 1

                else:
                    # to increase chance of finding a new policy, the training data are shuffled
                    random.shuffle(train_data)

            true_ranks, true_metrics = get_ranks(policy_list=trained_policies,
                                                 metric_fn=compute_classification_error,
                                                 metric_kwargs={'test_list': test_data})

            for data_round in range(NUM_DATA_ROUNDS):
                # the estimators use scarce data instead of test data
                scarce_data = make_single_interaction_dataset(data_list=test_data,
                                                              loss_index_map=loss_index_map,
                                                              policy=behavior_policy)

                # fix a constant baseline for all examples
                baseline_constant = self.get_baseline_constant(scarce_data=scarce_data)

                # previous versions used to utilize a dynamic r hat matrix, now we use a static one
                r_hat_matrix = baseline_constant * np.ones_like(r_hat_matrix)

                dim_ranks, dim_metrics = get_ranks(policy_list=trained_policies,
                                                   metric_fn=compute_direct_method,
                                                   metric_kwargs={'data_list': scarce_data})

                ips_ranks, ips_metrics = get_ranks(policy_list=trained_policies,
                                                   metric_fn=compute_inverse_propensity_score,
                                                   metric_kwargs={'scarce_data': scarce_data,
                                                                  'behavior_policy': behavior_policy})

                dre_ranks, dre_metrics = get_ranks(policy_list=trained_policies,
                                                   metric_fn=compute_doubly_robust_estimator,
                                                   metric_kwargs={'loss_index_map': loss_index_map,
                                                                  'scarce_data': scarce_data,
                                                                  'r_hat_matrix': r_hat_matrix,
                                                                  'behavior_policy': behavior_policy})

                lde_ranks, lde_metrics = get_ranks(policy_list=trained_policies,
                                                   metric_fn=compute_limited_data_estimator,
                                                   metric_kwargs={'loss_index_map': loss_index_map,
                                                                  'scarce_data': scarce_data,
                                                                  'r_hat_matrix': r_hat_matrix})

                exp_data.push_score(method='true', new_score=1.0)  # the truth is perfect
                exp_data.push_score(method='dim', new_score=score(true_ranks, dim_ranks))
                exp_data.push_score(method='ips', new_score=score(true_ranks, ips_ranks))
                exp_data.push_score(method='dre', new_score=score(true_ranks, dre_ranks))
                exp_data.push_score(method='lde', new_score=score(true_ranks, lde_ranks))
                exp_data.push_metric(method='true', new_metric=true_metrics)
                exp_data.push_metric(method='dim', new_metric=dim_metrics)
                exp_data.push_metric(method='ips', new_metric=ips_metrics)
                exp_data.push_metric(method='dre', new_metric=dre_metrics)
                exp_data.push_metric(method='lde', new_metric=lde_metrics)
                exp_data.push_rank(method='true', new_rank=true_ranks)
                exp_data.push_rank(method='dim', new_rank=dim_ranks)
                exp_data.push_rank(method='ips', new_rank=ips_ranks)
                exp_data.push_rank(method='dre', new_rank=dre_ranks)
                exp_data.push_rank(method='lde', new_rank=lde_ranks)

        # save the experiment result
        self.comp_data[data_name] = exp_data

    def report_results(self, data_names):
        '''report results of policy comparison and evaluation'''
        for data_name in data_names:
            exp_data = self.comp_data[data_name]
            print(f'\n---------------------')
            print(f'Policy Comparison on {data_name}')
            print(f'---------------------')
            print(f'LDE | {exp_data.average_score("lde"):.4f} ({exp_data.score_std("lde"):.3f})')
            print(f'DRE | {exp_data.average_score("dre"):.4f} ({exp_data.score_std("dre"):.3f})')
            print(f'IPS | {exp_data.average_score("ips"):.4f} ({exp_data.score_std("ips"):.3f})')
            print(f'DiM | {exp_data.average_score("dim"):.4f} ({exp_data.score_std("dim"):.3f})')
            print(f'---------------------')
            print(f'Policy Evaluation on {data_name}')
            print(f'---------------------')
            print(f'LDE | {exp_data.average_error("lde"): .2e} ({exp_data.error_std("lde"):.3f})')
            print(f'DRE | {exp_data.average_error("dre"): .2e} ({exp_data.error_std("dre"):.3f})')
            print(f'IPS | {exp_data.average_error("ips"): .2e} ({exp_data.error_std("ips"):.3f})')
            print(f'DiM | {exp_data.average_error("dim"): .2e} ({exp_data.error_std("dim"):.3f})')
        make_plots(data_names=data_names, comp_data=self.comp_data)

    @staticmethod
    def get_r_hat_matrix(testing_data, loss_index_map):
        """Returns a placeholder for the r hat matrix."""
        # the placeholder has size num_test_examples x num_classes
        num_test_examples = len(testing_data)
        num_classes = len(loss_index_map)
        return np.zeros((num_test_examples, num_classes))

    @staticmethod
    def get_baseline_constant(scarce_data: list):
        return np.mean([1 - t[1] for t in scarce_data])

    def save_variables(self, path='./save/'):
        '''save class variables to a file'''
        os.makedirs(path, exist_ok=True)
        with open(path + 'Classification.pkl', 'wb') as save_file:
            pickle.dump(self.__dict__, save_file)

    def load_variables(self, save_name, path='./save/'):
        '''load class variables from a file'''
        exp_data = ComparatorData()
        try:
            with open(path + save_name, 'rb') as save_file:
                self.__dict__.update(pickle.load(save_file))
        except:
            raise NameError(f'\ncannot load file {save_name}...')


def get_ranks(policy_list: list, metric_fn: Callable, metric_kwargs: dict):
    """returns ranking of the policies according to the provided computer.

    0 is worst len(policy_list)-1 is best
    """
    policy_metrics = [metric_fn(policy, **metric_kwargs)
                      for policy in policy_list]
    # converting numpy int64 to regular python ints when making policy_ranks
    policy_ranks = list(map(lambda x: int(x), np.argsort(policy_metrics)))
    return policy_ranks, policy_metrics


def score(true_ranks: list, computed_ranks: list):
    """compute a score for the provided computed ranking of policies.

    1 - (number of swaps required to get to truth)/(number of pairs of policies)
    """
    number_of_policies = len(true_ranks)
    number_of_pairs = number_of_policies * (number_of_policies - 1) / 2
    number_of_swaps = swap_count(true_ranks, computed_ranks)
    return 1 - number_of_swaps / number_of_pairs


def swap_count(list1: list, list2: list) -> float:
    """count the number of swaps required to transform array1 into array2"""
    L = list(list2)
    swaps = 0
    for element in list(list1):
        ind = L.index(element)
        L.pop(ind)
        swaps += ind
    return swaps
