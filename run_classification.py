"""
      ___       ___           ___
     /\__\     /\  \         /\  \
    /:/  /    /::\  \       /::\  \
   /:/  /    /:/\:\  \     /:/\:\  \
  /:/  /    /:/  \:\__\   /::\~\:\  \
 /:/__/    /:/__/ \:|__| /:/\:\ \:\__\
 \:\  \    \:\  \ /:/  / \:\~\:\ \/__/
  \:\  \    \:\  /:/  /   \:\ \:\__\
   \:\  \    \:\/:/  /     \:\ \/__/
    \:\__\    \::/__/       \:\__\
     \/__/     ~~            \/__/

Limited Data Estimator -- LDE

This module reproduces the results of the classification example from our NeurIPS 2021 submission.
"""

from classification.policies.dlm import DLMPolicySetup, DLMTrainStepStrategy
from classification.comparator import Comparator
from classification.policies.common import Trainer
from classification.data_utils.loader import load_df
import argparse


def setup():
    '''setup the experiment'''
    parser = argparse.ArgumentParser(description='Data processor argument parser')
    parser.add_argument('-d', '--data_names',
                        default=['abalone', 'algerian', 'ecoli', 'glass', 'winequality'],
                        nargs='+',
                        help='list of dataset names to load')
    parser.add_argument('-s', '--seed',
                        default=2021,
                        help='value of random seed')
    parser.add_argument('-save', '--save', action='store_true')
    parser.add_argument('-load', '--load', action='store_true')
    # parse the arguments
    args = parser.parse_args()
    print(f'Will {"load" if args.load else "perform"} the experiment for: ')
    for name in args.data_names:
        print(f'    {name}')
    print()
    return args.data_names, int(args.seed), args.save, args.load


if __name__ == '__main__':
    data_names, seed, save, load = setup()
    dlm_train_step = DLMTrainStepStrategy()
    trainer = Trainer(train_step_strategy=dlm_train_step)
    comparator = Comparator(DLMPolicySetup(), trainer)
    if load:
        comparator.load_variables('Classification.pkl')
    else:
        for data_name in data_names:
            data = load_df(data_name)
            comparator.compare(data, data_name, seed)
        if save:
            comparator.save_variables()
    comparator.report_results(data_names)
