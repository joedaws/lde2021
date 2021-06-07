"""
==============================================
===                                        ===
===     _____      ______     ________     ===
===    |_   _|    |_   _ `.  |_   __  |    ===
===      | |        | | `. \   | |_ \_|    ===
===      | |   _    | |  | |   |  _| _     ===
===     _| |__/ |  _| |_.' /  _| |__/ |    ===
===    |________| |______.'  |________|    ===
===                                        ===
===                                        ===
==============================================

Limited Data Estimator -- LDE

This module reproduces the results of the synthetic data example from our NeurIPS 2021 submission.
"""

from synthetic_data.SyntheticBanditEnv import SyntheticBanditEnv
import argparse


def setup():
    '''setup the experiment'''
    parser = argparse.ArgumentParser(description='argument parser for example 1')
    parser.add_argument('-d', '--example',
                        default='1',
                        help='number of the reeard function: 1 or 2')
    parser.add_argument('-s', '--seed',
                        default=2021,
                        help='value of random seed')
    parser.add_argument('-save', '--save', action='store_true')
    parser.add_argument('-load', '--load', action='store_true')
    # parse the arguments
    args = parser.parse_args()
    print(f'Will {"load" if args.load else "perform"} '
          + f'the experiment for Example 1.{args.example}')
    return int(args.example), int(args.seed), args.save, args.load


if __name__ == '__main__':
    example, random_seed, save, load = setup()
    params = {'num_s': 100, 'num_a': 100, 'dom_s': [0,1], 'dom_a': [-1,1], 'example': example}
    env = SyntheticBanditEnv(params)
    if load:
        env.reproduce_pictures(f'Synthetic_{example}.pkl')
    else:
        test_params = {
            'a': {'a_min': .01, 'a_max': .25, 'num_a_tests': 49, 'num_m': 1, 'num_sims': 1000},
            'm': {'alpha': .01, 'm_min': 1, 'm_max': 500, 'num_m_tests': 50, 'num_sims': 1000},
            '3d': {'a_min': .01, 'a_max': .10, 'num_a_tests': 10, 'm_min': 1, 'm_max': 10,
                   'num_m_tests': 10, 'num_sims': 1000},
            'grid': {'a_min': .05, 'a_max': .15, 'num_a_tests': 3, 'm_min': 1, 'm_max': 1000,
                     'num_m_tests': 5, 'num_sims': 1000}}
        env.produce_pictures(test_params, seed=random_seed)
        if save:
            env.save_variables()

