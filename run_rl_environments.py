"""

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     __           ______       ______
    /_/\         /_____/\     /_____/\
    \:\ \        \:::_ \ \    \::::_\/_
     \:\ \        \:\ \ \ \    \:\/___/\
      \:\ \____    \:\ \ \ \    \::___\/_
       \:\/___/\    \:\/.:| |    \:\____/\
        \_____\/     \____/_/     \_____\/

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Limited Data Estimator -- LDE

This module reproduces the results of the reinforcement learning example from our NeurIPS 2021 submission.
"""

from rl_environments.RLBanditEnv import RLBanditEnv
import argparse


def setup():
    '''setup the experiment'''
    parser = argparse.ArgumentParser(description='argument parser for example 3')
    parser.add_argument('-d', '--environments',
                        default=['main'],
                        nargs='+',
                        help='list of environments to perform policy comparison on')
    parser.add_argument('-s', '--seed',
                        default=2021,
                        help='value of random seed')
    parser.add_argument('-save', '--save', action='store_true')
    parser.add_argument('-load', '--load', action='store_true')
    # parse the arguments
    args = parser.parse_args()
    if args.environments == ['main']:
        args.environments = ['InvertedPendulumBulletEnv-v0',
                             'ReacherBulletEnv-v0',
                             'Walker2DBulletEnv-v0',
                             'AntBulletEnv-v0']
    if args.environments == ['all']:
        args.environments = ['InvertedPendulumBulletEnv-v0',
                             'InvertedPendulumSwingupBulletEnv-v0',
                             'ReacherBulletEnv-v0',
                             'Walker2DBulletEnv-v0',
                             'HalfCheetahBulletEnv-v0',
                             'AntBulletEnv-v0',
                             'HopperBulletEnv-v0',
                             'HumanoidBulletEnv-v0']
    print(f'Will {"load" if args.load else "perform"} the experiment for: ')
    for env in args.environments:
        print(f'    {env}')
    print()
    return args.environments, int(args.seed), args.save, args.load


if __name__ == '__main__':
    env_names, random_seed, save, load = setup()
    for env_name in env_names:
        params = {'env_name': env_name, 'train_steps': 1000, 'env_steps': 100}
        env = RLBanditEnv(params)
        if load:
            env.load_variables(env_name + '.pkl')
        else:
            env.run_tests(num_tests=15, num_sims=15, seed=random_seed)
            if save:
                env.save_variables()
        env.report_scores()

