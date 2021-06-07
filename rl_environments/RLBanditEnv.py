
import gym
import numpy as np
import torch
import stable_baselines3 as sb3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
import pybullet_envs
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid', palette=[sns.color_palette('colorblind')[i] for i in [0,3,4,2]])
np.set_printoptions(suppress=True, linewidth=100, precision=4)
pd.set_option('precision', 4)
gym.logger.set_level(40)
plt.rcParams['font.family'] = 'monospace'
plt.rcParams['font.weight'] = 'bold'


class RLBanditEnv:
    '''
        numerical experiment where the policies are trained on rl environments and
        then compared in the bandit setting via various policy evaluation methods
    '''

    def __init__(self, params):
        self.__dict__.update(params)
        self.make_env()

    def make_env(self):
        '''create the environment'''
        try:
            self.env = gym.make(self.env_name)
        except:
            self.env = make_vec_env(self.env_name, n_envs=1)
        self.low = self.env.action_space.low
        self.high = self.env.action_space.high

    def train_target_policies(self, seed=None):
        '''train policies to be ranked'''
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            self.env.seed(seed)
            self.env.action_space.seed(seed)
        models = {
            'A2C': sb3.A2C('MlpPolicy', self.env, seed=seed).learn(self.train_steps),
            'DDPG': sb3.DDPG('MlpPolicy', self.env, seed=seed).learn(self.train_steps),
            'PPO': sb3.PPO('MlpPolicy', self.env, seed=seed).learn(self.train_steps),
            'SAC': sb3.SAC('MlpPolicy', self.env, seed=seed).learn(self.train_steps),
            'TD3': sb3.TD3('MlpPolicy', self.env, seed=seed).learn(self.train_steps)}
        self.target_policies = {name: model.policy for name, model in models.items()}
        self.num_policy_pairs = len(models) * (len(models) - 1) / 2

    def evaluate_policy_rl(self, policy, num_sims=10):
        '''evaluate policy in rl environment'''
        reward_avg, reward_std = evaluate_policy(policy, self.env, n_eval_episodes=num_sims,
                                                 deterministic=False, warn=False)
        return reward_avg, reward_std

    def estimate_policy_value(self, policy, num_sims, seed=None):
        '''estimate policy value in bandit environment'''
        policy_value = 0
        for _ in range(num_sims):
            if seed is not None:
                self.env.seed(seed)
            obs = self.env.reset()
            for t in range(self.env_steps):
                action, _ = policy.predict(obs, deterministic=False)
                obs, reward, done, _ = self.env.step(action)
                policy_value += reward
                if done:
                    break
        policy_value /= num_sims
        return policy_value

    def evaluate_target_policies(self, num_sims=100):
        '''evaluate target policies in bandit environment'''
        self.value_true = {}
        for name, policy in self.target_policies.items():
            self.value_true[name] = self.estimate_policy_value(policy, num_sims)

    def probability_proxy(self, action1, action2):
        '''compute probability of taking action1 instead of action2'''
        action_delta = (action1 - action2) / (self.high - self.low)
        prob = np.exp((1 - 1 / (1 - action_delta**2 + 1e-08)).mean())
        return prob

    def generate_historical_data(self):
        '''sample historical data by deploying target policies'''
        self.historical_data, self.value_emp = [], {}
        for name, policy in self.target_policies.items():
            self.value_emp[name] = 0
            seed = np.random.randint(1e+06)
            self.env.seed(seed)
            obs = self.env.reset()
            actions, value, prob = [], 0, 1
            for t in range(self.env_steps):
                action, _ = policy.predict(obs, deterministic=False)
                actions.append(action)
                action_det, _ = policy.predict(obs, deterministic=True)
                prob *= self.probability_proxy(action, action_det)
                obs, reward, done, _ = self.env.step(action)
                value += reward
                if done:
                    break
            self.historical_data.append([seed, actions, value, prob])
            self.value_emp[name] += value
        self.rho = np.mean(list(self.value_emp.values()))

    def estimate_trajectory_probability(self, policy, trajectory):
        '''estimate proability that the policy follows the trajectory'''
        prob = 1.
        seed, actions, _, _ = trajectory
        self.env.seed(seed)
        obs = self.env.reset()
        for t in range(min(self.env_steps, len(actions))):
            action, _ = policy.predict(obs, deterministic=True)
            prob *= self.probability_proxy(action, actions[t])
            obs, _, done, _ = self.env.step(action)
        return prob

    def compute_value_dim(self, policy):
        '''evaluate the policy via the direct method'''
        value_dim = []
        for trajectory in self.historical_data:
            s, a, r, _ = trajectory
            prob = self.estimate_trajectory_probability(policy, trajectory)
            value_dim.append(r * prob)
        return np.mean(value_dim)

    def compute_value_lde(self, policy):
        '''evaluate the policy via the limited data estimator'''
        value_lde = []
        for trajectory in self.historical_data:
            s, a, r, _ = trajectory
            prob = self.estimate_trajectory_probability(policy, trajectory)
            value_lde.append((r - self.rho) * prob + self.rho)
        return np.mean(value_lde)

    def compute_value_dre(self, policy):
        '''evaluate the policy via the doubly robust estimator'''
        value_dre = []
        for trajectory in self.historical_data:
            s, a, r, p = trajectory
            prob = self.estimate_trajectory_probability(policy, trajectory)
            value_dre.append((r - self.rho) * prob / (p + 1e-06) + self.rho)
        return np.mean(value_dre)

    def compute_value_ips(self, policy):
        '''evaluate the policy via the inverse propensity scoring'''
        value_ips = []
        for trajectory in self.historical_data:
            s, a, r, p = trajectory
            prob = self.estimate_trajectory_probability(policy, trajectory)
            value_ips.append(r * prob / (p + 1e-06))
        return np.mean(value_ips)

    def swap_count(self, array1, array2):
        '''count the number of swaps required to transform array1 into array2'''
        L = list(array2)
        swaps = 0
        for element in list(array1):
            ind = L.index(element)
            L.pop(ind)
            swaps += ind
        return swaps

    def rank_target_policies(self):
        '''evaluate and rank target policies via various methods'''
        self.value_dim, self.value_lde, self.value_dre, self.value_ips = {}, {}, {}, {}
        for name, policy in self.target_policies.items():
            self.value_lde[name] = self.compute_value_lde(policy)
            self.value_dre[name] = self.compute_value_dre(policy)
            self.value_ips[name] = self.compute_value_ips(policy)
            self.value_dim[name] = self.compute_value_dim(policy)
        self.method_values = {'True': self.value_true, 'LDE': self.value_lde,
                              'DRE': self.value_dre, 'IPS': self.value_ips,
                              'DiM': self.value_dim, 'Emp': self.value_emp}
        self.values = pd.DataFrame.from_dict(self.method_values)
        self.ranks = {method: np.argsort(list(value.values()))
                      for method, value in self.method_values.items()}

    def score_ranking(self):
        '''compute scores of individual rankings'''
        scores = [1 - self.swap_count(self.ranks[method], self.ranks['True'])\
                  / self.num_policy_pairs for method in self.ranks]
        return scores

    def report_scores(self):
        '''print the resulting scores'''
        scores = np.array(self.scores, ndmin=2)[:,:-1]
        scores_med = np.median(scores, axis=0)
        scores_avg = np.mean(scores, axis=0)
        scores_std = np.std(scores, axis=0)
        print(f'average scores of policy evaluation methods on {self.env_name}:')
        for k in range(1,len(self.ranks)-1):
            print(f'  {list(self.ranks)[k]} = {scores_med[k]:.4f}',
                  f'/ {scores_avg[k]:.4f} ({scores_std[k]:.3f})')
        print()
        self.method_values.pop('Emp', None)
        data = pd.DataFrame(scores, columns=self.method_values.keys()).drop(columns='True')
        fig, ax = plt.subplots(figsize=(8,4))
        sns.violinplot(data=data, cut=0, gridsize=1000, bw=.5, linewidth=3)
        ax.set_title(self.env_name, fontname='monospace', fontweight='bold')
        ax.set_ylim(0,1)
        plt.tight_layout()
        os.makedirs('./images/', exist_ok=True)
        plt.savefig(f'./images/scores_{self.env_name}.pdf', format='pdf')
        plt.show()

    def run_simulation_explicit(self, seed=None):
        '''run a single ranking with verbose output'''
        print(f'\ntraining target policies...')
        self.train_target_policies(seed)
        print(f'rl-values of target policies:')
        for name, policy in self.target_policies.items():
            value_avg, value_std = self.evaluate_policy_rl(policy)
            print(f'  {name:>4s}-value = {value_avg:.4f} (std = {value_std:.4f})')
        self.evaluate_target_policies()
        print(f'\ngenerating historical data...')
        self.generate_historical_data()
        print(f'estimating values of target policies via policy evaluation methods...')
        self.rank_target_policies()
        print(f'estimated values:\n{self.values}')
        self.scores = self.score_ranking()

    def run_simulations(self, num_sims, seed=None):
        '''run multiple simulations'''
        self.train_target_policies(seed)
        self.evaluate_target_policies()
        self.scores = []
        for n in range(num_sims):
            self.generate_historical_data()
            self.rank_target_policies()
            self.scores.append(self.score_ranking())

    def run_tests(self, num_sims, num_tests, seed=None):
        '''run multiple tests'''
        if seed is not None:
            np.random.seed(seed)
        seeds = list(map(int, np.random.randint(1e+06, size=num_tests)))
        test_scores = []
        for n in range(num_tests):
            print(f'running test {n+1}/{num_tests} on {self.env_name}...')
            self.run_simulations(num_sims, seeds[n])
            test_scores += self.scores
        self.scores = test_scores

    def save_variables(self, path='./save/'):
        '''save class variables to a file'''
        os.makedirs(path, exist_ok=True)
        save_name = f'{self.env_name}.pkl'
        with open(path + save_name, 'wb') as save_file:
            pickle.dump(self.__dict__, save_file)

    def load_variables(self, save_name, path='./save/'):
        '''load class variables from a file'''
        try:
            with open(path + save_name, 'rb') as save_file:
                self.__dict__.update(pickle.load(save_file))
        except:
            raise NameError(f'\ncannot load file {save_name}...')

