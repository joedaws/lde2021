'''
Set up the synthetic bandit environment, run tests and analyze the results
'''

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
from matplotlib import animation
import seaborn as sns
from scipy import stats
import pickle
import os

np.set_printoptions(suppress=True, linewidth=100)
sns.set_theme(style='whitegrid', palette=[sns.color_palette('colorblind')[i] for i in [0,3,4,2]])


class SyntheticBanditEnv:

    def __init__(self, params):
        self.__dict__.update(params)
        self.methods = ['lde', 'dre', 'ips', 'dim']
        self.method_names = ['LDE', 'DRE', 'IPS', 'DiM']
        self.metrics = ['pred', 'pred_s', 'err_min', 'err_avg', 'err_max', 'err_std']
        os.makedirs('./images/', exist_ok=True)
        self.setup_env()
        self.compute_reward()
        self.derive_optimal_policy()
        print('\nenvironment is set up')
        print(f'min/max policy values: {self.V_min:.4f} / {self.V_max:.4f}\n')

    def reward(self, s, a):
        '''define the reward function'''
        if self.example == 1:
            r = (np.exp(s+a) + np.sin(2*np.pi * (s-a))) * (np.cos(2*np.pi * (s+a)) + np.exp(s-a))
        elif self.example == 2:
            r = (3/2 + np.cos(2*np.pi*(s + a)) + np.cos(10*np.pi*(s - a))/2)\
                * np.exp(-(a + np.cos(2*np.pi*s))**2)
        else:
            raise ValueError(f'\nreward function {self.example} is not defined...')
        return r

    def setup_env(self):
        '''generate discrete state and action spaces'''
        self.S = np.linspace(self.dom_s[0], self.dom_s[1], self.num_s)
        self.A = np.linspace(self.dom_a[0], self.dom_a[1], self.num_a)

    def compute_reward(self):
        '''compute the reward values for all state-action pairs'''
        self.R = np.zeros((self.S.size, self.A.size))
        for i in range(self.S.size):
            for j in range(self.A.size):
                self.R[i,j] = self.reward(self.S[i], self.A[j])

    def derive_optimal_policy(self):
        '''derive policies with the maximal and minimal rewards'''
        self.pi_max = np.zeros(self.R.shape)
        self.pi_max[np.arange(self.S.size), np.argmax(self.R, axis=1)] = 1
        self.V_max = self.compute_value(self.pi_max).mean()
        self.pi_min = np.zeros(self.R.shape)
        self.pi_min[np.arange(self.S.size), np.argmin(self.R, axis=1)] = 1
        self.V_min = self.compute_value(self.pi_min).mean()

    def compute_value(self, pi):
        '''compute the value of the policy pi on each state'''
        value = (self.R * pi).sum(axis=1)
        return value

    def set_random_seed(self, seed):
        '''fix random generator seed for reproducibility'''
        if seed is not None:
            np.random.seed(seed)

    def generate_policy(self, loc=1, seed=None):
        '''generate a random policy of specified localization'''
        self.set_random_seed(seed)
        pi = loc * np.random.randn(self.S.size, self.A.size)
        pi = np.exp(pi) / np.exp(pi).sum(axis=1, keepdims=True)
        return pi

    def generate_policy_diff(self, alpha=.01, seed=None):
        '''return difference of two generated policies whose values differ by alpha'''
        self.set_random_seed(seed)
        pi1, pi2 = self.generate_policy(), self.generate_policy()
        V1, V2 = self.compute_value(pi1).mean(), self.compute_value(pi2).mean()
        V_delta = (V1 - V2) / (self.V_max - self.V_min)
        beta = (alpha - V_delta) / (1 - V_delta)
        pi1 = (1 - beta) * pi1 + beta * self.pi_max
        pi2 = (1 - beta) * pi2 + beta * self.pi_min
        return pi1, pi2

    def sample_data(self, num_m, nu=None, loc=1, seed=None):
        '''sample the historical data from the behavioral policy nu'''
        self.set_random_seed(seed)
        if nu is None:
            nu = self.generate_policy(loc)
        D = np.zeros((self.S.size,self.A.size))
        for s in range(self.S.size):
            for a in np.random.choice(np.arange(self.A.size), num_m, p=nu[s]):
                D[s,a] += 1
        return D, nu

    def visualize_reward(self, show=True):
        '''visualize reward function for different states'''
        fig = plt.figure(figsize=(16,9))
        plt.subplots_adjust(left=.1, right=.9, bottom=.2, top=.9)
        reward_s, = plt.plot(self.A, self.R[0], linewidth=3, color='crimson')
        plt.xlim((self.A[0], self.A[-1]))
        plt.ylim((self.R.min(), self.R.max()))
        s_axes = plt.axes([.1, .05, .8, .05])
        s_slider = Slider(s_axes, 's', 0, self.S.size-1, valinit=0, valfmt='%d')
        self.pause = 0
        def plot_frame(frame):
            if self.pause == 0:
                s_slider.set_val(int(s_slider.val + 1) % self.S.size)
        def on_click(event):
            (x0,y0),(x1,y1) = s_slider.label.clipbox.get_points()
            if not (x0 < event.x < x1 and y0 < event.y < y1):
                self.pause = 1 - self.pause
        s_slider.on_changed(lambda s: reward_s.set_ydata(self.R[int(s)]))
        fig.canvas.mpl_connect('button_press_event', on_click)
        anim = animation.FuncAnimation(fig, plot_frame, frames=self.S.size, interval=50)
        plt.show() if show else plt.close()

    def visualize_reward_3d(self, show=True):
        '''visualize reward function on the state-action space'''
        fig, ax = plt.subplots(figsize=(10,8), subplot_kw={'projection': '3d'})
        S, A = np.meshgrid(self.S, self.A, indexing='ij')
        ax.plot_surface(S, A, self.R)
        ax.xaxis.set_rotate_label(False)
        ax.set_xlabel('$\mathcal{S}$', fontsize=20, labelpad=20)
        ax.yaxis.set_rotate_label(False)
        ax.set_ylabel('$\mathcal{A}$', fontsize=20, labelpad=20)
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel('$\mathcal{R}$', fontsize=20, labelpad=20)
        ax.view_init(elev=20, azim=-125)
        plt.tight_layout()
        plt.savefig(f'./images/{self.example}_reward_3d.pdf', format='pdf')
        plt.show() if show else plt.close()

    def evaluate_policy(self, method, pi, D, nu=None, rho_val='mean'):
        '''evaluate policy pi via the specified method'''
        if method == 'value':
            return (self.R * pi).sum(axis=1)
        elif method == 'dim':
            return self.compute_value_dim(pi, D)
        elif method == 'ips':
            return self.compute_value_ips(pi, D, nu)
        elif method == 'dre':
            return self.compute_value_dre(pi, D, nu, rho_val)
        elif method == 'lde':
            return self.compute_value_lde(pi, D, rho_val)
        else:
            raise NameError(f'\nmethod {method} is not implemented...')

    def compute_value_dim(self, pi, D):
        '''compute dim-value of the policy pi on each state'''
        value = (self.R * np.minimum(D,1) * pi).sum(axis=1)
        return value

    def compute_value_ips(self, pi, D, nu):
        '''compute ips-value of the policy pi on each state'''
        value = (self.R * D * pi / nu).sum(axis=1) / D.sum(axis=1)
        return value

    def compute_value_dre(self, pi, D, nu, rho_val='mean'):
        '''compute dre-value of the policy pi on each state'''
        rho = self.compute_baseline(D, rho_val)
        value = rho + ((self.R - rho) * D * pi / nu).sum(axis=1) / D.sum(axis=1)
        return value

    def compute_value_lde(self, pi, D, rho_val='mean'):
        '''compute caliber of the policy pi on each state'''
        rho = self.compute_baseline(D, rho_val)
        caliber = rho + ((self.R - rho) * np.minimum(D,1) * pi).sum(axis=1)
        return caliber

    def compute_baseline(self, D, rho_val='mean'):
        '''compute baseline based on the observed rewards'''
        if rho_val == 'mean':
            rho = np.mean(self.R[D.nonzero()])
        elif rho_val == 'median':
            rho = np.median(self.R[D.nonzero()])
        else:
            rho = float(rho_val)
        return rho

    def simulate_comparison(self, alpha=.01, num_m=1, loc=1, rho_val='mean', seed=None):
        '''sample data, generate policies, and compare them via the specified methods'''
        self.set_random_seed(seed)
        D, nu = self.sample_data(num_m, None, loc)
        pi1, pi2 = self.generate_policy_diff(alpha)
        vals = {}
        for method in ['value', *self.methods]:
            vals[method + '1'] = self.evaluate_policy(method, pi1, D, nu, rho_val)
            vals[method + '2'] = self.evaluate_policy(method, pi2, D, nu, rho_val)
        return vals

    def run_simulations(self, alpha=.01, num_m=1, num_sims=1000, loc=1, rho_val='mean', seed=None):
        '''compare policies with fixed amount of historical data and value difference'''
        self.set_random_seed(seed)
        keys = ['value', *self.methods]
        result = dict.fromkeys([k + i for k in keys for i in ['1','2']], [])
        for n in range(num_sims):
            vals = self.simulate_comparison(alpha, num_m, loc, rho_val)
            result = {key: np.concatenate([result[key], vals[key]]) for key in result}
        result['sim'] = np.concatenate([[n]*self.num_s for n in range(num_sims)])
        return pd.DataFrame(result)

    def run_tests_a(self, a_min=.01, a_max=.1, num_a_tests=10, num_m=1,
                    num_sims=1000, loc=1, rho_val='mean', seed=None):
        '''run policy comparison with various amount of value difference'''
        self.set_random_seed(seed)
        alpha = np.linspace(a_min, a_max, num_a_tests)
        data, metrics = pd.DataFrame(), []
        for n in range(num_a_tests):
            print(f'running a-test {n+1}/{num_a_tests}...')
            result = self.run_simulations(alpha[n], num_m, num_sims, loc, rho_val)
            data = data.append(result.groupby('sim').mean())
            metrics.append([alpha[n], *self.process_result(result)])
        metrics = pd.DataFrame(metrics, columns=['value_diff',
            *[method + '_' + metric for method in self.methods for metric in self.metrics]])
        return data, metrics

    def run_tests_m(self, alpha=.01, m_min=1, m_max=100, num_m_tests=10,
                    num_sims=1000, loc=1, rho_val='mean', seed=None):
        '''run policy comparison with various amount of historical data'''
        self.set_random_seed(seed)
        for m in range(num_m_tests, m_max+1):
            num_m = np.unique(np.geomspace(m_min, m_max, m, dtype=int))
            if num_m.size == num_m_tests:
                break
        data, metrics = pd.DataFrame(), []
        for n in range(num_m_tests):
            print(f'running d-test {n+1}/{num_m_tests}...')
            result = self.run_simulations(alpha, num_m[n], num_sims, loc, rho_val)
            data = data.append(result.groupby('sim').mean())
            metrics.append([num_m[n], *self.process_result(result)])
        metrics = pd.DataFrame(metrics, columns=['data_points',
            *[method + '_' + metric for method in self.methods for metric in self.metrics]])
        return data, metrics

    def run_tests_3d(self, a_min=.01, a_max=.1, num_a_tests=10, m_min=1, m_max=100, num_m_tests=10,
                     num_sims=1000, loc=1, rho_val='mean', seed=None):
        '''run policy comparison with various amounts of value difference / historical data'''
        self.set_random_seed(seed)
        alpha = np.linspace(a_min, a_max, num_a_tests)
        num_m = np.linspace(m_min, m_max, num_m_tests).astype(int)
        data, metrics = pd.DataFrame(), []
        for n in range(num_m_tests):
            for k in range(num_a_tests):
                print(f'running 3d-test {n*num_a_tests+k+1}/{num_a_tests*num_m_tests}...')
                result = self.run_simulations(alpha[k], num_m[n], num_sims, loc, rho_val)
                data = data.append(result.groupby('sim').mean())
                metrics.append([alpha[k], num_m[n], *self.process_result(result)])
        metrics = pd.DataFrame(metrics, columns=['value_diff', 'data_points',
            *[method + '_' + metric for method in self.methods for metric in self.metrics]])
        return data, metrics

    def process_result(self, result):
        '''compute stats for the provided test results'''
        diffs = self.compute_diffs(result)
        diffs_avg = diffs.groupby('sim').mean()
        value_appr = result.groupby('sim').mean() / (self.V_max - self.V_min)
        processed = []
        for method in self.methods:
            pred = self.evaluate_predictions(diffs_avg['value'], diffs_avg[method])
            pred_s = self.evaluate_predictions(diffs['value'], diffs[method])
            method_err = np.concatenate([value_appr[method + '1'] - value_appr['value1'],
                                         value_appr[method + '2'] - value_appr['value2']])
            err_min = np.percentile(method_err, 5)
            err_avg = np.mean(method_err)
            err_max = np.percentile(method_err, 95)
            err_std = np.std(method_err)
            processed += [pred, pred_s, err_min, err_avg, err_max, err_std]
        return processed

    def compute_diffs(self, result):
        '''compute differences on the provided results'''
        diffs = pd.DataFrame()
        for val in ['value', *self.methods]:
            diffs[val] = result[val + '1'] - result[val + '2']
        if 'sim' in result:
            diffs['sim'] = result['sim']
        if 'value_diff' in result:
            diffs['value_diff'] = result['value_diff']
        if 'data_points' in result:
            diffs['data_points'] = result['data_points']
        return diffs

    def evaluate_predictions(self, dV, dU):
        '''compute percentage of the correct predictions on the provided data'''
        pred = np.sum(dV * dU >= 0) / dV.size
        return pred

    def compute_correlation(self, dV, dU):
        '''compute correlation statistics on the provided data'''
        prs = stats.pearsonr(dV, dU)
        spr = stats.spearmanr(dV, dU)
        print(f'Pearson\'s correlation:  {prs[0]: .4f} ({prs[1]:.2e})')
        print(f'Spearman\'s correlation: {spr[0]: .4f} ({spr[1]:.2e})')

    def analyze_tests(self, result):
        '''compute and visualize relevant metrics on the provided test data'''
        processed = self.process_result(result)
        diffs = self.compute_diffs(result)
        for method in self.methods:
            for metric in self.metrics:
                print(f'{method}_{metric} = {100*processed.pop(0):.2f}%')
        self.plot_prods_hist(diffs)

    def plot_prediction_policy(self, data, show=True):
        '''plot the rate of policy predictions of different methods'''
        fig, ax = plt.subplots(figsize=(8,5))
        data.plot(x=data.columns[0],
                  y=[method + '_pred' for method in self.methods], linewidth=4, ax=ax)
        ax.set_ylim(.5, 1.05)
        ax.set_xlabel(None)
        ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.grid(b=True, which='major', linewidth=1.5)
        ax.grid(b=True, which='minor', linewidth=.5)
        legend = ax.legend(self.method_names, loc='lower right')
        plt.setp(legend.texts, family='monospace')
        plt.tight_layout()
        plt.savefig(f'./images/{self.example}_{data.columns[0]}_policy.pdf', format='pdf')
        plt.show() if show else plt.close()

    def plot_prediction_state(self, data, show=True):
        '''plot the rate of state predictions of different methods'''
        fig, ax = plt.subplots(figsize=(8,5))
        data.plot(x=data.columns[0],
                  y=[method + '_pred_s' for method in self.methods], linewidth=4, ax=ax)
        ax.set_ylim(.45, 1.)
        ax.set_xlabel(None)
        ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.grid(b=True, which='major', linewidth=1.5)
        ax.grid(b=True, which='minor', linewidth=.5)
        legend = ax.legend(self.method_names, loc='upper left')
        plt.setp(legend.texts, family='monospace')
        plt.tight_layout()
        plt.savefig(f'./images/{self.example}_{data.columns[0]}_state.pdf', format='pdf')
        plt.show() if show else plt.close()

    def plot_value_approximation(self, data, show=True):
        '''plot the value approximations via different methods'''
        fig, ax = plt.subplots(figsize=(8,5))
        for method in self.methods[::-1]:
            ax.fill_between(x=data[data.columns[0]], y1=data[method + '_err_min'],
                            y2=data[method + '_err_max'],
                           color=[sns.color_palette()[self.methods.index(method)]],
                           linewidth=2, alpha=.25)
        data.plot(x=data.columns[0], y=[method + '_err_avg' for method in self.methods],
                  linewidth=4, ax=ax)
        ax.set_xlabel(None)
        legend = ax.legend(self.method_names, loc='upper right')
        plt.setp(legend.texts, family='monospace')
        plt.tight_layout()
        plt.savefig(f'./images/{self.example}_{data.columns[0]}_appr.pdf', format='pdf')
        plt.show() if show else plt.close()

    def plot_value_approximation_avg(self, data, show=True):
        '''plot the average of value approximations via different methods'''
        fig, ax = plt.subplots(figsize=(8,5))
        data.plot(x=data.columns[0], y='lde_err_avg', linewidth=4, zorder=3, alpha=.9, ax=ax)
        data.plot(x=data.columns[0], y='dre_err_avg', linewidth=4, zorder=2, alpha=.9, ax=ax)
        data.plot(x=data.columns[0], y='ips_err_avg', linewidth=4, zorder=1, alpha=.9, ax=ax)
        ax.set_xlabel(None)
        legend = ax.legend(self.method_names, loc='upper right')
        plt.setp(legend.texts, family='monospace')
        plt.tight_layout()
        plt.savefig(f'./images/{self.example}_{data.columns[0]}_appr_avg.pdf', format='pdf')
        plt.show() if show else plt.close()

    def plot_prods_hist(self, diffs, show=True):
        '''plot the distribution of the products of per-state differences'''
        prods = [diffs['value'] * diffs[method] for method in self.methods]
        fig, ax = plt.subplots(figsize=(8,5))
        b = np.percentile(np.abs(prods), 90)
        num_bins = 50
        bins = np.linspace(-b, b, num_bins)
        pp = np.concatenate([[np.histogram(prods[i], bins)[0]] for i in range(4)], axis=0)
        pp = pp / pp.sum(axis=0, keepdims=True)
        for i in range(3,0,-1):
            pp[i-1] += pp[i]
        for i in range(4):
            ax.bar((bins[1:] + bins[:-1])/2, pp[i], width=2*b/(num_bins-1))
        ax.set_xlim(-b, b)
        ax.set_ylim(0, 1)
        legend = ax.legend(self.method_names, loc='upper left')
        plt.setp(legend.texts, family='monospace')
        plt.tight_layout()
        plt.savefig(f'./images/{self.example}_prods.pdf', format='pdf')
        plt.show() if show else plt.close()

    def plot_prediction_policy_3d(self, data, show=True):
        '''plot the rate of policy predictions of different methods'''
        m = data['data_points'].to_numpy()
        a = data['value_diff'].to_numpy()
        mesh_size = (np.unique(m).size, np.unique(a).size)
        M = m.reshape(mesh_size)
        A = a.reshape(mesh_size)
        for method in self.methods:
            fig, ax = plt.subplots(figsize=(10,8), subplot_kw={'projection': '3d'})
            P = data[f'{method}_pred'].to_numpy().reshape(mesh_size)
            ax.plot_surface(M, A, P, color=sns.color_palette()[self.methods.index(method)], alpha=.75)
            ax.set_zlim(.5, 1.)
            ax.set_xlabel('data points', fontsize=15, labelpad=20)
            ax.set_ylabel('value difference', fontsize=15, labelpad=20)
            ax.set_zlabel('policy prediction rate', fontsize=15, labelpad=20)
            ax.view_init(azim=-135, elev=20)
            plt.tight_layout()
            plt.savefig(f'./images/{self.example}_policy_3d_{method}.pdf', format='pdf')
            plt.show() if show else plt.close()

    def plot_value_approximation_avg_3d(self, data, show=True):
        '''plot the average of value approximations via different methods'''
        m = data['data_points'].to_numpy()
        a = data['value_diff'].to_numpy()
        mesh_size = (np.unique(m).size, np.unique(a).size)
        M = m.reshape(mesh_size)
        A = a.reshape(mesh_size)
        for method in self.methods:
            fig, ax = plt.subplots(figsize=(10,8), subplot_kw={'projection': '3d'})
            V = data[f'{method}_err_avg'].to_numpy().reshape(mesh_size)
            ax.plot_surface(M, A, V, color=sns.color_palette()[self.methods.index(method)], alpha=.75)
            ax.set_xlabel('data points', fontsize=15, labelpad=20)
            ax.set_ylabel('value difference', fontsize=15, labelpad=20)
            ax.set_zlabel('value approximation error', fontsize=15, labelpad=20)
            ax.view_init(azim=115, elev=15)
            plt.tight_layout()
            plt.savefig(f'./images/{self.example}_value_3d_{method}.pdf', format='pdf')
            plt.show() if show else plt.close()

    def plot_prods_hist_grid(self, num_bins=50, percentile=90, show=True):
        '''plot the distribution of the products of per-state differences on a grid'''
        fig, ax = plt.subplots(self.num_m_grid.size, self.alpha_grid.size, figsize=(12,16))
        for n in range(self.num_m_grid.size):
            for k in range(self.alpha_grid.size):
                prods = self.prods_grid[self.alpha_grid.size * n + k]
                b = np.percentile(np.abs(prods), percentile)
                bins = np.linspace(-b, b, num_bins)
                ax[n,k].set_ylim(0, 1)
                if k == 0:
                    ax[n,k].set_ylabel(fr'$m = {self.num_m_grid[n]:d}$')
                if n == self.num_m_grid.size - 1:
                    ax[n,k].set_xlabel(r'$\alpha = {:.2f}$'.format(self.alpha_grid[k]))
                dist = np.concatenate([[np.histogram(prods[i], bins)[0] + 1]\
                                       for i in range(len(self.methods))], axis=0)
                dist_norm = dist / dist.sum(axis=0, keepdims=True)
                for i in range(1,len(dist_norm)):
                    dist_norm[-i-1] += dist_norm[-i]
                for i in range(len(dist_norm)):
                    ax[n,k].bar((bins[1:] + bins[:-1])/2, dist_norm[i], width=2*b/(num_bins-1))
                ax[n,k].set_xlim(-b, b)
        fig.text(.5, .06, 'value difference', ha='center')
        fig.text(.05, .5, 'data points', va='center', rotation='vertical')
        legend = ax[0, self.alpha_grid.size//2].legend(
            self.method_names, loc='lower center', bbox_to_anchor=(.5, 1.1), ncol=len(self.methods))
        plt.setp(legend.texts, family='monospace')
        plt.savefig(f'./images/{self.example}_prods_grid.pdf', format='pdf')
        plt.show() if show else plt.close()

    def report_approximation_error(self):
        '''report policy evaluation errors'''
        print('\nvalue-difference approximation errors:')
        step_a = np.ceil(len(self.quals_a) / 5).astype(int)
        offset_a = (len(self.quals_a) - 1) % step_a
        columns_avg_a = [self.quals_a.columns[0]] + [method + '_err_avg' for method in self.methods]
        columns_std_a = [self.quals_a.columns[0]] + [method + '_err_std' for method in self.methods]
        print(self.quals_a[columns_avg_a][offset_a::step_a])
        print(self.quals_a[columns_std_a][offset_a::step_a])
        print('\ndata-points approximation errors:')
        step_m = np.ceil(len(self.quals_m) / 5).astype(int)
        offset_m = (len(self.quals_m) - 1) % step_m
        columns_avg_m = [self.quals_m.columns[0]] + [method + '_err_avg' for method in self.methods]
        columns_std_m = [self.quals_m.columns[0]] + [method + '_err_std' for method in self.methods]
        print(self.quals_m[columns_avg_m][offset_m::step_m])
        print(self.quals_m[columns_std_m][offset_m::step_m])

    def report_correlations(self):
        '''report correlation coefficients on the data obtained from 3d tests'''
        num_m = np.linspace(self.test_params['3d']['m_min'],
                            self.test_params['3d']['m_max'],
                            self.test_params['3d']['num_m_tests']).astype(int)
        num_a_sims = self.test_params['3d']['num_a_tests'] * self.test_params['3d']['num_sims']
        for method in self.methods:
            print()
            for m in range(self.test_params['3d']['num_m_tests']):
                print(f'{method}-value correlation stats with m = {num_m[m]}:')
                self.compute_correlation(self.diffs_3d['value'][m*num_a_sims : (m+1)*num_a_sims],
                                         self.diffs_3d[method][m*num_a_sims : (m+1)*num_a_sims])

    def run_dynamic_simulations_a(self, a_min=.01, a_max=.1, num_a_tests=10,
                                  num_m=1, num_sims=1000, loc=1, rho_val='mean', seed=None, show=True):
        '''run tests with various values of alpha'''
        self.data_a, self.quals_a = self.run_tests_a(a_min, a_max, num_a_tests, num_m,
                                                    num_sims, loc, rho_val, seed)
        print('\n', self.data_a)
        self.diffs_a = self.compute_diffs(self.data_a)
        for method in self.methods:
            print(f'\n{method}-value correlation stats:')
            self.compute_correlation(self.diffs_a['value'], self.diffs_a[method])
        print('\n', self.quals_a)
        self.plot_prediction_policy(self.quals_a, show)
        self.plot_value_approximation(self.quals_a, show)
        self.plot_value_approximation_avg(self.quals_a, show)

    def run_dynamic_simulations_m(self, m_min=1, m_max=100, num_m_tests=10,
                                  alpha=.01, num_sims=1000, loc=1, rho_val='mean', seed=None, show=True):
        '''run tests with various values of historical data'''
        self.data_m, self.quals_m = self.run_tests_m(alpha, m_min, m_max, num_m_tests,
                                                    num_sims, loc, rho_val, seed)
        print('\n', self.data_m)
        self.diffs_m = self.compute_diffs(self.data_m)
        for method in self.methods:
            print(f'\n{method}-value correlation stats:')
            self.compute_correlation(self.diffs_m['value'], self.diffs_m[method])
        print('\n', self.quals_m)
        self.plot_prediction_policy(self.quals_m, show)
        self.plot_prediction_state(self.quals_m, show)
        self.plot_value_approximation(self.quals_m, show)
        self.plot_value_approximation_avg(self.quals_m, show)

    def run_dynamic_simulations_3d(self, a_min=.01, a_max=.1, num_a_tests=10,
                                   m_min=1, m_max=100, num_m_tests=10,
                                   num_sims=1000, loc=1, rho_val='mean', seed=None, show=True):
        '''run tests with various values of alpha / num_m'''
        self.data_3d, self.quals_3d = self.run_tests_3d(a_min, a_max, num_a_tests,
                                                       m_min, m_max, num_m_tests,
                                                       num_sims, loc, rho_val, seed)
        print('\n', self.data_3d)
        self.diffs_3d = self.compute_diffs(self.data_3d)
        for method in self.methods:
            print(f'\n{method}-value correlation stats:')
            self.compute_correlation(self.diffs_3d['value'], self.diffs_3d[method])
        print('\n', self.quals_3d)
        self.plot_prediction_policy_3d(self.quals_3d, show)
        self.plot_value_approximation_avg_3d(self.quals_3d, show)

    def run_prods_hist_grid(self, a_min=.01, a_max=.1, num_a_tests=3, m_min=1, m_max=100,
                             num_m_tests=5, num_sims=1000, loc=1, rho_val='mean', seed=None, show=True):
        '''compute the distribution of the products of per-state differences on a grid'''
        self.set_random_seed(seed)
        self.alpha_grid = np.linspace(a_min, a_max, num_a_tests)
        for m in range(num_m_tests, m_max+1):
            self.num_m_grid = np.unique(np.geomspace(m_min, m_max, m, dtype=int))
            if self.num_m_grid.size == num_m_tests:
                break
        self.diffs_grid, self.prods_grid = [], []
        for n in range(num_m_tests):
            for k in range(num_a_tests):
                print(f'running grid-test {n*num_a_tests+k+1}/{num_a_tests*num_m_tests}...')
                result = self.run_simulations(self.alpha_grid[k], self.num_m_grid[n],
                                             num_sims, loc, rho_val)
                diffs = self.compute_diffs(result)
                prods = [diffs['value'] * diffs[method] for method in self.methods]
                self.diffs_grid.append(diffs)
                self.prods_grid.append(prods)
        self.plot_prods_hist_grid(show=show)

    def save_variables(self):
        '''save class variables to a file'''
        os.makedirs('./save/', exist_ok=True)
        save_name = f'Synthetic_{self.example}.pkl'
        with open('./save/' + save_name, 'wb') as save_file:
            pickle.dump(self.__dict__, save_file)

    def load_variables(self, save_name):
        '''load class variables from a file'''
        try:
            with open('./save/' + save_name, 'rb') as save_file:
                self.__dict__.update(pickle.load(save_file))
        except:
            raise NameError(f'\ncannot load file {save_name}...')

    def produce_pictures(self, test_params, seed, show=True):
        '''run tests and produce the pictures presented in the paper'''
        self.test_params = test_params
        self.visualize_reward_3d(show=show)
        self.run_dynamic_simulations_a(**self.test_params['a'], seed=seed, show=show)
        self.run_dynamic_simulations_m(**self.test_params['m'], seed=seed, show=show)
        self.run_dynamic_simulations_3d(**self.test_params['3d'], seed=seed, show=show)
        self.run_prods_hist_grid(**self.test_params['grid'], seed=seed, show=show)
        self.report_approximation_error()
        self.report_correlations()

    def reproduce_pictures(self, save_name, show=True):
        '''load saved test results and reproduce the pictures presented in the paper'''
        self.load_variables(save_name)
        self.visualize_reward_3d(show=show)
        self.plot_prediction_policy(self.quals_a, show=show)
        self.plot_value_approximation(self.quals_a, show=show)
        self.plot_value_approximation_avg(self.quals_a, show=show)
        self.plot_prediction_policy(self.quals_m, show=show)
        self.plot_prediction_state(self.quals_m, show=show)
        self.plot_value_approximation(self.quals_m, show=show)
        self.plot_value_approximation_avg(self.quals_m, show=show)
        self.plot_prediction_policy_3d(self.quals_3d, show=show)
        self.plot_value_approximation_avg_3d(self.quals_3d, show=show)
        self.plot_prods_hist_grid(show=show)
        self.report_approximation_error()
        self.report_correlations()

