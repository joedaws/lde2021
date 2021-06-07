"""
Plot policy comparison and evaluation scores
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style='whitegrid', palette=[sns.color_palette('colorblind')[i] for i in [0, 3, 4, 2]])


def make_comparison_plot(data_names, comp_data):
    '''plot policy comparison scores'''
    lde_means, dre_means, ips_means, dim_means = [], [], [], []
    lde_vars, dre_vars, ips_vars, dim_vars = [], [], [], []
    for data_name in data_names:
        # compute average and std
        lde_means.append(comp_data[data_name].average_score(method='lde'))
        dre_means.append(comp_data[data_name].average_score(method='dre'))
        ips_means.append(comp_data[data_name].average_score(method='ips'))
        dim_means.append(comp_data[data_name].average_score(method='dim'))
        lde_vars.append(comp_data[data_name].score_std(method='lde'))
        dre_vars.append(comp_data[data_name].score_std(method='dre'))
        ips_vars.append(comp_data[data_name].score_std(method='ips'))
        dim_vars.append(comp_data[data_name].score_std(method='dim'))
    # data labels
    methods = ['LDE', 'DRE', 'IPS', 'DiM']
    labels = data_names
    # plot parameters
    ind = np.arange(len(labels))
    width = 0.2
    error_kw = dict(lw=4, capsize=5, capthick=3)
    ecolor = 'darkslategray'
    # create bar plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(ind - 1.5 * width, lde_means, width=width, yerr=lde_vars, error_kw=error_kw, ecolor=ecolor)
    ax.bar(ind - 0.5 * width, dre_means, yerr=dre_vars, width=width, error_kw=error_kw, ecolor=ecolor)
    ax.bar(ind + 0.5 * width, ips_means, yerr=ips_vars, width=width, error_kw=error_kw, ecolor=ecolor)
    ax.bar(ind + 1.5 * width, dim_means, yerr=dim_vars, width=width, error_kw=error_kw, ecolor=ecolor)
    # configure axes
    ax.set_ylim(.5, 1.05)
    plt.ylabel('Policy comparison scores')
    ax.tick_params(labelright=True)
    plt.xticks(ind, labels, fontfamily='monospace')
    legend = ax.legend(methods, loc='lower center', bbox_to_anchor=(.5, 1.01), ncol=len(methods))
    plt.setp(legend.texts, family='monospace')
    # save and show
    plt.tight_layout()
    plt.savefig('./images/classification_scores.pdf', format='pdf')
    plt.show()


def make_evaluation_plot(data_names, comp_data):
    '''plot policy evaluation errors'''
    lde_error, dre_error, ips_error, dim_error = [], [], [], []
    lde_vars, dre_vars, ips_vars, dim_vars = [], [], [], []
    for data_name in data_names:
        # compute average and std
        lde_error.append(np.mean(np.array(comp_data[data_name].metrics['lde'])
                         - np.array(comp_data[data_name].metrics['true'])))
        dre_error.append(np.mean(np.array(comp_data[data_name].metrics['dre'])
                         - np.array(comp_data[data_name].metrics['true'])))
        ips_error.append(np.mean(np.array(comp_data[data_name].metrics['ips'])
                         - np.array(comp_data[data_name].metrics['true'])))
        dim_error.append(np.mean(np.array(comp_data[data_name].metrics['dim'])
                         - np.array(comp_data[data_name].metrics['true'])))
        lde_vars.append(np.std(np.array(comp_data[data_name].metrics['lde'])
                        - np.array(comp_data[data_name].metrics['true'])))
        dre_vars.append(np.std(np.array(comp_data[data_name].metrics['dre'])
                        - np.array(comp_data[data_name].metrics['true'])))
        ips_vars.append(np.std(np.array(comp_data[data_name].metrics['ips'])
                        - np.array(comp_data[data_name].metrics['true'])))
        dim_vars.append(np.std(np.array(comp_data[data_name].metrics['dim'])
                        - np.array(comp_data[data_name].metrics['true'])))
    # data labels
    methods = ['LDE', 'DRE', 'IPS', 'DiM']
    labels = data_names
    # plot parameters
    ind = np.arange(len(labels))
    width = 0.2
    avg_height = .005
    avg_color = 'darkslategray'
    # averge bias
    lde_avg_upper = np.array(lde_error) + avg_height
    lde_avg_lower = np.array(lde_error) - avg_height
    dre_avg_upper = np.array(dre_error) + avg_height
    dre_avg_lower = np.array(dre_error) - avg_height
    ips_avg_upper = np.array(ips_error) + avg_height
    ips_avg_lower = np.array(ips_error) - avg_height
    dim_avg_upper = np.array(dim_error) + avg_height
    dim_avg_lower = np.array(dim_error) - avg_height
    # rmse distribution
    lde_upper = np.array(lde_error) + np.array(lde_vars)
    lde_lower = np.array(lde_error) - np.array(lde_vars)
    dre_upper = np.array(dre_error) + np.array(dre_vars)
    dre_lower = np.array(dre_error) - np.array(dre_vars)
    ips_upper = np.array(ips_error) + np.array(ips_vars)
    ips_lower = np.array(ips_error) - np.array(ips_vars)
    dim_upper = np.array(dim_error) + np.array(dim_vars)
    dim_lower = np.array(dim_error) - np.array(dim_vars)
    # create bar plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(ind - 1.5 * width, lde_upper - lde_lower, bottom=lde_lower, width=width)
    ax.bar(ind - 0.5 * width, dre_upper - dre_lower, bottom=dre_lower, width=width)
    ax.bar(ind + 0.5 * width, ips_upper - ips_lower, bottom=ips_lower, width=width)
    ax.bar(ind + 1.5 * width, dim_upper-dim_lower, bottom=dim_lower, width=width)
    ax.bar(ind - 1.5 * width, lde_avg_upper - lde_avg_lower, bottom=lde_avg_lower,
           width=width, color=avg_color)
    ax.bar(ind - 0.5 * width, dre_avg_upper - dre_avg_lower, bottom=dre_avg_lower,
           width=width, color=avg_color)
    ax.bar(ind + 0.5 * width, ips_avg_upper - ips_avg_lower, bottom=ips_avg_lower,
           width=width, color=avg_color)
    ax.bar(ind + 1.5 * width, dim_avg_upper - dim_avg_lower, bottom=dim_avg_lower,
           width=width, color=avg_color)
    # configure axes
    plt.ylabel('Policy evaluation errors')
    ax.tick_params(labelright=True)
    plt.xticks(ind, labels, fontfamily='monospace')
    legend = ax.legend(methods, loc='lower center', bbox_to_anchor=(.5, 1.01), ncol=len(methods))
    plt.setp(legend.texts, family='monospace')
    # save and show
    plt.tight_layout()
    plt.savefig('./images/classification_values.pdf', format='pdf')
    plt.show()


def make_plots(data_names: list, comp_data: dict):
    """Make and save plots associated to the classifier examples."""
    os.makedirs('./images/', exist_ok=True)
    make_comparison_plot(data_names, comp_data)
    make_evaluation_plot(data_names, comp_data)
