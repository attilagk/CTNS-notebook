import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def drop_genes_notin_network(genes, network):
    kept_genes = [y for y in genes if y in network.nodes]
    dropped_genes = set(genes).difference(set(kept_genes))
    return((kept_genes, dropped_genes))

def plot_proximity_results(prox):
    fig, ax = plt.subplots(1, 3, sharey=True, figsize=(8, 4))
    sns.stripplot(x='d', y='target', hue='condition', data=prox, ax=ax[0])
    sns.stripplot(x='z', y='target', hue='condition', data=prox, ax=ax[1])
    sns.stripplot(x='p', y='target', hue='condition', data=prox, ax=ax[2])
    ax[2].set_xscale('log')
    ax[0].get_legend().remove()
    ax[1].get_legend().remove()
    ax[2].get_legend().remove()
    ax[0].set_xlabel('distance')
    ax[1].set_xlabel('z-score')
    ax[2].set_xlabel('p-value')
    ax[0].grid(True, axis='y')
    ax[1].grid(True, axis='y')
    ax[2].grid(True, axis='y')
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2)
    #fig.legend(handles, labels, loc='upper center', ncol=2, title='AD genes')
    return((fig, ax))
