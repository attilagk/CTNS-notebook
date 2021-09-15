import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def drop_genes_notin_network(genes, network):
    kept_genes = [y for y in genes if y in network.nodes]
    dropped_genes = set(genes).difference(set(kept_genes))
    return((kept_genes, dropped_genes))

def plot_proximity_results(prox, jitter=True):
    fig, ax = plt.subplots(1, 3, sharey=True, figsize=(8, 4))
    for quantity, axi in zip(list('dzp'), ax):
        sns.stripplot(x=quantity, y='target', hue='condition', data=prox, ax=axi, jitter=jitter)
        axi.get_legend().remove()
        axi.grid(True, axis='y')
    ax[2].set_xscale('log')
    ax[0].set_xlabel('distance')
    ax[1].set_xlabel('z-score')
    ax[2].set_xlabel('p-value')
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2)
    return((fig, ax))
