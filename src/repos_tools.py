import pandas as pd

def drop_genes_notin_network(genes, network):
    kept_genes = [y for y in genes if y in network.nodes]
    dropped_genes = set(genes).difference(set(kept_genes))
    return((kept_genes, dropped_genes))
