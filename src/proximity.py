import pandas as pd
import numpy as np
import time
from toolbox import wrappers
import concurrent.futures
import repos_tools

network = wrappers.get_network('../../resources/PPI/Cheng2019/network.sif', only_lcc = True)
id_mapping_file = '../../resources/PPI/geneid_to_symbol.txt'
gset_names = ['knowledge', 'knowledge-TWAS2plus', 'knowledge-TWAS', 'knowledge-TWAS2plus-IAPS']
def read_geneset(name):
    file_name = '../../results/2021-07-01-high-conf-ADgenes/AD-genes-' + name
    gset = wrappers.convert_to_geneid(file_name=file_name, id_type='symbol', id_mapping_file=id_mapping_file)
    gset, gset_dropped = repos_tools.drop_genes_notin_network(gset, network)
    return(gset)

gene_sets = {k: read_geneset(k) for k in gset_names}
dis_genes = gene_sets['knowledge']

def process_drug(item):
    drugbank_id, targets = item
    res = wrappers.calculate_proximity(network=network, nodes_from=targets, nodes_to=dis_genes)
    return((drugbank_id, res))


def calculate_proximities(drugbank_prot, asynchronous=True):
    start = time.time()
    gb = drugbank_prot.groupby('drugbank_id')
    l = gb.apply(lambda row: (row.index.get_level_values(0)[0], set(row.entrez_id))).to_list()
    def proc_d(item):
        res = process_drug(*item, dis_genes, network)
        return(res)
    preparation = time.time() - start
    if asynchronous:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            val = list(executor.map(process_drug, l))
    else:
        val = list(map(process_drug, l)) # for testing synchronous execution
    total = time.time() - start
    calculation =  total - preparation
    print(preparation, 's preparation', calculation, 's calculation')
    return(val)
