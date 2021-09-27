import os
import pandas as pd
import numpy as np
import time
from toolbox import wrappers
import concurrent.futures
import repos_tools
import scipy

main_dirpath = '../../'

gset_names = ['knowledge', 'knowledge-TWAS2plus', 'knowledge-TWAS', 'knowledge-TWAS2plus-IAPS']
def read_geneset(name):
    id_mapping_file = main_dirpath + 'resources/PPI/geneid_to_symbol.txt'
    file_name = '../../results/2021-07-01-high-conf-ADgenes/AD-genes-' + name
    gset = wrappers.convert_to_geneid(file_name=file_name, id_type='symbol', id_mapping_file=id_mapping_file)
    gset, gset_dropped = repos_tools.drop_genes_notin_network(gset, network)
    return(gset)


def process_drug(item):
    start = time.time()
    drugbank_id, targets = item
    res = wrappers.calculate_proximity(network=network, nodes_from=targets, nodes_to=dis_genes)
    runtime = time.time() - start
    print(drugbank_id, f'processed in {runtime:.1f}s')
    return((drugbank_id, res))


def calculate_proximities(drugbank_prot, dis_genes_name='knowledge', asynchronous=True):
    '''
    '''
    start = time.time()
    network_fpath = main_dirpath + 'resources/PPI/Cheng2019/network.sif'
    global network
    network = wrappers.get_network(network_fpath, only_lcc = True)
    global dis_genes
    dis_genes = read_geneset(dis_genes_name)
    gb = drugbank_prot.groupby('drugbank_id')
    l = gb.apply(lambda row: (row.index.get_level_values(0)[0], set(row.entrez_id))).to_list()
    def proc_d(item):
        res = process_drug(*item, dis_genes, network)
        return(res)
    if asynchronous:
        max_workers = os.cpu_count() - 1
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            val = list(executor.map(process_drug, l))
    else:
        val = list(map(process_drug, l)) # for testing synchronous execution
    dfval = postprocess_prox_results(val, drugbank_prot)
    runtime = time.time() - start
    print(f'total runtime: {runtime:.1f}s')
    return(dfval)


def prox_result_to_df(inval):
    drugbank_id, prox_res = inval
    d, z, H0 = prox_res
    avg_d_H0, sdev_d_H0 = H0
    p = scipy.stats.norm.sf(-z)
    d = {'d': d, 'avg_d_H0': avg_d_H0, 'sdev_d_H0': sdev_d_H0, 'z': z, 'p': p}
    ix = pd.Index([drugbank_id], name='drugbank_id')
    val = pd.DataFrame(d, index=ix)
    return(val)


def postprocess_prox_results(val, drugbank_prot):
    # convert proximity results into a data frame
    df_prox = pd.concat(map(prox_result_to_df, val), axis=0)
    # drug information
    drugbank_all_drugs_fpath = main_dirpath + 'results/2021-08-11-drugbank/drugbank-all-drugs.csv'
    usecols = ['drugbank_id', 'name', 'type', 'groups']
    drugbank_drug = pd.read_csv(drugbank_all_drugs_fpath, usecols=usecols, index_col='drugbank_id')
    drugbank_drug = drugbank_drug.loc[df_prox.index]
    # gene/protein information
    l = [repos_tools.collapse_drugbank_proteins_group(drugbank_prot, col=col) for col in ['symbol', 'hgnc_id']]
    drugbank_prot_collapsed = pd.concat(l, axis=1).loc[df_prox.index]
    # putting all together
    dfval = pd.concat([drugbank_drug, drugbank_prot_collapsed, df_prox], axis=1).sort_index()
    return(dfval)
