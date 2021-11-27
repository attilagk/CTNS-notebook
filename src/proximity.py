#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import time
import datetime
from toolbox import wrappers
import concurrent.futures
import repos_tools
from scipy import stats
import pickle
import itertools
import functools


main_dirpath = '../../'
#network = wrappers.get_network(main_dirpath + 'resources/PPI/Cheng2019/network.sif', only_lcc = True)

def read_geneset(dis_genes_fpath, id_mapping_file):
    gset = wrappers.convert_to_geneid(file_name=dis_genes_fpath, id_type='symbol', id_mapping_file=id_mapping_file)
    gset, gset_dropped = repos_tools.drop_genes_notin_network(gset, network)
    return(gset)

#dis_genes = read_geneset(main_dirpath + 'results/2021-07-01-high-conf-ADgenes/AD-genes-knowledge', main_dirpath + 'resources/PPI/geneid_to_symbol.txt')

def read_data(dis_genes_fpath=main_dirpath + 'results/2021-07-01-high-conf-ADgenes/AD-genes-knowledge',
              network_fpath=main_dirpath + 'resources/PPI/Cheng2019/network.sif',
              id_mapping_file=main_dirpath + 'resources/PPI/geneid_to_symbol.txt'):
    global network
    network = wrappers.get_network(network_fpath, only_lcc = True)
    global dis_genes
    dis_genes = read_geneset(dis_genes_fpath, id_mapping_file)
    return((dis_genes, network))


def preprocess_chembl_dtn(dtn_chembl,
                          uniprot2x_path=main_dirpath + 'resources/UniProt/idmapping/HUMAN_9606_idmapping_selected.tab'):
    uniprot2entrez = pd.read_csv(uniprot2x_path, sep='\t', usecols=[0, 1, 2], dtype='str',
                                 names=['uniprot_ac', 'uniprot_name', 'entrez_id'], index_col=0)
    # remove invariant _HUMAN suffix from UniProtKB entry names
    uniprot2entrez['uniprot_name'] = uniprot2entrez.uniprot_name.str.replace('_HUMAN', '')
    # creating sets of entrez_ids from ';' separated strings
    uniprot2entrez['entrez_id'] = uniprot2entrez.entrez_id.apply(lambda x: set() if pd.isna(x)
                                                                 else set(x.split(';')))
    df = uniprot2entrez.loc[dtn_chembl.index.get_level_values(1), :]
    df.index = dtn_chembl.index
    dtn_chembl = pd.concat([dtn_chembl, df], axis=1)
    return(dtn_chembl)


def process_drug(item, network, dis_genes):
    start = time.time()
    drug_id, targets = item
    targets, dropped = repos_tools.drop_genes_notin_network(targets, network)
    try:
        res = wrappers.calculate_proximity(network=network, nodes_from=targets, nodes_to=dis_genes)
    except:
        res = (np.nan, np.nan, (np.nan, np.nan))
    runtime = time.time() - start
    print(f'{drug_id} processed in {runtime:.1f}s')
    return((drug_id, res))


def calculate_proximities(drug_target_network,
                          dis_genes_fpath=main_dirpath + 'results/2021-07-01-high-conf-ADgenes/AD-genes-knowledge',
                          network_fpath=main_dirpath + 'resources/PPI/Cheng2019/network.sif',
                          id_mapping_file=main_dirpath + 'resources/PPI/geneid_to_symbol.txt',
                          uniprot2x_path=main_dirpath + 'resources/UniProt/idmapping/HUMAN_9606_idmapping_selected.tab',
                          drugbank_all_drugs_fpath=main_dirpath + 'results/2021-08-11-drugbank/drugbank-all-drugs.csv',
                          asynchronous=True,
                          pickle_path='default',
                          max_workers=os.cpu_count() - 1):
    '''
    '''
    start = time.time()
    dis_genes, network = read_data(dis_genes_fpath=dis_genes_fpath,
              network_fpath=network_fpath,
              id_mapping_file=id_mapping_file)
    if drug_target_network.index.names[0] == 'drug_chembl_id':
        drug_target_network = preprocess_chembl_dtn(drug_target_network, uniprot2x_path)
    gb = drug_target_network.groupby(axis=0, level=0)
    if drug_target_network.index.names[0] == 'drugbank_id':
        l = gb.apply(lambda row: (row.index.get_level_values(0)[0], set(row.entrez_id))).to_list()
    elif drug_target_network.index.names[0] == 'drug_chembl_id':
        def multi_union(row):
            # takes union of sets of entrez_ids across multiple rows
            res = functools.reduce(lambda a, b: a.union(b), row.entrez_id)
            return(res)
        l = gb.apply(lambda row: (row.index.get_level_values(0)[0], multi_union(row))).to_list()
    if asynchronous:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            val = list(executor.map(process_drug, l, itertools.repeat(network), itertools.repeat(dis_genes)))
    else:
        val = list(map(process_drug, l, itertools.repeat(network), itertools.repeat(dis_genes))) # for testing synchronous execution
    if pickle_path == 'default':
        pickle_path = '/tmp/' + datetime.datetime.utcnow().isoformat() + '.p'
    print('dumping results to', pickle_path)
    pickle.dump(val, open(pickle_path, 'wb'))
    if drug_target_network.index.names[0] == 'drugbank_id':
        dfval = postprocess_drugbank_prox_results(val, drug_target_network, drugbank_all_drugs_fpath)
    elif drug_target_network.index.names[0] == 'drug_chembl_id':
        dfval = postprocess_chembl_prox_results(val, drug_target_network)
    else:
        dfval = val
    runtime = time.time() - start
    print(f'total runtime: {runtime:.1f}s')
    return(dfval)


def postprocess_prox_result(inval):
    drug_id, prox_res = inval
    try:
        d, z, H0 = prox_res
        avg_d_H0, sdev_d_H0 = H0
        p = stats.norm.sf(-z)
    except:
        d, avg_d_H0, sdev_d_H0, z, p = (np.nan, ) * 5
    d = {'d': d, 'avg_d_H0': avg_d_H0, 'sdev_d_H0': sdev_d_H0, 'z': z, 'p': p}
    ix = pd.Index([drug_id], name='drug_id')
    val = pd.DataFrame(d, index=ix)
    return(val)


def postprocess_chembl_prox_results(val, drug_target_network):
    sel_cols = ['drug_name', 'max_phase', 'indication_class']
    df_drug = drug_target_network.groupby(axis=0, level=0).first()[sel_cols]
    df_prox = pd.concat(map(postprocess_prox_result, val), axis=0)
    sel_cols1 = ['uniprot_name', 'target_name']
    l = [repos_tools.collapse_drugbank_proteins_group(drug_target_network,
                                                      col=col) for col in sel_cols1]
    drug_target_network_collapsed = pd.concat(l, axis=1).loc[df_prox.index]
    dfval = pd.concat([df_drug, drug_target_network_collapsed, df_prox], axis=1).sort_index()
    return(dfval)


def postprocess_drugbank_prox_results(val, drug_target_network, drugbank_all_drugs_fpath):
    # convert proximity results into a data frame
    df_prox = pd.concat(map(postprocess_prox_result, val), axis=0)
    # drug information
    usecols = ['drugbank_id', 'name', 'type', 'groups']
    drugbank_drug = pd.read_csv(drugbank_all_drugs_fpath, usecols=usecols, index_col='drugbank_id')
    drugbank_drug = drugbank_drug.loc[df_prox.index]
    # gene/protein information
    l = [repos_tools.collapse_drugbank_proteins_group(drug_target_network, col=col) for col in ['symbol', 'hgnc_id']]
    drug_target_network_collapsed = pd.concat(l, axis=1).loc[df_prox.index]
    # putting all together
    dfval = pd.concat([drugbank_drug, drug_target_network_collapsed, df_prox], axis=1).sort_index()
    return(dfval)

if __name__ == '__main__':
    import configparser
    import sys
    config = configparser.ConfigParser()
    config.read(sys.argv[1])
    try:
        dtn_path = config['DEFAULT']['drug_target_network_fpath']
        drugbank_all_drugs_fpath = None
        index_col = ('drug_chembl_id', 'target_uniprot_ac')
    except KeyError:
        dtn_path = config['DEFAULT']['drugbank_prot_fpath']
        drugbank_all_drugs_fpath = config['DEFAULT']['drugbank_all_drugs_fpath']
        index_col = (0, 1)
    drug_target_network = pd.read_csv(dtn_path, index_col=index_col, dtype={'entrez_id': 'str'})
    if config.getboolean('DEFAULT', 'test_run'):
        drug_target_network = drug_target_network.iloc[0:9]
    try:
        max_workers = int(os.environ['SLURM_CPUS_PER_TASK'])
    except KeyError:
        max_workers = config.getint('DEFAULT', 'max_workers')
    result = calculate_proximities(drug_target_network,
                                   dis_genes_fpath=config['DEFAULT']['dis_genes_fpath'],
                                   network_fpath=config['DEFAULT']['network_fpath'],
                                   id_mapping_file=config['DEFAULT']['id_mapping_file'],
                                   uniprot2x_path=config['DEFAULT']['uniprot_map'],
                                   drugbank_all_drugs_fpath=drugbank_all_drugs_fpath,
                                   asynchronous=config.getboolean('DEFAULT', 'asynchronous'),
                                   pickle_path=config['DEFAULT']['out_csv'] + '.p',
                                   max_workers=max_workers)
    result.to_csv(config['DEFAULT']['out_csv'])
    print('Results written to', config['DEFAULT']['out_csv'])
