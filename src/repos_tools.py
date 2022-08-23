import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xml.etree.ElementTree as ET
import collections
import re
import time
import os
from toolbox import wrappers, network_utilities
import concurrent.futures

def drop_genes_notin_network(genes, network):
    '''
    Drop genes that are not in network

    Parameters
    ----------
    genes: entrez IDs in an iterable
    network: a network object returned by wrappers.get_network

    Value: a tuple of two; the first element is the set of kept genes, while
    the second element is the set of dropped genes.
    '''
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


def get_drugbank_xml_root(xml_path='/home/attila/CTNS/resources/drugbank/drugbank.5.1.8.xml'):
    '''
    Get root element of drugbank.xml; see
    https://github.com/dhimmel/drugbank/blob/gh-pages/parse.ipynb
    '''
    with open(xml_path) as xml_file:
        tree = ET.parse(xml_file)
    root = tree.getroot()
    return(root)


def get_drugbank_drugs(root):
    '''
    Get all drugs from drugbank; see
    https://github.com/dhimmel/drugbank/blob/gh-pages/parse.ipynb
    '''
    ns='{http://www.drugbank.ca}'
    inchikey_template = "{ns}calculated-properties/{ns}property[{ns}kind='InChIKey']/{ns}value"
    inchi_template = "{ns}calculated-properties/{ns}property[{ns}kind='InChI']/{ns}value"
    rows = list()
    for i, drug in enumerate(root):
        row = collections.OrderedDict()
        assert drug.tag == ns + 'drug'
        row['type'] = drug.get('type')
        row['drugbank_id'] = drug.findtext(ns + "drugbank-id[@primary='true']")
        row['name'] = drug.findtext(ns + "name")
        row['description'] = drug.findtext(ns + "description")
        row['groups'] = [group.text for group in drug.findall("{ns}groups/{ns}group".format(ns = ns))]
        row['atc_codes'] = [code.get('code') for code in
                            drug.findall("{ns}atc-codes/{ns}atc-code".format(ns = ns))]
        row['categories'] = [x.findtext(ns + 'category') for x in
                             drug.findall("{ns}categories/{ns}category".format(ns = ns))]
        row['inchi'] = drug.findtext(inchi_template.format(ns = ns))
        row['inchikey'] = drug.findtext(inchikey_template.format(ns = ns))
        # Add drug aliases
        aliases = {
            elem.text for elem in
            drug.findall("{ns}international-brands/{ns}international-brand".format(ns = ns)) +
            drug.findall("{ns}synonyms/{ns}synonym[@language='English']".format(ns = ns)) +
            drug.findall("{ns}international-brands/{ns}international-brand".format(ns = ns)) +
            drug.findall("{ns}products/{ns}product/{ns}name".format(ns = ns))
        }
        aliases.add(row['name'])
        row['aliases'] = sorted(aliases)
        rows.append(row)
    def collapse_list_values(row):
        for key, value in row.items():
            if isinstance(value, list):
                row[key] = '|'.join(value)
                return(row)
    rows = list(map(collapse_list_values, rows))
    columns = ['drugbank_id', 'name', 'type', 'groups', 'atc_codes',
               'categories', 'inchikey', 'inchi', 'description']
    drugbank_df = pd.DataFrame.from_dict(rows)[columns]
    drugbank_df = drugbank_df.set_index('drugbank_id')
    return(drugbank_df)


def get_drugbank_proteins(root):
    '''
    Get all proteins from drugbank; see
    https://github.com/dhimmel/drugbank/blob/gh-pages/parse.ipynb
    '''
    ns='{http://www.drugbank.ca}'
    protein_rows = list()
    for i, drug in enumerate(root):
        drugbank_id = drug.findtext(ns + "drugbank-id[@primary='true']")
        for category in ['target', 'enzyme', 'carrier', 'transporter']:
            proteins = drug.findall('{ns}{cat}s/{ns}{cat}'.format(ns=ns, cat=category))
            for protein in proteins:
                row = {'drugbank_id': drugbank_id, 'category': category}
                row['organism'] = protein.findtext('{}organism'.format(ns))
                row['known_action'] = protein.findtext('{}known-action'.format(ns))
                actions = protein.findall('{ns}actions/{ns}action'.format(ns=ns))
                row['actions'] = '|'.join(action.text for action in actions)
                row['name'] = protein.findtext('{}name'.format(ns))
                uniprot_ids = [polypep.text for polypep in protein.findall(
                    "{ns}polypeptide/{ns}external-identifiers/{ns}external-identifier[{ns}resource='UniProtKB']/{ns}identifier".format(ns=ns))]
                if len(uniprot_ids) != 1:
                    continue
                row['uniprot_id'] = uniprot_ids[0]
                polypeptide = protein.find(ns + 'polypeptide')
                for identifier in polypeptide.find(ns + 'external-identifiers'):
                    resource = identifier.findtext(ns + 'resource')
                    if resource == 'HUGO Gene Nomenclature Committee (HGNC)':
                        row['hgnc_id'] = identifier.findtext(ns + 'identifier')
                protein_rows.append(row)
    protein_df = pd.DataFrame.from_dict(protein_rows)
    protein_df = protein_df.set_index(['drugbank_id', 'uniprot_id'])
    return(protein_df)


def drugbank_drugs_simplify_groups(drugbank_df):
    df = drugbank_df.copy()
    df['group'] = 'other'
    is_approved = df.groups.apply(lambda s: bool(re.match('^approved.*', s)))
    df.loc[is_approved, 'group'] = 'approved'
    is_investigational = df.groups.apply(lambda s: bool(re.match('.*investigational.*', s)))
    df.loc[is_investigational & ~ is_approved, 'group'] = 'investigational'
    return(df)


def filter_drugbank_proteins(protein_df, drugbank_df):
    '''
    Keep only human proteins that fulfill all below:
    * are human proteins 
    * are targets 
    * possess a HGNC ID
    * targeted by small molecules
    '''
    protein_df = protein_df.loc[protein_df.organism == 'Humans', :]
    protein_df = protein_df.loc[protein_df.category == 'target', :]
    protein_df = protein_df.dropna(axis=0, subset=['hgnc_id'])
    df = drugbank_drugs_simplify_groups(drugbank_df)
    protein_df['group'] = df.loc[protein_df.index.get_level_values(0), 'group'].to_list()
    ix2drop = df.index[df.type != 'small molecule']
    protein_df = protein_df.drop(ix2drop, axis=0, level='drugbank_id')
    return(protein_df)


def extend_with_entrez_id(protein_df,
                          hgnc_fpath='/home/attila/CTNS/resources/hgnc/hgnc_complete_set.txt'):
    '''
    Extend filtered protein_df with symbol, entrez id, and group

    Important: protein_df must be filtered with filter_drugbank_proteins.
    '''
    usecols = ['hgnc_id', 'entrez_id', 'symbol']
    entrez = pd.read_csv(hgnc_fpath, sep='\t', usecols=usecols,
                         index_col='hgnc_id', dtype={'entrez_id': np.str})
    l = protein_df.columns.to_list()
    l.remove('name')
    columns = ['symbol', 'name'] + l + ['entrez_id']
    protein_df['symbol'] = entrez.loc[protein_df.hgnc_id, 'symbol'].to_list()
    protein_df['entrez_id'] = entrez.loc[protein_df.hgnc_id, 'entrez_id'].to_list()
    protein_df = protein_df.reindex(columns=columns)
    return(protein_df)

def collapse_drugbank_proteins_group(proteins_f, col='entrez_id'):
    val = proteins_f.groupby(axis=0, level=0)[col].apply(lambda x: '|'.join(x.to_list()))
    #val = proteins_f.groupby('drugbank_id')[col].apply(lambda x: '|'.join(x.to_list()))
    val = val.to_frame(name=col)
    return(val)

def read_chembl_screen_results(csvpath):
    df = pd.read_csv(csvpath, index_col=0)
    df = df.sort_values('z').dropna(subset=['z'])
    df['rank'] = np.arange(len(df), dtype=np.int64) + 1
    df = df.rename_axis('drug_chembl_id', index=True)
    return(df)

def add_b3db_permeabilities(chembl_results,
                            bbbpath=os.environ['HOME'] + '/CTNS/results/2021-12-13-chembl-drug-info/drug-info-bbb.csv'):
    dtypes = {'logBB': np.float64, 'BBB+/BBB-': 'category', 'group': 'category'}
    chembl_bbb = pd.read_csv(bbbpath, index_col=0, dtype=dtypes)
    chembl_bbb = chembl_bbb.loc[chembl_results.index].drop(['drug_name'], axis=1)
    val = chembl_results.join(chembl_bbb)
    return(val)

def symbol2entrez_gmt(gmt_fpath,
                      hgn_fpath='~/CTNS/resources/hgnc/hgnc_complete_set.txt'):
    usecols = ['entrez_id', 'symbol']
    hgn = pd.read_csv(hgn_fpath, sep='\t', usecols=usecols,
                      index_col='symbol', dtype={'entrez_id': str})
    hgns = hgn.squeeze()
    def symbol2entrezid_line(line, hgns):
        lin = line.rstrip().split('\t')
        lout = lin[:2] + hgns.loc[lin[2:]].to_list()
        sout = '\t'.join(lout)# + '\n'
        return(sout)
    with open(gmt_fpath, 'r') as f:
        for line in f:
            sout = symbol2entrezid_line(line, hgns)
            print(sout)
    return(None)
