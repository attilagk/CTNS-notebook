import pandas as pd

def read_data(datapath, codebookpath, liberal_filter=True, return_m2exclude=False):
    data = pd.read_csv(datapath, index_col='idno')
    data = data.rename({'LIScore': 'LIscore'}, axis=1)
    metabolites = data.columns[4:]
    m2exclude_lib = [metabolite for metabolite in data.columns[4:]
                     if any([sum(~ data.loc[data.Group == group, metabolite].isna()) < 2 for group in ['AI', 'AU', 'Y']])]
    m2exclude_cons = [m for m in metabolites if data.loc[:, m].isna().sum() > len(data) * 0.2]
    metabolites2exclude = m2exclude_lib if liberal_filter else m2exclude_cons
    codebook = pd.read_csv(codebookpath, index_col='metabolite')
    if return_m2exclude:
        metabolites2exclude_in_codebook = set(metabolites2exclude).intersection(set(codebook.index))
        excluded_metabolites = codebook.loc[metabolites2exclude_in_codebook, 'MetaboliteName'].to_list()
        return(excluded_metabolites)
    data = data.drop(metabolites2exclude, axis=1)
    data = data.rename(columns=dict(zip(codebook.index, codebook.MetaboliteName)))
    #data = data.rename(columns=dict(zip(codebook.index, tuple(zip(codebook.AnalyteClass, codebook.MetaboliteName)))))
    try:
        data = data.drop(['material', 'species'], axis=1)
    except KeyError:
        data = data
    A = set(data.columns[2:])
    data = data.drop(A.difference(set(codebook.MetaboliteName)), axis=1)
    return(data)

def read_all_data(prefix='/Users/jonesa7/CTNS/resources/rat-metabolites/Rat_',
                  codebookpath='/Users/jonesa7/CTNS/resources/rat-metabolites/Rat_codebook_27_Oct_2022.csv'):
    datapath = {
        'blood new': 'blood_27_Oct_2022',
        'brain new': 'brain_27_Oct_2022',
        'blood old': 'old_blood_17_Nov_2022',
        'brain old': 'old_brain_17_Nov_2022',
    }
    data = {k: read_data(prefix + v + '.csv', codebookpath, liberal_filter=True) for k, v in datapath.items()}
    data.update({tissue: pd.concat([data[tissue + ' new'], data[tissue + ' old']], axis=0, join='outer') for tissue in ['blood', 'brain']})
    return(data)

