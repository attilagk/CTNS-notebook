import pandas as pd

def read_active_reactions_group(group='all_control', cohort='MSBB'):
    bname = '../../resources/tunahan/Busra-2023-02-05/active_reactions/'
    suffix = '.xlsx'
    fpath = bname + cohort + '/' + group + suffix
    df = pd.read_excel(fpath, 'Sheet1', index_col='rxn_ID', dtype=bool)
    return(df)

def read_active_reactions(groupdict={'m-control': ('all_control', 'MSBB'),
                                     'm-AD-B2': ('SubtypeB2_AD', 'MSBB')}):
    ar = {k: read_active_reactions_group(*v) for k, v in groupdict.items()}
    return(ar)

def read_gem_excel(index_col='ID', usecols=['ID', 'SUBSYSTEM'],
                   fpath='/Users/jonesa7/CTNS/resources/human-GEM/v1.11.0/model/Human-GEM.xlsx'):
    gem = pd.read_excel(fpath, usecols=usecols, index_col=index_col)
    return(gem)
