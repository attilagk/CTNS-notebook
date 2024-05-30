import pandas as pd
import numpy as np
import re


def read_nfl_data(fpath, sheet_name, treatments, study, fpath_subjects=None):
    dataw = pd.read_excel(fpath, sheet_name=sheet_name)
    dataw['IRN'] = dataw.IRN.astype('str')
    dataw['Treatment'] = dataw.Group.apply(lambda x: treatments[x])
    dataw['Sex'] = pd.Categorical(dataw.Sex, categories=['m', 'f'], ordered=True)
    dataw['Study'] = study
    # calculate change of Nfl
    value_vars = [x for x in dataw.columns if re.match('^NF-L week \d+$', x)]
    id_vars = [x for x in dataw.columns if not re.match('^NF-L week \d+$', x)]
    df = dataw[value_vars].apply(lambda r: r - r.loc['NF-L week 0'], axis=1)
    df = df.rename({x: 'ΔNfl week ' + re.sub('NF-L week ', '', x) for x in df.columns}, axis=1)
    df['max_ΔNfl'] = df.drop('ΔNfl week 0', axis=1).max(axis=1)
    dataw = pd.concat([dataw, df], axis=1)
    dataw = dataw.set_index('IRN')
    # read and add info on subjects ("animal-list" file)
    if fpath_subjects is not None:
        df = pd.read_csv(fpath_subjects, dtype={'IRN': str}, index_col='IRN')
        df['Lifespan (weeks)'] = df['Lifespan (days)'] / 7
        dataw = pd.concat([dataw, df[['Curriculum vitae', 'Lifespan (weeks)']]], axis=1, join='inner')
    return(dataw)
