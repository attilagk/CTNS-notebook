import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import scipy.stats


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


def plot_dataw_ax(ax, treatment, dataw, nfl_prefix='^NF-L week '):
    data = dataw.loc[dataw.Treatment == treatment]
    nfl_columns = [x for x in dataw.columns if re.match(nfl_prefix + '\d{1,2}$', x)]
    weeks = [np.int64(re.sub(nfl_prefix, '', x)) for x in nfl_columns]
    colord = {'m': 'blue', 'f': 'red'}
    for ix in data.index:
        datum = data.loc[ix]
        nfls = datum.loc[nfl_columns]
        color = colord[datum.Sex]
        ax.plot(weeks, nfls, color=color)
    ax.set_xticks(weeks)
    ax.set_title(treatment)
    return(ax)


def plot_dataw(dataw, treatmentl=None, nfl_prefix='^NF-L week '):
    treatmentl = dataw.Treatment.cat.categories if treatmentl is None else treatmentl
    fig, ax = plt.subplots(1, len(treatmentl), figsize=(2.4 * len(treatmentl), 4.8), sharey=True)
    for axi, treatment in zip(ax, treatmentl):
        axi = plot_dataw_ax(axi, treatment, dataw, nfl_prefix=nfl_prefix)
        axi.grid(axis='y')
    axlabel_fontsize=12
    ax[0].set_ylabel('Nfl, pg/ml', fontsize=axlabel_fontsize)
    fig.supxlabel('Time after treatment, weeks', fontsize=axlabel_fontsize)
    return((fig, ax))


def extract_treatment_data(treatment, dataw, var='NF-L week 0'):
    data = dataw.loc[dataw.Treatment == treatment, var]
    return(data)


def my_ttest(treatment_a, treatment_b, dataw_a, dataw_b=None, var='NF-L week 0'):
    dataw_b = dataw_a if dataw_b is None else dataw_b
    Z = zip([treatment_a, treatment_b], [dataw_a, dataw_b])
    a, b = [extract_treatment_data(treatment, dataw, var=var) for treatment, dataw in Z]
    res = scipy.stats.ttest_ind(a, b)
    return(res)
