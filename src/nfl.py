import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines
import re
import seaborn as sns
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


def results2df(resultd, ref_treatment='Saline TG', study='CO28152', drop_intercept=False):
    def helper(duration):
        results = resultd[duration]
        df = pd.concat([results.params.to_frame('mean'),
                        results.bse.to_frame('bse'),
                        results.pvalues.to_frame('pval')], axis=1)
        df['treatment'] = [x.replace('Treatment[T.', '').replace(']', '') for x in df.index]
        df['reference treatment'] = ref_treatment
        df['study'] = study
        df['duration'] = duration
        ix = pd.MultiIndex.from_frame(df[['duration', 'treatment']])
        df = pd.DataFrame(df.to_numpy(), columns=df.columns, index=ix)
        return(df)
    l = [helper(d) for d in resultd.keys()]
    val = pd.concat(l, axis=0)
    val = val.drop('Intercept', axis=0, level=2) if drop_intercept else val
    return(val)


def result_plotter_ax(ax, df, suptitle=''):
    full_effect_mean, full_effect_se = df.loc['Intercept', ['mean', 'bse']]
    df = df.drop('Intercept', axis=0)
    rectL, rectR = [Rectangle(xy=(- full_effect_mean, -0.5), width=full_effect_se,
                              height=df.shape[0] + 1, alpha=0.5, color='lightgreen', angle=a,
                              rotation_point=(- full_effect_mean, df.shape[0]/2)) for a in [180, 0]]
    ax[0].add_patch(rectL)
    ax[0].add_patch(rectR)
    ax[0].axvline(0, color='k', linewidth=1)
    ax[0].axvline(- full_effect_mean, color='green', linewidth=1, linestyle='solid')
    ax[0].errorbar(y=np.arange(df.shape[0]), x=df['mean'], xerr=df['bse'], linewidth=0, elinewidth=1, marker='d', capsize=0)
    ax[0].set_title(r'effect size, $\hat{\beta}_j \pm \mathrm{SE}_j$')
    ax[1].axvline(1, color='k', linewidth=1)
    sns.stripplot(x='pval', y='treatment', ax=ax[1], data=df)
    ax[1].set_xscale('log')
    ax[1].set_xlabel('')
    ax[1].set_ylabel('')
    ax[1].set_title(r'$p$-value for no effect $H_{0j}: \; \beta_j = 0$')
    for j in range(2):
        axi = ax[j]
        axi.grid(linestyle='dotted')
        axi.set_yticks(list(range(len(df))))
        axi.set_ylim(len(df) - 0.5, -0.5)
    ax[0].set_yticklabels(df.treatment)
    ax[1].set_yticklabels('')
    return(ax)

def result_plotter(l_resultsdf, duration='week 0-4', hspace_denom=None, x0_lim=None, x1_lim=None):
    resl = [df.xs(duration, axis=0, level=0) for df in l_resultsdf]
    height_ratios = [len(df) for df in resl]
    hspace_denom = 40 if hspace_denom is None else hspace_denom
    gridspec_kw = {'hspace': sum(height_ratios) / hspace_denom}
    fig, ax = plt.subplots(len(resl), 2, figsize=(6.4, sum(height_ratios) * 0.5),
                           height_ratios=height_ratios, squeeze=False, gridspec_kw=gridspec_kw)
    for i, res in enumerate(resl):
        axes = ax[i, :]
        axes = result_plotter_ax(axes, res)
        if x0_lim is not None:
            axes[0].set_xlim(*x0_lim)
        if x1_lim is not None:
            axes[1].set_xlim(*x1_lim)
        if i > 0:
            [axes[j].set_title('') for j in range(2)]
    black_line, green_line = [mlines.Line2D([], [], color=color, marker=None, linewidth=1, label=label)
                              for color, label in zip(['black', 'green'], ['no effect', 'full effect'])]
    fig.legend(handles=[black_line, green_line], loc='upper left')
    return((fig, ax))
