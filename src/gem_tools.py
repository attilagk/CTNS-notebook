import pandas as pd
import numpy as np
import seaborn as sns
import itertools
import matplotlib.patches as mpatches
import concurrent.futures
import warnings
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

warnings.filterwarnings('ignore', category=RuntimeWarning)


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
                   fpath='../../resources/human-GEM/v1.11.0/model/Human-GEM.xlsx'):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        gem = pd.read_excel(fpath, usecols=usecols, index_col=index_col, engine='openpyxl')
    return(gem)

def extract_subsystem(subsystems, ar):
    gemsubsys = read_gem_excel()['SUBSYSTEM']
    l = [ar[k] if subsystems is None else ar[k].loc[gemsubsys.loc[gemsubsys.isin(subsystems)].index] for k in ar.keys()]
    df = pd.concat(l, axis=1)
    return(df)


def ar_clustermap(subsystems, ar, col_cluster=False, row_cluster=False,
                  row_colors=None, row_cmap=None, col_palette=None, draw_cbar=True,
                  draw_col_legend=True, figsize=(7,7)):
    '''
    Active reactions cluster map
    '''
    df = extract_subsystem(subsystems, ar=ar)
    col_palette = col_palette if col_palette is not None else ['C' + str(i) for i, g in enumerate(ar.keys())]
    coldict = dict(zip(ar.keys(), col_palette))
    col_colors = list(itertools.chain(*[[v] * ar[k].shape[1] for k, v in coldict.items()]))
    cmap = ['white', 'gray']
    cbar_pos = (0 - 0.125 * row_cluster, 0.4, 0.05, 0.2) if draw_cbar else None
    g = sns.clustermap(df, row_cluster=row_cluster, col_cluster=col_cluster,
                       row_colors=row_colors, col_colors=col_colors,
                       cmap=cmap, figsize=figsize, dendrogram_ratio=0.15,
                       cbar_pos=cbar_pos)
    if draw_col_legend:
        handles = [mpatches.Patch(color=c) for c in coldict.values()]
        col_bbox_to_anchor = (0.3, 0.9 + 0.1 * col_cluster, 0.5, 0.1)
        g.fig.legend(handles, ar.keys(), loc='lower center', bbox_to_anchor=col_bbox_to_anchor, ncol=2)
    if row_colors is not None:
        row_handles = [mpatches.Patch(color=c) for c in row_cmap]
        #row_bbox_to_anchor = (0 - 1.50 * row_cluster, 0.5, 0.5, 0.2)
        row_bbox_to_anchor = (0.15 - 0.25 * row_cluster, 0.5)
        g.fig.legend(row_handles, row_cmap.index, loc='center right',
                     bbox_to_anchor=row_bbox_to_anchor, ncol=1,
                     fontsize='x-small')
    g.ax_heatmap.set_xlabel(str(df.shape[1]) + ' samples')
    g.ax_heatmap.set_xticklabels('')
    g.ax_heatmap.set_xticks(range(df.shape[1]))
    if subsystems is not None:
        subsys_str = ', '.join(subsystems)
        suffix = ' in ' + subsys_str if len(subsys_str) <= 50 else ''
    else:
        suffix = ''
    g.ax_heatmap.set_ylabel(str(df.shape[0]) + ' reactions' + suffix)
    g.ax_heatmap.set_yticklabels('')
    g.ax_heatmap.set_yticks(range(df.shape[0]) if df.shape[0] <= 100 else [])
    for spine in g.ax_heatmap.spines.values():
        spine.set_visible(True)
    if draw_cbar:
        g.ax_cbar.set_position((0 - 0.125 * row_cluster, 0.4, 0.05, 0.2))
        g.ax_cbar.set_title('reaction state')
        g.ax_cbar.set_yticks([0.25, 0.75])
        g.ax_cbar.set_yticklabels(['inactive', 'active'])
        g.ax_cbar.set_ylim([0, 1])
        for spine in g.ax_cbar.spines.values():
            spine.set_visible(True)
    return(g)


'''
Model fitting
'''

def reshape2long(df, disease_state):
    long = df.stack().to_frame('rxn_state')
    long['rxn_state'] = np.int8(long.rxn_state)
    long['disease_state'] = disease_state
    long['rxn_ID'] = long.index.get_level_values(0)
    long['subject_ID'] = long.index.get_level_values(1)
    return(long)


def long_ar_subsys(subsystems, ar, gemsubsys):
    l = [reshape2long(ar[k].loc[gemsubsys.loc[gemsubsys.isin(subsystems)].index], k) for k in ar.keys()]
    long = pd.concat(l, axis=0)
    return(long)


def myBinomialBayesMixedGLM(subsys, ar, control_group='m-control', AD_group='m-AD-B2', vcp_p=0.2, fe_p=2,
                            fit_method='fit_vb', gemsubsys=read_gem_excel()['SUBSYSTEM']):
    data = long_ar_subsys(subsys, ar=ar, gemsubsys=gemsubsys)
    if data.rxn_state.std() == 0:
        return(None)
    random = {'Reactions': 'rxn_ID', 'Subjects': 'subject_ID'}
    formula = 'rxn_state ~ C(disease_state, levels=["' + control_group + '", "' + AD_group + '"])'
    md = BinomialBayesMixedGLM.from_formula(formula, random, data, vcp_p=vcp_p, fe_p=fe_p)
    fit = getattr(md, fit_method)
    m = fit()
    return(m)


def fit_all_subsystems(AD_group='m-AD-B2', AD_name='SubtypeB2_AD', AD_cohort='MSBB'):
    control_group = 'm-control' if AD_cohort == 'MSBB' else 'r-control'
    control_name = 'all_control'
    control_cohort = AD_cohort
    groupdict = {control_group: (control_name, control_cohort),
                 AD_group: (AD_name, AD_cohort)}
    return(groupdict)
    ar = read_active_reactions(groupdict)
    gemsubsys = gem_tools.read_gem_excel()['SUBSYSTEM']
    subsystems = gemsubsys.unique()
    args = {'ar': ar,
            'control_group': control_group
            }
    #d = {subsys: myBinomialBayesMixedGLM(, 'm-control', 'm-AD-B2', vcp_p=0.2, fe_p=2, fit_method='fit_map') for subsys in subsystems}
    return(None)




'''
Bayesian computation
'''

def prepare_data(m):
    '''
    Brings fitted data to a format that helps repeated likelihood
    calculations

    Parameter: m, a fitted model object (statsmodels.genmod.bayes_mixed_glm.BayesMixedGLMResults)

    Value: a tuple of three elements:
    pdata: the prepared data
    ref_rxn: the rxn_ID used as reference level in dummy coding
    ref_subject: the subject_ID used as reference level in dummy coding
    '''
    pdata = m.model.data.frame.copy()
    pdata['rxn_ID'] = pdata.rxn_ID.apply(lambda s: 'rxn_ID[T.' + s + ']')
    pdata['subject_ID'] = pdata.subject_ID.apply(lambda s: 'subject_ID[T.' + s + ']')
    ref_rxn = list(set(pdata.rxn_ID.unique()).difference(m.model.names))[0]
    ref_subject = list(set(pdata.subject_ID.unique()).difference(m.model.names))[0]
    pdata['Y'] = m.model.endog
    pdata['Dx'] = m.model.exog[:, 1]
    return((pdata, ref_rxn, ref_subject))


def get_log_likelihood(params, pdata, ref_rxn, ref_subject):
    '''
    Calculate log-likelihood for given data and parameters
    '''
    pars = params.copy()
    pars[ref_rxn] = 0
    pars[ref_subject] = 0
    df = pd.DataFrame({'Intercept': pars.loc['Intercept'].sum(),
                       'Dx': pars.iloc[1] * pdata.Dx,
                       'rxn_ID': pars.loc[pdata.rxn_ID].to_numpy(),
                       'subject_ID': pars.loc[pdata.subject_ID].to_numpy(),
                       })
    # sum terms of linear predictor, then apply logistic function to get non-linear predictor
    pi = df.sum(axis=1).apply(lambda x: 1 / (1 + np.exp(-x))).to_frame('pi')
    # get likelihoods
    likelihoods = pd.concat([pdata['Y'], pi], axis=1).apply(lambda r: r.pi if r.Y else 1 - r.pi, axis=1)
    LL = likelihoods.apply(np.log).sum()
    return(LL)


def sample_params(mean, cov, m, null=False):
    '''
    Sample parameters from the posterior normal distribution
    '''
    # mean[1] is the posterior mean of the fixed effect of Dx
    meanc = mean.copy()
    meanc[1] = 0 if null else meanc[1]
    params = np.random.multivariate_normal(meanc, cov)
    params = pd.Series(params, index=m.model.names)
    return(params)


def BF_from_marginal_likelihoods(LLs):
    replicas = LLs.shape[0]
    twice_log_BF = - 2 * LLs.mean().diff().loc['M0']
    return(twice_log_BF)


def get_marginal_likelihoods(m, replicas=12, returnBF=True, asynchronous=False, max_workers=6):
    pdata, ref_rxn, ref_subject = prepare_data(m)
    param_samples_l = [[sample_params(mean=m.params, cov=m.cov_params(), m=m, null=null)
                     for i in range(replicas)] for null in [False, True]]
    if asynchronous:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            ll = [list(executor.map(get_log_likelihood,
                                    param_samples,
                                    itertools.repeat(pdata),
                                    itertools.repeat(ref_rxn),
                                    itertools.repeat(ref_subject)))
                  for param_samples in param_samples_l]
    else:
        ll = [[get_log_likelihood(param, pdata, ref_rxn, ref_subject)
               for param in param_samples] for param_samples in param_samples_l]
    a = np.array(ll).transpose()
    LLs = pd.DataFrame(a, columns=['M1', 'M0'])
    if not returnBF:
        return(LLs)
    twice_log_BF = BF_from_marginal_likelihoods(LLs)
    return(twice_log_BF)
