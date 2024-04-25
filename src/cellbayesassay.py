import arviz as az
import pymc as pm
import pandas as pd
import numpy as np
import os
import os.path
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import string
import itertools
import seaborn as sns
import re
from matplotlib.gridspec import GridSpec

my_var = 3
mcmc_random_seed = [1947, 1949, 1976, 2021]
gamma_shape = 5
default_H1_prior_prob = 0.2


def estimate_y_0(y_obs, x_obs):
    s = pd.Series(y_obs, index=x_obs)
    s = s.loc[np.unique(x_obs)[:2]]
    val = np.mean(s)
    return(val)

def sample_linear_1(y_obs, x_obs, return_model=False):
    mymodel = pm.Model()
    with mymodel:
        σ = pm.HalfStudentT("σ", 20, 100)
        β_0 = pm.Normal("β_0", mu=0, sigma=20)
        β_1 = pm.Normal("β_1", mu=0, sigma=10)
        μ = pm.Deterministic("μ", β_0 + β_1 * x_obs)
        y = pm.Normal("y", mu=μ, sigma=σ, observed=y_obs)
        if return_model:
            return(mymodel)
        mcmc = pm.sample(return_inferencedata=True, idata_kwargs={'log_likelihood': True}, random_seed=mcmc_random_seed)
        return(mcmc)

def sample_sigmoid_1(y_obs, x_obs, return_model=False):
    EC_50_mu = np.quantile(x_obs, 0.5)
    y_0_alpha = estimate_y_0(y_obs, x_obs)
    mymodel = pm.Model()
    with mymodel:
        EC_50 = pm.Normal('EC_50', EC_50_mu, 0.5)
        y_0 = pm.Gamma('y_0', y_0_alpha, 1)
        FC_y = pm.Gamma('FC_y', gamma_shape, gamma_shape)
        y_1 = pm.Deterministic('y_1', FC_y * y_0)
        k = pm.HalfStudentT("k", nu=1, sigma=1)
        σ = pm.HalfStudentT("σ", nu=20, sigma=100)
        μ = pm.Deterministic("μ", y_1 + (y_0 - y_1) / (1 + np.exp(k * (x_obs - EC_50))))
        y = pm.Normal("y", mu=μ, sigma=σ, observed=y_obs)
        if return_model:
            return(mymodel)
        mcmc = pm.sample(return_inferencedata=True, idata_kwargs={'log_likelihood': True}, random_seed=mcmc_random_seed)
        return(mcmc)

def sample_sigmoid_2(y_obs, x_obs, return_model=False):
    EC_50_mu = np.quantile(x_obs, 0.5)
    y_0_alpha = estimate_y_0(y_obs, x_obs)
    mymodel = pm.Model()
    with mymodel:
        EC_50 = pm.Normal('EC_50', EC_50_mu, 0.5)
        y_0 = pm.Gamma('y_0', y_0_alpha, 1)
        FC_y = pm.Gamma('FC_y', gamma_shape, gamma_shape)
        y_1 = pm.Deterministic('y_1', FC_y * y_0)
        k = pm.HalfStudentT("k", nu=1, sigma=1)
        μ = pm.Deterministic("μ", abs(y_1 + (y_0 - y_1) / (1 + np.exp(k * (x_obs - EC_50)))))
        σ = pm.HalfStudentT("σ", 10, 20)
        σ_y = pm.Deterministic("σ_y", σ * μ)
        y = pm.Normal("y", mu=μ, sigma=σ_y, observed=y_obs)
        if return_model:
            return(mymodel)
        mcmc = pm.sample(return_inferencedata=True, idata_kwargs={'log_likelihood': True}, random_seed=mcmc_random_seed)
        return(mcmc)

def sample_sigmoid_3(y_obs, x_obs, return_model=False):
    EC_50_mu = np.quantile(x_obs, 0.5)
    y_0_alpha = estimate_y_0(y_obs, x_obs)
    mymodel = pm.Model()
    with mymodel:
        EC_50 = pm.Normal('EC_50', EC_50_mu, 0.5)
        y_0 = pm.Gamma('y_0', y_0_alpha, 1)
        FC_y = pm.Gamma('FC_y', gamma_shape, gamma_shape)
        y_1 = pm.Deterministic('y_1', FC_y * y_0)
        k = pm.HalfStudentT("k", nu=1, sigma=1)
        μ = pm.Deterministic("μ", abs(y_1 + (y_0 - y_1) / (1 + np.exp(k * (x_obs - EC_50)))))
        α = pm.HalfStudentT("α", nu=1, sigma=1)
        β = pm.HalfStudentT("β", nu=1, sigma=1)
        y = pm.Gamma('y', alpha=α * μ, beta=β, observed=y_obs)
        if return_model:
            return(mymodel)
        mcmc = pm.sample(return_inferencedata=True, idata_kwargs={'log_likelihood': True}, random_seed=mcmc_random_seed)
        return(mcmc)

def sample_sigmoid_4(y_obs, x_obs, return_model=False):
    EC_50_mu = np.quantile(x_obs, 0.5)
    y_0_alpha = estimate_y_0(y_obs, x_obs)
    mymodel = pm.Model()
    with mymodel:
        EC_50 = pm.Normal('EC_50', EC_50_mu, 0.5)
        y_0 = pm.Gamma('y_0', y_0_alpha, 1)
        FC_y = pm.Gamma('FC_y', gamma_shape, gamma_shape)
        y_1 = pm.Deterministic('y_1', FC_y * y_0)
        k = pm.HalfStudentT("k", nu=1, sigma=1)
        μ = pm.Deterministic("μ", abs(y_1 + (y_0 - y_1) / (1 + np.exp(k * (x_obs - EC_50)))))
        α = pm.HalfStudentT("α", nu=1, sigma=1)
        β = pm.HalfStudentT("β", nu=1, sigma=1)
        y = pm.Gamma('y', alpha=α, beta=β / μ, observed=y_obs)
        if return_model:
            return(mymodel)
        mcmc = pm.sample(return_inferencedata=True, idata_kwargs={'log_likelihood': True}, random_seed=mcmc_random_seed)
        return(mcmc)

my_models = {
    'linear 1': sample_linear_1,
    'sigmoid 1': sample_sigmoid_1,
    'sigmoid 2': sample_sigmoid_2,
    'sigmoid 3': sample_sigmoid_3,
    'sigmoid 4': sample_sigmoid_4,
}


def read_data(compound='TI26', assay='TNF', bname='../../resources/cell-based-assays/Tina-email-2023-08-31/'):
    fpath = bname + compound + '_' + assay + '.txt'
    data = pd.read_csv(fpath, sep='\t')
    return(data)


def reshape_data(data, concentrations):
    df = data[concentrations.keys()].stack().to_frame('activity')
    df['conc'] = [concentrations[c] for c in df.index.get_level_values(1)]
    concentrations_log10 = {k: np.log10(v) for k, v in concentrations.items()}
    df['conc_log10'] = [concentrations_log10[c] for c in df.index.get_level_values(1)]
    return(df)

def read_reshape_data(compound, assay, concentrations,
                      bname='../../resources/cell-based-assays/Tina-email-2023-08-31/',
                      standardize=True):
    df = read_data(compound=compound, assay=assay, bname=bname)
    df = reshape_data(df, concentrations=concentrations)
    if standardize:
        df['activity'] = df['activity'] / df['activity'].std() * 10
    return(df)


exp_name = 'Abeta clearance, iPSC'
drug_concentrations_Ab = {
    'TI1': {
        'c1': 100e-6,
        'c2': 10e-6,
        'c3': 1e-6,
        'VC': 1e-9,
    },
    'TI21': {
        'c1': 100e-6,
        'c2': 10e-6,
        'c3': 1e-6,
        'VC': 1e-9,
    },
    'TI26': {
        'c1': 10e-6,
        'c2': 3e-6,
        'c3': 1e-6,
        'VC': 1e-9,
    },
}
assays_Ab = ['pHrodo-number-4h', 'supernatant']
H1_increase_Ab = [True, False] # scalar bool or the same length as assays
bname_Ab = '../../resources/cell-based-assays/Tina-email-2023-08-29/'

def get_experiment_conditions(exp_name, assays, drug_concentrations, H1_increase, bname):
    d = {(exp_name, a, d): c for d, c in drug_concentrations.items() for a in assays}
    experiments = pd.Series(d.values(), index=d.keys()).to_frame('concentrations')
    experiments['compound'] = experiments.index.get_level_values(2)
    experiments['assay'] = experiments.index.get_level_values(1)
    if isinstance(H1_increase, bool):
        experiments['H1_increase'] = H1_increase
    else:
        d = dict(zip(assays, H1_increase))
        experiments['H1_increase'] = [d[e] for e in experiments.index.get_level_values(level=1)]
    experiments['bname'] = bname
    experiments = experiments.sort_index(axis=0, level=1)
    return(experiments)


def fit_model_experiment(compound, assay, concentrations, model='sigmoid 2',
                         n_chains=4, return_model=False, my_models=my_models,
                         bname='../../resources/cell-based-assays/Tina-email-2023-08-31/'):
    data_reshaped = read_reshape_data(compound=compound, assay=assay, concentrations=concentrations, bname=bname)
    x_obs = data_reshaped['conc_log10'].values
    y_obs = data_reshaped['activity'].values
    fun = my_models[model]
    val = fun(y_obs, x_obs, return_model=return_model)
    return(val)


def fit_models(models, experiments):
    def fit_one_model(model):
        sel_cols = ['compound', 'assay', 'concentrations', 'bname']
        l = [experiments.loc[:, sel_cols].apply(lambda r:
                                                fit_model_experiment(**r.to_dict(), model=model, return_model=b),
                                                axis=1).to_frame((model, colname))
             for b, colname in zip([True, False], ['model', 'idata'])]
        return(pd.concat(l, axis=1))
    l = [fit_one_model(m) for m in models]
    results = pd.concat(l, axis=1)
    return(results)


def get_H1_posterior_prob(posterior_idata, H1_prior_prob=default_H1_prior_prob,
                          H1_increase=True, prior_gamma_loc=gamma_shape,
                          prior_gamma_scale=1/gamma_shape):
    if H1_increase:
        t = scipy.stats.gamma.ppf(1 - H1_prior_prob, prior_gamma_loc, scale=prior_gamma_scale)
        H1_posterior_prob = sum(sum(posterior_idata > t))
    else:
        t = scipy.stats.gamma.ppf(H1_prior_prob, prior_gamma_loc, scale=prior_gamma_scale)
        H1_posterior_prob = sum(sum(posterior_idata < t))
    H1_posterior_prob = np.float64(H1_posterior_prob)
    H1_posterior_prob /= len(posterior_idata.to_numpy().ravel())
    return(H1_posterior_prob)


def get_H1_posterior_prob_batch(idata_series, H1_increase_series, H1_prior_prob=default_H1_prior_prob):
    df = pd.concat([idata_series.to_frame('idata'),
                    H1_increase_series.to_frame('H1_increase')], axis=1)
    res = df.apply(lambda r: get_H1_posterior_prob(r.loc['idata'].posterior['FC_y'],
                                                   H1_increase=r.loc['H1_increase'],
                                                   H1_prior_prob=H1_prior_prob), axis=1)
    return(res)


def plot_data(ax, data_reshaped):
    if 'std_activity' in data_reshaped:
        data_reshaped = data_reshaped.copy()
        data_reshaped = data_reshaped.rename({'std_activity': 'activity'}, axis=1)
    xx = np.linspace(data_reshaped.conc_log10.min(), data_reshaped.conc_log10.max() + 1, 200)
    ax.scatter(x='conc_log10', y='activity', data=data_reshaped, marker='+', color='k')
    ax.set_xlabel(r'$\log_{10}$ conc')
    ax.set_ylabel(r'activity')
    return(ax)


def plot_sampled_curves_sigmoid(ax, idata, data_reshaped, color='C0', alpha=0.5,
                                plot_sampled_curves=True, draw_y0_y1=False,
                                H1_prior_prob=default_H1_prior_prob,
                                H1_increase=False, ylim_top=None, H_text=True,
                                H_yticks=True, linewidth=0.2):
    t_1 = scipy.stats.gamma.ppf(H1_prior_prob, gamma_shape, scale=1/gamma_shape)
    t_2 = scipy.stats.gamma.ppf(1 - H1_prior_prob, gamma_shape, scale=1/gamma_shape)
    xx = np.linspace(data_reshaped.conc_log10.min(), data_reshaped.conc_log10.max() + 1, 200)
    chain = 0 # use samples from only one chain
    if plot_sampled_curves:
        for i in range(idata.dims['draw']):
            EC_50 = idata['EC_50'][chain][i].to_numpy()
            k = idata['k'][chain][i].to_numpy()
            y_0 = idata['y_0'][chain][i].to_numpy()
            y_1 = idata['y_1'][chain][i].to_numpy()
            yy = y_1 + (y_0 - y_1) / (1 + np.exp(k * (xx - EC_50)))
            ax.plot(xx, yy, linewidth=linewidth, color=color, alpha=alpha)
    EC_50_mean = idata.mean().to_dict()['data_vars']['EC_50']['data']
    k_mean = idata.mean().to_dict()['data_vars']['k']['data']
    y_0_mean = idata.mean().to_dict()['data_vars']['y_0']['data']
    y_1_mean = idata.mean().to_dict()['data_vars']['y_1']['data']
    y_sigmoid_1_mean = y_1_mean + (y_0_mean - y_1_mean) / (1 + np.exp(k_mean * (xx - EC_50_mean)))
    ax.plot(xx, y_sigmoid_1_mean, color='red', linewidth=3, label='sigmoid 1')
    if draw_y0_y1:
        def add_H_region(is_upper=True, H1_increase=H1_increase):
            if is_upper and (ax.get_ylim()[1] <= y_0_mean * t_2):
                ax.set_ylim(ax.get_ylim()[0], y_0_mean * t_2 * 1.05)
            if (not is_upper) and (ax.get_ylim()[0] >= y_0_mean * t_1):
                ax.set_ylim(y_0_mean * t_1 * 0.9, ax.get_ylim()[1])
            bottom = y_0_mean * t_2 if is_upper else 0
            if ylim_top is None:
                height = ax.get_ylim()[1] - y_0_mean * t_2 if is_upper else y_0_mean * t_1
            else:
                height = ylim_top - y_0_mean * t_2 if is_upper else y_0_mean * t_1
            color = 'green' if is_upper and H1_increase or not is_upper and not H1_increase else 'red'
            H = '$H_1$: protective' if is_upper and H1_increase or not is_upper and not H1_increase else '$H_2$: adverse'
            ax.add_patch(plt.Rectangle((xx[0], bottom), xx[-1] - xx[0], height, ls=None, lw=0, ec="c", fc=color, alpha=0.2))
            if H_text:
                ax.text(xx[0] + 0.1 * (xx[-1] - xx[0]), bottom + height / 2, H,
                        color=color, backgroundcolor='white', ha='left', va='center')
            ax.axhline(y_0_mean * t_2 if is_upper else y_0_mean * t_1, linestyle='solid', color='k', linewidth=0.5)
            return(ax)
        ax.axhline(y_0_mean, linestyle='solid', color='k', linewidth=2)
        ax.axhline(y_1_mean, linestyle='dashed', color='k', linewidth=1)
        ax = add_H_region(True, H1_increase)
        ax = add_H_region(False, H1_increase)
        if H_text:
            ax.text(xx[0] + 0.1 * (xx[-1] - xx[0]), y_0_mean, '$H_0$: neutral',
                    color='gray', backgroundcolor='white', ha='left', va='center')
        if H_yticks:
            labels = ['$y_0$', '$y_0 t_1$', '$y_0 \mathrm{FC}_y$', '$y_0 t_2$']
            ax.set_yticks([y_0_mean, y_0_mean * t_1, y_1_mean, y_0_mean * t_2], labels=labels)
        #ax.text(EC_50_mean - 2, y_0_mean / 4, '$H_1$',
        #        color='green', backgroundcolor='white')
    ax.set_xlabel(r'$\log_{10}$ conc')
    return(ax)


def prior_posterior_curves_sigmoid(model, prior_samples, idata):
    prior = prior_samples[model].prior
    fig, ax = plt.subplots(1, 2, sharey=True)
    fig.suptitle('Model: ' + model)
    for axi, data, title in zip(ax, [prior, idata[model].posterior], ['prior sample', 'posterior sample']):
        if title == 'posterior sample':
            axi = plot_data(axi)
        axi = plot_sampled_curves_sigmoid(ax=axi, idata=data, alpha=0.2)
        axi.set_title(title)
        axi.axhline(0, color='k', linewidth=0.5)
    return((fig, ax))


def prior_posterior_density_plot(ax, x_max, t_0, t_1, idata, prior_shape=gamma_shape):
    alpha = 0.5
    xx = np.linspace(0, x_max, num=200)
    yy_prior = scipy.stats.gamma.pdf(xx, prior_shape, scale=1/prior_shape)
    ax.plot(xx, yy_prior, label='prior')
    ax.fill_between(xx, yy_prior, alpha=alpha)
    az.plot_density(idata, group='posterior', var_names=['FC_y'], ax=ax, colors='C1', shade=0.2, point_estimate=None)
    where = np.repeat(True, len(xx))
    if t_0 is not None:
        where = where & (xx > t_0)
        ax.axvline(t_0, linewidth=0.5, color='k')
    if t_1 is not None:
        where = where & (xx < t_1)
        ax.axvline(t_1, linewidth=0.5, color='k')
    ax.fill_between(xx, ax.get_ylim()[1], where=where, color='w')
    ax.axvline(1, linewidth=2, color='k', linestyle='solid')
    t = np.repeat(1e3, len(xx))
    ax.set_title('')
    ax.set_xlim(0, x_max)
    ax.set_xlabel('$\mathrm{FC}_y = y_1 / y_0$: fold change of activity')
    ax.margins(y=0)
    return(ax)

def prior_posterior_density_plot_complex(idata, prior_shape=gamma_shape, H1_prior_prob=default_H1_prior_prob):
    alpha = 0.5
    fig, ax = plt.subplots(1, 2)
    t_0 = scipy.stats.gamma.ppf(H1_prior_prob, prior_shape, scale=1/prior_shape)
    t_1 = None
    ax[0] = prior_posterior_density_plot(ax[0], x_max=2, t_0=t_0, t_1=t_1,
                                         idata=idata, prior_shape=prior_shape)
    d = {'prior': scipy.stats.gamma.cdf(t_0, prior_shape, scale=1/prior_shape),
         'posterior': get_H1_posterior_prob(posterior_idata=idata.posterior['FC_y'],
                                               H1_prior_prob=H1_prior_prob, H1_increase=False)}
    pd.Series(d).plot(kind='bar', ax=ax[1], color=['C0', 'C1'], alpha=alpha)
    pd.Series(d).plot(kind='bar', ax=ax[1], edgecolor=['C0', 'C1'], fill=False, linewidth=1)
    ax[1].set_xticklabels(['$P(H_1)$', '$P(H_1|\mathrm{data})$'], rotation=0)
    ax[0].set_xticklabels(ax[0].get_xticklabels(), fontsize=10)
    ax[0].set_title('Parameter\'s prob. density')
    ax[1].set_title('Hypothesis testing')
    handles = [mpatches.Patch(edgecolor=c, color=c, alpha=alpha) for c in ['C0', 'C1']]
    fig.legend(handles=handles, labels=d.keys(), loc='upper left', bbox_to_anchor=(0.7, 1.05))
    #fig.suptitle('Model:')
    return((fig, ax))

def nice_assay_names(data, index_cols=['experiment', 'assay'], nice_cols=['experiment (nice)', 'assay (nice)']):
    fpath = '/Users/jonesa7/CTNS/resources/cell-based-assays/ideal-effects.csv'
    mapper = pd.read_csv(fpath, index_col=index_cols)
    df = data.copy()
    k_experiment, k_assay = [mapper.index.get_level_values(i) for i in [0, 1]]
    v_experiment, v_assay = [mapper[c] for c in ['experiment (nice)', 'assay (nice)']]
    d_experiment = dict(zip(k_experiment, v_experiment))
    d_assay = dict(zip(k_assay, v_assay))
    df = df.rename(d_experiment, axis=0, level=0).rename(d_assay, axis=0, level=1)
    df = df.sort_index(axis=0, level=[0,1])
    return(df)


def idata_to_netcdf_helper(data, dirname):
    l = [dirname + 'idata-' + str(i) + '.nc' for i in np.arange(len(data))]
    fpathdf = pd.DataFrame({'fpath': l}, index=data.index)
    fpathdf.to_csv(dirname + 'fpaths.csv')
    idata_saveloc = pd.concat([data, fpathdf], axis=1)
    idata_saveloc.apply(lambda r: r.loc['idata'].to_netcdf(r.loc['fpath']), axis=1)
    return(fpathdf)


def idatadf_to_netcdf(idatadf, subdir='idatadf/', maindir='../../results/2023-09-26-cell-bayes-assays/'):
    dirname = maindir + subdir
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    data = idatadf.stack().to_frame('idata')
    fpathdf = idata_to_netcdf_helper(data, dirname)
    return(fpathdf)


def idatas_to_netcdf(idatas, subdir='idatas/', maindir='../../results/2024-02-14-cell-bayes/'):
    dirname = maindir + subdir
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    fpathdf = idata_to_netcdf_helper(idatas, dirname)
    return(fpathdf)


def idatadf_from_netcdf(subdir='idatadf/', maindir='../../results/2023-09-26-cell-bayes-assays/'):
    fpathdf = pd.read_csv(maindir + subdir + 'fpaths.csv', index_col=[0,1,2])
    val = fpathdf.apply(lambda row: az.from_netcdf(row.loc['fpath']), axis=1)
    val = val.unstack(level=2).reindex(fpathdf.xs(fpathdf.index.get_level_values(2)[0], axis=0, level=2).index)
    val = nice_assay_names(val, index_cols=['experiment', 'assay'], nice_cols=['experiment (nice)'])
    return(val)

def idatas_from_netcdf(subdir='idatas/', maindir='../../results/2024-02-14-cell-bayes/'):
    fpathdf = pd.read_csv(maindir + subdir + 'fpaths.csv', index_col=[0,1,2,3])
    val = fpathdf.apply(lambda row: az.from_netcdf(row.loc['fpath']), axis=1)
    #val = nice_assay_names(val, index_cols=['experiment', 'assay'], nice_cols=['experiment (nice)'])
    return(val)

def read_ideal_H1_increase(fpath='../../resources/cell-based-assays/ideal-effects.csv'):
    ideal_H1_increase = pd.read_csv(fpath,
                                    index_col=['experiment (nice)', 'assay (nice)'],
                                    usecols=['experiment', 'assay', 'experiment (nice)', 'assay (nice)', 'H1_increase', 'ideal effect'])
    ideal_H1_increase['H2_increase'] = ~ ideal_H1_increase.H1_increase
    return(ideal_H1_increase)


def remove_poorly_fitted(df, poor_fits, replace_val=None):
    data = df.copy()
    for x in poor_fits:
        if isinstance(df, pd.DataFrame):
            ix, column = x
            data.loc[ix, column] = replace_val
        if isinstance(df, pd.Series):
            data.loc[x] = replace_val
    return(data)


def get_H1_posterior_from_idatadf(idatadf, poor_fits, hypothesis='H1'):
    ideal_H1_increase = read_ideal_H1_increase()
    data = remove_poorly_fitted(idatadf, poor_fits)
    ll = [[get_H1_posterior_prob(data.loc[ix, drug].posterior['FC_y'], H1_increase=ideal_H1_increase.loc[ix, hypothesis + '_increase'])
           if data.loc[ix, drug] not in [None, np.nan] else None for drug in data.columns] for ix in data.index]
    H1_posteriors = pd.DataFrame(ll, index=data.index, columns=pd.MultiIndex.from_product([data.columns, [hypothesis]]))
    return(H1_posteriors)


def get_H1_posterior_from_idatas(idatas, poor_fits, hypothesis='H1'):
    data = remove_poorly_fitted(idatas, poor_fits)
    ideal_H1_increase = read_ideal_H1_increase()
    H1 = ideal_H1_increase.loc[zip(data.index.get_level_values(1),
                                   data.index.get_level_values(2)),
                               ['H1_increase', 'H2_increase']]
    H1 = pd.DataFrame(H1.to_numpy(), index=idatas.index, columns=H1.columns)
    df = pd.concat([data.to_frame('idata'), H1], axis=1)
    H1_posteriors = df.apply(lambda x:
                             get_H1_posterior_prob(x.loc['idata'].posterior['FC_y'],
                                                   H1_increase=x.loc[hypothesis + '_increase'])
                             if x.loc['idata'] is not None else None, axis=1)
    return(H1_posteriors)


def get_H102_posterior_from_idatadf(idatadf, poor_fits):
    H1_posteriors, H2_posteriors = [get_H1_posterior_from_idatadf(idatadf, poor_fits, hypothesis=h) for h in ['H1', 'H2']]
    a = 1 - (H1_posteriors.to_numpy() + H2_posteriors.to_numpy())
    H0_posteriors = pd.DataFrame(a, index=idatadf.index, columns=pd.MultiIndex.from_product([idatadf.columns, ['H0']]))
    H102_posteriors = pd.concat([H1_posteriors, H0_posteriors, H2_posteriors], axis=1).sort_index(axis=1, level=0)
    hypotheses = ['H1', 'H0', 'H2']
    columns = pd.MultiIndex.from_product([idatadf.columns, pd.CategoricalIndex(hypotheses, categories=hypotheses, ordered=True)])
    H102_posteriors = H102_posteriors.reindex(columns=columns)
    return(H102_posteriors)


def get_H102_posterior_from_idatas(idatas, poor_fits):
    H1_posteriors, H2_posteriors = [get_H1_posterior_from_idatas(idatas, poor_fits, hypothesis=h) for h in ['H1', 'H2']]
    H0_posteriors = 1 - (H1_posteriors + H2_posteriors)
    H102_posteriors = pd.DataFrame({'H1': H1_posteriors, 'H0': H0_posteriors, 'H2': H2_posteriors})
    return(H102_posteriors)


def get_diagnostics(idatadf, fun=az.ess, var_names=['EC_50', 'y_0', 'FC_y', 'k', 'y_1'], return_df=False, nice_assay_names=False):
    def helper(x, var):
        if x in [np.nan, None]:
            return(x)
        else:
            return(fun(x, var_names=var).to_dict()['data_vars'][var]['data'])
    #idat = idatadf.xs('idata', axis=1, level=1)
    def my_applymap(var, idatadf):
        if isinstance(idatadf, pd.DataFrame):
            df = idatadf.applymap(lambda x: helper(x, var))
            df.columns = pd.MultiIndex.from_product([pd.CategoricalIndex(df.columns, categories=df.columns, ordered=True), [var]])
        if isinstance(idatadf, pd.Series):
            df = idatadf.apply(lambda x: helper(x, var))
        return(df)
    df = pd.concat([my_applymap(var, idatadf) for var in var_names], axis=1)
    df.columns = var_names
    if isinstance(idatadf, pd.DataFrame):
        df = df.sort_index(axis=1, level=0)
    if nice_assay_names:
        df = nice_assay_names(df)
    if return_df:
        return(df)
    precision = np.int64(3 - np.round(np.log10(df.mean().mean())))
    val = df.style.format(precision=precision).background_gradient(axis=None, vmin=df.min().min(), vmax=df.max().max(), cmap='hot')
    return(val)


def get_diagnostics_series(idatas, fun=az.ess, vmax=None, return_df=False,
                           TI_cols=True):
    df = get_diagnostics(idatas, fun=fun, return_df=True)
    df = df.stack().to_frame('value')
    index_labels = list(df.index.to_frame().columns)
    df = df.rename_axis(index_labels[:-1] + ['parameter'])
    #df = pd.concat([df.index.to_frame(), df], axis=1).pivot(index=['study', 'experiment', 'assay', 'parameter'], columns='TI', values=df.columns)
    df = pd.concat([df.index.to_frame(), df], axis=1).pivot(index=['experiment', 'assay', 'parameter'], columns=['study', 'TI'], values=df.columns)
    df = df.droplevel(0, axis=1)
    if TI_cols:
        df = df.sort_index(axis=1, level=0)
        df = df.sort_index(axis=1, level=1, key=lambda x: np.int64(x.str.replace('TI', '')))
    if return_df:
        return(df)
    precision = np.int64(3 - np.round(np.log10(df.mean().mean())))
    vmax = df.max().max() if vmax is None else vmax
    val = df.style.format(precision=precision).background_gradient(axis=None, vmin=df.min().min(), vmax=vmax, cmap='hot')
    return(val)


def diagnostics_series_heatmap(idatas, fun=az.ess, vmax=None, yticklabels=True, TI_cols=True):
    fundict = {az.ess: r'bad $\leftarrow$ Effective sample size $\rightarrow$ good',
               az.rhat: r'good $\leftarrow$ $\hat{R}$ (convergence) $\rightarrow$ bad',
               az.mcse: r'good $\leftarrow$ Markov chain standard error $\rightarrow$ bad'}
    df = get_diagnostics_series(idatas, fun=fun, return_df=True,
                                TI_cols=TI_cols)#.droplevel('study', axis=1)
    fig, ax = plt.subplots(figsize=tuple(x / 4 for x in df.shape)[::-1]) 
    ax = sns.heatmap(df, square=True, cbar_kws={'label': fundict[fun]},
                     yticklabels=yticklabels, vmax=vmax, ax=ax)
    ylabel = ax.get_ylabel() if yticklabels else ''
    ax.set_ylabel(ylabel)
    ax.set_xlabel('')
    # separate units
    # x axis
    stop = df.shape[1]
    ax.set_xticks(np.arange(0, stop=stop, step=5), minor=True)
    # y axis
    stop = df.shape[0]
    step = df.index.to_frame().parameter.unique().size
    ax.set_yticks(np.arange(0, stop=stop, step=step), minor=True)
    ax.grid(axis='both', which='minor', color='green')
    return(ax)



def my_legend(g, colors, labels, title='Hypotheses', loc='center left', bbox_to_anchor=(0.5, -0.05, 0.5, 0.5), ncols=3, reverse_labels=False):
    handles = [mpatches.Patch(color=c) for c in colors]
    interpretations = ['protective', 'neutral', 'adverse']
    if reverse_labels:
        handles = reversed(handles)
        labels = reversed(labels)
    g.legend(handles, labels, title=title, loc=loc, bbox_to_anchor=bbox_to_anchor, ncols=ncols)
    return(g)


def barchart_H102_posteriors_ax(axi, compound, H102_posteriors, df_prior, df_mean_posterior, exper2letter_d):
    df = H102_posteriors.xs(compound, axis=1, level=0).copy()
    df_mean_posterior_compound = df_mean_posterior.xs(compound, axis=1, level=0) if df_mean_posterior is not None else None
    df = pd.concat([df_prior, df, df_mean_posterior_compound], axis=0)
    df_cum = df.cumsum(axis=1)
    k = H102_posteriors.index.get_level_values(0).unique()
    v = string.ascii_lowercase[:len(k)]
    exper2letter_d = dict(zip(k, v))
    if H102_posteriors.index.nlevels > 1:
        y = [a + ' (' + exper2letter_d[e] + ')' for e, a in H102_posteriors.index]
    else:
        y = [e + ' (' + exper2letter_d[e] + ')' for e in H102_posteriors.index]
    y = ['prior'] + y + (['avg. posterior'] if df_mean_posterior is not None else [])
    box_alpha = 1
    sns.barplot(ax=axi, data=df, y=y, x='H1', left=0, color='green', alpha=box_alpha)
    sns.barplot(ax=axi, data=df, y=y, x='H0', left=df_cum['H1'], color='lightgray', alpha=box_alpha)
    sns.barplot(ax=axi, data=df, y=y, x='H2', left=df_cum['H0'], color='red', alpha=box_alpha)
    axi.set_title(compound)
    axi.set_xlabel('')
    axi.set_xticks([0, 0.5, 1])
    axi.set_xticklabels(['0', '0.5', '1'])
    for x, color in zip((default_H1_prior_prob, 1 - default_H1_prior_prob), ('green', 'red')):
        axi.axvline(x, linestyle='solid', color=color, linewidth=0.5)
    return(axi)


def barchart_H102_posteriors(H102_posteriors, e2l_textbox=True, legend=True, plot_avg=False):
    compounds = H102_posteriors.xs('H0', axis=1, level=1).columns
    fig, ax = plt.subplots(1, len(compounds), figsize=(4.8, 2.5 * np.sqrt(H102_posteriors.shape[0]/4)), sharey=True)
    d = {'H1': default_H1_prior_prob, 'H0': 1 - 2 * default_H1_prior_prob, 'H2': default_H1_prior_prob}
    df_prior = pd.DataFrame(d, index=pd.MultiIndex.from_product([[''], ['prior']]))
    df_mean_posterior = H102_posteriors.mean(axis=0).to_frame(pd.MultiIndex.from_product([[''], ['avg. posterior']])).transpose() if plot_avg else None
    # labeling experiments with letters
    k = H102_posteriors.index.get_level_values(0).unique()
    v = string.ascii_lowercase[:len(k)]
    exper2letter_d = dict(zip(k, v))
    for axi, compound in zip(ax, compounds):
        axi = barchart_H102_posteriors_ax(axi, compound, H102_posteriors, df_prior, df_mean_posterior, exper2letter_d)
    if e2l_textbox:
        #fig = exper2letter_textbox(fig, exper2letter_d, x=1, y=0.5, horizontalalignment='left', verticalalignment='center') # Bug: this miscplaces the text box
        s = '\n'.join(['(' + v + ') ' + k for k, v in exper2letter_d.items()])
        s = 'Experiments:\n' + s
        fig.text(1, 0.5, s, ha='left', va='center')
    fig.supxlabel('$P(H_i|\mathrm{data})$', y=0.03 if H102_posteriors.shape[0] > 15 else -0.01)
    b = H102_posteriors.index.nlevels > 1
    fig.supylabel('assays' if b else 'experiments', x=-0.2 if b else -0.6)
    if legend:
        bbox_to_anchor = (1, 0.9) if e2l_textbox else (0.5, -0.01)
        title = 'Hypotheses $H_i:$'
        loc = 'upper left' if e2l_textbox else 'upper center'
        ncols = 1 if e2l_textbox else 3
        colors = ['green', 'lightgray', 'red']
        labels = ['$H_' + str(i) + '$: ' + interpret for i, interpret in zip([1, 0, 2], ['protective', 'neutral', 'adverse'])]
        my_legend(fig, colors=colors, labels=labels, loc=loc, bbox_to_anchor=bbox_to_anchor, title=title, ncols=ncols)
    return((fig, ax))


def get_FC_y_posterior_sample(exper, assay, compound, idatadf, ideal_H1_increase):
    H1_increase = ideal_H1_increase.loc[(exper, assay), 'H1_increase']
    idata = idatadf.loc[(exper, assay), compound]
    if idata in [np.nan, None]:
        idata = idatadf.dropna().iloc[0,0]
        npoints = len(idata.sample_stats['chain']) * len(idata.sample_stats['draw'])
        l = np.repeat(np.nan, npoints)
    else:
        posterior = idata.posterior
        l = list(itertools.chain(*posterior['FC_y'].to_numpy()))
    df = pd.DataFrame({'FC_y': l, 'exper': exper, 'assay': assay, 'compound': compound, 'H1_increase': H1_increase})
    return(df)


def get_FC_y_posterior_sample_all(idatadf):
    ideal_H1_increase = read_ideal_H1_increase(fpath='../../resources/cell-based-assays/ideal-effects.csv')
    ll = [[get_FC_y_posterior_sample(exper, assay, compound, idatadf, ideal_H1_increase)
           for exper, assay in idatadf.index] for compound in idatadf.columns]
    l = list(itertools.chain(*ll))
    df = pd.concat(l, axis=0)
    return(df)


def violin_compound(ax, compound, idatadf, ideal_H1_increase, exper2letter_d, plot_avg=False):
    FC_y_max=3
    npoints=400
    delta_y = 5
    t_1 = scipy.stats.gamma.ppf(default_H1_prior_prob, gamma_shape, scale=1/gamma_shape)
    t_2 = scipy.stats.gamma.ppf(1 - default_H1_prior_prob, gamma_shape, scale=1/gamma_shape)
    prior_stdev = gamma_shape ** (-1/2)
    xx = np.linspace(0, FC_y_max, npoints)
    yy_prior = scipy.stats.gamma.pdf(xx, gamma_shape, scale=1/gamma_shape)
    b_left = xx <= t_1
    b_right = xx > t_2
    b_center = (~ b_left) & (~ b_right)
    y_bases = np.arange(idatadf.shape[0], step=1) * delta_y
    for t, lw in zip([t_1, t_2, 1], [0.5, 0.5, 2]):
        ax.axvline(t, color='k', linewidth=lw, linestyle='solid')
    df = idatadf.index.to_frame().rename(columns={0: 'experiment', 1:'assay'})
    ticklabels = df.apply(lambda r: r.loc['assay'] + ' (' + exper2letter_d[r.loc['experiment']] + ')', axis=1).to_list()
    ticklabels = ['prior, desired increase', 'prior, desired decrease'] + ticklabels 
    ticklabels += ['avg, desired increase', 'avg, desired decrease'] if plot_avg else []
    yticks_prior = [y_bases.max() + delta_y * 2, y_bases.max() + delta_y]
    yticks_avg = [- delta_y, -2 *  delta_y]
    yticks = yticks_prior + list(y_bases[::-1]) + (yticks_avg if plot_avg else [])
    ax.set_yticks(yticks)
    ax.set_yticklabels(ticklabels)
    ax.set_ylim(min(yticks) - delta_y, max(yticks) + delta_y)
    scatter_c = 'blue'
    scatter_edgecolors = 'yellow'
    linewidths_c = 1
    
    def one_violin_helper(ax, l, H1_increase):
        k = scipy.stats.gaussian_kde(l)
        yy = k.evaluate(xx)
        c_left = 'red' if H1_increase else 'green'
        c_right = 'red' if not H1_increase else 'green'
        c_center = 'gray'
        for b, c in zip([b_left, b_center, b_right], [c_left, c_center, c_right]):
            ax.fill_between(xx[b], y_base + yy[b], y_base - yy[b], alpha=.5, linewidth=0.5, color=c)
        return(ax)

    def one_errorbar_helper(ax, l, y_base):
        x = np.mean(l)
        ax.errorbar(x=x, y=y_base, xerr=np.std(l), capsize=3, capthick=1, ecolor=scatter_edgecolors)
        ax.scatter(x=x, y=y_base, c=scatter_c, marker='o', linewidths=linewidths_c, edgecolors=scatter_edgecolors)
        return(ax)
    
    def one_prior_errorbar_helper(ax, y_base):
        ax.errorbar(x=1, y=y_base, xerr=prior_stdev, capsize=3, capthick=1, ecolor=scatter_edgecolors)
        ax.scatter(x=1, y=y_base, c=scatter_c, marker='o', linewidths=linewidths_c, edgecolors=scatter_edgecolors)
        return(ax)

    def one_violin(ax, exper, assay, y_base):
        H1_increase = ideal_H1_increase.loc[(exper, assay), 'H1_increase']
        if idatadf.loc[(exper, assay), compound] in [None, np.nan]:
            return(None)
        posterior = idatadf.loc[(exper, assay), compound].posterior
        l = list(itertools.chain(*posterior['FC_y'].to_numpy()))
        ax = one_violin_helper(ax, l, H1_increase)
        ax = one_errorbar_helper(ax, l, y_base)

    def one_avg_violin(ax, y_base, H1_increase):
        samples = get_FC_y_posterior_sample_all(idatadf=idatadf).dropna(axis=0)
        samples = samples.loc[samples.compound == compound]
        l = samples.loc[samples.H1_increase == H1_increase, 'FC_y'].to_list()
        ax = one_violin_helper(ax, l, H1_increase)
        ax = one_errorbar_helper(ax, l, y_base)
        
    def one_prior_violin(ax, y_base, H1_increase):
        c_left = 'red' if H1_increase else 'green'
        c_right = 'red' if not H1_increase else 'green'
        c_center = 'gray'
        for b, c in zip([b_left, b_center, b_right], [c_left, c_center, c_right]):
            ax.fill_between(xx[b], y_base + yy_prior[b], y_base - yy_prior[b], alpha=.5, linewidth=0.5, color=c)
        ax = one_prior_errorbar_helper(ax, y_base)

    for ix, y_base in zip(idatadf.index, y_bases[::-1]):
        one_violin(ax, *ix, y_base)

    if plot_avg:
        for y_base, H1_increase in zip(yticks_avg, [True, False]):
            one_avg_violin(ax, y_base, H1_increase)
        
    for y_base, H1_increase in zip(yticks_prior, [True, False]):
        one_prior_violin(ax, y_base, H1_increase)
        
    ax.set_xlim(0, FC_y_max)
    return(ax)

def violin_posterior_pdf(idatadf, poor_fits, text_box=True, H_legend=True, plot_avg=False):
    idatadf = remove_poorly_fitted(idatadf, poor_fits)
    ideal_H1_increase = read_ideal_H1_increase(fpath='../../resources/cell-based-assays/ideal-effects.csv')
    df = idatadf.index.to_frame().rename(columns={0: 'experiment', 1:'assay'})
    k = df.experiment.unique()
    v = string.ascii_lowercase[:len(k)]
    exper2letter_d = dict(zip(k, v))
    fig, ax = plt.subplots(1, idatadf.shape[1], figsize=(8, idatadf.shape[0] / 4), sharey=True)
    for axi, compound in zip(ax, idatadf.columns):
        axi.set_title(compound)
        violin_compound(axi, compound, idatadf, ideal_H1_increase, exper2letter_d=exper2letter_d, plot_avg=plot_avg)
    fig.supxlabel('$\mathrm{FC}_y$: fold change', y=0.025)
    fig.supylabel('assays', x=-0.1)
    if H_legend:
        bbox_to_anchor = (1, 0.8)
        title = 'Hypotheses $H_i$'
        loc = 'center left'
        colors = ['green', 'lightgray', 'red']
        labels = ['$H_' + str(i) + '$: ' + interpret for i, interpret in zip([1, 0, 2], ['protective', 'neutral', 'adverse'])]
        my_legend(fig, colors=colors, labels=labels, loc=loc, bbox_to_anchor=bbox_to_anchor, title=title, ncols=1)
    if text_box:
        #fig = exper2letter_textbox(fig, exper2letter_d, x=1.0, y=0.25, horizontalalignment='left', verticalalignment='center') # Bug: this miscplaces the text box
        s = '\n'.join(['(' + v + ') ' + k for k, v in exper2letter_d.items()])
        s = 'Experiments:\n' + s
        fig.text(1, 0.5, s, ha='left', va='center')
    return((fig, ax))


def get_TI_conc(fpath='/Users/jonesa7/CTNS/resources/cell-based-assays/test-items.csv'):
    df = pd.read_csv(fpath, index_col=['Study', 'TI']).drop(['TI ID', 'Name'], axis=1)
    df = df * 1e-6 # from microM to M
    df = df.stack()
    df = df.rename_axis(['Study', 'TI', 'conc'])
    return(df)


def get_TI_name(fpath='/Users/jonesa7/CTNS/resources/cell-based-assays/test-items.csv'):
    df = pd.read_csv(fpath, usecols=['Study', 'TI', 'Name'], index_col=['Study', 'TI'])
    return(df)


def get_control_conc(data, controls_fpath='/Users/jonesa7/CTNS/resources/cell-based-assays/experiment-controls2.csv'):
    controls = pd.read_csv(controls_fpath, index_col='Experiment')
    def helper(r):
        control_conc = controls.loc[r.loc['Experiment'], 'concentration']
        drug_conc = r.loc['concentration']
        is_control = re.match(controls.loc[r.loc['Experiment'], 'Control'], r.loc['TI'])
        val = control_conc if is_control else drug_conc
        return(val)
    concentration = data.apply(helper, axis=1)
    return(concentration)


def get_data(data_fpath, sheet_name='Data', TI_fpath='/Users/jonesa7/CTNS/resources/cell-based-assays/test-items.csv',
             controls_fpath='/Users/jonesa7/CTNS/resources/cell-based-assays/experiment-controls2.csv'):
    data = pd.read_excel(data_fpath, sheet_name=sheet_name)
    TI2name = get_TI_name(TI_fpath)
    data_name = data.apply(lambda r: TI2name.loc[*r.loc[['Study', 'TI']]]
                           if re.match('^TI.*', r.loc['TI']) else
                           pd.Series('', index=['Name']), axis=1)
    TI2conc = get_TI_conc(TI_fpath)
    data_conc = data.apply(lambda r: TI2conc.loc[*r.loc[['Study', 'TI', 'conc']]]
                           if re.match('^TI.*', r.loc['TI']) else np.nan,
                           axis=1).to_frame('concentration')
    data = pd.concat([data.loc[:, :'TI'], data_name, data[['conc']],
                      data_conc, data.loc[:, 'Activity':]], axis=1)
    data['concentration'] = get_control_conc(data, controls_fpath)
    data = pd.concat([data.loc[:, :'concentration'],
               data.concentration.apply(np.log10).to_frame('conc_log10'),
               data.loc[:, ['Activity']]], axis=1)
    return(data)


def extract_regr_data(study, exper, assay, TI, data, batchvars=['Batch', 'Plate'],
                      return_data_reshaped=False, batch_corr=True,
                      controls_fpath='/Users/jonesa7/CTNS/resources/cell-based-assays/experiment-controls2.csv',
                      accept_multi_batches=False):
    b1 = (data.Study == study) & (data.Experiment == exper) & (data.Assay == assay)
    TI_data = data.loc[b1 & (data.TI == TI)]
    # ensure that all treatment TI data are from the same batch:plate
    if len(TI_data.groupby(batchvars)) != 1:
        print(study, exper, assay, TI)
        print('treatment with multiple batches')
        if not accept_multi_batches:
            return(None)
    TI_data_row0 = TI_data.iloc[0]
    b2 = data.Plate == TI_data_row0.loc['Plate'] if not TI_data_row0.isna().loc['Plate'] else True
    #b2 = data.Plate == TI_data.iloc[0].loc['Plate']
    # if there's information on Batch, update bool vector b with it
    if not TI_data.iloc[0].isna().loc['Batch']:
        b2 = b2 & (data.Batch == TI_data.iloc[0].loc['Batch'])
    data_reshaped = data.loc[b1 & b2]
    controls = pd.read_csv(controls_fpath, index_col='Experiment')
    control_TI = controls.loc[exper, 'Control']
    data_reshaped_control = data_reshaped.loc[data_reshaped.TI == control_TI].copy()
    #if not len(data_reshaped_control):
    #    data_reshaped_control = data_reshaped.loc[data_reshaped.conc_log10 == -9].copy()
    # if there's no control for the same batch:plate, use controls from all other batch:plate combinations
    if len(data_reshaped_control) == 0:
        data_reshaped_control = data.loc[b1 & (data.TI == control_TI)].copy()
    if (len(data_reshaped_control) == 0) or not batch_corr:
        data_reshaped_control = data.loc[b1 & (data.conc_log10 == -9)].copy()
    data_reshaped_control['conc'] = control_TI
    data_reshaped_TI = data_reshaped.loc[data_reshaped.TI == TI].copy()
    data_reshaped = pd.concat([data_reshaped_control, data_reshaped_TI], axis=0)
    data_reshaped['std_activity'] = data_reshaped['Activity'] / data_reshaped['Activity'].std() * 10
    if return_data_reshaped:
        return(data_reshaped)
    y_obs = data_reshaped['std_activity'].values
    x_obs = data_reshaped['conc_log10'].values
    return((y_obs, x_obs))


def fit_single_unit(study, exper, assay, TI, data, do_fit=True, do_print=True,
                    accept_multi_batches=False):
    unitv = [study, exper, assay, TI]
    if do_print:
        print(*unitv)
    y_obs, x_obs = extract_regr_data(study, exper, assay, TI, data,
                                     return_data_reshaped=False,
                                     accept_multi_batches=accept_multi_batches)
    if not do_fit:
        d = dict(zip(['study', 'exper', 'assay', 'TI'], unitv))
        arrays = [[x] for x in unitv]
        df = pd.DataFrame(d, index=pd.MultiIndex.from_arrays(arrays))
        return(df)
    try:
        model, idata = [sample_sigmoid_2(y_obs, x_obs, return_model=b) for b in [True, False]]
    except pm.SamplingError:
        model, idata = (None, None)
    index = pd.MultiIndex.from_product([[study], [exper], [assay], [TI]])
    idatadf = pd.DataFrame({'model': [model], 'idata': [idata]}, index=index)
    idatadf = idatadf.rename_axis(['study', 'experiment', 'assay', 'TI'], axis=0)
    return(idatadf)


def fit_multiple_units(data, unit_list=None, do_fit=True, do_print=True,
                       accept_multi_batches=False):
    unitv = ['Study', 'Experiment', 'Assay', 'TI']
    if unit_list is None:
        dat = data.loc[data.TI.apply(lambda x: bool(re.match('TI[0-9]+', x)))]
        dat = dat.groupby(unitv).first()
        unit_list = dat.index.to_numpy()
    l = [fit_single_unit(*args, data, do_fit=do_fit, do_print=do_print, accept_multi_batches=accept_multi_batches) for args in unit_list]
    idatadf = pd.concat(l, axis=0)
    return(idatadf)


def plot_single_unit(ax, study, exper, assay, TI, data, idatas,
                     plot_sampled_curves=True, draw_y0_y1=True,
                     H1_increase=False, compound_name_title=True,
                     linewidth=0.2):
    data_reshaped = extract_regr_data(study, exper, assay, TI, data,
                                      return_data_reshaped=True,
                                      accept_multi_batches=True)
    ax = plot_data(ax, data_reshaped)
    posterior = idatas.loc[(study, exper, assay, TI)].posterior
    ax = plot_sampled_curves_sigmoid(ax, posterior, data_reshaped,
                                     plot_sampled_curves=plot_sampled_curves,
                                     draw_y0_y1=draw_y0_y1,
                                     H1_increase=H1_increase, H_text=False,
                                     H_yticks=False, linewidth=linewidth)
    #ax.set_ylim(0, data_reshaped.std_activity.quantile(0.8) * 5)
    l = list(data_reshaped.Name.unique())
    try:
        l.remove('')
    except ValueError:
        pass
    compound = l[0]
    #title = compound[:20] + '\n' + compound[20:] if compound_name_title else TI + ', ' + study
    title = compound[:30] if compound_name_title else TI + ', ' + study
    pad = -25 if compound_name_title else 0
    ax.set_title(title, fontsize=10, backgroundcolor='white',
                 bbox={'pad': 0, 'color': 'white'})
    return(ax)


def plot_multiple_units(unit_list, data, idatas, plot_sampled_curves=True,
                        draw_y0_y1=True, compound_name_title=True, nrows=None,
                        ncols=None, figsize=None, linewidth=0.1,
                        fpath_ideal_H1_increase='/Users/jonesa7/CTNS/resources/cell-based-assays/ideal-effects.csv'):
    n_units = len(unit_list)
    # read ideal H1 increase info
    if draw_y0_y1:
        ideal_H1_increase = pd.read_csv(fpath_ideal_H1_increase, index_col=['experiment (nice)', 'assay (nice)'],
                                        usecols=['experiment', 'assay', 'experiment (nice)', 'assay (nice)',
                                                 'H1_increase', 'ideal effect'])
    nrows = np.int64(np.ceil(np.sqrt(n_units))) if nrows is None else nrows
    ncols = nrows if ncols is None else ncols
    figscaler = 1.5
    figsize = (6.4 * figscaler, 4.8 * figscaler) if figsize is None else figsize
    fig, ax = plt.subplots(nrows, ncols, sharex=True, figsize=figsize)
    for axi, unit in zip(ax.ravel()[:n_units], unit_list):
        H1_increase = False
        if draw_y0_y1:
            experiment = unit[1]
            assay = unit[2]
            H1_increase = ideal_H1_increase.loc[(experiment, assay),
                                                'H1_increase']
        try:
            axi = plot_single_unit(axi, *unit, data, idatas,
                                   plot_sampled_curves,
                                   draw_y0_y1=draw_y0_y1,
                                   H1_increase=H1_increase,
                                   compound_name_title=compound_name_title, linewidth=linewidth)
        except IndexError:
            pass
        axi.set_xlabel('')
        axi.set_ylabel('')
    for axi in ax.ravel()[n_units:]:
        axi.remove()
    fig.supxlabel(r'$\log_{10}$ conc')
    #fig.supylabel(r'activity')
    return((fig, ax))


def sort_index_TI(val):
    is_series = isinstance(val, pd.Series)
    is_dataframe = isinstance(val, pd.DataFrame)
    df = val.index.to_frame()
    df['TIn'] = [np.int64(x.replace('TI', '')) for x in
                 val.index.get_level_values(3)]
    constructor = pd.Series if is_series else pd.DataFrame
    val = constructor(val, index=pd.MultiIndex.from_frame(df))
    val = val.sort_index(level=[1, 2, 0, 4]).droplevel(4, axis=0)
    return(val)

def check_H102_posteriors_for_zero(H102_posteriors):
    if any(H102_posteriors.loc[:, [x is np.dtype('float64') for x in H102_posteriors.dtypes]].min()):
        raise ValueError('Zero probability found.\nRegularize with pseudocount_to_H102_posteriors before computing Bayes factors!')
    return(None)

def BF10_from_H102_posteriors(H102_posteriors):
    #check_H102_posteriors_for_zero(H102_posteriors)
    BF10 = H102_posteriors.stack(level=0, dropna=False).apply(lambda r: r.loc['H1'] / r.loc['H0'], axis=1)
    BF10 = pd.concat([BF10.index.to_frame(name=['Experiment', 'Assay', 'Drug']), BF10.to_frame('BF')], axis=1)
    BF10['2 log BF'] = BF10.BF.apply(lambda x: 2 * np.log(x))
    return(BF10)


def BF10_from_H102_posteriors_long(H102_posteriors):
    #check_H102_posteriors_for_zero(H102_posteriors)
    BF10 = H102_posteriors.apply(lambda r: r.loc['H1'] / r.loc['H0'], axis=1)
    BF10 = pd.concat([BF10.index.to_frame(name=['Study', 'Experiment', 'Assay', 'TI']), BF10.to_frame('BF')], axis=1)
    BF10['2 log BF'] = BF10.BF.apply(lambda x: 2 * np.log(x))
    return(BF10)


def plot_BF(BF10, anonym_drugs=False):
    n_assays = BF10.xs(BF10.Drug.unique()[0], axis=0, level=-1).groupby(level=0).apply(len)
    n_exper = n_assays.size
    drugs = BF10['BF'].unstack(level=-1).columns
    drug_d = dict(zip(drugs, ['drug ' + x for x in  string.ascii_uppercase[:len(drugs)]]))
    titlepanel_j = 0
    #gs = GridSpec(n_exper, len(drugs), height_ratios=n_assays.to_list(), left=0.05, right=0.48, hspace=0.75)
    gs = GridSpec(n_exper + 1, len(drugs), height_ratios=n_assays.to_list() + [7], left=0.05, right=0.48, hspace=0.75)
    fig = plt.figure(figsize=(len(drugs) * 4.8, 10.8))
    def BF_ticks(ax):
        xticks_major = [0, 2, 6, 10]
        xticks_minor = [-2, 1, 4, 8, 12]
        ax.set_xticks(xticks_major, minor=False)
        #ax.set_xticks(xticks_minor, minor=True)
        xticklabels_minor = len(xticks_minor) * ['']
        ax.grid(axis='x')
        return(ax)
    for j, drug in enumerate(drugs):
        for i, x in enumerate(n_assays):
            exper = n_assays.index[i]
            if i == 0:
                ax = fig.add_subplot(gs[i, j])
                drug_s = drug_d[drug] if anonym_drugs else drug
                ax.text(0.5, 5 / x, drug_s, transform=ax.transAxes, fontsize=12, horizontalalignment='center')
                ax0 = ax
            else:
                ax = fig.add_subplot(gs[i, j], sharex=ax0)
            df = BF10.xs(exper, axis=0, level=0).xs(drug, axis=0, level=-1)
            df['color'] = df['2 log BF'].apply(lambda x: 'green' if x > 2 else 'gray')
            sns.barplot(data=df, y='Assay', x='2 log BF', ax=ax, palette=df['color'])
            #ax.barh(data=df, y='Assay', width='2 log BF', color=df['color'].to_list()) # issues with y axis scaling
            ax.set_ylabel('')
            if j == titlepanel_j:
                title = string.ascii_lowercase[i] + ') ' + exper
                ax.set_title(title, x=-0.5, fontsize=10, horizontalalignment='left', transform=ax.transAxes, weight='bold')
            ax.set_xlabel('')
            ax = BF_ticks(ax)
            #ax.set_xticklabels(xticklabels_minor, minor=True, rotation=90)
            if j > 0:
                ax.set_yticklabels(len(ax.get_yticks()) * [''])
    axes = np.array(fig.get_axes()).reshape(len(drugs), n_exper)
    fig.supxlabel(r'$2 \times \log$ Bayes factor: evidence for protective effect', horizontalalignment='right', x=0.4, y=0.05)
    ax = fig.add_subplot(gs[n_exper, 1], sharex=ax0)
    ax = BF_ticks(ax)
    #ax.set_xlim(-4, 14)
    ax.set_yticks([])
    evidence_l = ['nonexistent', 'negligible', 'moderate', 'strong', 'very strong']
    xticks_minor = [-2, 1, 4, 8, 12]
    for x, evidence in zip(xticks_minor, evidence_l):
        ax.text(x=x, y=0.5, s=evidence, horizontalalignment='center', verticalalignment='center', rotation=90, fontsize=10)
    ax.set_ylabel('Evidence:', rotation=0, horizontalalignment='right', verticalalignment='center')
    return(fig)


def pseudocount_to_H102_posteriors(H102_posteriors):
    n_mcmc_chains = 4
    n_mcmc_samples = 1000
    pseudocount = 0.5
    val = H102_posteriors.map(lambda x: pseudocount / (n_mcmc_chains * n_mcmc_samples) if x == 0 else x)
    return(val)
