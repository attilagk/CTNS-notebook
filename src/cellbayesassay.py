import arviz as az
import pymc as pm
import pandas as pd
import numpy as np
import os
import os.path
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
    xx = np.linspace(data_reshaped.conc_log10.min(), data_reshaped.conc_log10.max() + 1, 200)
    ax.scatter(x='conc_log10', y='activity', data=data_reshaped, marker='+', color='k')
    ax.set_xlabel(r'$\log_{10}$ conc')
    ax.set_ylabel(r'activity')
    return(ax)


def plot_sampled_curves_sigmoid(ax, idata, data_reshaped, color='C0', alpha=0.5,
                                plot_sampled_curves=True, draw_y0_y1=False,
                                H1_prior_prob=default_H1_prior_prob,
                                H1_increase=False, ylim_top=None, H_text=True):
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
            ax.plot(xx, yy, linewidth=0.2, color=color, alpha=alpha)
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


def idatadf_to_netcdf(idatadf, subdir='idatadf/', maindir='../../results/2023-09-26-cell-bayes-assays/'):
    dirname = maindir + subdir
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    data = idatadf.stack().to_frame('idata')
    l = [dirname + 'idata-' + str(i) + '.nc' for i in np.arange(len(data))]
    fpathdf = pd.DataFrame({'fpath': l}, index=data.index)
    fpathdf.to_csv(maindir + subdir + 'fpaths.csv')
    idata_saveloc = pd.concat([data, fpathdf], axis=1)
    idata_saveloc.apply(lambda r: r.loc['idata'].to_netcdf(r.loc['fpath']), axis=1)
    return(fpathdf)


def idatadf_from_netcdf(subdir='idatadf/', maindir='../../results/2023-09-26-cell-bayes-assays/'):
    fpathdf = pd.read_csv(maindir + subdir + 'fpaths.csv', index_col=[0,1,2])
    val = fpathdf.apply(lambda row: az.from_netcdf(row.loc['fpath']), axis=1)
    val = val.unstack(level=2).reindex(fpathdf.xs(fpathdf.index.get_level_values(2)[0], axis=0, level=2).index)
    val = nice_assay_names(val, index_cols=['experiment', 'assay'], nice_cols=['experiment (nice)'])
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
        ix, column = x
        data.loc[ix, column] = replace_val
    return(data)


def get_H1_posterior_from_idatadf(idatadf, poor_fits, hypothesis='H1'):
    ideal_H1_increase = read_ideal_H1_increase()
    data = remove_poorly_fitted(idatadf, poor_fits)
    ll = [[get_H1_posterior_prob(data.loc[ix, drug].posterior['FC_y'], H1_increase=ideal_H1_increase.loc[ix, hypothesis + '_increase'])
           if data.loc[ix, drug] not in [None, np.nan] else None for drug in data.columns] for ix in data.index]
    H1_posteriors = pd.DataFrame(ll, index=data.index, columns=pd.MultiIndex.from_product([data.columns, [hypothesis]]))
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


def get_diagnostics(idatadf, fun=az.ess, var_names=['EC_50', 'y_0', 'FC_y', 'k', 'y_1'], return_df=False, nice_assay_names=False):
    def helper(x, var):
        if x in [np.nan, None]:
            return(x)
        else:
            return(fun(x, var_names=var).to_dict()['data_vars'][var]['data'])
    #idat = idatadf.xs('idata', axis=1, level=1)
    def my_applymap(var, idatadf):
        df = idatadf.applymap(lambda x: helper(x, var))
        df.columns = pd.MultiIndex.from_product([pd.CategoricalIndex(df.columns, categories=df.columns, ordered=True), [var]])
        return(df)
    df = pd.concat([my_applymap(var, idatadf) for var in var_names], axis=1)
    df = df.sort_index(axis=1, level=0)
    if nice_assay_names:
        df = cba.nice_assay_names(df)
    if return_df:
        return(df)
    precision = np.int64(3 - np.round(np.log10(df.mean().mean())))
    val = df.style.format(precision=precision).background_gradient(axis=None, vmin=df.min().min(), vmax=df.max().max(), cmap='hot')
    return(val)



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
    for x, color in zip((cba.default_H1_prior_prob, 1 - cba.default_H1_prior_prob), ('green', 'red')):
        axi.axvline(x, linestyle='solid', color=color, linewidth=0.5)
    return(axi)


def barchart_H102_posteriors(H102_posteriors, e2l_textbox=True, legend=True, plot_avg=False):
    compounds = H102_posteriors.xs('H0', axis=1, level=1).columns
    fig, ax = plt.subplots(1, len(compounds), figsize=(4.8, 2.5 * np.sqrt(H102_posteriors.shape[0]/4)), sharey=True)
    d = {'H1': cba.default_H1_prior_prob, 'H0': 1 - 2 * cba.default_H1_prior_prob, 'H2': cba.default_H1_prior_prob}
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
