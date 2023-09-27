import pymc as pm
import pandas as pd
import numpy as np
import scipy.stats

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
                                t=scipy.stats.gamma.ppf(0.1, gamma_shape,
                                                        scale=1/gamma_shape)):
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
    if draw_y0_y1:
        ax.add_patch(plt.Rectangle((xx[0], 0), xx[-1] - xx[0], y_0_mean * t, ls=None, lw=0, ec="c", fc='green', alpha=0.2))
        linestyle = 'dashed'
        color = 'k'
        linewidth = 1
        ax.axhline(y_0_mean, linestyle='solid', color=color, linewidth=2)
        ax.axhline(y_0_mean * t, linestyle='solid', color=color, linewidth=0.5)
        ax.axhline(y_1_mean, linestyle=linestyle, color=color, linewidth=linewidth)
        labels = ['$y_0$', '$y_0 t$: relevant effect size', '$y_1 = y_0 \mathrm{FC}_y$']
        ax.set_yticks([y_0_mean, y_0_mean * t, y_1_mean], labels=labels)
        ax.text(EC_50_mean - 2, y_0_mean / 4, '$H_1: \mathrm{FC}_y < t$', color='green')
    ax.plot(xx, y_sigmoid_1_mean, color='red', linewidth=3, label='sigmoid 1')
    return(ax)
