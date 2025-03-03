import arviz as az
import pymc as pm
import pandas as pd
import numpy as np
import pytensor.tensor as at
import bambi as bmb
import matplotlib.pyplot as plt
import os.path
from cellbayesassay import idatas_to_netcdf
#import patsy


mcmc_random_seed = 2021
# Hyperparameters
y_0_alpha = 2
y_inf_alpha = y_0_alpha
λ_mu = 0
λ_sigma = 1
α_alpha = 1.5
α_mean = 2
α_beta = α_alpha / α_mean
censoring_upper_lim = 60


def get_Weibull_beta(mu, alpha=2):
    arg = 1.0 + 1.0 / alpha
    beta = mu / at.gamma(arg)
    return(beta)


def model_A(y_obs, x_obs, return_model=False, y_inf_A_val=0, censored=True):
    #data.loc[:, 'Day'] = data.Day - 1
    #y_obs, x_obs = patsy.dmatrices('Latency ~ 1 + Day', data=data)
    mymodel = pm.Model()
    with mymodel:
        y_0 = pm.Weibull('y_0', y_0_alpha, get_Weibull_beta(60, alpha=y_0_alpha))
        y_inf_A = pm.Deterministic('y_inf_A', at.as_tensor_variable(y_inf_A_val)) if\
            y_inf_A_val is not None else None
        y_inf_B = pm.Weibull('y_inf_B', y_inf_alpha, get_Weibull_beta(10, alpha=y_inf_alpha))
        y_inf = pm.Deterministic('y_inf', y_inf_A if y_inf_A_val is not None else y_inf_B)
        λ = pm.Normal('λ', λ_mu, λ_sigma)
        μ = pm.Deterministic('μ', y_inf + (y_0 - y_inf) * pm.math.exp(λ * (x_obs - 1)))
        #μ = pm.Deterministic('μ', y_inf + (y_0 - y_inf) * pm.math.exp(λ * (x_obs - 1)))
        α = pm.Gamma('α', α_alpha, α_beta)
        β = pm.Deterministic('β', get_Weibull_beta(μ, alpha=α))
        if censored:
            y_latent = pm.Weibull.dist(α, β)
            y = pm.Censored('y', y_latent, lower=0, upper=censoring_upper_lim, observed=y_obs)
        else:
            y = pm.Weibull('y', α, β, observed=y_obs)
        if return_model:
            return(mymodel)
        idata = pm.sample(return_inferencedata=True,
                          idata_kwargs={'log_likelihood': True},
                          random_seed=mcmc_random_seed,
                          init='jitter+adapt_diag_grad')
        return(idata)


def read_data_train(fpath, treatments, sheet_name='MWM day 1-4'):
    data_train = pd.read_excel(fpath, sheet_name=sheet_name, header=[0,1], index_col=None)
    ix = pd.MultiIndex.from_frame(data_train.iloc[:, :4].xs('Covariates', axis=1, level=0))
    data_train = pd.DataFrame(data_train.iloc[:, 4:].to_numpy(), columns=data_train.iloc[:, 4:].columns, index=ix)
    data_train = data_train.stack(level=1)
    data_train = data_train.rename_axis(data_train.index.names[:-1] + ['Day'], axis=0)
    data_train = pd.concat([data_train, data_train.index.to_frame()], axis=1)
    data_train['Day'] = data_train.Day.str.replace('Day ', '').astype('float64')
    #data_train['Day'] = data_train.Day.str.replace('Day ', '').astype(pd.Int64Dtype())
    data_train['Condition'] = data_train.Group.apply(lambda x: treatments[x])
    data_train['Status'] = np.where(data_train['Latency (s)'] < 60, 'none', 'right') # censoring status
    d = {'Latency (s)': 'Latency', 'Velocity (m/s)': 'Velocity', 'Thigmotaxis %': 'Thigmotaxis', 'Floating %': 'Floating', 'Distance (m)': 'Distance'}
    data_train = data_train.rename(d, axis=1)
    data_train = data_train.rename_axis(['cohort', 'group', 'sex', 'irn', 'day'])
    #data_train = pd.DataFrame(data_train.to_numpy(), index=list(range(data_train.shape[0])), columns=data_train.columns)
    return(data_train)


# Instead of None we have data, a DataFrame read with the read_data function
_experiments_example = {
    'Amiloride 10': (None, ['5xFAD', '5xFAD + Amiloride', 'WT'], [21947, 21949, 21976, 22021]),
    'TUDCA WT': (None, ['WT', 'WT + TUDCA', '5xFAD'], [21947, 21949, 21976, 22021]),
}


def data_train_plotter(yname, data_train, lvl=['5xFAD', '5xFAD + Amiloride', 'WT'],
                       colors={'f': 'red', 'm': 'blue'}, extra_subplot=False):
    ncol = len(lvl) + (1 if extra_subplot else 0)
    fig, ax = plt.subplots(1, ncol, sharey=True, figsize=(len(lvl) * 4.8, 4.8))
    for condition, axi in zip(lvl, ax):
        axi.set_title(condition)
        axi.set_ylabel(yname)
        axi.set_xlabel('Day')
        axi.set_xticks(range(5))
        axi.set_xticklabels(range(5))
        axi.grid(axis='y')
        df1 = data_train.loc[data_train.Condition == condition]
        for sex in df1.Sex.unique():
            color = colors[sex]
            df2 = df1.xs(sex, level=2)
            for irn in df2.IRN.unique():
                df3 = df2.xs(irn, level=2)
                for cohort in df3.Cohort.unique():
                    s = sex + cohort
                    y = df3[yname]
                    x = df3.Day
                    axi.plot(x, y, color=color, label=irn, marker='$' + cohort
                             + '$', linewidth=0.5)
                pass
            pass
        #axi.legend()
    return((fig, ax))


def escape_latency_plotter(exper, data_train, lvl, colors={'f': 'red', 'm': 'blue'}, extra_subplot=False):
    fig, ax = data_train_plotter(yname='Latency', data_train=data_train,
                                 lvl=lvl, colors=colors, extra_subplot=extra_subplot)
    fig.suptitle(exper, fontsize=16, va='bottom')
    return((fig, ax))


def fit_one(data, lvl, random_seed):
    dat = data.loc[data.Condition.isin(lvl)]
    model = bmb.Model(
        'censored(Latency, Status) ~ 1 + C(Condition, levels=lvl) + Day + (1 | IRN)',
        dat, 
        family='weibull',
        link='log',
        center_predictors=False
    )
    try:
        idata = model.fit(idata_kwargs={'log_likelihood': True}, random_seed=random_seed)
    except(pm.SamplingError):
        idata = None
    return(idata)



'''
Fit multiple data sets

The following type of dictionary, experimentsd, is used as input:

experiments3_CO28154 = {
    'TUDCA + HCQ': (data_CO28154, ['Saline', 'TUDCA + HCQ', 'Saline WT'], [21947, 21949, 21976, 22021]),
    'Arundine low dose': (data_CO28154, ['Vehicle (Arundine)', 'Arundine low dose', 'Vehicle (Arundine) WT'], [21947, 21949, 21976, 22021]),
}
'''
def fit_multiple(experimentsd):
    idatad = {k: fit_one(*v) for k, v in experimentsd.items()}
    idatas = pd.Series(idatad)
    return(idatas)


def idatas_from_netcdf(subdir='idatas/', maindir='../../results/2024-02-14-cell-bayes/'):
    fpath = os.path.join(maindir, subdir, 'fpaths.csv')
    fpathdf = pd.read_csv(fpath, index_col=0)
    val = fpathdf.apply(lambda row: az.from_netcdf(row.loc['fpath']), axis=1)
    #val = nice_assay_names(val, index_cols=['experiment', 'assay'], nice_cols=['experiment (nice)'])
    return(val)


def get_diagnostics(idatas, fun=az.ess):
    titled = {
        az.ess: 'effective sample size',
        az.mcse: 'Markov chain std error',
        az.rhat: r'$\hat{r}$',
    }
    title = titled[fun]
    def get_one(exper):
        idata = idatas.loc[exper]
        var_name = 'C(Condition, levels=lvl)'
        diagnostic = list(fun(idata, var_names=var_name).to_dict()['data_vars'][var_name]['data'])
        df = idata.posterior.to_dataframe()
        df = df.drop([var_name, '1|IRN'], axis=1)
        diagnostic = [fun(idata, var_names=v).to_dict()['data_vars'][v]['data'] for v in df.columns] + diagnostic
        treatments = idata.posterior.coords[var_name + '_dim'].to_numpy()
        ix = df.columns.to_list() + list(treatments.astype('object'))
        df = pd.DataFrame({exper: diagnostic}, index=ix)
        s = pd.Series(diagnostic, index=pd.MultiIndex.from_product([[exper], ix]))
        return(s)
    l = [get_one(exper) for exper in idatas.index]
    df = pd.concat(l, axis=0).to_frame(title)
    precision = np.int64(3 - np.round(np.log10(df.mean().mean())))
    val = df.style.format(precision=precision).background_gradient(axis=None, vmin=df.min().min(), vmax=df.max().max(), cmap='hot')
    return(val)
