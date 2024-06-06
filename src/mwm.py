import arviz as az
import pymc as pm
import pandas as pd
import numpy as np
import pytensor.tensor as at
import bambi as bmb
import matplotlib.pyplot as plt
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


def data_train_plotter(yname, data_train, lvl=['5xFAD', '5xFAD + Amiloride', 'WT']):
    fig, ax = plt.subplots(1, len(lvl), sharey=True, figsize=(3 * 4.8, 4.8))
    for condition, axi in zip(lvl, ax):
        axi.set_title(condition)
        axi.set_ylabel(yname)
        axi.set_xlabel('Day')
        axi.set_xticks(range(5))
        axi.set_xticklabels(range(5))
        axi.grid(axis='y')
        df1 = data_train.loc[data_train.Condition == condition]
        #df1 = data_train.xs(group, level=1)
        colors = {'f': 'red', 'm': 'blue'}
        #for sex, color in zip(df1.Sex.unique(), ['red', 'blue']):
        for sex in df1.Sex.unique():
            color = colors[sex]
            df2 = df1.xs(sex, level=2)
            for irn in df2.IRN.unique():
                df3 = df2.xs(irn, level=2)
                for cohort in df3.Cohort.unique():
                    s = sex + cohort
                    y = df3[yname]
                    x = df3.Day
                    axi.plot(x, y, color=color, label=irn, marker='$' + cohort + '$', linewidth=1)
                pass
            pass
        #axi.legend()
    return((fig, ax))


def escape_latency_plotter(exper, data_train, lvl):
    fig, ax = data_train_plotter(yname='Latency', data_train=data_train, lvl=lvl)
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


def get_diagnostics(idatas, fun=az.ess):
    def diagnose_one(exper):
        idata = idatas[exper]
        var_name = 'C(Condition, levels=lvl)'
        l = list(fun(idata, var_names=var_name).to_dict()['data_vars'][var_name]['data'])
        var_names = ['Intercept', 'Day', '1|IRN_sigma']
        l += [fun(idata, var_names=v).to_dict()['data_vars'][v]['data'] for v in var_names]
        var_names = ['Drug effect', 'Genotype effect'] + var_names
        df = pd.DataFrame(l, index=var_names, columns=[exper])
        return(df)

    l = [diagnose_one(exper) for exper in idatas.keys()]
    df = pd.concat(l, axis=1).transpose()
    precision = np.int64(3 - np.round(np.log10(df.mean().mean())))
    val = df.style.format(precision=precision).background_gradient(axis=None, vmin=df.min().min(), vmax=df.max().max(), cmap='hot')
    return(val)
