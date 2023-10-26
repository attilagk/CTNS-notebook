import arviz as az
import pymc as pm
import pandas as pd
import pytensor.tensor as at
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
