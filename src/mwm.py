import arviz as az
import pymc as pm
import pandas as pd
import pytensor.tensor as at


mcmc_random_seed = 2021


def get_Weibull_beta(mu, alpha=2):
    arg = 1.0 + 1.0 / alpha
    print('arg:', arg)
    beta = mu / at.gamma(arg)
    return(beta)


def model_1(y_obs, x_obs, return_model=False):
    mymodel = pm.Model()
    # Hyperparameters
    y_0_alpha = 2
    λ_mu = 0
    λ_sigma = 1
    α_alpha = 1.5
    α_mean = 2
    α_beta = α_alpha / α_mean
    with mymodel:
        y_0 = pm.Weibull('y_0', y_0_alpha, get_Weibull_beta(60, alpha=y_0_alpha))
        λ = pm.Normal('λ', λ_mu, λ_sigma)
        μ = pm.Deterministic('μ', y_0 * pm.math.exp(λ * (x_obs - 1)))
        #μ = pm.Deterministic('μ', y_0 * np.exp(λ * (x_obs - 1)))
        α = pm.Gamma('α', α_alpha, α_beta)
        β = pm.Deterministic('β', get_Weibull_beta(μ, alpha=α))
        y = pm.Weibull('y', α, β, observed=y_obs)
        if return_model:
            return(mymodel)
        idata = pm.sample(return_inferencedata=True, idata_kwargs={'log_likelihood': True}, random_seed=mcmc_random_seed)
        return(idata)
