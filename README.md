# Bayesian Adpative Models

[![Github All Releases](https://img.shields.io/github/downloads/hugo1005/bayesianadaptivemodels/total.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

## Working with short time series data with complex functional forms
A sparse time series process is a stochastic sequence whoses functional form has a sparse dependency on high order MA or AR lags. 
Under the assumption that the data generating process dependency structure is known this package provides a method for bayesian estimation
of only the sparse parameters rather than a full ARMA(p,q) model which would overfit and also use up precious 
degree's of freedom when the observed time series is short and inflate variance. 

This procedure does not require the likelihood of these sparse parameter models to be known, the only assumption that is required is that 
the residuals / errors of such a model be distributed normally and thus bayesian methods can be applied by treating Y_{t} = F(Y,T) + epsilon_{t}
as a problem in which epsilon_{t} = Y_{t}  - F(Y,T) where Y_{t} is observed from data and F(Y,T) is the sparse functional form with observations Y for t in T = {1,...,t-1} with parameter values obtained NUTS / other specified sampling procedure. Under this framework the parameters will converge to the true distribution so long as the functional form is correct and the errors are in fact normally distributed.

## Warning
This package is in development, subject to modification and further testing.

## Installing the package
```
python -m pip install git+https://github.com/hugo1005/bayesianadaptivemodels.git
```

## Specifying Models
The process may be specifed as a string where any capital letter eg. Y_{t} denotes a time series variable, with expection of E_{t} which is reserved for lagged errors.
White noise may be specified as epsilon_k where k is any integer and any lower case letter denotes a parameter to be estimated eg. p_k where k is any integer. Multiple time series may be specified but only the final equation is assumed to be observed. 

An example of an ARMA(1,1) model

```
data_gen_process = """
E_{t} = epsilon_1 
Y_{t} = p_1 * Y_{t-1} + p_2 * E_{t-1} + E_{t};
"""
```

Currently only multiplication and addition operations are supported and processes of 1 Dimension.

Note currently latent variables are not supported (eg. where X is not observed)

```
data_gen_process = """
X_{t} = p_2 * X_{t-1} + epsilon_2;       
Y_{t} = p_1 * Y_{t-1} + X_{t} + epsilon_1;
"""
```

Vector processes and Exogenous variable support under consideration.

Note that the DataGenerationProcess class can be used to simulate any of the processes accepted by this model

## Generating Fake Data
```
dgp = """
Y_{t} = p_1 * Y_{t-1} + p_2 * Y_{t-7} + p_3 * Y_{t-8} + epsilon_1;
"""

rho = 0.1
gamma = 0.5
parameter_values = {'p_1': gamma, 'p_2': rho, 'p_3': - 1 * rho * gamma}
datagen = dg.DataGenerationProcess(dgp, parameter_values)

# 2 parameters, the number of samples and a burn in period to throw away
Y = np.array(datagen.run_for_n_iters(400,200)['Y'])
Y_train = Y[:200]
Y_test = Y[200:]
```

## Fitting Bayesian Models
Fitting models can be conducted using the following process, note that the backend used for NUTS sampling is pymc3,
so trace objects returned will be the same as the PyMc3 API.

```
import pymc3 as pm
import numpy as np
from bayesianadaptivemodels import adaptive_models as am

dgp = """
Y_{t} = p_1 * Y_{t-1} + p_2 * Y_{t-7} + p_3 * Y_{t-8} + epsilon_1;
"""

rho = 0.1
gamma = 0.5
parameter_values = {'p_1': gamma, 'p_2': rho, 'p_3': - 1 * rho * gamma}
datagen = am.DataGenerationProcess(dgp, parameter_values, sigma=0.1)

Y = np.array(datagen.run_for_n_iters(400,200)['Y'])
Y_train = Y[:200]
Y_test = Y[200:]

atsm = am.AdaptiveTimeSeriesModel(dgp, {'p_1': 0, 'p_2': 0, 'p_3': 0})
atsm.create_model(Y_train)
trace = atsm.sample(draws=6000, chains=4)
pm.summary(trace)
```

## In sample predictions
```
results = atsm.predict(Y_train, n_samples=1000, show_plot=True)
```

## Out of sample forecasts
```
k_periods = 8
forecasts = atsm.forecast(k_periods)
```
