import numpy as np
import matplotlib.pyplot as plt
from pymc3.model import Model
import theano as T
import pandas as pd
import pymc3 as pm

import re
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Helper functions
def diagnostics(series):
    plt.figure(figsize=(20,5))
    ax1 = plt.subplot2grid((1,3),(0,0))
    ax1.plot(series)
    ax1.axhline(series.mean(), linestyle='dashed', c='r')
    ax2 = plt.subplot2grid((1,3),(0,1))
    plot_acf(series, ax=ax2, zero=False)
    ax3 = plt.subplot2grid((1,3),(0,2))
    plot_pacf(series, ax=ax3, zero=False)
    plt.show()

def fmt_axis(ax, title=''):
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_facecolor((0,0,0,0.05))

    font_title = {'family': 'roboto',
        'color':  '#143d68',
        'weight': 'bold',
        'size': 16,
        'alpha': 0.7
    }

    ax.set_title(title, fontdict=font_title,loc='left')

class Noise():
    def __init__(self, name, mean=0, variance=1):
        self.name = name
        self.mean = mean
        self.var = variance

    def eval(self):
        return np.random.normal(self.mean, self.var)

class Parameter():
    def __init__(self, name, init_value):
        self.name = name
        self.value = init_value
        self._trace = None

    def eval(self):
        return self.value

    def create_normal_prior(self, sigma = 3):
        self.prior = pm.Normal(self.name, mu = self.value, sigma=sigma)

    def get_prior(self):
        if type(self.prior) == type(None):
            self.create_normal_prior()
        
        return self.prior

    def set_trace(self, trace):
        self._distribution = trace.get_values(self.name)

    def set_value_as_trace_sample(self):
        pos = np.random.random_integers(0, len(self._distribution) - 1)
        self.value = self._distribution[pos]

class TSVariable():
    def __init__(self, name, max_lags=1, history=None):
        self.name = name
        self.create_history(max_lags, history)

    def create_history(self, max_lags, history):
        if type(history) != type(None):
            self.history = history
        else:
            self.history = self.init_random_history(max_lags)

    def init_random_history(self, max_lags):
        return [Noise(0,1).eval() for i in range(max_lags+1)]

    def get_history(self):
        return self.history

    # DO this first
    def update_value(self, new_val):
        self.current_value = new_val

    # THEN do this at end of loop
    def update_history(self):
        self.history.append(self.current_value)
        self.current_value = None

    def eval(self, lag):
        if lag == 0:
            return self.current_value
        else:
            return self.history[-(lag)]

class Expression():
    def __init__(self, expression_str):
        self._expression_str = expression_str

        self._TS_variable_strs = list(set(re.findall('[A-Z]_{t-?\d*?}', self._expression_str)))
        self.TS_variable_names = [var.split('_')[0] for var in self._TS_variable_strs]
        self.parameter_names = list(set(re.findall('(?:^|\W)([a-z]_\d*)', self._expression_str)))
        self.noise_names = list(set(re.findall('epsilon_\d*', self._expression_str)))

        self.TS_variable_max_lags = self.get_TS_vars_max_lags()

        all_lags = list(self.TS_variable_max_lags.values())
        self.max_lags = max(all_lags) if len(all_lags) > 0 else 0
        
    def get_TS_var_lags(self, TS_variable_strs):
        return [int(re.search("\d+",var)[0]) for var in TS_variable_strs]

    def get_TS_vars_max_lags(self):
        get_indexes = lambda var_name: [idx for idx, name in enumerate(self.TS_variable_names) if name == var_name]
        TS_variable_lags = [int(re.search("\d+",var)[0]) for var in self._TS_variable_strs]
        TS_variable_max_lags = {var_name : max(*[TS_variable_lags[i] for i in get_indexes(var_name)]) for var_name in set(self.TS_variable_names)} # X,Y

        return TS_variable_max_lags

    def eval_expression(self, node_eval_function, use_index_i=-1):
        add_together = self._expression_str.split('+')
        components = [self.multiply(sub_eqn.split('*'), node_eval_function, use_index_i) for sub_eqn in add_together]
        return sum(components)

    def multiply(self, split_eqn, node_eval_function, use_index_i=-1):
        def node_eval_wrapped(split_eqn_comp):
            if use_index_i < 0:
                return node_eval_function(split_eqn_comp) 
            else:
                return node_eval_function(split_eqn_comp, use_index_i) 

        if len(split_eqn) > 1:
            return node_eval_wrapped(split_eqn[0]) * node_eval_wrapped(split_eqn[1]) 
        else:
            return node_eval_wrapped(split_eqn[0])

class Equation():
    def __init__(self, equation_str):
        equation_str = equation_str.replace(' ','')
        lhs, rhs = equation_str.split('=')

        self.TS_variable_name = lhs.split('_')[0]
        self.expression = Expression(rhs)

class EquationInterpreter():
    """
    params
    ------
    equations: list of string equations
    lhs_rhs: {VarName: rhs of equation string}
    TS_variable_strs, parameter_strs, noise_strs: Variables extracted from strings as typed

    TS_variable_names: X,Y,Z etc (Note E is reserved for error time series)
    TS_variable_lags: List of variable lags (not labelled)
    TS_variable_max_lags: {X: max_lag, ...}
    max_lags: maximum lag present in any variable

    TS_variables, parameters, noises: Dict referencing the object by TS_variable_names, parameter_strs, noise_strs
    """

    def __init__(self, data_generating_process, parameter_values, sigma=0.1):
        self.data_generating_process = data_generating_process
        self.equations = [ Equation(eq_str) for eq_str in self.data_generating_process.replace('\n','').replace('{t}','{t-0}').split(';')[:-1]]
        self.max_lags = max([eq.expression.max_lags for eq in self.equations])

        self.TS_variable_names = [eq.TS_variable_name for eq in self.equations]
        self.TS_variable_max_lags = {ts_var: 0 for ts_var in self.TS_variable_names}
        self.parameter_names = []
        self.noise_names = []

        for eq in self.equations:
            self.parameter_names += eq.expression.parameter_names
            self.noise_names += eq.expression.noise_names

            for ts_var, lags in eq.expression.TS_variable_max_lags.items():
                old_val = self.TS_variable_max_lags[ts_var]
                self.TS_variable_max_lags[ts_var] = lags if self.TS_variable_max_lags[ts_var] < lags else old_val

        self.parameter_names = list(set(self.parameter_names))
        self.noise_names = list(set(self.noise_names))
        
        self.TS_variables = {var: TSVariable(var, max_lags=lags) for var, lags in self.TS_variable_max_lags.items()}
        self.parameters = {var: Parameter(var, parameter_values[var]) for var in self.parameter_names}
        self.noises = {var: Noise(var, 0, sigma) for var in self.noise_names}

    def get_variables_in_string(self, param_str):
        TS_variable_strs = list(set(re.findall('[A-Z]_{t-?\d*?}', param_str)))
        
        TS_variable_names = [var.split('_')[0] for var in TS_variable_strs]
        parameter_names = re.findall('(?:^|\W)([a-z]_\d*)', param_str)
        noise_names  = re.findall('epsilon_\d*', param_str)

        return TS_variable_strs, TS_variable_names, parameter_names, noise_names

class DataGenerationProcess(EquationInterpreter):
    def __init__(self, data_generating_process, parameter_values, sigma=0.1):
        """
        Warning Data Generating Process Must be written in the sequence in which the variables must be evaluated
        i.e write the equation for X before the Equation for Y if Y = F(X)
        """
        super().__init__(data_generating_process, parameter_values, sigma)

    def run_for_n_iters(self, n_iters, burn_in=100):
        """
        Generates a time series stored in the history of the variables. Burn in is used to ensure no strange behaviour and removed the first batch of entries from the output series
        """
        for _ in range(burn_in):
            self.update_all_ts_variables()

        for _ in range(n_iters):
            self.update_all_ts_variables()

        return {var_name: ts_var.get_history()[burn_in:burn_in+n_iters] for var_name, ts_var in self.TS_variables.items()}

    def update_all_ts_variables(self):
        self.evaluate_equations()
        self.update_variable_histories()
        
    def evaluate_equations(self):
        for eq in self.equations:
            eqn_eval = eq.expression.eval_expression(self.get_value)
            self.TS_variables[eq.TS_variable_name].update_value(eqn_eval)

    def update_variable_histories(self):
        # End of loop update
        for eq in self.equations:
            self.TS_variables[eq.TS_variable_name].update_history()

    def get_value(self, param_str):
        TS_variable_strs = list(set(re.findall('[A-Z]_{t-?\d*?}', param_str)))
        
        TS_variable_names = [var.split('_')[0] for var in TS_variable_strs]
        parameter_names = re.findall('(?:^|\W)([a-z]_\d*)', param_str)
        noise_names  = re.findall('epsilon_\d*', param_str)

        if len(TS_variable_names):
            return self.evaluate_time_series_variable(TS_variable_strs, TS_variable_names)
        elif len(noise_names):
            return self.evaluate_noise_variable(noise_names)
        else:
            return self.evaluate_parameter_variable(parameter_names)

    def evaluate_time_series_variable(self, TS_variable_strs, TS_variable_names):
        TS_variable_lags = [int(re.search("\d+",var)[0]) for var in TS_variable_strs]
        lag = TS_variable_lags[0]
        var_name = TS_variable_names[0]

        return self.TS_variables[var_name].eval(lag)

    def evaluate_noise_variable(self, noise_names):
        var_name = noise_names[0]
        return self.noises[var_name].eval()

    def evaluate_parameter_variable(self, parameter_names):
        var_name = parameter_names[0]
        return self.parameters[var_name].eval()    

class ModelData():
    def __init__(self, Y, max_lags):
        Y_df = pd.Series(Y)
        self.max_lags = max_lags
        self.length = len(Y_df[max_lags:])
        self.base = np.array(Y_df[max_lags:])
        self.lags = {i: np.array(Y_df.shift(i)[max_lags:]) for i in range(max_lags+1)}
        #self.base = pm.Data('Y', Y_df[max_lags:])
        #self.lags = [pm.Data('Y_{t-%i}' % i, Y_df.shift(i)[max_lags:]) for i in range(max_lags+1)]
        
    def replace_data(self, Y):
        Y_df = pd.Series(Y)
        self.base = np.array(Y_df[self.max_lags:])
        self.lags = {i: np.array(Y_df.shift(i)[self.max_lags:]) for i in range(self.max_lags+1)}
        #pm.set_data({ 'Y':  Y_df[self.max_lags:] })
        #pm.set_data({ 'Y_{t-%i}' % i: Y_df.shift(i)[self.max_lags:] for i in range(self.max_lags+1) })

    def __len__(self):
      return self.length

class AdaptiveTimeSeriesModel(EquationInterpreter):
    def __init__(self, time_series_process, prior_means):
        """
        Warning Data Generating Process Must be written in the sequence in which the variables must be evaluated
        i.e write the equation for X before the Equation for Y if Y = F(X)
        """
        super().__init__(time_series_process, prior_means)
        self.contains_MA_terms = 'E' in self.TS_variable_names

    def compute_expectation(self, use_index_i=-1, inference=False):
        expr = self.equations[-1].expression # Use the last equation

        if inference:
            return expr.eval_expression(self.eval_node_with_posterior_distribution, use_index_i)
        else:
            return expr.eval_expression(self.eval_node_with_priors, use_index_i)

    def eval_node_with_priors(self, param_str, use_index_i=-1):
        TS_variable_strs, TS_variable_names, parameter_names, noise_names = self.get_variables_in_string(param_str)
        return self.evaluate_node(TS_variable_strs, TS_variable_names, parameter_names, noise_names, use_index_i)

    def eval_node_with_posterior_distribution(self, param_str, use_index_i=-1):
        TS_variable_strs, TS_variable_names, parameter_names, noise_names = self.get_variables_in_string(param_str)
        return self.evaluate_node(TS_variable_strs, TS_variable_names, parameter_names, noise_names, use_index_i=use_index_i, inference=True)

    def evaluate_node(self, TS_variable_strs, TS_variable_names, parameter_names, noise_names, use_index_i=-1, inference=False):
        
        if len(TS_variable_names):
           
            # Return the model data
            TS_variable_lags = [int(re.search("\d+",var)[0]) for var in TS_variable_strs]
            lag = TS_variable_lags[0]
            
            return_all_model_data_for_lag = use_index_i < 0

            if return_all_model_data_for_lag:
                # returns all observed time series for given lag
                return self.model_data.lags[lag]
            else:
                # E case is only considered here as we only need to worry about residuals in MA case
                # and MA case always requires indexing
                TS_variable_is_error_term = TS_variable_names[0] == 'E'

                if TS_variable_is_error_term:
                    is_observed_error =  lag > 0 and use_index_i-lag >= 0 and use_index_i-lag < len(self.residuals)

                    if is_observed_error:
                        return self.residuals[use_index_i-lag]
                    else:
                        return 0
                else:
                    is_in_sample = use_index_i >= 0 and use_index_i < len(self.model_data.base)

                    if is_in_sample:
                        return self.model_data.lags[lag][use_index_i]
                    else:
                        return self.predicted[use_index_i - lag]


        elif len(noise_names):
            # Ignore
            return 0


        else:
            if inference:
                # Return the prior
                return self.parameters[parameter_names[0]].eval()
            else:
                # Return the prior
                return self.parameters[parameter_names[0]].get_prior() 

    def create_model(self, Y):
        self.model = pm.Model()

        with self.model:
            self.set_model_data(Y)
            max_n = len(self.model_data)
         
            for _, param in self.parameters.items():
                param.create_normal_prior()

            if self.contains_MA_terms:
                # Computing expected values
                self.residuals = []

                # Feedforward through time computing observed errors based on current param estimates
                for i in range(0, max_n):
                    expected_val = self.compute_expectation(use_index_i=i)
                    error_estimate = (expected_val - self.model_data.base[i]) * -1
                    self.residuals.append(error_estimate) 
                
                # Compute errors
                sigma = pm.HalfCauchy('sigma',3) 
                pm.Normal('residuals', mu=0, sigma = sigma, observed=T.tensor.stack(*self.residuals))

            else:
                # Computing expected values
                expected_vals = self.compute_expectation()
                
                # Computing the errors
                self.residuals = (expected_vals - self.model_data.base) * -1
                sigma = pm.HalfCauchy('sigma',3)
                pm.Normal('residuals', mu=0, sigma=sigma, observed=self.residuals)
                
    def set_model_data(self, Y, out_of_sample=False):
        if out_of_sample:
            self.model_data.replace_data(Y)
        else:
            self.model_data = ModelData(Y, self.max_lags)

    def sample(self, *args, **kwargs):
        with self.model:
            trace = pm.sample(*args, **kwargs)

            for param in self.parameters:
                self.parameters[param].set_trace(trace)

            return trace
    
    # One step ahead forecasts
    def predict(self, n_samples=100, show_plot=True):
        max_n = len(self.model_data)
        
        predicted = np.zeros((max_n,n_samples))
        
        for j in range(n_samples):
            
            for param in self.parameters:
                self.parameters[param].set_value_as_trace_sample()
            
            self.residuals = []
            for i in range(max_n):
                predicted[i,j] = self.compute_expectation(use_index_i=i, inference=True)
                error_estimate = (predicted[i,j] - self.model_data.base[i]) * -1
                self.residuals.append(error_estimate) 
        
        if show_plot:
          predictions = pd.DataFrame(predicted)
          plt.figure(figsize=(10,5))
          ax = plt.gca()
          predictions.plot(alpha=0.02,c='k',ax=ax)
          pd.DataFrame(self.model_data.base).plot(alpha=0.2,c='r',ax=ax, linestyle='-')
          ax.legend([])
          fmt_axis(ax, "Predicted vs Actual")

        return predicted

    def forecast(self, k_steps, n_samples=100):
        """
        Forecasts k-steps out of sample
        param Y: a 1D array of in-sample observations
        param k_steps: An nteger number of steps to forecast
        param n_samples: number times to draw parameters from posterior distribution

        return forecasts: a k_steps x n_samples matrix of forecast values
        """

        max_n = len(self.model_data)

        predicted = np.zeros((max_n + k_steps,n_samples))
      
        for j in range(n_samples):
            
            for param in self.parameters:
                self.parameters[param].set_value_as_trace_sample()
            
            # Predict in sample first to get the residuals and error estimates
            self.residuals = []
            for i in range(max_n):
                predicted[i,j] =  self.compute_expectation(use_index_i=i, inference=True)
                error_estimate = (predicted[i,j] - self.model_data.base[i]) * -1
                self.residuals.append(error_estimate) 

            # Out of sample forecast using whatever information is available (i.e previous residuals + predictions)
            self.predicted = []

            for i in range(max_n, max_n + k_steps):
                self.predicted = predicted[:i,j]
                predicted[i,j] = self.compute_expectation(use_index_i = i, inference = True)

        forecasts = predicted[max_n:,:]
        return forecasts