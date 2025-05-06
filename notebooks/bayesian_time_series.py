# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fonnesbeck/ts_pydata_london_2025/blob/master/notebooks/Bayesian_Time_Series.ipynb)
# 
# # Introduction to Bayesian Time Series Analysis
# 
# Time series analysis is one of the most crucial applications in data science, given the importance placed on forecasting and prediction. 
# 
# A sound approach to time series analysis presents a new set of challenges to analysts:
# 
# 1. Many of the typical statistical assumptions do not apply
# 2. Time series data are typically sparser than static data
# 3. Model validation is more difficult
# 
# At its simplest, 
# 
# > Time series data are sequences of observations, indexed by time.
# 
# It introduces the concept of inter-temportal dependence: An observation at time $t_i$ can be related to previous observations $t_{i-1}, t_{i-2} ...$. 
# 
# This implies a lack of independence among the observations across time; specifically, the order of the observations is important, and must be taken into account for any analysis.

# %%
import arviz as az
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import polars as pl
import pymc as pm
import pytensor as pt

# %%
az.style.use("arviz-darkgrid")
RANDOM_SEED = 20090426
RNG = np.random.default_rng(RANDOM_SEED)

# %% [markdown]
# ## Stochastic Volatility model
# 
# Often we don't need a full mechanistic model, but rather seek to build simple models which capture the time series behaviour of the data. These may be used to provide an adequate basis for forecasting. 
# 
# Asset prices have time-varying volatility (variance of day-over-day `returns`). In some periods, returns are highly variable, while in others very stable. Stochastic volatility models model this with a latent volatility variable, modeled as a stochastic process.
# 
# Let's first look at the data:

# %%
file_path = pm.get_data("SP500.csv")
returns = pl.read_csv(file_path, columns=["Date", "change"], try_parse_dates=True)
returns.head()

# %%
fig = px.line(returns, x='Date', y='change', title='SP500 Daily Returns')
fig.update_layout(width=1000, height=500)
fig.show()

# %% [markdown]
# We get the average daily returns of the SP500 in $\frac{percent}{100}$.

# %% [markdown]
# As you can see, the volatility seems to change over time quite a bit but these changes cluster around certain time-periods. For example, the 2008 financial crisis is easy to pick out.
# 
# We will use a **Gaussian Random Walk** (GRW) as a starting point of our volatility model, specifically a discrete time version using log-volatility.
# 
# $$\Large
# \begin{aligned}
# \sigma_t &= \sigma_{t-1} + \epsilon_{t-1} \\
# \epsilon_t &\sim \mathcal{N}(0, \sigma_{\text{step}}^2) \\
# \sigma_{\text{step}} &\sim \operatorname{Exp}(10) \\
# \sigma_0 &\sim \mathcal{N}(0, 1) \\
# \end{aligned}
# $$
# 
# 
# We have the *evolution equation*, $\sigma_t = \sigma_{t-1} + \epsilon_{t-1}$, which tells us that our *log-volatility* at time $t$ depends on the *log-volatility* $\sigma_{t-1}$ at time $t-1$ plus a random perturbation (or *innovation*), which is Gaussian (hence GRW). 
# 
# Lastly, we need to provide a prior on the *standard deviation of the innovations* $\sigma_{\text{step}}$, for which we specify an *Exponential* distribution. 
# 
# The choices of distributions for $\sigma_{\text{step}}$ and $\sigma_0$ are subjective. But for the purposes of our example, they will do the job.
# 

# %% [markdown]
# While we have defined the log-volatility as a Gaussian Random Walk (GRW), our primary goal is to model the actual returns. In other words, we have defined a distribution over latent variables (a prior, in Bayesian terms), but we still need to define the observation model.
# 
# For our observation model, we choose the Student's t-distribution:
# 
# $$\Large
# \begin{aligned}
# \text{returns}_t &\sim \operatorname{StudentT}(\nu, \lambda = \exp(-2 * \sigma_t)) \\
# \nu &\sim \operatorname{Exponential}(0.1) \\
# \end{aligned}
# $$
# 
# 
# We assume that our actual observations, the returns, are derived from a Student's t-distribution, with precision $\lambda$ coming from our latent GRW. Additionally, we place a prior on the degrees of freedom $\nu$ of our Student's t-distribution, which is described by an Exponential distribution. This Student's t-distribution serves as our likelihood.
# 
# The choice of the specific distribution for either the prior on $\nu$ or the likelihood is a matter of judgment. However, with this observation model, our overall model is now complete, and we can proceed with PyMC implementation.
# 
# 

# %% [markdown]
# ### The `GaussianRandomWalk` Distribution
# 
# ## The GaussianRandomWalk Distribution
# 
# The `GaussianRandomWalk` distribution provides a prior distribution for the vector of latent incidence. As the name suggests, this distribution represents a vector-valued random process where the elements form a Gaussian random walk of length `n`, determined by the `shape` (or `dims`) argument. 
# 
# Let's build the model:

# %%
# Extract the two columns from the returns DataFrame
dates = returns.select(["Date"]).to_numpy().flatten()
changes = returns.select(["change"]).to_numpy().flatten()

# %%
with pm.Model(coords={"date": dates}) as stochastic_vol_model:
    # Priors
    sigma_volatility = pm.Exponential("sigma_volatility", 5)
    log_volatility = pm.GaussianRandomWalk(
        "volatility", sigma=sigma_volatility, init_dist=pm.Normal.dist(0, 1), dims="date"
    )

    nu = pm.Exponential("nu", 0.1)

    # Likelihood
    obs = pm.StudentT(
        "returns", nu=nu, lam=np.exp(-2 * log_volatility), observed=changes, dims="date"
    )

pm.model_to_graphviz(stochastic_vol_model)

# %% [markdown]
# ### Prior Predictive Check

# %%
with stochastic_vol_model:
    idata_prior_pred = pm.sample_prior_predictive(random_seed=RANDOM_SEED)

# %% [markdown]
# Let's plot simulated returns from the prior predictive. As we have learned throughout the course, this is a useful check for us to understand whether our prior assumptions are reasonable.

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Calculate maximums
max_observed = np.max(np.abs(returns))
max_simulated = np.max(
    np.abs(idata_prior_pred.prior_predictive["returns"])
)

# Create figure
fig = make_subplots()

# Add original returns trace
fig.add_trace(
    go.Scatter(
        x=returns["Date"],
        y=returns["change"],
        mode="lines",
        line=dict(color="black", width=1),
        name="Original Returns"
    )
)

# Add simulated returns from prior
fig.add_trace(
    go.Scatter(
        x=returns["Date"],
        y=idata_prior_pred.prior_predictive["returns"].sel(chain=0, draw=10).T,
        mode="lines",
        line=dict(color="green", width=1),
        opacity=0.5,
        name="Simulated Returns"
    )
)

# Update layout
fig.update_layout(
    title=f"Maximum observed: {max_observed:.2g}<br>Maximum simulated: {max_simulated:.2g}(!)",
    width=1000,
    height=400,
    showlegend=True
)

fig.show()

# %% [markdown]
# 
# The prior predictive plot reveals that the initial assumptions of the `GaussianRandomWalk` distribution are significantly overestimated compared to the actual observed returns. 
# 
# #### Potential Prior Adjustment
# 
# This discrepancy suggests that adjusting our prior assumptions may be necessary. A return that our model considers plausible would violate various constraints by a substantial margin. For instance, the total value of all goods and services produced globally is approximately $10^9, so it would be reasonable to exclude returns above this magnitude from our prior assumptions.
# 
# #### Proceeding with the Standard Model
# 
# Despite the potential need for prior adjustments, we proceed with fitting this standard model as is. It is worth noting that this model can be challenging to fit even with the No-U-Turn Sampler (NUTS). Consequently, we sample and tune the model for a longer duration than the default settings to ensure accurate results.
# 
# By refining the prior assumptions and acknowledging the model's complexity, we can improve the accuracy and interpretability of the `GaussianRandomWalk` distribution.

# %%
with stochastic_vol_model:
    trace = pm.sample(tune=2000, draws=1000, random_seed=RANDOM_SEED)

# %%
az.plot_trace(trace, var_names=["sigma_volatility", "nu"]);

# %% [markdown]
# Note that the `sigma_volatility` parameter does not look ideal: the different chains look somewhat different and draws within a chain are strongly autocorrelated. This again indicates some weakness in our model: it probably makes sense to allow `sigma_volatility` to change over time, especially over an 11-year time span.
# 
# Despite this limitation, we will proceed with the example.

# %% [markdown]
# ### Posterior Predictive
# 
# Now let's take a look at our posterior estimates of the volatility in S&P 500 returns over time. We will also use the posterior predictive distribution to see how the learned volatility could have affected returns:

# %%
with stochastic_vol_model:
    trace_post_pred = pm.sample_posterior_predictive(trace=trace, random_seed=RANDOM_SEED)
trace.extend(trace_post_pred)

# %%
post = trace.posterior.stack(sample=("chain", "draw"))
post_pred = trace.posterior_predictive.stack(sample=("chain", "draw"))
random_draws = RNG.choice(len(post.sample), 100)

# Create subplots with 2 rows and 1 column
from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                   vertical_spacing=0.1, subplot_titles=("True Returns (black) and Posterior Predictive Returns (green)", 
                                                        "Inferred posterior log-volatility"))

# Plot returns - using polars DataFrame
fig.add_trace(
    go.Scatter(x=returns["Date"].to_numpy(), y=returns["change"].to_numpy(), 
               mode='lines', line=dict(color='black', width=1), 
               opacity=0.8, name='True Returns'),
    row=1, col=1
)

# Plot posterior predictive returns
for i in random_draws:
    fig.add_trace(
        go.Scatter(x=returns["Date"].to_numpy(), y=post_pred["returns"].isel(sample=i).values, 
                  mode='lines', line=dict(color='green', width=0.5), 
                  opacity=0.4, showlegend=False),
        row=1, col=1
    )

# Plot posterior predictive volatility
for i in random_draws:
    fig.add_trace(
        go.Scatter(x=returns["Date"].to_numpy(), y=np.exp(post["volatility"].isel(sample=i).values), 
                  mode='lines', line=dict(color='orange', width=0.5), 
                  opacity=0.5, showlegend=False),
        row=2, col=1
    )

# Update layout
fig.update_layout(height=700, width=1400, showlegend=True)
fig.show()

# %% [markdown]
# ## Autoregressive Models

# %% [markdown]
# Let's take a step back and focus on modeling only the properties of our GRW (forget about the *observation model* for now). 
# 
# It turns out that the GRW is a special case of an **autoregressive model**, which is specified by:
# 
# 
# $$\Large y_t = \rho y_{t-1} + \epsilon_t,$$
# 
#    
# where $\epsilon_t \overset{iid}{\sim} {\cal N}(0,1)$. In the case of the **GRW**, the parameter $\rho$ is fixed to 1; consequentially, the random **increments alone drive the evolution** of the state (hence the name, "random walk"). 
# 
# The form above is also a specific subclass of an autoregressive model, **the first-order autoregressive, or AR(1), process**. 
# 
# This is a *Markovian* model (recall Markov chains), which means that state $y_t$, only depends on $y_{t-1}$.
# 
# A more general form of autoregressive model is the **nth-order autoregressive process, AR(n)**:
# 
# 
# $$\Large y_t = \rho_{1} y_{t-1} + \rho_2 y_{t-2} + \ldots + \rho_n y_{t-n} + \epsilon_t$$
# 
# 
# ### AR(1) models with PyMC
# 
# First, let's generate some data from the $\operatorname{AR}(1)$ process. To gain intuition, let's generate and plot data form a few different setting for the `rho` parameter.
# 
# Let's begin with setting $\rho = 0$; this turns off the random walk behavior. Our data now consists of independent random Gaussian variates.

# %%
T = 200
y = np.zeros(T)
rho = 0

# Generate AR(1) random draws
for i in range(1, T):
    y[i] = rho * y[i - 1] + RNG.normal()

# %% [markdown]
# We can also use PyMC directly like so,

# %%
y = pm.draw(pm.AR.dist([0.0], steps=200, init_dist=pm.Normal.dist(0, 1, shape=1)))

# %% [markdown]
# which is doing the equivalent under the hood.

# %%
plt.plot(y)

# %% [markdown]
# Next, let's turn $\rho$ negative, which gives us anticorrelated examples:

# %%
y = pm.draw(pm.AR.dist([-1.0], steps=200, init_dist=pm.Normal.dist(0, 1, shape=1)))

# %%
plt.plot(y);

# %% [markdown]
# The basic GRW is returned (as discussed above), by setting $\rho = 1$

# %%
y = pm.draw(pm.AR.dist([1.0], steps=200, init_dist=pm.Normal.dist(0, 1, shape=1)))

# %%
plt.plot(y)

# %% [markdown]
# Lastly, consider $\rho > 1$, $\rho < -1$. Such settings make the process **non-stationary** (or **unstable**). Let's look at what some data would looks like for each case.

# %%
# non-stationary anti-correlated
y = pm.draw(pm.AR.dist([-1.2], steps=200, init_dist=pm.Normal.dist(0, 1, shape=1)))

# %%
plt.plot(y);

# %%
# non-stationary correlated
y = pm.draw(pm.AR.dist([1.2], steps=200, init_dist=pm.Normal.dist(0, 1, shape=1)))

# %%
plt.plot(y)

# %% [markdown]
# Let's now simulate some data which we actually want to use for fitting.

# %%
y_obs = pm.draw(pm.AR.dist([0.8], steps=200, init_dist=pm.Normal.dist(0, 10, shape=1)))

# %% [markdown]
# As with all Bayesian models, the first step is to choose our priors; here we consider the coefficient $\rho$. 
# 
# Let's use a standard normal $\rho \sim {\cal N}(0,1)$.

# %%
with pm.Model() as ar1:
    rho = pm.Normal("rho", mu=0, sigma=1.0)
    ts = pm.AR("ts", rho, sigma=1.0, observed=y_obs)

    trace = pm.sample()

# %%
az.plot_trace(trace);

# %%
mu_rho = ((y_obs[:-1] ** 2).sum() + 1**-2) ** -1 * np.dot(y_obs[:-1], y_obs[1:])
var_rho = ((y_obs[:-1] ** 2).sum() + 1**-2) ** -1

print(
    "Mean: {:5.3f} (exact = {:5.3f})".format(trace.posterior["rho"].mean().data, mu_rho)
)
print(
    "Std: {:5.3f} (exact = {:5.3f})".format(
        trace.posterior["rho"].std().data, np.sqrt(var_rho)
    )
)

# %%
az.plot_posterior(trace, ref_val=mu_rho);

# %% [markdown]
# ### Extension to AR(n)
# 
# Let's extend this to an AR(2) process:
# 
# $$
#  y_t = \rho_1 y_{t-1} + \rho_2 y_{t-2} + \epsilon_t.
# $$
# 
# The `AR` distribution infers the order of the process through the size the of the $\rho$ argmument passed to `AR`, so our model looks very similar to the previous one:

# %%
with pm.Model() as ar2:
    rho = pm.Normal("rho", mu=0, sigma=1, shape=2)
    likelihood = pm.AR("likelihood", rho, sigma=1.0, observed=y_obs)

    trace = pm.sample(tune=2000)

# %%
az.plot_trace(trace, compact=False);

# %% [markdown]
# Notice that the estimate of $\rho_1$ is close to zero, which is expected since the data was simulated from an AR(1) model.

# %% [markdown]
# ## Adding a moving average: ARMA
# 
# More complex time series models are can be implemented by adding other components to the basic **AR** model. 
# 
# A common extension is to use a **moving average**; a moving average model uses past forecast errors in a regression-like model:
# 
# $$\Large y_{t}=\mu+\varepsilon_{t}+\theta_{1} \varepsilon_{t-1}+\theta_{2} \varepsilon_{t-2}+\cdots+\theta_{q} \varepsilon_{t-q}$$
# 
# Notice that the observations $y_t$ can be viewed as **a weighted moving average of the past several errors**. So a first-order MA process is:
# 
# 
# $$\Large y_{t}=\mu+\varepsilon_{t}+\theta_{1} \varepsilon_{t-1}$$
# 
# 
# This is homologous to smoothing, but a moving average model is used for forecasting future values, whereas smoothing is used for estimating the trend-cycle of past values.
# 
# The motivation for the MA model is that we can explain shocks in the error process directly by fitting a model to the error terms.
# 
# 
# > As a general rule, a low order AR process will give rise to a high order MA process and a low order MA process will give rise to a high order AR process.
# >
# > $$x_{t}=\lambda x_{t-1}+\varepsilon_{t}, \quad \lambda<1$$
# >
# > by successively lagging this equation and substituting out the lagged value of x we may rewrite this as, 
# >
# > $$x_{t}=\sum_{j=1}^{\infty} \lambda^{j} \varepsilon_{t-j} \quad \text { where } \lambda^{\infty} x_{t-\infty} \rightarrow 0$$
# >
# > So the first order AR process has been recast as an infinite order MA one. 

# %% [markdown]
# An AR(p) and a MA(q) process can be combined to yield an **autoregressive moving average (ARMA)** model as follows:
# 
# $$\Large y_{t}=c+\phi_{1} y_{t-1}+\cdots+\phi_{p} y_{t-p}+\varepsilon_{t}+\theta_{1} \varepsilon_{t-1}+\cdots+\theta_{q} \varepsilon_{t-q}$$
# 
# Why would we want such similar components in the same model? The AR process accounts for **trends** in the stochastic process, while the MA component will soak up **unexpected events** in the time series.
# 
# A common data transformation that is applied to non-stationary time series to render them stationary is **differencing**. The differenced series is the change between consecutive observations in the original series, and can be written as,
# 
# $$\Large y_{t}^{\prime}=y_{t}-y_{t-1}$$
# 
# 
# The differenced series will have only T-1 values, since it is not possible to calculate a difference for the first observation. 
# 
# Applying the ARMA to differenced data yields an **autoregressive _integrated_ moving average (ARIMA)** model:
# 
# $$\Large y_{t}^{\prime}=c+\phi_{1} y_{t-1}^{\prime}+\cdots+\phi_{p} y_{t-p}^{\prime}+\varepsilon_{t}+\theta_{1} \varepsilon_{t-1}+\cdots+\theta_{q} \varepsilon_{t-q}$$
# 
# For our purposes though, we will stick to the ARMA model.

# %% [markdown]
# ### Air Passengers Data

# %% [markdown]
# Implementing an ARMA model in PyMC is trickier than for the AR(n) process. It involves generating variables in a loop, which requires coding in PyTensor directly ☠️.

# %% [markdown]
# Let's try to fit an ARMA model to a sample dataset. We will use a common time series dataset, which is just a summary of monthly totals of international airline passengers between 1949 and 1960.

# %%
import polars as pl

air_passengers = pl.read_csv(
    pm.get_data("AirPassengers.csv")
)
# Convert Month column to datetime with explicit format
air_passengers = air_passengers.with_columns(
    pl.col("Month").str.to_datetime("%Y-%m")
).sort("Month")

# Convert to pandas for plotting with Month as index
air_passengers.to_pandas().set_index("Month").plot();

# %% [markdown]
# We can start, as always, by declaring our priors, which here consist of:
# 
# - observational noise: $\sigma$
# - initial state: $\mu$
# - moving average coefficient: $\theta$
# - autoregression coefficient: $\rho$
# 
# For simplicity, we will model an ARMA(1, 1) process, so first order for both the moving average and autoregression:
# 
# $$\Large y_{t} \sim \mathrm{Normal}(\mu + \rho_{1} y_{t-1} + \varepsilon_{t} + \theta_{1} \varepsilon_{t-1}, \sigma)$$
# 
# Next, for the target variable, we divide by the maximum. We do this, rather than standardising, so that the sign of the observations is unchanged - this will be necessary later on, when we explicitly model seasonality.

# %%
y = air_passengers["#Passengers"].to_numpy()
y_max = np.max(y)
y = y / y_max

# %%
air_passengers

# %% [markdown]
# Now we're ready to set our model up:

# %%
coords = {"months": air_passengers.index, "months_m_1": air_passengers.index[1:]}

with pm.Model(coords=coords) as arma_model:
    mu = pm.Normal("mu", sigma=0.5)
    rho = pm.Normal("rho", sigma=0.5)
    theta = pm.Normal("theta", sigma=0.5)
    sigma = pm.HalfNormal("sigma", sigma=0.1)

# %% [markdown]
# The tricky part comes with calculating the sequence of states (recall that we need the sequence of $\epsilon_{t}$). We cannot simply use a python `for` loop because tensor libraries have trouble with structures that are cyclic, like loops. instead, we need to write an PyTensor `scan` function. 
# 
# ### `scan`
# 
# The `scan` functions provides the basic functionality needed to do loops in PyTensor. Scan comes with many whistles and bells, which we will introduce by way of examples.
# 
# #### Simple loop with accumulation: Computing $A^k$
# Assume that, given $k$ you want to get $A^k$ using a loop. More precisely, if $A$ is a tensor you want to compute $A^k$ elemwise. The python code might look like:
# 
# ```python
# result = 1
# for i in range(k):
#     result = result * A
# ```
# 
# There are three things here that we need to handle: the initial value assigned to `result`, the accumulation of results in `result`, and the unchanging variable `A`. Unchanging variables are passed to `scan` as `non_sequences`. Initialization occurs in `outputs_info`, and the accumulation happens automatically.
# 
# The equivalent PyTensor code would be:
# 
# ```python
# import pytensor
# import pytensor.tensor as pt
# 
# k = pt.iscalar("k")
# A = pt.vector("A")
# 
# # Symbolic description of the result
# result, updates = pytensor.scan(fn=lambda prior_result, A: prior_result * A,
#                               outputs_info=pt.ones_like(A),
#                               non_sequences=A,
#                               n_steps=k)
# 
# # We only care about A**k, but scan has provided us with A**1 through A**k.
# # Discard the values that we don't care about. Scan is smart enough to
# # notice this and not waste memory saving them.
# final_result = result[-1]
# 
# # compiled function that returns A**k
# power = pytensor.function(inputs=[A,k], outputs=final_result, updates=updates)
# 
# print(power(range(10),2))
# print(power(range(10),4))
# ```
# 
# ```
# [  0.   1.   4.   9.  16.  25.  36.  49.  64.  81.]
# [  0.00000000e+00   1.00000000e+00   1.60000000e+01   8.10000000e+01
#    2.56000000e+02   6.25000000e+02   1.29600000e+03   2.40100000e+03
#    4.09600000e+03   6.56100000e+03]
# ```
# Let us go through the example line by line. What we did is first to construct a function (using a lambda expression) that given `prior_result` and `A` returns `prior_result * A`. The order of parameters is fixed by scan: the output of the prior call to `fn` (or the initial value, initially) is the first parameter, followed by all non-sequences.
# 
# Next we initialize the output as a tensor with same shape and dtype as `A`, filled with ones. We give `A` to `scan` as a non sequence parameter and specify the number of steps `k` to iterate over our lambda expression.
# 
# Scan returns a tuple containing our result (`result`) and a dictionary of updates (empty in this case). Note that the result is not a matrix, but a 3D tensor containing the value of `A**k` for each step. We want the last value (after `k` steps) so we compile a function to return just that. Note that there is a rewrite that at compile time will detect that you are using just the last value of the result and ensure that `scan` does not store all the intermediate values that are used. So do not worry if `A` and `k` are large.
# 
# ---
# 
# So, we need to account for:
# 
# 1. The initial value assigned to the result
# 2. The accumulation of results
# 3. The non-sequence values required by the calculation in the loop 
# 
# Scan returns a tuple containing our result (`err`) and a dictionary of updates, which we do not need so it is assigned to the throwaway variable `_`.

# %%
with arma_model:
    y_ = pm.ConstantData("y", y, dims="months")

    # intial error
    err0 = y_[0] - (mu + rho * mu)

    # function to calculate next error
    def calc_next(last_y, this_y, err, mu, rho, theta):
        nu_t = mu + rho * last_y + theta * err
        return this_y - nu_t

    # pytensor for loop over errors
    err, _ = pt.scan(
        fn=calc_next,
        sequences=dict(input=y_, taps=[-1, 0]),
        outputs_info=[err0],
        non_sequences=[mu, rho, theta],
    )

    # predictions
    pred = pm.Deterministic("pred", err + y_[1:], dims="months_m_1")

    # observation model
    obs = pm.Normal("likelihood", mu=err, sigma=sigma, observed=np.zeros_like(y[1:]))

# %% [markdown]
# Notice that, for convenience, we are modeling the residuals in our likelihood function, hence the observations are all set to zero.

# %%
with arma_model:
    trace = pm.sample(tune=2000, draws=2000, target_accept=0.9)

# %%
az.plot_trace(trace, var_names="~pred");

# %%
def _sample(array, n_samples):
    """Little utility function to sample n_samples with replacement"""
    idx = RNG.choice(np.arange(len(array)), n_samples, replace=True)
    return array[idx]

# %%
fig, ax = plt.subplots(figsize=(10, 5))

posterior_pred = trace.posterior["pred"].stack(sample=("chain", "draw")).T
ax.plot(
    air_passengers.index[1:],
    _sample(posterior_pred, 1000).T * y_max,
    color="C0",
    alpha=0.01,
)

air_passengers.reset_index().plot.scatter(
    x="Month", y="#Passengers", color="C1", ax=ax, label="observed"
)

ax.set_title("Posterior predictive samples");

# %% [markdown]
# The model works quite well, but the use of a for-loop (executed via the `pytensor.scan` function) makes it somewhat slow to sample, and most importantly, it's not easy to interpret (especially the $\rho$ and $\theta$ parameters). One should moreover be concerned about overfitting, due to the general level of flexibility of the model.
# 
# Generally, its preferable to take a **generative** approach to model this kind of data.

# %% [markdown]
# ## A generative airplane model
# 
# Actually, we could model these data with a more generative structure, which is an integral component of the [Bayesian workflow](https://arxiv.org/abs/2011.01808). Indeed, look again at the raw data. You'll see that there's an increasing trend, with multiplicative seasonality (i.e the seasonality increases with time). This means that we can fit a linear trend and add a multiplicative seasonality part to it.
# 
# Let's scale our time values to be between 0 and 1 -- this will make sampling easier and it is more reasonable given the way we think about our data here.

# %%
t = (air_passengers.index - pd.Timestamp("1900-01-01")).total_seconds().to_numpy()
t_min = np.min(t)
t_max = np.max(t)
t = (t - t_min) / (t_max - t_min)

# %% [markdown]
# ### Linear Trend
# 
# A simple starting point for this model is:
# 
# $$\text{Passengers} = \alpha + \beta\ \text{time}$$
# 
# Let's assing some weak priors and see what the prior check looks like.

# %%
with pm.Model(check_bounds=False) as linear:
    # Priors
    # regression betas
    alpha = pm.Normal("alpha", mu=0, sigma=5)
    beta = pm.Normal("beta", mu=0, sigma=5)
    # trend
    trend = pm.Deterministic("trend", alpha + beta * t)
    # observation noise
    sigma = pm.HalfNormal("sigma", sigma=5)

    # likelihood
    pm.Normal("likelihood", mu=trend, sigma=sigma, observed=y)

    # sample
    trace = pm.sample_prior_predictive()

# %%
fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 6))

# Real data
air_passengers.reset_index().plot.scatter(
    x="Month", y="#Passengers", color="C1", alpha=0.8, ax=ax[0]
)
ax[0].set_title("Prior predictive samples")

air_passengers.reset_index().plot.scatter(
    x="Month", y="#Passengers", color="C1", alpha=0.8, ax=ax[1]
)
ax[1].set_title("Prior trend lines")


# Prior predictive observations
ax[0].plot(
    air_passengers.index,
    _sample(trace.prior_predictive["likelihood"].squeeze(), 100).T * y_max,
    color="C0",
    alpha=0.05,
)

# Prior predictive trend
ax[1].plot(
    air_passengers.index,
    _sample(trace.prior["trend"].squeeze(), 100).T * y_max,
    color="C0",
    alpha=0.05,
);

# %% [markdown]
# We can easily improve upon this. The priors are very wide, as we end up with implausible passenger numbers. 
# 
# Let's try setting tighter priors.

# %%
with pm.Model(check_bounds=False) as linear:
    # Priors
    alpha = pm.Normal("alpha", mu=0, sigma=0.5)
    beta = pm.Normal("beta", mu=0, sigma=0.5)
    trend = pm.Deterministic("trend", alpha + beta * t)
    # This changed!
    sigma = pm.HalfNormal("sigma", sigma=0.1)

    # Likelihood
    pm.Normal("likelihood", mu=trend, sigma=sigma, observed=y)

    trace = pm.sample_prior_predictive()

# %%
fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 6))

# Real data
air_passengers.reset_index().plot.scatter(
    x="Month", y="#Passengers", color="C1", alpha=0.8, ax=ax[0]
)
ax[0].set_title("Prior predictive samples")

air_passengers.reset_index().plot.scatter(
    x="Month", y="#Passengers", color="C1", alpha=0.8, ax=ax[1]
)
ax[1].set_title("Prior trend lines")

# Prior predictive observations
ax[0].plot(
    air_passengers.index,
    _sample(trace.prior_predictive["likelihood"].squeeze(), 100).T * y_max,
    color="C0",
    alpha=0.05,
)

# Prior predictive trend
ax[1].plot(
    air_passengers.index,
    _sample(trace.prior["trend"].squeeze(), 100).T * y_max,
    color="C0",
    alpha=0.05,
);

# %% [markdown]
# Looks much better (ignore the negative passenger counts for now). Let's proceed.

# %%
with linear:
    posterior_trace = pm.sample()
    posterior_pred_trace = pm.sample_posterior_predictive(trace=posterior_trace)
trace.extend(posterior_trace)
trace.extend(posterior_pred_trace)

# %%
az.plot_trace(trace, var_names="~trend");

# %%
fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 6))

posterior_pred_like = (
    trace.posterior_predictive["likelihood"].stack(sample=("draw", "chain")).T
)

# Real Data
air_passengers.reset_index().plot.scatter(
    x="Month", y="#Passengers", color="C1", alpha=0.8, ax=ax[0]
)
ax[0].set_title("Posterior predictive samples")


air_passengers.reset_index().plot.scatter(
    x="Month", y="#Passengers", color="C1", alpha=0.8, ax=ax[1]
)
ax[1].set_title("Posterior trend lines")

# Posterior Predictive observations
ax[0].plot(
    air_passengers.index,
    _sample(posterior_pred_like, 100).T * y_max,
    color="C0",
    alpha=0.01,
)

# Posterior Predictive trend
posterior_trend = trace.posterior["trend"].stack(sample=("draw", "chain")).T
ax[1].plot(
    air_passengers.index,
    _sample(posterior_trend, 100).T * y_max,
    color="C0",
    alpha=0.01,
);

# %% [markdown]
# Not a bad start; the model does pick up the upward trend, but it underestimates the number of passengers late in the time series.
# 
# This is when multiplicative seasonality starts to manifest itself, as evidenced by the increasing amplitude of the oscillations? This model won't yet be able to deal with this multiplicative part.
# 
# ### Seasonality
# 
# To model seasonality, we'll use the same approach as in the popular Prophet model (see [the paper](https://peerj.com/preprints/3190/) for details). The idea is to make a matrix of [Fourier features](https://en.wikipedia.org/wiki/Fourier_series) which get multiplied by a vector of coefficients. 
# 
# As we're modeling multiplicative seasonality, the final model will be:
# 
# $$\text{Passengers} = (\alpha + \beta\ \text{time}) (1 + \text{seasonality})$$

# %%
n_order = 10
periods = air_passengers.index.dayofyear / 365.25
fourier_features = pd.DataFrame(
    {
        f"{func}_order_{order}": getattr(np, func)(2 * np.pi * periods * order)
        for order in range(1, n_order + 1)
        for func in ("sin", "cos")
    }
)

fourier_features

# %% [markdown]
# Let's take a look at these Fourier features before we put them to use.

# %%
_, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(8, 6))

fourier_features.iloc[:, 0:2].plot(ax=axs[0]).legend(loc=(1, 0.5))
fourier_features.iloc[:, 2:4].plot(ax=axs[1]).legend(loc=(1, 0.5));

# %% [markdown]
# 
# The Fourier features represent periodic oscillations with increasing frequencies. Observe that the sine and cosine features are out-of-phase versions of each other. The key property that makes Fourier features powerful is that any linear combination of these features will be periodic. The period of the resulting function is determined by the lowest frequency used in the feature set.
# 
# Let's examine how the summation of all the Fourier features appears.As you can see, the Fourier features are periodic oscilations of increasing frequency. We can also see that the sine and cosine features are phased out versions of each other. The property that make Fourier features so powerful is that any linear combination of Fourier features will be periodic, with a period that is given by the lowest frequency used by the features. Let's take a look at how summing all the Fourier features looks like:

# %%
fourier_features.sum(axis=1).plot();

# %% [markdown]
# For our particular use case, we will try to infer the weight of the linear combination of Fourier features in order to best fit the periodic pattern that is observed in the data.

# %% [markdown]
# Here we will use our past experience and specify some better priors.

# %%
coords = {"fourier_features": np.arange(2 * n_order)}

with pm.Model(check_bounds=False, coords=coords) as linear_with_seasonality:

    # Priors
    alpha = pm.Normal("alpha", mu=0, sigma=0.5)
    beta = pm.Normal("beta", mu=0, sigma=0.5)
    trend = pm.Deterministic("trend", alpha + beta * t)

    β_fourier = pm.Normal("β_fourier", mu=0, sigma=0.1, dims="fourier_features")
    seasonality = pm.Deterministic(
        "seasonality", pm.math.dot(β_fourier, fourier_features.to_numpy().T)
    )

    μ = trend * (1 + seasonality)
    sigma = pm.HalfNormal("sigma", sigma=0.1)

    # Likelihood
    pm.Normal("likelihood", mu=μ, sigma=sigma, observed=y)

    # Sample prior predictive
    trace = pm.sample_prior_predictive()

# %%
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(12, 9))

# Prior predictive observations
ax[0].plot(
    air_passengers.index,
    _sample(trace.prior_predictive["likelihood"].squeeze(), 100).T * y_max,
    color="C0",
    alpha=0.05,
)
air_passengers.reset_index().plot.scatter(
    x="Month", y="#Passengers", color="C1", alpha=0.8, ax=ax[0]
)
ax[0].set_title("Prior predictive samples")

# Prior predictive trend
ax[1].plot(
    air_passengers.index,
    _sample(trace.prior["trend"].squeeze(), 100).T * y_max,
    color="C0",
    alpha=0.05,
)
air_passengers.reset_index().plot.scatter(
    x="Month", y="#Passengers", color="C1", alpha=0.8, ax=ax[1]
)
ax[1].set_title("Prior trend lines")


# Prior predictive seasonality
ax[2].plot(
    air_passengers.index[:12],
    _sample(trace.prior["seasonality"].squeeze()[:, :12], 100).T * 100,
    color="C0",
    alpha=0.05,
)
ax[2].set_title("Prior seasonality")
ax[2].set_ylabel("Percent change")
formatter = mdates.DateFormatter("%b")
ax[2].xaxis.set_major_formatter(formatter);

# %% [markdown]
# Prior checks don't reveal any serious issues.

# %%
with linear_with_seasonality:
    posterior = pm.sample()
    posterior_predictive = pm.sample_posterior_predictive(trace=posterior)
trace.extend(posterior)
trace.extend(posterior_predictive)

# %%
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(12, 9))

posterior_pred_like = (
    trace.posterior_predictive["likelihood"].stack(sample=("draw", "chain")).T
)

# Posterior Predictive Observations
ax[0].plot(
    air_passengers.index,
    _sample(posterior_pred_like, 100).T * y_max,
    color="C0",
    alpha=0.05,
)

# Real data
air_passengers.reset_index().plot.scatter(
    x="Month", y="#Passengers", color="C1", alpha=0.8, ax=ax[0]
)
ax[0].set_title("Posterior predictive samples")

# Posterior Predictive Trend
posterior_trend = trace.posterior["trend"].stack(sample=("draw", "chain")).T
ax[1].plot(
    air_passengers.index,
    _sample(posterior_trend, 100).T * y_max,
    color="C0",
    alpha=0.05,
)

# Real data
air_passengers.reset_index().plot.scatter(
    x="Month", y="#Passengers", color="C1", alpha=0.8, ax=ax[1]
)
ax[1].set_title("Posterior trend lines")

# Posterior Predictive Seasonality
posterior_seasonality = trace.posterior["seasonality"].stack(sample=("draw", "chain")).T
ax[2].plot(
    air_passengers.index[:12],
    _sample(posterior_seasonality[:, :12], 100).T * 100,
    color="C0",
    alpha=0.05,
)
ax[2].set_title("Posterior seasonality")
ax[2].set_ylabel("Percent change")
formatter = mdates.DateFormatter("%b")
ax[2].xaxis.set_major_formatter(formatter);

# %% [markdown]
# 
# The **Trend + Season** model offers superior interpretability compared to the **ARMA** model when making forecasting decisions. Although the **ARMA** model effectively tracks the flight data, the **Trend + Season** model provides a more transparent and well-reasoned approach for forecasting future demand.
# 
# For instance, when projecting demand for a specific month like July in the following year, the **Trend + Season** model allows for more informed judgments by explicitly accounting for seasonal patterns and long-term trends. This interpretability advantage enables more reliable and justifiable forecasts, making the **Trend + Season** model a preferable choice for decision-making processes that require clear explanations and rationales.
# 
# We can reason that: 
# 
# 1. The long term trend towards increasing passenger numbers will be continues
# 2. July is a busy month so it makes sense to increase our demand expectations due to the specific seasonal of July
# 

# %% [markdown]
# ## Bayesian Structural Time Series Models

# %% [markdown]
# Another approach to time series modeling involves the use of **state-space models**, which has its origins in control engineering. For example, in navigation systems one requires continuous updating of a user's position, based on noisy data. This is analogous to what time series models try to do: make inferences about a *latent state*, based on a sequence of data. 
# 
# In this context, they are known as **structural time series models**. They are generally more transparent than ARIMA-type models because it is not based on autoregression or moving averages, which are not intuitive. Moreover they are flexible and modular, making them widely-applicable to a variety of settings.
# 
# The modularity of structural time series models is their key feature. Specifically, they are comprised of an **observation equation** that specifies how the data are related to the unobserved state, and a **state dynamics equation**, which describes how the latent state evolves over time.
# 
# ### Observation equation
# 
# $$\Large y_t = \mu_t + \epsilon_t$$
# 
# The observation equation relates the observed data with the concurrent value of the unobserved state $\mu_t$. The observation error is typially assumed to be Gaussian:
# 
# $$\Large \epsilon_t \sim N(0, \sigma_{\epsilon})$$
# 
# *More generally* (and used below), we can allow the Observation equation to depend on $\mu$, but in a distributional sense only, 
# 
# $$\Large y_t \sim P(\mu_t, ...) $$
# 
# Observation $y_t$ here depends on $\mu_t$ (and potentially other parameters), via distribution $P$.
# 
# ### State dynamics equation
# 
# $$\Large \mu_{t+1} = \mu_t + \beta X_t + S_t + \eta_t$$
# 
# The state dynamics equation models the temporal dynamics of the baseline mean $\mu_t$ and is sometimes called the **unobserved trend**, since we never observe $\mu$ (though it is typically what we want to infer). Thus, we are assuming that the state is somehow changing over time.
# 
# This regession component optionally models the influence of a set of predictor variables $X_t$, as well as a seasonality component, $S_t$ on an observed time series of data $\{y_t\}$.
# 
# Analogous to the observation error, we typically assume the system errors $\eta_t$ are drawn from some random, zero-centered distribution:
# 
# $$\Large \eta_t \sim N(0, \sigma_{\eta})$$
# 
# Additionally, we assume $\epsilon_t$ and $\eta_t$ are uncorrelated.
# 
# ![state space model](images/state_space.png)
# 
# This modular structure allows the uncertainty in constituent components to be handled separately. Yet, using a Bayesian approach for inference, it allows all components to be estimated **simultaneously**. All estimated quantities will have posterior distributions that can be used for inference.

# %% [markdown]
# ### Example: Snowshoe hare population dynamics
# 
# We can use structural time series modeling to create a phenomenological model of Snowshoe hare (*Lepus americanus*) data. We will use a dataset consisting of 7 years of regular counts as our time series, modeling the latent population and the observation process simultaneously.

# %%
hare_data = pd.read_csv("../data/hare-data-kluane.csv", parse_dates=["date"])
hare_data

# %%
plt.scatter(hare_data.date, hare_data["# Indiv"], alpha=0.6)
plt.ylabel("# observed hares");

# %% [markdown]
# We are going to use the following model to account for the Snowshoe Hare sightings. 
# 
# **Observation and state dynamics model**
# 
# $$\Large 
# \begin{aligned}
# y &\sim \operatorname{NegativeBinomial}(\mu = \exp(\mu_{\text{state}}), \alpha = \alpha_{\text{obs}}) \\
# \mu_{\text{state}} &\sim \operatorname{AR}(\rho = \rho_{\mu}, \sigma = \sigma_{\text{ar}}, n = 1) \\
# \end{aligned}
# $$
# 
# **Priors**
# 
# $$\Large 
# \begin{aligned}
# \sigma_{\text{ar}} &\sim \operatorname{HalfNormal}(\sigma = 1) \\
# \rho_{\mu} &\sim \operatorname{Normal}(\mu = 0, \sigma = 1) \\
# \alpha_{\text{obs}} &\sim \operatorname{HalfNormal}(\sigma = 1) \\
# \end{aligned}
# $$
# 
# Note, that our **state dynamics model** according to our **AR(1)** process is,
# 
# $$\Large \mu_t = \rho *  \mu_{t-1} + \eta_t $$ 
# 
# 
# We can relate this to our structural time series approach, 
# 
# 
# $$\Large \mu_{t+1} = \mu_t + \beta X_t + S_t + \eta_t$$
# 
# 
# if we set that $X_t = \mu_{t}$ , $\beta = 1 - \rho$, and $S_t = 0$.
# 
# Our **observation model** here takes the form,
# 
# 
# $$\Large y \sim \operatorname{NegativeBinomial}(\mu = \exp(\mu_{\text{state}}), \alpha = \alpha) $$
# 
# 
# in other words, our *state dynamics model* enters into the *observation model* via the $\mu$ parameter of the $\operatorname{NegativeBinomial}$ observation distribution. The *observation model* is moreover affected by the $\alpha$ shape parameter **separately from the state dynamics**. 
# 
# Thus, we specify the unknoqn population of snow-shoe hares as a **latent state**. We observe part of this population as a number of **hare sightings** which we model via a $\operatorname{NegativeBinomial}$ distribution...

# %%
with pm.Model(coords={"timesteps": hare_data.date.values}) as hare_model:
    # Priors
    sigma_ar = pm.HalfNormal("sigma_ar", 1)
    rho = pm.Normal("rho", 1, sigma=1)
    mu = pm.AR(
        "mu", rho, sigma=sigma_ar, init_dist=pm.Normal.dist(0, 10), dims="timesteps"
    )  # Poisson rate
    alpha = pm.HalfNormal("alpha", 1)  # gamma param

    # Likelihood
    obs = pm.NegativeBinomial(
        "obs",
        mu=pm.math.exp(mu),
        alpha=alpha,
        observed=hare_data["# Indiv"].values,
        dims="timesteps",
    )

    trace = pm.sample()

# %%
az.plot_trace(trace, var_names="~μ");

# %%
post = trace.posterior.stack(sample=("chain", "draw"))
mu_mean = np.exp(post["mu"].mean("sample"))
mu_hdi = az.hdi(np.exp(trace.posterior))["mu"]

fig, ax = plt.subplots()
ax.plot(post.timesteps, mu_mean, color="black", label="Post. mean")
ax.fill_between(
    post.timesteps,
    mu_hdi.sel(hdi="lower"),
    mu_hdi.sel(hdi="higher"),
    color="C0",
    alpha=0.8,
    label="94% HDI",
)
ax.set(ylabel="# hares", title="Inferred mean latent population of showshoe hare")
plt.legend();

# %% [markdown]
# Let's evaluate the model's performance using a posterior predictive check.

# %%
with hare_model:
    post_pred_trace = pm.sample_posterior_predictive(trace)
trace.extend(post_pred_trace)

# %%
fig, ax = plt.subplots()
post_pred_obs = trace.posterior_predictive["obs"].stack(sample=("draw", "chain")).T
ax.plot(post.timesteps, _sample(post_pred_obs, 500).T, color="C0", alpha=0.01)
ax.scatter(
    hare_data.date,
    hare_data["# Indiv"],
    color="C1",
    alpha=0.5,
    label="Observation",
    zorder=2,
)
ax.set(ylabel="# observed hares", title="PPC of observed showshoe hare")
plt.legend();

# %% [markdown]
# ## Exercise: Demand forecasting and inventory optimization
# 
# Stored in `data/demand.csv`, the data have four columns:
# - `date` - Month the sale was made in. There are no holiday effects or store closures.
# - `store` - Store ID
# - `item` - Item ID
# - `sales` - Number of items sold at a particular store on a particular date.

# %%
data_dir = pathlib.Path("..") / "data"
data_dir
df = pd.read_csv(data_dir / "demand.csv", parse_dates=["date"], index_col=0)
df.head()

# %%
sns.relplot(
    data=df,
    x="date",
    y="sales",
    hue="item",
    col="store",
    kind="line",
    col_wrap=3,
);

# %% [markdown]
# Here, we define a function that defines each time points to the start of the time series. Feel free to explore the different variables defined to understand them. That will help you when modeling.

# %%
def date_to_timeindex(dates):
    return (dates.year - 2013 + (dates.month - 1) / 12).values

store_idx, stores = df.store.factorize(sort=True)
item_idx, items = df.item.factorize(sort=True)
date_idx, date = df.date.factorize(sort=True)
t = date_to_timeindex(date)
COORDS = {
    "obs": df.index,
    "store": stores,
    "item": items,
    "date": date,
}

# %% [markdown]
# The objective is to forecast the monthly demand for the next year (2018) and plan the optimum item inventory for each store accordingly.
# 
# Here is a sequence of tasks towards that overall goal, each task being slightly harder than the last one.
# 
# ### Modeling and forecasting
# 
# 1. Our goal is to model and ultimately predict sales of each item in each store. Look at the `sales` column of our dataframe. Which likelihood distribution do you think you can use? [Easy]
# 2. Once you have your likelihood, write a first simple model that doesn't differentiate between items not stores. It should be of the form `baseline + trend * time`. Don't forget to do some prior / posterior predictive checks before / after sampling. [Easy]
# 3. Write down a second model, very similar your first one, but this time differentiating between items and stores. Again, use prior and posterior predictive checks to give you hints as whether your model goes in the right direction. [Easy]
# 4. What do you think is missing from this model? [Easy]
# 5. Add seasonality to your previous model. For simplicty, only differentiate the baseline by store and item, and leave the trend and seasonality component common to all stores and items. _Hint: refer to the lesson about time series modeling if you don't remember how to add seasonality_. [Medium]
# 6. Forecast new sales for each store and item over the following year after the end of the training dataset. _Hint: use `pd.date_range("2018-01", "2019-01", freq="M")` to define the new dates_. [Medium]
# 
# ### Inventory optimization
# 
# 7. Write a loss function that, depending on the demand forecast, inventory, sales price and holding cost, returns the loss you expect to see. _Hint: go back to the Bayesian decision making lesson if you don't remember how to setup such a loss function_. [Medium]
# 8. As we did in the Bayesian decision making lesson, write an objective function to find the optimum inventory that each store should held for each item over the forecasted period, in order to minimize the expected loss. Use the `items_prices_costs.csv` file to get each item's sales price and holding cost. [Hard]
# 9. Do the same thing, but this time adding the constraint that each store can't store more items that what's indicated in `stores_storage_limits.csv`. [Hard]

# %%
# Write your answer here

# %% [markdown]
# ## Exercise: Gaussian Processes for Time Series
# 
# Replicate the analysis of the `air_passengers` dataset using Gaussian Processes. Take advantage of additive and multiplicative kernels!

# %%
# Write your answer here

# %% [markdown]
# ## References
# 
# - Lyle Broemeling [Bayesian Analysis of Time Series](https://www.amazon.com/Bayesian-Analysis-Time-Lyle-Broemeling/dp/1138591521)
# 
# - Tingting Yu,  [Structural Time Series Models](http://oliviayu.github.io/post/2019-03-21-bsts/)

# %%
%load_ext watermark
%watermark -n -u -v -iv -w


