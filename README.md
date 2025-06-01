# Introduction to Bayesian Time Series Analysis with PyMC

## Setup

This course can be set up using either Anaconda/Miniforge or Pixi for Python package management.

### Option 1: Using Anaconda/Miniforge

If you prefer using Anaconda, you'll need [Anaconda](https://www.anaconda.com/products/individual#download-section) Python (with Python version 3.11) installed on your system. We recommend using the [Miniforge](https://github.com/conda-forge/miniforge#download) distribution, which is a lightweight version of Anaconda that is easier to work with.

### Option 2: Using Pixi 

Alternatively, you can use Pixi, a modern package management tool. To install Pixi:

On Linux/macOS:
    curl -fsSL https://pixi.sh/install.sh | bash

On Windows:
    iwr -useb https://pixi.sh/install.ps1 | iex

You may need to restart your terminal after installation.

### Getting the Course Materials

The next step is to clone or download the course materials. If you are familiar with Git, run:

    git clone https://github.com/fonnesbeck/ts_pydata_london_2025.git

otherwise you can [download a zip file](https://github.com/fonnesbeck/ts_pydata_london_2025/archive/main.zip) of its contents, and unzip it on your computer.

### Setting up the Environment

If using Anaconda/Miniforge:
The repository contains an `environment.yml` file with all required packages. Run:

    mamba env create

from the main course directory (use `conda` instead of `mamba` if you installed Anaconda). Then activate the environment:

    mamba activate ts_course
    # or
    conda activate ts_course

If using Pixi:
The repository contains a `pixi.toml` file. From the main course directory, simply run:

    pixi install
    pixi shell

Then, you can start **JupyterLab** to access the materials:

    jupyter lab

For those who like to work in VS Code, you can also run Jupyter notebooks from within VS Code. To do this, you will need to install the [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter). Once this is installed, you can open the notebooks in the `notebooks` subdirectory and run them interactively.

## Course Outline

### Section 1: Introduction (20 minutes)
- Motivation for Bayesian time series analysis
  - Handling uncertainty
  - Incorporating prior knowledge 
  - Flexibility for complex patterns
- Key characteristics of time series data
  - Trend, seasonality, and noise components
  - Stationarity and autocorrelation
- Data preprocessing for time series analysis
  - Handling missing values
  - Normalization/standardization
  - Trend and seasonality decomposition

### Section 2: Bayesian Inference and PyMC (20 minutes)
- Brief review of Bayesian inference concepts
  - Bayes' theorem
  - Prior, likelihood, and posterior distributions
  - Markov Chain Monte Carlo (MCMC) sampling
- Introduction to PyMC
  - API overview
  - Basic PyMC model structure
  - Model fitting and inference

### Section 3: Bayesian Time Series Models (30 minutes)
- Gaussian Random Walks in PyMC
  - Model specification and implementation
  - Parameter interpretation
  - Applications to time-varying parameters
- Bayesian regression for time series
  - Linear trend models
  - Polynomial trends
  - Seasonal components with Fourier terms
- State-Space Models in PyMC
  - Dynamic Linear Models (DLMs)
  - Time-varying parameters using random walks
  - Incorporating external regressors
- Stochastic Volatility Models
  - Modeling changing variance
  - Applications in finance and economics
- Gaussian Process regression for time series
  - Flexible non-parametric approach
  - Kernel selection for time series

### Section 5: Model Evaluation and Selection (10 minutes)
- Posterior predictive checks
- Information criteria (WAIC, LOO)
- Cross-validation for time series
- Forecast evaluation metrics

### Section 6: Forecasting with Bayesian Models (15 minutes)
- Generating predictions
- Obtaining prediction intervals
- Incorporating forecast uncertainty
