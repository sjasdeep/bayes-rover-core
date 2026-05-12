# Bayes RoVer-CoRe

A probabilistic safety analysis framework that uses Bayesian inference and Monte Carlo simulation to quantify conservatism in formal verification of perception-based autonomous controllers.

## Paper

[[Bayesian Risk Analysis of Perception-Based Controllers via RoVer-CoRe]](Bayesian_Risk_Analysis_of_Perception_Based_Controllers_via_RoVer_CoRe_Preprint__DRAFT__.pdf)

## Setup

This project builds on the official [RoVer-CoRe repository](https://github.com/albertklin/rover-core). Please follow the installation and setup instructions in that repository to generate the simulation data used in this analysis.

## Reproducing Results

After completing the RoVer-CoRe setup and generating the required simulation outputs, run the following commands from the root of this repository:

```bash
python scripts/bayesian_analysis/extract_data.py
python scripts/bayesian_analysis/run_analysis.py
python scripts/bayesian_analysis/plot_results.py
```

These scripts will:
1. Extract perception error data from the RoVer-CoRe simulation logs.
2. Fit Bayesian models using Markov Chain Monte Carlo (MCMC).
3. Generate posterior summaries, posterior predictive checks, and risk visualizations.


## Project Status
This project is an active research effort and remains under development. Results, scripts, and documentation may change as the methodology is refined.

