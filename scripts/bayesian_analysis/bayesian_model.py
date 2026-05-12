"""
bayesian_model.py
-----------------
Bayesian model for RoVer-CoRe perception error analysis.

Two backends:
  1. PyMC (MCMC via NUTS) — preferred, richer diagnostics
  2. Analytic Normal-InvGamma conjugate — fallback if PyMC unavailable

Model:
    e_i | mu, sigma ~ Normal(mu, sigma²)   [per state dim, independent]
    mu              ~ Normal(0, 0.5)
    sigma           ~ HalfNormal(0.3)

Posterior predictive failure probability:
    p_fail(x0) = E_{(mu,sigma)~posterior} [ P(fail | x0, mu, sigma) ]
    estimated by: for each posterior draw (mu_k, sigma_k),
                  simulate N_rollouts trajectories with e~N(mu_k, sigma_k²),
                  count fraction that fail.
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False


# ============================================================
# Part 1: Bayesian inference
# ============================================================

def fit_error_model_pymc(errors: np.ndarray, n_draws=2000, n_tune=1000,
                          chains=2, target_accept=0.9, random_seed=42):
    """
    MCMC posterior for perception error parameters using PyMC.

    errors: (N, D) array of perception errors
    Returns (trace, model)
    """
    if not PYMC_AVAILABLE:
        raise ImportError("Install PyMC: conda install -c conda-forge pymc arviz")

    N, D = errors.shape
    print(f"\nFitting PyMC model: N={N} error samples, D={D} dims")

    with pm.Model() as model:
        # Priors: weakly informative based on observed error magnitudes (~0.09-0.12)
        mu    = pm.Normal("mu",    mu=0.0, sigma=0.5,  shape=D)
        sigma = pm.HalfNormal("sigma", sigma=0.3,       shape=D)

        # Likelihood
        _ = pm.Normal("obs", mu=mu, sigma=sigma, observed=errors)

        trace = pm.sample(
            draws=n_draws, tune=n_tune, chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=True,
            return_inferencedata=True,
        )

    print("\n=== Posterior Summary ===")
    print(az.summary(trace, var_names=["mu", "sigma"], round_to=5))

    return trace, model


def fit_error_model_analytic(errors: np.ndarray) -> list[dict]:
    """
    Closed-form Normal-InvGamma conjugate posterior (no PyMC needed).

    Prior:  mu | sigma² ~ N(mu0, sigma²/kappa0)
            sigma²      ~ InvGamma(alpha0, beta0)

    Returns list of per-dimension posterior parameter dicts.
    """
    N, D = errors.shape
    print(f"\nFitting analytic conjugate model: N={N} samples, D={D} dims")

    # Weakly informative hyperparameters
    mu0    = 0.0
    kappa0 = 1.0
    alpha0 = 2.0
    beta0  = 0.05   # small → prior allows sigma ~ sqrt(beta/(alpha-1)) ~ 0.22

    dim_labels = ["px", "py", "theta"] + [f"dim{i}" for i in range(3, D)]
    results = []

    for d in range(D):
        e_d   = errors[:, d]
        n     = len(e_d)
        e_bar = e_d.mean()
        SS    = ((e_d - e_bar)**2).sum()   # sum of squared deviations

        # Posterior hyperparameters
        kappa_n = kappa0 + n
        mu_n    = (kappa0*mu0 + n*e_bar) / kappa_n
        alpha_n = alpha0 + n/2
        beta_n  = beta0 + 0.5*SS + (kappa0*n*(e_bar - mu0)**2) / (2*kappa_n)

        post_sigma_mean = np.sqrt(beta_n / (alpha_n - 1)) if alpha_n > 1 else np.nan
        post_mu_std = np.sqrt(beta_n / (alpha_n * kappa_n))

        results.append({
            "dim":             d,
            "label":           dim_labels[d] if d < len(dim_labels) else f"dim{d}",
            "mu_n":            mu_n,
            "mu_std":          post_mu_std,
            "kappa_n":         kappa_n,
            "alpha_n":         alpha_n,
            "beta_n":          beta_n,
            "sigma_post_mean": post_sigma_mean,
        })
        print(f"  [{dim_labels[d] if d < len(dim_labels) else d}]  "
              f"mu = {mu_n:.5f} ± {post_mu_std:.5f},  "
              f"sigma ≈ {post_sigma_mean:.5f}")

    return results


def get_posterior_samples_pymc(trace, n_samples=500) -> list[dict]:
    """Draw (mu, sigma) from PyMC trace."""
    mu_arr    = trace.posterior["mu"].values.reshape(-1, trace.posterior["mu"].values.shape[-1])
    sigma_arr = trace.posterior["sigma"].values.reshape(-1, trace.posterior["sigma"].values.shape[-1])
    idx = np.random.default_rng(0).choice(len(mu_arr), size=min(n_samples, len(mu_arr)), replace=False)
    return [{"mu": mu_arr[i], "sigma": sigma_arr[i]} for i in idx]


def get_posterior_samples_analytic(params: list[dict], n_samples=500,
                                    rng_seed=0) -> list[dict]:
    """Draw (mu, sigma) from analytic Normal-InvGamma posterior."""
    from scipy.stats import invgamma, norm

    rng = np.random.default_rng(rng_seed)
    D = len(params)
    samples = []

    for _ in range(n_samples):
        mu_draw    = np.zeros(D)
        sigma_draw = np.zeros(D)
        for d, p in enumerate(params):
            # Sample sigma² ~ InvGamma(alpha_n, beta_n)
            sigma2 = invgamma.rvs(a=p["alpha_n"], scale=p["beta_n"],
                                   random_state=rng.integers(0, 2**31))
            sigma_draw[d] = np.sqrt(sigma2)
            # Sample mu | sigma² ~ Normal(mu_n, sigma²/kappa_n)
            mu_draw[d] = norm.rvs(loc=p["mu_n"],
                                   scale=sigma_draw[d] / np.sqrt(p["kappa_n"]),
                                   random_state=rng.integers(0, 2**31))
        samples.append({"mu": mu_draw, "sigma": sigma_draw})

    return samples


# ============================================================
# Part 2: RoverDark simulator for failure probability
# ============================================================

def simulate_roverdark(x0: np.ndarray, mu_err: np.ndarray, sigma_err: np.ndarray,
                        T: int = 500, dt: float = 0.01, v: float = 1.5,
                        failure_bound_py: float = 1.5,
                        rng: np.random.Generator = None) -> bool:
    """
    Simulate one RoverDark Dubins trajectory with Gaussian perception noise.

    State: x = (px, py, theta)
    Dynamics (Euler):
        px_dot = v * cos(theta)
        py_dot = v * sin(theta)
        theta_dot = u

    Control: simple proportional feedback on perceived state
        u = -k_py * py_hat - k_theta * theta_hat
        (drives py→0, theta→0, mimicking the MPC behavior qualitatively)

    Failure: |py| >= failure_bound_py at any step.

    Returns True if trajectory fails.

    Note: This is a simplified stand-in for the full MPC controller. For
    posterior predictive comparison vs the BRT, the key quantity is whether
    the Bayesian failure probability tracks the HJ worst-case prediction.
    """
    if rng is None:
        rng = np.random.default_rng()

    x = x0.copy().astype(float)
    k_py    = 1.2
    k_theta = 2.0
    u_max   = 1.5   # control saturation

    for _ in range(T):
        # Sample perception noise
        e = rng.normal(loc=mu_err, scale=sigma_err)

        # Perceived state (only px, py, theta; ignore light dim if present)
        x_hat = x[:3] + e[:3]
        py_hat, theta_hat = x_hat[1], x_hat[2]

        # Control
        u = -k_py * py_hat - k_theta * theta_hat
        u = float(np.clip(u, -u_max, u_max))

        # Euler step
        px, py, theta = float(x[0]), float(x[1]), float(x[2])
        x[0] += v * np.cos(theta) * dt
        x[1] += v * np.sin(theta) * dt
        x[2] += u * dt

        # Check failure
        if abs(x[1]) >= failure_bound_py:
            return True

    return False


def estimate_failure_prob(x0: np.ndarray, mu_err: np.ndarray, sigma_err: np.ndarray,
                           n_rollouts: int = 200, **sim_kwargs) -> float:
    """Monte Carlo estimate of P(fail | x0, mu, sigma)."""
    rng = np.random.default_rng()
    failures = sum(
        simulate_roverdark(x0, mu_err, sigma_err, rng=rng, **sim_kwargs)
        for _ in range(n_rollouts)
    )
    return failures / n_rollouts


def posterior_predictive_failure(x0: np.ndarray,
                                  posterior_samples: list[dict],
                                  n_rollouts_per_sample: int = 80,
                                  **sim_kwargs) -> dict:
    """
    Posterior predictive P(fail | x0) averaged over posterior draws.

    Returns dict: mean, std, lower (2.5%), upper (97.5%), all_samples
    """
    p_fail_samples = np.array([
        estimate_failure_prob(x0, s["mu"], s["sigma"],
                               n_rollouts=n_rollouts_per_sample, **sim_kwargs)
        for s in posterior_samples
    ])
    return {
        "mean":    float(p_fail_samples.mean()),
        "std":     float(p_fail_samples.std()),
        "lower":   float(np.percentile(p_fail_samples, 2.5)),
        "upper":   float(np.percentile(p_fail_samples, 97.5)),
        "median":  float(np.median(p_fail_samples)),
        "samples": p_fail_samples,
    }
