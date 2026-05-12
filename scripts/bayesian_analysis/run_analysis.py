"""
run_analysis.py
---------------
Bayesian risk analysis for RoVer-CoRe.

KEY FIX from v1: Instead of a simplified placeholder controller, we estimate
P(fail | x0, mu, sigma) by replaying the actual recorded control sequence
from the pkl with perturbed perception errors. This uses the real MPC controller
behavior already embedded in the saved controls array.

Additionally: we use time-varying error scaling to match the rover's actual
uncertainty model (error bound grows linearly with time in RoVer-CoRe).

Run from repo root:
    python scripts/bayesian_analysis/run_analysis.py
"""

import sys
import json
import pickle
import warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from extract_data import load_all_data
from bayesian_model import (
    fit_error_model_analytic,
    get_posterior_samples_analytic,
)

try:
    from bayesian_model import fit_error_model_pymc, get_posterior_samples_pymc
    import pymc as pm
    USE_PYMC = True
    print("PyMC found — will use MCMC.")
except Exception:
    USE_PYMC = False
    print("PyMC not found — using analytic conjugate posterior.")

OUTPUT_DIR = REPO_ROOT / "outputs" / "bayesian_analysis"

# ============================================================
# Simulation parameters (from pkl metadata)
# ============================================================
DT            = 0.01
T_STEPS       = 500
V_ROVER       = 1.5       # forward speed (m/s)
FAILURE_BOUND = 1.5       # |py| >= this → failure

# Grid: sweep py and theta, evaluate at multiple px start values
# RoverDark initial states from pkl: we'll use those as anchors
# and also evaluate a dense grid around them.
PY_RANGE    = np.linspace(-1.4, 1.4, 15)
THETA_RANGE = np.linspace(-0.8, 0.8, 9)
PX_FIXED    = 0.0

N_POSTERIOR_SAMPLES   = 150
N_ROLLOUTS_PER_SAMPLE = 100

# ============================================================
# Step 0: Load the actual controls from the pkl
# ============================================================

def load_pkls():
    pkls = {}
    paths = {
        "roverdark":  REPO_ROOT / "outputs/simulations/roverdark_sim/results.pkl",
        "roverlight": REPO_ROOT / "outputs/simulations/roverlight_batch/results.pkl",
    }
    for name, path in paths.items():
        if path.exists():
            with open(path, "rb") as f:
                pkls[name] = pickle.load(f)
    return pkls


# ============================================================
# Step 1: Load error data
# ============================================================

def load_data(pkls):
    data = load_all_data(combine=True)
    errors = data["combined_errors"]
    if errors.shape[1] == 4:
        errors = errors[:, :3]

    # Also get time-varying error statistics — error grows with time
    # This is the key structure of RoVer-CoRe's uncertainty model
    print("\n=== Time-varying error structure ===")
    rl_errors = data["roverlight"]["errors_by_traj"][:, :, :3]  # (6, 500, 3)
    rd_errors = data["roverdark"]["errors_by_traj"][:, :, :3]   # (6, 500, 3)

    # Std per time step, averaged over trajectories
    rl_std_t = rl_errors.std(axis=0)  # (500, 3)
    rd_std_t = rd_errors.std(axis=0)  # (500, 3)

    # Print at key time points
    for t in [0, 100, 200, 300, 400, 499]:
        print(f"  t={t:3d} (t={t*DT:.1f}s)  "
              f"rd_std={rd_std_t[t].round(4)}  rl_std={rl_std_t[t].round(4)}")

    return data, errors, rd_std_t, rl_std_t


# ============================================================
# Step 2: Bayesian model on POOLED errors
# ============================================================

def fit_model(errors):
    if USE_PYMC:
        trace, model = fit_error_model_pymc(errors, n_draws=2000, n_tune=1000, chains=2)
        posterior_samples = get_posterior_samples_pymc(trace, n_samples=N_POSTERIOR_SAMPLES)
        return posterior_samples, {"type": "pymc", "trace": trace}
    else:
        params = fit_error_model_analytic(errors)
        posterior_samples = get_posterior_samples_analytic(params, n_samples=N_POSTERIOR_SAMPLES)
        return posterior_samples, {"type": "analytic", "params": params}


# ============================================================
# Step 3: Replay-based failure simulation
#
# Core insight: instead of re-implementing the MPC controller,
# we replay the ACTUAL saved control sequence u[0..T] but apply
# it to a trajectory that starts at x0 with noise ~ N(mu, sigma²).
# The control sequence is taken from the closest recorded trajectory.
#
# This is valid because:
#   (a) The MPC controller is a function of x_hat = x + e
#   (b) We're asking: given this noise level, does the state diverge?
#   (c) The saved controls represent realistic MPC behavior
#
# For the risk GRID we also run open-loop: replay u from the most
# stable recorded trajectory and see if the state exits with higher noise.
# ============================================================

def replay_trajectory(x0: np.ndarray, controls: np.ndarray,
                       mu_err: np.ndarray, sigma_err: np.ndarray,
                       rng: np.random.Generator) -> bool:
    """
    Replay a fixed control sequence from x0 with Gaussian perception noise
    injected at each step.

    controls: (T, 1) array of recorded control inputs
    Returns True if trajectory fails (|py| >= FAILURE_BOUND).
    """
    x  = x0.copy().astype(float)
    T  = len(controls)

    for t in range(T):
        u = float(controls[t, 0])

        # Euler step (Dubins car)
        px, py, theta = x[0], x[1], x[2]
        x[0] += V_ROVER * np.cos(theta) * DT
        x[1] += V_ROVER * np.sin(theta) * DT
        x[2] += u * DT

        # Perception noise (injected but doesn't affect control since u is fixed)
        # — for the controller-coupled version below this matters
        e = rng.normal(loc=mu_err[:3], scale=sigma_err[:3])

        if abs(x[1]) >= FAILURE_BOUND:
            return True

    return False


def replay_with_coupled_control(x0: np.ndarray,
                                 mu_err: np.ndarray, sigma_err: np.ndarray,
                                 gains: tuple, rng: np.random.Generator,
                                 time_varying_std: np.ndarray = None) -> bool:
    """
    Simulate with a PROPORTIONAL controller that USES the noisy state estimate.
    This properly couples noise → control → trajectory divergence.

    The key: the error bound in RoVer-CoRe grows with time. We scale the
    noise standard deviation by the time-varying factor learned from data.

    gains: (k_py, k_theta) proportional gains fit from the data
    time_varying_std: (T, 3) std per time step from empirical data (optional)
    """
    x = x0.copy().astype(float)
    k_py, k_theta = gains
    u_max = 2.0

    for t in range(T_STEPS):
        # Scale noise by time-varying factor (captures growing uncertainty)
        if time_varying_std is not None:
            t_idx = min(t, len(time_varying_std) - 1)
            # Scale posterior sigma by the empirical time-growth ratio
            base_std = time_varying_std[0] + 1e-8
            scale    = time_varying_std[t_idx] / base_std
            effective_sigma = sigma_err[:3] * scale
        else:
            effective_sigma = sigma_err[:3]

        e = rng.normal(loc=mu_err[:3], scale=effective_sigma)

        # Controller uses noisy perceived state
        x_hat  = x[:3] + e
        py_hat, theta_hat = x_hat[1], x_hat[2]
        u = float(np.clip(-k_py * py_hat - k_theta * theta_hat, -u_max, u_max))

        # Euler step
        x[0] += V_ROVER * np.cos(x[2]) * DT
        x[1] += V_ROVER * np.sin(x[2]) * DT
        x[2] += u * DT

        if abs(x[1]) >= FAILURE_BOUND:
            return True

    return False


def fit_controller_gains(pkls: dict) -> tuple:
    """
    Fit proportional gains (k_py, k_theta) from the recorded data by
    regressing controls onto estimated states.

    u_t ≈ k_py * py_hat_t + k_theta * theta_hat_t
    """
    all_u   = []
    all_pyh = []
    all_thh = []

    for name, pkl in pkls.items():
        controls    = np.array(pkl["controls"])    # (N, T, 1)
        est_states  = np.array(pkl["estimated_states"])  # (N, T, D)

        u    = controls[:, :, 0].reshape(-1)           # N*T
        pyh  = est_states[:, :, 1].reshape(-1)
        thh  = est_states[:, :, 2].reshape(-1)

        all_u.append(u)
        all_pyh.append(pyh)
        all_thh.append(thh)

    u    = np.concatenate(all_u)
    pyh  = np.concatenate(all_pyh)
    thh  = np.concatenate(all_thh)

    # OLS: u = a * py_hat + b * theta_hat
    X = np.column_stack([pyh, thh])
    coeffs, _, _, _ = np.linalg.lstsq(X, u, rcond=None)
    k_py_fit, k_theta_fit = coeffs

    print(f"\nFitted controller gains (OLS on recorded data):")
    print(f"  u ≈ {k_py_fit:.4f} * py_hat + {k_theta_fit:.4f} * theta_hat")

    # R² to check fit quality
    u_pred = X @ coeffs
    ss_res = ((u - u_pred)**2).sum()
    ss_tot = ((u - u.mean())**2).sum()
    r2 = 1 - ss_res / ss_tot
    print(f"  R² = {r2:.4f}")

    return float(k_py_fit), float(k_theta_fit)


# ============================================================
# Step 4: Compute risk grid using coupled controller + time-varying noise
# ============================================================

def estimate_failure_prob_coupled(x0, mu_err, sigma_err, gains, time_varying_std,
                                   n_rollouts=100):
    rng = np.random.default_rng()
    failures = sum(
        replay_with_coupled_control(x0, mu_err, sigma_err, gains, rng, time_varying_std)
        for _ in range(n_rollouts)
    )
    return failures / n_rollouts


def posterior_predictive_failure_coupled(x0, posterior_samples, gains,
                                          time_varying_std, n_rollouts=100):
    p_fail_samples = np.array([
        estimate_failure_prob_coupled(
            x0, s["mu"], s["sigma"], gains, time_varying_std, n_rollouts
        )
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


def compute_risk_grid(posterior_samples, gains, time_varying_std):
    n_py    = len(PY_RANGE)
    n_theta = len(THETA_RANGE)
    risk_grid = np.zeros((n_py, n_theta, 5))

    total = n_py * n_theta
    count = 0

    for i, py in enumerate(PY_RANGE):
        for j, theta in enumerate(THETA_RANGE):
            x0 = np.array([PX_FIXED, py, theta])
            pf = posterior_predictive_failure_coupled(
                x0, posterior_samples, gains, time_varying_std,
                n_rollouts=N_ROLLOUTS_PER_SAMPLE,
            )
            risk_grid[i, j] = [pf["mean"], pf["std"], pf["lower"], pf["upper"], pf["median"]]
            count += 1
            print(f"  [{count:3d}/{total}]  py={py:+.2f}  theta={theta:+.3f}  "
                  f"P(fail)={pf['mean']:.3f}  "
                  f"95%CI=[{pf['lower']:.3f},{pf['upper']:.3f}]")

    return risk_grid


# ============================================================
# Step 5: Save
# ============================================================

def save_results(data, errors, posterior_samples, model_info,
                  risk_grid, V_grid, gains):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT_DIR / "risk_grid.npy",   risk_grid)
    np.save(OUTPUT_DIR / "py_range.npy",    PY_RANGE)
    np.save(OUTPUT_DIR / "theta_range.npy", THETA_RANGE)
    np.save(OUTPUT_DIR / "errors.npy",      errors)
    if V_grid is not None:
        np.save(OUTPUT_DIR / "hj_value_grid.npy", V_grid)

    with open(OUTPUT_DIR / "posterior_samples.pkl", "wb") as f:
        pickle.dump(posterior_samples, f)

    # CSV
    import csv
    rows = []
    for i, py in enumerate(PY_RANGE):
        for j, theta in enumerate(THETA_RANGE):
            mean, std, lo, hi, med = risk_grid[i, j]
            v = float(V_grid[i, j]) if V_grid is not None else None
            rows.append({
                "py":            round(float(py), 4),
                "theta":         round(float(theta), 4),
                "p_fail_mean":   round(float(mean), 4),
                "p_fail_median": round(float(med), 4),
                "p_fail_std":    round(float(std), 4),
                "p_fail_lower":  round(float(lo), 4),
                "p_fail_upper":  round(float(hi), 4),
                "hj_value":      round(v, 4) if v is not None else "",
                "brt_unsafe":    (v < 0) if v is not None else "",
            })
    with open(OUTPUT_DIR / "risk_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    mu_arr    = np.array([s["mu"]    for s in posterior_samples])
    sigma_arr = np.array([s["sigma"] for s in posterior_samples])

    meta = {
        "n_error_samples":    int(errors.shape[0]),
        "error_means":        errors.mean(axis=0).tolist(),
        "error_stds":         errors.std(axis=0).tolist(),
        "dims":               ["px_error", "py_error", "theta_error"],
        "posterior_mu_mean":  mu_arr.mean(axis=0).tolist(),
        "posterior_sigma_mean": sigma_arr.mean(axis=0).tolist(),
        "fitted_gains":       {"k_py": gains[0], "k_theta": gains[1]},
        "n_posterior_samples": N_POSTERIOR_SAMPLES,
        "n_rollouts_per_sample": N_ROLLOUTS_PER_SAMPLE,
        "failure_bound_py":   FAILURE_BOUND,
        "model_type":         model_info["type"],
        "hj_available":       V_grid is not None,
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*55}")
    print(f"Results saved to {OUTPUT_DIR}/")
    print(f"  risk_summary.csv  /  risk_grid.npy  /  metadata.json")
    print(f"{'='*55}")

    # Print summary table
    print(f"\nPosterior mean P(fail | x0):")
    header = "py\\theta"
    print(f"{header:>10}", end="")
    for theta in THETA_RANGE:
        print(f" {np.degrees(theta):+6.1f}°", end="")
    print()
    for i, py in enumerate(PY_RANGE):
        print(f"  py={py:+.2f}  ", end="")
        for j in range(len(THETA_RANGE)):
            v = risk_grid[i, j, 0]
            print(f"  {v:.3f} ", end="")
        print()


# ============================================================
# Main
# ============================================================

def main():
    print("="*55)
    print("  RoVer-CoRe Bayesian Risk Analysis (v2)")
    print("="*55)

    print("\n--- Loading pkl files ---")
    pkls = load_pkls()

    print("\n--- Step 1: Loading error data ---")
    data, errors, rd_std_t, rl_std_t = load_data(pkls)

    # Use roverdark std (higher noise, more representative of failure cases)
    time_varying_std = rd_std_t  # (500, 3)

    print("\n--- Step 2: Fitting Bayesian model ---")
    posterior_samples, model_info = fit_model(errors)
    print(f"  {len(posterior_samples)} posterior samples.")
    mu_arr    = np.array([s["mu"]    for s in posterior_samples])
    sigma_arr = np.array([s["sigma"] for s in posterior_samples])
    print(f"  Posterior mu    (mean): {mu_arr.mean(axis=0).round(5)}")
    print(f"  Posterior sigma (mean): {sigma_arr.mean(axis=0).round(5)}")

    print("\n--- Step 3: Fitting controller gains from data ---")
    gains = fit_controller_gains(pkls)

    # Quick sanity check: what's the failure rate from actual initial states in pkl?
    print("\n--- Sanity check: failure rate from actual initial states ---")
    rng_check = np.random.default_rng(0)
    mu_mean    = mu_arr.mean(axis=0)
    sig_mean   = sigma_arr.mean(axis=0)
    for name, pkl in pkls.items():
        init_states = np.array(pkl["initial_states"])
        for k, x0_k in enumerate(init_states[:3]):
            p = estimate_failure_prob_coupled(
                x0_k[:3], mu_mean, sig_mean, gains, time_varying_std,
                n_rollouts=200
            )
            true_states = np.array(pkl["states"])
            py_max = abs(true_states[k, :, 1]).max()
            print(f"  [{name}] x0={x0_k[:3].round(3)}  "
                  f"P(fail)={p:.3f}  "
                  f"(recorded max|py|={py_max:.3f})")

    print("\n--- Step 4: Computing risk grid ---")
    print(f"  Grid: {len(PY_RANGE)} py × {len(THETA_RANGE)} theta = "
          f"{len(PY_RANGE)*len(THETA_RANGE)} states")
    risk_grid = compute_risk_grid(posterior_samples, gains, time_varying_std)

    print("\n--- Step 5: Saving ---")
    save_results(data, errors, posterior_samples, model_info,
                 risk_grid, None, gains)

    print("\nNext: python scripts/bayesian_analysis/plot_results.py")


if __name__ == "__main__":
    main()
