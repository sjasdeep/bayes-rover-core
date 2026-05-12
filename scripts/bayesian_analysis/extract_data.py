"""
extract_data.py
---------------
Loads RoVer-CoRe simulation outputs using the exact pkl structure:

    states:           (N_traj, T+1, state_dim)  — true states x
    estimated_states: (N_traj, T,   state_dim)  — perceived states x_hat
    uncertainties:    (N_traj, T,   state_dim)  — perception error e = x_hat - x
    initial_states:   (N_traj, state_dim)

RoverDark:  state_dim=3, dims = (px, py, theta)
RoverLight: state_dim=4, dims = (px, py, theta, s)  — s is light state (ignored for error modeling)

Usage:
    python scripts/bayesian_analysis/extract_data.py
    -- or --
    from scripts.bayesian_analysis.extract_data import load_all_data
"""

import pickle
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

SIM_PATHS = {
    "roverdark":  REPO_ROOT / "outputs/simulations/roverdark_sim/results.pkl",
    "roverlight": REPO_ROOT / "outputs/simulations/roverlight_batch/results.pkl",
}

# Failure definition for RoverDark (Dubins car in obstacle field).
# The true failure is encoded in the HJ value function, but for simulation-based
# labeling we use the summary.txt reported outcomes + a heuristic boundary check.
# RoverDark obstacle region: based on paper Fig, the rover starts around px~0-5
# and navigates; failure = leaving navigable region (|py| > threshold or hitting obstacle).
# We use conservative bounds here — the HJ value function is the ground truth.
ROVERDARK_FAILURE_BOUND_PY = 1.5   # |py| >= this → failure (lateral excursion)
ROVERLIGHT_FAILURE_BOUND_PY = 1.5  # same

DIM_NAMES_3 = ["px", "py", "theta"]
DIM_NAMES_4 = ["px", "py", "theta", "s"]


def load_pkl(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_errors_and_labels(pkl: dict, failure_bound_py: float = 1.5,
                               use_dims: list = None) -> dict:
    """
    Extract perception errors and failure labels from a single simulation pkl.

    Parameters
    ----------
    pkl         : loaded pickle dict
    failure_bound_py : |py| >= this at any step → trajectory labeled failed
    use_dims    : which state dimensions to use for error modeling (default: first 3)

    Returns
    -------
    dict with:
        errors          : (N_steps_total, n_dims)  — all error samples pooled across trajs & time
        errors_by_traj  : (N_traj, T, n_dims)
        initial_states  : (N_traj, state_dim)
        failed          : (N_traj,) bool
        states          : (N_traj, T+1, state_dim)
        metadata        : dict
    """
    states      = np.array(pkl["states"],           dtype=float)  # (N, T+1, D)
    est_states  = np.array(pkl["estimated_states"],  dtype=float)  # (N, T,   D)
    uncertainties = np.array(pkl["uncertainties"],   dtype=float)  # (N, T,   D)
    init_states = np.array(pkl["initial_states"],    dtype=float)  # (N, D)

    N, T, D = uncertainties.shape

    # use_dims: for RoverLight (D=4) we drop the light state s (dim 3) for error modeling
    if use_dims is None:
        use_dims = list(range(min(D, 3)))  # use px, py, theta only

    # Perception errors: uncertainties IS e = x_hat - x (verified: identical to est-true)
    errors_full = uncertainties[:, :, use_dims]       # (N, T, n_dims)
    errors_flat = errors_full.reshape(-1, len(use_dims))  # (N*T, n_dims)

    # Failure labels: did py ever exceed the bound?
    py_traj = states[:, :, 1]  # (N, T+1)
    failed  = np.any(np.abs(py_traj) >= failure_bound_py, axis=1)  # (N,)

    print(f"  Loaded: {N} trajectories, {T} steps each, {D}D state, "
          f"using dims {use_dims} = {[DIM_NAMES_4[i] for i in use_dims]}")
    print(f"  Error samples: {errors_flat.shape[0]} total")
    print(f"  Failures: {failed.sum()}/{N} "
          f"(py bound = ±{failure_bound_py})")
    print(f"  Error means: {errors_flat.mean(axis=0).round(5)}")
    print(f"  Error stds:  {errors_flat.std(axis=0).round(5)}")

    return {
        "errors":          errors_flat,
        "errors_by_traj":  errors_full,
        "initial_states":  init_states,
        "failed":          failed,
        "states":          states,
        "est_states":      est_states,
        "use_dims":        use_dims,
        "dim_names":       [DIM_NAMES_4[i] for i in use_dims],
        "metadata": {
            "tag":          pkl.get("tag", ""),
            "system":       pkl.get("system_name", ""),
            "n_traj":       N,
            "T":            T,
            "dt":           pkl.get("dt", None),
            "time_horizon": pkl.get("time_horizon", None),
            "state_dim":    D,
            "use_dims":     use_dims,
        }
    }


def load_all_data(combine: bool = True) -> dict:
    """
    Load both simulation outputs and optionally combine their error samples.

    Returns dict with keys:
        "roverdark"     : extracted data for roverdark_sim
        "roverlight"    : extracted data for roverlight_batch
        "combined_errors": all error samples pooled (N_dark*T + N_light*T, 3)
    """
    results = {}

    for name, path in SIM_PATHS.items():
        if not path.exists():
            print(f"WARNING: {path} not found — skipping {name}")
            continue
        print(f"\n=== Loading {name} ({path.name}) ===")
        pkl = load_pkl(path)
        results[name] = extract_errors_and_labels(pkl)

    if combine and len(results) >= 2:
        # Both have 3 usable dims (px, py, theta) — pool error samples
        all_errors = np.concatenate(
            [results[k]["errors"] for k in results], axis=0
        )
        print(f"\n=== Combined error dataset: {all_errors.shape[0]} samples, "
              f"{all_errors.shape[1]} dims ===")
        print(f"  Combined means: {all_errors.mean(axis=0).round(5)}")
        print(f"  Combined stds:  {all_errors.std(axis=0).round(5)}")
        results["combined_errors"] = all_errors
    elif len(results) == 1:
        key = list(results.keys())[0]
        results["combined_errors"] = results[key]["errors"]

    return results


if __name__ == "__main__":
    data = load_all_data()

    # Extra: print time-varying error statistics (error grows with time per paper)
    print("\n=== Time-varying error statistics (roverlight) ===")
    if "roverlight" in data:
        e_traj = data["roverlight"]["errors_by_traj"]  # (6, 500, 3)
        T = e_traj.shape[1]
        # Print std at t=0, t=T/4, t=T/2, t=3T/4, t=T-1
        for t in [0, T//4, T//2, 3*T//4, T-1]:
            std_t = e_traj[:, t, :].std(axis=0)
            print(f"  t={t:3d}  std(px,py,theta) = {std_t.round(5)}")
