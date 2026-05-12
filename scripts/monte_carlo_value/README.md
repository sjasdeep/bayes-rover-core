# Monte Carlo Value Scripts

Scripts for computing an under-approximate value function V_N(x) via Monte Carlo rollouts on a 2D state slice, inspecting caches, and visualizing zero-level contours.

## Workflow

1. **run.py** – Compute V_N(x) on a 2D slice using random disturbance and uncertainty; save to cache
2. **inspect_cache.py** – List and inspect saved Monte Carlo value caches
3. **visualize.py** – Overlay zero-level contours across snapshot counts (and optional final), or plot the full final value field

## run.py
Computes the Monte Carlo under-approximate value over a 2D slice by simulating trajectories with uniformly sampled disturbance and uncertainty (within system limits) and your chosen control input. No early stopping is used; we take the minimum signed distance over the full horizon and aggregate a min across samples.

Config-driven parameters (slice, grid, dt, samples, snapshots) are read from `config/monte_carlo_value.yaml`.

```bash
# Basic run with a direct control class
python scripts/monte_carlo_value/run.py \
  --system RoverDark \
  --control RoverDark_MPC \
  --tag mc_dubins_mpc

# Using a cache-backed control (e.g., OptimalInputFromValue with a GridValue tag)
python scripts/monte_carlo_value/run.py \
  --system RoverDark \
  --control OptimalInputFromValue \
  --control-tag <grid_value_tag> \
  --tag mc_dubins_optval

# Use a non-default config path or preset
python scripts/monte_carlo_value/run.py \
  --system RoverDark \
  --control RoverDark_MPC \
  --mc-config config/monte_carlo_value.yaml \
  --preset default \
  --tag mc_custom
```

Notes:
- `dt` comes from `config/monte_carlo_value.yaml`; the script adjusts step count to span the system time horizon exactly.
- Disturbance and uncertainty are sampled uniformly each step within the system’s limits.
- Control input is specified like in `scripts/simulation/simulate.py`. For cache-backed controls (`GridInput`, `NNInput`, `OptimalInputFromValue`), pass `--control-tag`.

**Cache output:** `.cache/monte_carlo_values/{TAG}.pkl` (+ `.meta.json`)

## Config file: `config/monte_carlo_value.yaml`
Per-system structure:

```yaml
SystemName:
  default:
    slices:
      vary_dims: [i, j]        # two state indices to vary
      fixed: { k: value, ... } # fixed values for remaining dims
    grid_resolution: [nx, ny]  # grid points along the two varying dims
    dt: 0.05                   # simulation step (seconds)
    total_samples_per_state: 100
    snapshot_samples: [5, 10, 20, 50, 100]
```

- `vary_dims`: which two state components define the 2D slice
- `fixed`: mapping from index → value for all other dimensions
- `grid_resolution`: size of the slice grid
- `dt`: integration time step; horizon comes from the System
- `total_samples_per_state`: samples per grid point
- `snapshot_samples`: cumulative sample counts at which to save intermediate V_N snapshots

## inspect_cache.py
List and inspect Monte Carlo caches saved under `.cache/monte_carlo_values/`.

```bash
# List all caches
python scripts/monte_carlo_value/inspect_cache.py --list

# Inspect a specific tag
python scripts/monte_carlo_value/inspect_cache.py --tag mc_dubins_mpc
```

## visualize.py
Overlay zero-level contours across snapshot counts; optionally show the final contour or the full final value function.

```bash
# Default behavior: save a PNG (no window)
python scripts/monte_carlo_value/visualize.py --tag mc_dubins_mpc

# Interactive display (requires GUI backend). Saving is still enabled by default.
python scripts/monte_carlo_value/visualize.py --tag mc_dubins_mpc --interactive

# Disable saving explicitly (only show interactive window)
python scripts/monte_carlo_value/visualize.py --tag mc_dubins_mpc --interactive --no-save

# Overlay snapshots and the final contour
python scripts/monte_carlo_value/visualize.py --tag mc_dubins_mpc --show-final

# Plot the full final value field (with colorbar) and overlay its zero contour
python scripts/monte_carlo_value/visualize.py --tag mc_dubins_mpc --show-final-field
```

**Visualization output (default save):**
- Zero-contours view: `outputs/visualizations/monte_carlo/{TAG}.png`
- Full-field view: `outputs/visualizations/monte_carlo/{TAG}_field.png`
